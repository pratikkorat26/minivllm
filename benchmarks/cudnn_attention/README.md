# cuDNN attention dispatch — November 4, 2026

This pass adds an **opt-in** switch that gives PyTorch's fused cuDNN
scaled-dot-product-attention kernel priority over the memory-efficient kernel
that the default dispatcher selects for additive masks. It uses resident BF16
LFM3-1.7B weights and KV on one RTX 4070 Laptop 8 GB, with PyTorch compilation
and library kernels. No speculative decoding or handwritten CUDA/Triton kernels
are involved, and no weights are quantized.

The option is off by default because the cuDNN kernel is not bitwise
reproducible at some decode shapes; see [Determinism](#determinism).

## Results

End-to-end streaming runs use the same harness, prompt lengths, batch sizes and
warmup rules as the [overlap record](../overlap_optimization/README.md). Rates
are aggregate output tokens/s over prefill, decode, scheduling and the streaming
harness. "Baseline" is the default backend (`fmha_cutlass` memory-efficient
attention) on commit `e298fec`; "cuDNN" passes `--cudnn-attention`.

| Workload | Baseline | cuDNN attention | Change |
|---|---:|---:|---:|
| 256 requests, 64-token cap, S1 | 5,016 | **5,896** | +17.5% |
| 256 requests, 64-token cap, S8 | 5,250 | **6,206** | +18.2% |
| 128 requests, 512-token cap, S1 | 2,990 | **3,710** | +24.1% |
| 128 requests, 512-token cap, S8 | 3,104 | **3,855** | +24.2% |

Median worker step time at batch 256/S1 fell from 40.0 ms to 33.0 ms. Every run
used a fresh process, the `performance` platform profile, four OMP/MKL threads,
a 2,048-token scheduling budget and `LFM3_FUSED_PAGED_ATTENTION=0`. A first
sweep measured 5,734/6,427/3,732/3,862 tokens/s; repeated sweeps vary by up to
about 3.5% because clocks and temperatures are not locked.

## Why the default dispatcher misses cuDNN

LFM3 builds an additive `[B,1,Q,S]` mask that combines causality, sliding
windows and per-row lengths. With a mask present, PyTorch's priority order
places `FLASH_ATTENTION` before `CUDNN_ATTENTION`, and flash cannot consume that
mask, so the memory-efficient kernel runs. cuDNN accepts the additive mask
directly and is several times faster at the shapes this engine uses.

Measured with `torch.nn.functional.scaled_dot_product_attention` on the same GPU
(BF16, GQA groups 4, head dim 128):

| Shape | Memory-efficient | cuDNN | Speedup |
|---|---:|---:|---:|
| B=256, S=128, decode | 1,075 µs | 331 µs | 3.2× |
| B=256, S=512, decode | 3,740 µs | 1,180 µs | 3.2× |
| B=128, S=1,024, decode | 3,286 µs | 1,169 µs | 2.8× |
| B=1, S=8,192, decode | 438 µs | 23 µs | 19× |
| B=4, Q=S=512, prefill | 492 µs | 308 µs | 1.6× |
| B=8, Q=S=1,024, prefill | 3,517 µs | 2,036 µs | 1.7× |

Prefill outputs at these sizes were bit-identical to the memory-efficient
kernel. In a compiled B=256/64-token wave profile, attention kernels fell from
about 600 ms of `fmha_cutlass` plus 204 ms of mask Triton work to about 150 ms
of cuDNN plus the same 203 ms of mask construction. Recorded CUDA time fell from
3.06 s to 2.58 s.

## Implementation

`lfm3.attention.prefer_cudnn_attention` installs a process-wide backend
priority with `torch.nn.attention._sdpa_kernel(..., set_priority=True)`:

```
CUDNN_ATTENTION, FLASH_ATTENTION, EFFICIENT_ATTENTION, MATH
```

`PagedModelExecutor.__init__` calls it before the first forward, so Dynamo,
Inductor and CUDA graph capture see the cuDNN kernels. The dispatcher still
checks per-shape support and falls back to the remaining backends on GPUs where
cuDNN attention is unavailable.

Enable it with any of:

- `EngineConfig(enable_cudnn_attention=True)`
- server `--enable-cudnn-attention` or `LFM3_API_ENABLE_CUDNN_ATTENTION=true`
- benchmark harness `--cudnn-attention`

The priority is process-wide, so choose the setting before constructing the
first executor; a later `False` cannot undo an installed priority.

### Compile-cache isolation

The backend choice is baked into Inductor's generated graph, but the FX-graph
cache key does not include the SDPA backend priority. An artifact compiled
before this change was therefore reused with the memory-efficient kernel even
after the priority was installed; `torch.compiler.config.cache_key_tag` did not
invalidate it. The helper redirects `TORCHINDUCTOR_CACHE_DIR` to a
`minivllm-cudnn-attention` subdirectory of the configured cache, so cuDNN and
non-cuDNN artifacts never mix. The first opt-in run rebuilds graphs (roughly
2–3 minutes for these workloads); later runs reuse them.

## Determinism

cuDNN reduces in a different order and is not bitwise reproducible at every
shape. Repeating an identical `scaled_dot_product_attention` call on unchanged
buffers, BF16 with a fully valid additive mask:

| Batch | Context | Memory-efficient | cuDNN |
|---:|---:|---:|---:|
| 128 | 128 | 1 unique | 1 unique |
| 128 | 256 | 1 unique | 3 unique |
| 128 | 384 | 1 unique | 2 unique |
| 128 | 512 | 1 unique | 3 unique |
| 128 | 640 | 1 unique | 1 unique |
| 128 | 768 | 1 unique | 2 unique |
| 128 | 1,024 | 1 unique | 4 unique |
| 256 | 128 | 1 unique | 1 unique |
| 1 | 8,192 | 1 unique | 1 unique |

Each cell is the number of distinct bitwise outputs across 10–15 identical
calls. cuDNN's alternative results differ by about one BF16 ulp (0.001 at these
magnitudes) and appear intermittently; `torch.backends.cudnn.deterministic` does
not change the behaviour. The memory-efficient kernel returned one result in
every cell.

That 1-ulp difference can flip a near-tied greedy argmax and change the rest of
the continuation. In the engine, the three measured 512-token waves produced
three different greedy hashes at both S1 and S8; the five 64-token waves matched.
For real workloads with different prompts a flip usually changes one
conversation, not all of them, but temperature=0 is no longer run-to-run
identical. The option is therefore off by default; it does not change the
sampling algorithm itself.

## Numerical validation with the option enabled

The quality and token comparisons below were captured with cuDNN enabled.

**Token divergence.** At batch 256, all 256 requests share one prompt, so the
requests are not independent samples. Comparing a fresh-cache control run
(cuDNN off, which reproduces the recorded `3517e89560ea` hash exactly) with the
cuDNN run (`d32eae370ee6`):

- 254 of 256 rows diverge, but all at token 35 or 42, inside otherwise
  equivalent paraphrases.
- The control run itself emitted **four** different 64-token variants of that
  single prompt depending on prefill cohort; the cuDNN run emitted **one**.
- That cuDNN continuation is byte-identical to one of the four baseline
  variants (the one the default backend produced for 2 of 256 rows), so cuDNN
  selected an existing trajectory rather than generating new content.
- The first 35 tokens, including the whole answer's structure, are identical.

Details, per-request first-divergence indices and the variant counts are in
[`token_divergence.json`](token_divergence.json).

**Quality audit.** The repository's 18-task audit
([method and limitations](../quality_assessment/README.md)) was rerun with cuDNN
against the recorded pre-change results:

| Configuration | Strict contracts | Identical response texts |
|---|---:|---:|
| Compile + prefill batching, cuDNN off | 8/14 | — |
| Compile + prefill batching, cuDNN on | 8/14 | 16/18 |
| Eager, cuDNN off | 9/14 | — |
| Eager, cuDNN on | 9/14 | 17/18 |

The two differing optimized answers are an arithmetic explanation with the same
failure mode (extra prose) and a French translation that chooses *reportée*
instead of *déplacée*; both remain valid. Raw rows are in
[`quality_optimized_greedy.json`](quality_optimized_greedy.json) and
[`quality_eager_greedy.json`](quality_eager_greedy.json).

These checks establish parity on the tested cases. They are not a proof of
output identity, and the model's own
[quality limitations](../quality_assessment/README.md) still apply.

## Power and telemetry

The final S8 batch-256 run recorded 26 busy samples: median GPU utilization
97.5%, median load power 104.1 W against a 115 W limit, SM clock 2,445 MHz,
memory clock 8,001 MHz. No power or thermal limiting reason was active. Clocks
and temperatures were not locked, so percentages are workload-specific
observations.

## Reproduction

```bash
PY=/home/pratik/pyenvs/inference/.venv/bin/python
omarchy powerprofiles set ac performance

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 LFM3_FUSED_PAGED_ATTENTION=0 \
  "$PY" benchmarks/execution_timing/handoff.py \
  --batch-size 256 --output-tokens 64 --variant full --decode-steps 1 \
  --cudnn-attention --stream --require-platform-profile performance \
  --repeats 5 --output benchmarks/cudnn_attention/final/full_b256_n64.json
```

The harness already samples greedily (`temperature=0`, `top_p=1`, `top_k=0`,
`repetition_penalty=1`). Use `--decode-steps 8` for the S8 configuration and add
`--prompt "$LONG_PROMPT"` from `run_final_overlap.py` for the 512-token cases.
For the server, the equivalent settings are `--compile-model
--enable-cudnn-attention --decode-steps-per-round 8 --temperature 0
--repetition-penalty 1`. `final/` holds the raw JSON waves and `.gpu.csv`
telemetry. The `source_sha256` and `config` blocks record which files and
switches produced each result.
