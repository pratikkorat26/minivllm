# Measurements and methodology

## Current model: LFM3-1.7B

The runtime currently supports `LiquidAI/LFM3-1.7B` only. The checked-in engine
and attention JSON files predate the LFM3 migration and are retained in the
**historical Gemma measurements** section below. They are not LFM3 results.

### Model memory and quantization

Measured on the RTX 4070 Laptop 8 GB GPU with the real BF16 checkpoint, greedy
decoding, batch size 1, `torch.cuda.memory_allocated()` after load:

| Mode | Weights/model allocation | Short-prompt result |
|---|---:|---|
| BF16 | 3.22 GiB | `2+2 equals 4.` |
| int8 | 1.87 GiB | identical token IDs |
| int4 NF4 | 1.22 GiB | experimental; stop behavior changed |

BF16 generation peaked at 3.32 GiB allocated including KV and live temporaries.
The int8/int4 conversions transiently peak near the BF16 model size because the
BF16 checkpoint is loaded before replacement. These figures exclude the CUDA
context, allocator reservation and other processes.

int8 uses symmetric per-output-channel weights and PyTorch's
`torch._weight_int8pack_mm` on CUDA when available. Embeddings and the tied output
head remain BF16. On CPU, int8 dequantizes before the matmul. int4 uses
bitsandbytes NF4 with double quantization and is marked experimental for a reason:
on the arithmetic prompt above it emitted `</think>` and repeated the answer
instead of stopping. Prefer int8 unless the extra ~0.65 GiB is essential.

### Decode profiling and `torch.compile`

The fixed-batch decode profiler showed matrix multiplies taking most CUDA time,
with many small kernel launches per model step. Compiled execution fused much of
the surrounding pointwise work: across four profiled forwards, host-side
`cudaLaunchKernel` calls fell from 7,448 to 756. The `aten::mm` count stayed at
616, and its CUDA time changed only slightly (72.5 ms eager, 71.7 ms compiled), so
the compiler is reducing launch/elementwise overhead rather than fusing away the
main matrix multiplications. The executor marks its KV and convolution cache
buffers as stable-address inputs, allowing CUDA graph capture/replay despite
in-place cache updates. Decode batches and context lengths are padded into
power-of-two buckets; dynamic prefill shapes may still produce several graphs.
The profiler script can emit a Chrome trace for closer inspection.

The engine-level throughput test used `LLMEngine.generate` (not HTTP/SSE), four
short prompts, 32 output tokens per request, three measured waves, and a full warmup
wave excluded from the steady-state timer:

| Mode | Decode batch | Concurrent requests | Warmup wave | Steady-state aggregate |
|---|---:|---:|---:|---:|
| Eager BF16 | 128 | 128 | 8.1 s | 677 tokens/s |
| `torch.compile` + CUDA graphs BF16 | 128 | 128 | 154.7 s | **1,058 tokens/s** |

The compiled run returned the same sample output on the benchmark and the same
greedy IDs and EOS stop reason on the checked arithmetic prompt. CUDA graph capture
succeeded after marking engine-owned cache tensors as stable addresses. The
128-request warmup still reported nine distinct shapes, so some re-recording
remains; the long compile warmup makes this experimental. HTTP throughput and other
prompt mixes have not been measured. Compilation remains opt-in.

Reproduce or profile locally:

```bash
python scripts/benchmark_tps.py --concurrency 128 --decode-batch-size 128 \
  --max-batch-tokens 256 --max-kv-cache-tokens 16384 --compile
python scripts/profile_decode.py --batch-size 128 --steps 16 \
  --trace /tmp/lfm3-decode.json
```

### Projection packing and optional direct-page decode

Q/K/V weights are packed in Q, K, V order; FFN gate/up weights in w1, w3 order.
Checkpoint loading copies individual tensors into views of the packed weights,
avoiding an extra concatenated checkpoint copy. Quantization runs after loading
and sees ordinary Linear modules. The attention output gate stays separate.

Three sequential runs on the same RTX 4070 Laptop, BF16, compile enabled, batch
and concurrency 128, KV budget 16,384, four short prompts, 32 generated tokens per
request, three measured waves (12,288 generated tokens each):

| Variant | Aggregate tokens/s | Warmup seconds | Peak allocated GiB |
|---|---:|---:|---:|
| Separate projections + SDPA | 1,074.63 | 157.86 | 3.72 |
| Packed projections + SDPA | 1,111.64 | 155.39 | 3.72 |
| Packed projections + paged decode | 1,173.19 | 140.58 | 3.72 |

Packing improved this sample by 3.4%; paged attention added 5.5% (9.2% total).
Clocks/power were not locked, so these are indicative single-run comparisons,
not statistically established speedups. Sample text changed after projection
packing: BF16 GEMM ordering is not guaranteed token-identical. The packed SDPA
and paged runs returned the same displayed sample, not a full quality evaluation.

The direct-page kernel is opt-in via `LFM3_FUSED_PAGED_ATTENTION=1`. It supports
single-token CUDA decode (head dimension <=256, FP16/BF16/FP32) when Triton is
installed. Other shapes and prefill use SDPA. Online softmax reads KV pages in
64-token tiles without dense gathers. Sliding attention reads only its live
window. Long full-attention contexts can also use the split-KV path described
below.

Synthetic one-layer microbenchmark, batch 16, BF16, 16 query heads / 4 KV heads,
head dimension 128, 16-token pages, five warmups / 30 measurements:

| Context | Window | Dense gather + eager SDPA ms | Paged decode ms |
|---|---:|---:|---:|
| 128 | full | 0.268 | 0.023 |
| 512 | full | 0.491 | 0.063 |
| 2048 | full | 1.754 | 0.382 |
| 2048 | 256 | 1.835 | 0.032 |

These isolate attention, exclude cache writes/projections, and compare against
**eager** SDPA, not compiled SDPA. Do not extrapolate their speedups to the model.
Tests cover ragged contexts, permuted pages, unused invalid page entries, sliding
windows, graph replay with changed KV/metadata, and the independent dense result.
Local real-checkpoint serial/batched/streaming/chunked tests also pass with paged
decode enabled. Tiny-model Transformers reference and int8/int4 tests pass.

```bash
python scripts/benchmark_tps.py --concurrency 128 --decode-batch-size 128 \
  --repeats 3 --compile --fused-paged-attention
python scripts/benchmark_paged_attention.py
```

### Split-KV decode for long contexts

The paged decode kernel can partition a request's context across several GPU
blocks. Each partition writes an FP32 unnormalized numerator, denominator and
softmax maximum. A second kernel rescales these partials to a common maximum and
reduces them. Empty partitions contribute zero, including for short requests in a
ragged batch. No dense KV gather or full attention-score tensor is constructed.

This runs automatically inside the existing opt-in paged-attention path:
contexts/windows <=1,024 tokens remain single-pass. Above that, the heuristic
targets 512 tokens per partition for small batches (<128 query heads total),
otherwise 1,024, capped at 16 partitions. The Python kernel wrapper accepts
`num_splits=1` for comparisons or an explicit count up to 32. This is a simple
heuristic, not autotuning; the fastest count varies by workload and GPU.

Measured on the RTX 4070 Laptop with the same BF16 one-layer synthetic setup,
five warmups / 100 timed CUDA graph replays per configuration:

| Batch | Full context | Single pass ms | Auto split count | Split-KV ms |
|---:|---:|---:|---:|---:|
| 1 | 2,048 | 0.064 | 4 | 0.023 |
| 1 | 8,192 | 0.251 | 16 | 0.049 |
| 16 | 2,048 | 0.345 | 2 | 0.296 |
| 16 | 8,192 | 1.536 | 8 | 1.188 |

These compare against the previous direct-page kernel, not dense SDPA. They
exclude projections and cache writes and are **not whole-model throughput**.
Short-window measurements did not justify always splitting, so the default
256-token sliding layers retain the single-pass path. FP32/BF16 numerical tests
cover non-power-of-two partition counts, empty partitions, changed metadata and
KV during graph replay, and fullgraph compilation.

```bash
python scripts/benchmark_paged_attention.py --batch-size 1 \
  --contexts 2048 8192 --splits 1 2 4 8 16 --repeats 100 --cuda-graphs
```

### Split-KV whole-engine measurements

Real LFM3 checkpoint, BF16, packed projections, fused paged attention and
compilation enabled in both modes; concurrency/decode batch 1, 64 output tokens,
256-token prefill budget, two warmup waves, three measured waves. Synthetic chats
contain repeated text truncated to the stated prompt token counts. Each
context/mode ran in a fresh process to avoid compiler guard-cache exhaustion.
The initial attempt with insufficient warmup and multiple modes/contexts in one
process suffered recompilation and is excluded from these results.

| Prompt tokens | Single-pass request tokens/s | Split-KV request tokens/s | Change |
|---:|---:|---:|---:|
| 2,048 | 44.34 | 45.18 | +1.9% |
| 8,192 | 27.89 | 28.73 | +3.0% |

The request timer includes prefill, scheduling and decoding, but not model load
or warmup. Instrumenting the engine's decode worker separately measured:

| Prompt tokens | Single-pass decode tokens/s | Split-KV decode tokens/s | Change |
|---:|---:|---:|---:|
| 2,048 | 54.49 | 55.87 | +2.5% |
| 8,192 | 51.26 | 54.79 | +6.9% |

Decode-worker timing excludes prefill/idle time and counts only tokens generated
by decode, not the first prefill token. These are **not HTTP results**, and clocks
were not locked. Three measured request waves were stable within about 2%, but
there is only one process run per mode: small differences need more repetitions.
At 8,192 tokens the generated-token hashes matched; at 2,048 they differed,
consistent with BF16 near-tie sensitivity to reduction order. This is not a
quality evaluation. Short contexts do not activate split-KV, so their previous
high-concurrency throughput should not be interpreted as improved by this change.

This run also exposed RoPE growth retaining graph-owned buffers beyond a graph
step; the executor now grows these persistent buffers outside compilation.

Raw files: `lfm3_split_kv_engine_2048_single.json`,
`lfm3_split_kv_engine_2048_auto.json`, `lfm3_split_kv_engine_8192_single.json`,
`lfm3_split_kv_engine_8192_auto.json`.

Run the following commands separately (repeat for `--contexts 2048`):

```bash
python scripts/benchmark_split_kv_engine.py --contexts 8192 --mode single_pass
python scripts/benchmark_split_kv_engine.py --contexts 8192 --mode auto_split_kv
```

### Reference validation

`lfm3_reference.json` compares the local checkpoint with the Transformers remote
implementation of LFM3, recomputing the full reference context at every step. The
native path uses paged KV plus per-request conv state.

- Chat prompt: 8/8 greedy tokens match; maximum absolute logit error 0.375.
- Raw prompt: 7/8 greedy tokens match. The differing position is a BF16 near-tie:
  native top-2 gap `0.0`, reference gap `0.125`.
- The tiny FP32 LFM3 reference test matches all logits with about `1.27e-7`
  maximum absolute error, including chunked prefill and conv-state isolation.

The validator records per-step logits and top-2 gaps. It only exits nonzero when
`--strict` is passed.

```bash
python scripts/validate_reference.py --model-dir LFM3-1.7B --dtype bfloat16
```

### Current benchmark commands

```bash
python scripts/benchmark_engine.py --output benchmarks/my_lfm3_engine.json
python scripts/benchmark_engine.py --quantization int8 --output benchmarks/my_lfm3_int8.json
python scripts/benchmark_tps.py --concurrency 128 --decode-batch-size 128 --compile
python scripts/profile_decode.py --batch-size 128 --steps 16
python scripts/benchmark_attention.py --output benchmarks/my_lfm3_attention.json
```

No full LFM3 closed-loop sweep, long-context power-controlled study or vLLM
comparison has been recorded. The `benchmark_engine.py` result includes
`model`, `quantization`, `dtype`, engine config and raw per-request latencies.

## Historical Gemma measurements

> These results were collected before the LFM3 migration on
> `google/gemma-3-270m`, PyTorch 2.7.1+cu128, BF16, eager execution. They document
> the scheduling/attention changes made to this repository and must not be read as
> LFM3 performance. The corresponding model files are no longer part of the
> supported configuration.

Environment: RTX 4070 Laptop (8,188 MiB reported), SM89, NVIDIA driver 595.79,
Windows 11 WDDM, Python 3.12.10. Base checkpoint revision
`9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`.

The baseline was recorded after KV/lifecycle fixes, before attention/scheduling
performance changes; prefix caching was off. Improved data comes from the changed
working tree. BF16 norm arithmetic was corrected as part of the work, so exact
baseline token hashes are not a correctness oracle.

`benchmark_engine.py` used 8,192 KV slots (16-token pages), one 16-token warmup,
then three waves at concurrency 1/2/4/8. Decode batch cap 4, token budget 256,
maximum active requests 16, greedy sampling. No prefix caching or compilation.
All recorded requests generated 16 tokens. Peak allocated memory was about
728.5-731.5 MiB.

| Concurrency | Baseline output tok/s | Improved output tok/s | Change |
|---:|---:|---:|---:|
| 1 | 28.47 | 26.01 | -8.6% |
| 2 | 46.59 | 47.53 | +2.0% |
| 4 | 73.29 | 80.91 | +10.4% |
| 8 | 76.93 | 91.47 | +18.9% |

The single-request regression is retained. These are small samples with no locked
GPU clocks or isolated desktop workload. No result supports a claim of parity or
superiority to vLLM.

![Historical Gemma engine measurements](../docs/figures/engine.png)

### Historical attention microbenchmark

`attention_baseline.json` and `attention_vectorized.json` measure one synthetic
Gemma-shaped attention layer: embedding 640, 4 query heads, 1 KV head, head dim
256, BF16, page size 16; batch 1/4, context 128/512, query length 1/32. Ten
warmups and 50 timed iterations per shape. Random weights/KV, no RoPE or QK norm;
not end-to-end model performance.

Batch-4 latency fell from about 1.36-1.47 ms to 0.83-0.98 ms. Batch-1 results were
mixed. No custom kernel was introduced.

![Historical Gemma attention measurements](../docs/figures/attention.png)

### Historical Gemma numerical validation

`reference.json` compares local Gemma weights with Transformers 4.52.4 eager FP32
attention: three prompts, eight positions each. All 24 tested greedy positions
matched; maximum absolute logit error about 2e-5. `reference_bfloat16.json` and
`reference_bfloat16_sdpa.json` retain strict BF16 validation failures: 23/24
positions matched, with maximum absolute errors 0.5 (eager) / 0.375 (SDPA). At the
SDPA mismatch the native top two logits tied; the reference gap was 0.0625.

## Equivalent vLLM workload (not measured here)

Use a compatible Linux/CUDA host with a vLLM build that supports LFM3. Use the
same checkpoint and tokenizer revision on both servers. Run one server at a time
on the same GPU; record versions, clocks, power, model revision and memory
settings.

```bash
vllm serve LiquidAI/LFM3-1.7B --served-model-name LFM3-1.7B --dtype bfloat16 \
  --max-model-len 4096 --max-num-seqs 16 --max-num-batched-tokens 256 \
  --gpu-memory-utilization 0.7 --enforce-eager --no-enable-prefix-caching
python scripts/benchmark_serving.py --url http://127.0.0.1:8000 --label vllm \
  --model LFM3-1.7B --output benchmarks/vllm_http.json
```

Stop vLLM, start this engine with equivalent limits, then run the same client:

```bash
python -m adapters.openai.run --decode-batch-size 16 \
  --max-concurrent-requests 16 --max-batch-tokens 256
python scripts/benchmark_serving.py --url http://127.0.0.1:8000 --label minivllm \
  --model LFM3-1.7B --output benchmarks/minivllm_http.json
```

The client sends identical prompts, sampling, token caps and closed-loop arrival
waves to `/v1/chat/completions`. Confirm prompt token counts/chat templates and
actual output lengths match before attributing differences to the engine. Compare
p50/p95 TTFT, TPOT, output throughput, failures and GPU memory. Longer prompts and
realistic arrivals require additional workloads.

Reference: [vLLM serve options](https://docs.vllm.ai/en/stable/cli/serve/).
