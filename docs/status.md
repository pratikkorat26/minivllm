# Project status — updated November 23, 2026

Latest: end-to-end LFM3-1.7B memory, decode, and executor optimization pass
verified on RTX 4070 Laptop (8 GB). See [OPTIMIZATION_REPORT.md](../OPTIMIZATION_REPORT.md)
for full before/after tables and parity-gate results.

## November 23: LFM3-1.7B memory, decode & executor optimization

- **Bounded sliding-window KV ring buffers** (`lfm3/paged_kv.py`, `lfm3/model.py`, `lfm3/attention.py`, `inference/executor.py`): 5 sliding-window attention layers allocate per-slot ring buffers (`window_size=256`) instead of full-context paged blocks, cutting weights + KV allocation from `3.917 GB` to `3.642 GB` (`-0.275 GB`).
- **Default Triton paged decode for long contexts** (`lfm3/attention.py`): `LFM3_FUSED_PAGED_ATTENTION=1` is enabled by default for `max_seq_len >= 1024` (with precomputed RoPE/write indices in `Lfm3Model.forward`), boosting decode throughput at `16384` (`38.9 -> 51.7 tok/s`, `1.33x`) and `32768` (`30.5 -> 49.6 tok/s`, `1.63x`) while reducing `32768` total latency from `7.348s` to `6.544s`.
- **Quantization** (`lfm3/quantization.py`): `int8` now skips `gate_proj` for precision parity, and `fp8` (`torch.float8_e4m3fn` + `torch._scaled_mm`) is supported on SM89+ CUDA.
- **Executor, sampling & streaming fast-paths** (`inference/executor.py`, `inference/sampling/sampler.py`, `app/service.py`): pre-uploaded prompt tensors for zero-copy prefill slicing, GPU boolean repetition-penalty masks, top-k-first nucleus sampling, and O(N) tail-bounded streaming stop-sequence scanning.
- **Validation**: `RUN_REAL_ENGINE_TESTS=1 pytest -v` passes **225 tests** (5 opt-in skipped), including bit-exact greedy parity against `baseline_parity.json` in `tests/test_lfm3_parity_gate.py`.

## November 4: opt-in cuDNN attention

LFM3 passes an additive `[B,1,Q,S]` mask to SDPA. Flash attention rejects that
mask and the default priority places it first, so the dispatcher used the
memory-efficient kernel. Opting in to cuDNN priority
(`EngineConfig.enable_cudnn_attention`, default false) cut attention kernel time
at the measured shapes and reduced a compiled batch-256 decode wave from 3.06 s
to 2.58 s of recorded CUDA time. End-to-end streaming results:

| Validated workload | Default backend | cuDNN attention |
|---|---:|---:|
| 256 requests, 64-token cap, S1 | 5,016 | **5,896 (+17.5%)** |
| 256 requests, 64-token cap, S8 | 5,250 | **6,206 (+18.2%)** |
| 128 requests, 512-token cap, S1 | 2,990 | **3,710 (+24.1%)** |
| 128 requests, 512-token cap, S8 | 3,104 | **3,855 (+24.2%)** |

Units are aggregate output tokens/s with the same harness, warmup and
performance-profile rules as November 3. cuDNN is not bitwise reproducible at
some batch/context shapes: repeated identical calls can differ by one BF16 ulp,
which flips near-tied greedy tokens, and the three 512-token waves produced
three different hashes. The memory-efficient kernel returned one bitwise result
in every measured cell. The 18-task quality audit kept 8/14 strict passes in
compiled mode and 9/14 in eager mode, with 16/18 and 17/18 response texts
identical. Inductor's FX-graph key does not include the SDPA priority, so cuDNN
artifacts live in a separate `minivllm-cudnn-attention` cache subdirectory.
The option is therefore off by default. Details and raw data are in the
[cuDNN report](../benchmarks/cudnn_attention/README.md).

## November 1: PyTorch throughput optimization

The software optimization pass is implemented and benchmarked with real
`LiquidAI/LFM3-1.7B` BF16 weights on an RTX 4070 Laptop 8 GB, using Python 3.12.16
and PyTorch 2.15.0+cu132. The reference baseline is commit `83c1329`. This pass
adds no custom CUDA/Triton kernels; the custom paged-attention option was disabled
for every recorded software throughput measurement.

| Validated workload | Prefill input tokens/s | Decode output tokens/s | End-to-end output tokens/s |
|---|---:|---:|---:|
| One short conversation | 860 | 61 | 60 |
| One conversation, 8,118-token prompt | 9,191 | 51 | 30 |
| 128 concurrent short prompts | 8,242 | 4,657 | 3,983 |
| 256 concurrent short prompts | 8,808 | 6,161 | 5,004 |
| 512 concurrent short prompts | 8,848 | 6,071 | 5,038 |

Batched rates are aggregate. Measurements use five waves with a 64-token output
cap and a 2,048-token budget; model loading and compilation warmup are excluded.
They measure the engine, not HTTP serving. Clocks and power were not locked.
These are workload-specific results, not a proven hardware maximum. At the same
concurrency of 128, compiled end-to-end output throughput increased from 1,769
to 3,983 tokens/s with code and scheduler-setting changes.

Batch 256 is the recommended throughput setting from this sweep: batch 512 added
less than 1% throughput while roughly doubling per-conversation generation time.
The exact configurations, raw waves, token hashes, memory peaks, numerical caveats,
and reproduction commands are in [benchmarks](../benchmarks/README.md). A compact
machine-readable report is [software_summary.json](../benchmarks/software_summary.json).

## Key architectural decisions

1. **Compatible prefill batching is opt-in.** `EngineConfig.enable_prefill_batching`,
   the server's `--enable-prefill-batching`, and
   `LFM3_API_ENABLE_PREFILL_BATCHING` enable equal-length chunks on models declaring
   `supports_batched_prefill`. The oldest chunk anchors a cohort with matching
   sampling settings and completion phase. Prior KV lengths can differ. The
   scheduler enforces the shared token budget and at most one chunk per request
   per round. The synchronous executor runs the cohort in one model forward.
2. **Sliding attention bounds physical storage via per-slot ring buffers.**
   `Lfm3Model` declares `supports_bounded_sliding_kv = True`, allocating
   `[max_slots, num_kv_groups, window_size, head_dim]` ring buffers for the 5
   sliding-window layers instead of full-context paged blocks. Full-attention
   layers continue to use paged KV blocks and dispatch to Triton paged decode
   when `max_seq_len >= 1024`.
3. **Compiled one-token convolution uses a short dot product.** FP32 accumulation
   allows PyTorch to fuse surrounding pointwise work. Eager and multi-token
   execution use the existing depthwise convolution. Per-request state slots
   remain the ownership boundary.
4. **Compilation has a bounded, scoped specialization budget.** Persistent buffers
   retain stable addresses and decode uses shape buckets. Up to 32 compiled
   specializations are allowed during each model call, with the framework setting
   restored afterward. Forced `dynamic=True` was measured and rejected as slower.
5. **Sampling reduces transfers and copies.** Repetition penalties use batched
   indices or device gather/scatter, and logits are copied only for mutation.
   Sampling results transfer to CPU once per completed prefill/decode batch.
6. **Timing separates phases and units.** Optional CUDA events measure worker
   prefill input tokens/s and decode output tokens/s. The first output token is
   sampled in prefill. End-to-end throughput includes both phases and scheduling.

The [architecture guide](architecture.md) describes execution, lifecycle, and
ownership in more detail. Model/KV mutation stays serialized on one worker.
BF16 GEMM shapes and reduction orders can change near-tied greedy choices;
prefill batching and compilation are explicit performance options.

## November 1 validation record

- GPU suite: 178 passed, 9 skipped after adding the compilation-scope check.
- Local-checkpoint suite: 3 passed, including batched/serial prefill-logit tolerance.
- Real-engine regression suite: 4 passed with default configuration.
- Actual Inductor convolution checks: 8 passed across CPU/CUDA and FP32/BF16.
- Targeted Ruff correctness checks and `git diff --check` passed.
- Every final phase report reconciles decode-token counts with total generated
  tokens, accounting for the first token produced during prefill.

Run real-model suites in separate processes: their independent cached runtimes
can exceed 8 GB if both opt-in suites are enabled in one process. Runtime, dev,
and benchmark requirements are installed in `/home/pratik/pyenvs/inference/.venv`.
`httpx` is included in the dev extra for API tests. Downloaded weights, caches,
and local dependency directories are ignored and are not part of the source changes.

## Next objective: longer conversations using system RAM

CPU-backed KV offload and persistent conversation sessions are not implemented.
The proposed next stages are bounded physical storage for sliding-window layers,
CPU backing for full-attention KV with GPU staging, and session ownership of both
KV pages and convolution state. Extra RAM does not extend the model's configured
262,144-token context limit. LFM3 prefix reuse stays disabled until convolution
state snapshots can be restored with matching KV history.

No CPU-offload throughput, HTTP capacity, or vLLM comparison is claimed here.

## November 2 follow-up: execution timing and overlap

The [execution timing investigation](../benchmarks/execution_timing/README.md)
separates actual GPU copy duration from CPU waits for GPU completion. Transfers
are small relative to the GPU kernels; a blocking sampled-ID copy is largely
waiting for its preceding forward, not moving data over PCIe.

The RoPE fix canonicalizes CUDA device aliases so compiled steps
reuse their caches. Compiled preparation restores the configured cache precision
once; eager precision behavior is preserved. Fixed-work paired decode medians
were 51.40 to 52.11 output tokens/s for one 8,118-token prompt, and 6,538 to 6,563
aggregate tokens/s for batch 256 short prompts. Short-context changes are within
noise; these microbenchmarks exclude prefill and scheduler overhead and should
not replace the serving-throughput table above. All paired token hashes match.

After the scoped fix, the standard GPU suite passed 180 tests (9 skipped), the
local-checkpoint suite passed 3, and the real-engine suite passed 4. No speculative
decoding, custom kernels, or concurrent mutation of model/KV state was added.

Real engine timing identifies CPU text emission as a possible next overlap
target: batch 256 has a roughly 40 ms worker step, plus median scheduler time of
0.39 ms without streaming or 2.15 ms with streaming. Actual GPU copies take
about 0.004 ms per batched step. Any expected gain from overlapping emission is
small at this investigation stage; the measurements did not justify adding copy
streams merely to hide these transfers. The subsequent implementation is recorded
below.

## November 3: implemented execution overlap

The worker now queues GPU computation and a pinned sampled-ID copy before the
event loop emits the previous immutable text prefix. A CUDA event fences host
reads and buffer reuse. Stream errors and cancellation release pages after the
worker returns. A bounded metadata cache avoids repeated uploads for stable
compiled decode cohorts, and native incremental decoding avoids repeatedly
decoding the full generated history.

Optional `decode_steps_per_round` chains exact greedy steps before one host
transfer and bulk history update. Its default is one. Eight-step rounds require
compilation, temperature zero, repetition penalty 1 and a settled cohort; budgets,
reservations and context cap each chunk. This trades finer streaming cadence and
cancellation granularity for throughput. No speculative decoding is involved.

<!-- overlap-final-results:start -->
Final warmed streaming measurements: batch 256 with a 64-token output cap improved from 4,813 to 5,250 aggregate output tokens/s (+9.1%) with exact greedy S8. Batch 128 with a 512-token output cap improved from 2,791 to 3,104 tokens/s (+11.2%). All compared token/text hashes match; no compilation or capture occurred during measurement. Rates include prefill and the streaming harness. See the [overlap report](../benchmarks/overlap_optimization/README.md) for default S1 results, power telemetry, cadence tradeoffs and reproduction commands.
<!-- overlap-final-results:end -->

Validation: 195 standard tests passed (9 skipped), local checkpoint tests 3
passed, real-engine regressions 4 passed. Twenty-four varied real streams matched
baseline token IDs, text and stop reasons, including seeded sampling and early
completion. Native incremental decoding matched 1,847 valid UTF-8 prefixes across
56 saved/text cases.

The laptop's supported performance profile is selected on AC. Benchmarks verify
it before loading and measuring, reject changes during a run, and record GPU
power, clocks, temperature and throttling reasons. The GPU power limit is not
raised. The implementation, validation code and raw measurements are recorded
together in this repository.
