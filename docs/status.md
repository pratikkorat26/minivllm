# Project status — November 1, 2026

## Completed: PyTorch throughput optimization

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
2. **Sliding attention bounds reads, not physical storage.** Gather width is at
   most `window + chunk_length - 1`, with independent starting positions for each
   row. Masks retain absolute causal positions. Full-attention gathers and the
   physical page layout remain unchanged; sliding layers still reserve full KV
   history in this version.
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

## Validation record

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
