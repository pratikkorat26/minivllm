# Repository audit — 2026-10-15

Completed before implementation. Baseline: 51 passed, 5 skipped (real-model
tests opt in and require instruct assets). Windows, Python 3.12.10, PyTorch
2.7.1+cu128, RTX 4070 Laptop 8 GB, SM89. No custom CUDA/Triton kernels exist.

## Existing architecture and lifecycle

HTTP schema → adapter mapper → ChatCompletionService (prompt + sampling) →
LLMEngine → AsyncScheduler → PagedModelExecutor → Gemma3Model → SDPA.
The runtime loads weights/tokenizer and chooses device/dtype. Submit tokenizes,
creates a future, queues FIFO; admission checks concurrency and prompt capacity.
Prefill runs one request per step, optionally chunked; all prefill takes priority
over decode. Decode selects the largest equal-length/equal-sampling cohort.
Executor allocates blocks, transfers metadata, runs 18 layers, samples and reads
each token to CPU. Completion releases request references; JSON/SSE returns output.

KV tensors are real fixed physical pages [blocks, KV heads, tokens, head dim],
but attention gathers them into dense tensors before SDPA. This is paged storage,
not a fused paged attention kernel. Prefix entries own block references; requests
share whole pages. There is entry-count LRU eviction but no pressure eviction.
Model execution runs synchronously on the ASGI event loop; streaming queues are
unbounded. Cancellation is polled between steps. There is no captured CUDA graph;
automatic torch.compile is not evidence of graph replay.

## Component classification

| Component | Decision | Finding / action |
|---|---|---|
| Public API, engine facade, types | KEEP | Stable useful seam; add latency fields compatibly |
| Scheduler | REFACTOR | Prefill starvation, duplicated branches, caller-future cancellation leak, initialization outside exception handling |
| Admission policy | REMOVE from hot path | Class + result object only wrap concurrency/status checks; retain compatibility export |
| Decode selector | OPTIMIZE | Equal lengths unnecessarily prevent ragged decode; largest cohort can starve oldest |
| Executor | OPTIMIZE | Per-row metadata transfers and token scalar reads; prompt-only admission overcommits future decode |
| Block manager | REFACTOR | Duplicate cache insertion increments wrong refs; no pressure reclamation; reserve request growth |
| Prefix cache | REFACTOR | Exact entries only, linear search acceptable at 64 entries; no need for a radix-tree framework here |
| Attention append | REPLACE | Physical [N,G,T,D] viewed as [G,N*T,D] corrupts G>1 caches |
| Attention gather / RoPE | OPTIMIZE | Per-layer scalar synchronization and Python loops; vectorize batch indexing |
| SDPA, MLP, norms, last-token head | KEEP | Use supported PyTorch kernels; validate numerically against independent reference |
| Runtime | REFACTOR | Implicit compilation is fragile on Windows and dynamic shapes; make opt in |
| HTTP adapter / app separation | KEEP | Existing import-boundary tests useful; avoid a repository rewrite |
| Streaming / stop handling | REFACTOR | Token-at-a-time decode can corrupt Unicode; split stop sequences already emitted; EOS usage undercount |
| Metrics / benchmarks | OPTIMIZE | No true TTFT/TPOT; decode time divided by batch is not request latency; no saved reproducible results |
| Tests | KEEP + EXTEND | Good fake-runtime contracts; missing real attention and cache ownership invariants |
| Config parser | KEEP | Verbose but preserves CLI/env contract; validate core config rather than add knobs |
| Existing demo scripts | KEEP | Preserve entry points; add one reusable measurement path |

## Prioritized incremental changes

1. Regression tests for KV layout, cache refs/pressure/exact reuse, cancelled
   callers, initialization failure; then fix ownership and cleanup.
2. Bounded prefill chunks from token budget and fair prefill/decode turns;
   ragged decode cohorts; conservative KV admission that reserves maximum growth.
3. Vectorized KV append/gather and RoPE, CPU-known sequence bound once per forward,
   batched metadata/token transfers; compare logits with dense reference on CPU/CUDA.
4. Keep event loop responsive with one execution worker and serialize mutation;
   bounded stream buffering, stop suffix buffering, explicit iterator cleanup.
5. Reproducible JSON benchmarks + graphs, reference validation, honest README.

## Proposed compact structure

Keep `adapters/openai`, `app`, `inference`, `runtime`, `gemma3`, `config` because
their boundaries already work. Core: `scheduler/{scheduler,batch}.py`,
`executor.py`, `kv/{block_manager,prefix_cache}.py`, `sampling/sampler.py`,
`config.py`, `types.py`. Keep admission compatibility import, remove its use from
the scheduler. Add focused tests, a benchmark runner and plotting script, results
under `benchmarks/`, and study diagrams in `docs/`. No new serving framework.

## Missing techniques and tradeoffs

No mixed ragged batched prefill, fused paged attention, graph capture/replay,
preemption/recompute, speculative decoding, quantization, distributed serving,
or overlapping scheduler/GPU pipeline. These should not be claimed as implemented.
Conservative growth reservation sacrifices admission density to guarantee progress
without adding a preemption subsystem. Fair alternating phase turns are simpler
than mixed prefill/decode forwards. Prefix reuse remains bounded exact-entry
lookup rather than SGLang's radix tree. Dense gather remains an O(context) copy.

## Expected impact — hypotheses until measured

Vectorized attention should reduce CPU launch/synchronization overhead, especially
at higher concurrency. Ragged decode should increase batch occupancy on unequal
prompts. Bounded prefill and phase fairness should lower decode tail stalls under
long-prompt arrivals, potentially increasing TTFT. Cache reclamation should prevent
false capacity rejection, not necessarily improve speed. Worker handoff should
improve HTTP responsiveness but adds CPU overhead. No numerical speedup forecast.

## Primary references

- [vLLM token budget and chunked prefill](https://docs.vllm.ai/en/stable/configuration/optimization/)
- [vLLM cache ownership and eviction](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- [SGLang scheduling policy](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_policy.py)

Comparison concerns architectural principles, not feature or performance parity.
