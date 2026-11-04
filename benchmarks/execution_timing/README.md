# Decode execution and overlap investigation — November 2, 2026

This records the initial investigation. The implemented follow-up and final
warmed measurements are in [overlap optimization](../overlap_optimization/README.md).

Hardware: RTX 4070 Laptop, 8 GB. BF16 LFM3-1.7B, compiled PyTorch
2.15.0+cu132, four OMP/MKL threads. Custom paged-attention kernels are disabled.
Weights and KV remain on the GPU; these results do not cover CPU KV offload.

## Executor measurements

Each case uses a fresh process, warms 66 decode steps, then measures five
32-step repetitions without a profiler. EOS is disabled only for this bounded
fixed-work comparison. Throughput includes executor metadata construction,
forward, greedy sampling and the sampled-ID copy, but excludes prefill,
scheduler, detokenization and HTTP. Batched rates are aggregate output tokens/s.

| Workload | Baseline tokens/s | Cache fix tokens/s | Observed change |
|---|---:|---:|---:|
| One request, 19-token prompt | 61.55 | 61.73 | +0.31% |
| 256 requests, 19-token prompts | 6,538.49 | 6,562.89 | +0.37% |
| One request, 8,118-token prompt | 51.40 | 52.11 | +1.38% |

These are medians from one paired experiment, not confidence intervals or a
proven hardware maximum. Clocks/power are not locked. The short-context changes
are within normal run-to-run noise; the long-context result is a small observed
gain requiring more repetitions before a strong speedup claim.

All five generated-token hashes match between baseline and candidate for each
workload. This checks these deterministic cases, not general model quality.

Separate 16-step CUPTI traces show the following candidate execution time:

| Workload | GPU kernels per step | Actual GPU copies per step | GPU busy fraction |
|---|---:|---:|---:|
| One short request | 15.382 ms | 0.0331 ms | 93.96% |
| 256 short requests | 36.889 ms | 0.0038 ms | 94.95% |
| One long request | 18.647 ms | 0.0278 ms | 95.90% |

Profiler timings are diagnostic and perturb execution; they are not substituted
for unprofiled throughput. Busy fraction is the union of kernels, copies and
memsets divided by the first-to-last GPU activity span. It is not SM utilization
or memory-bandwidth utilization. Kernel and copy columns omit memsets and gaps.
Current traces show zero copy/kernel overlap. Summing operator rows would double
count nested compiled regions; the table above uses raw GPU activity events.

The apparent large CPU time in `aten::copy_`/`cudaMemcpyAsync` mainly waits for
preceding GPU computation before returning sampled IDs. It does not mean PCIe
transfers consume that time. B1's two GEMV kernel families account for roughly
97% of kernel time. Raw copies are a very small fraction of the critical path.

## Implemented change

`torch.device("cuda") != torch.device("cuda:0")` caused both RoPE caches to
rebuild on every compiled executor step. Canonicalizing the device preserves
cache reuse. A 16-step trace drops from 32 cosine and 32 sine operations to zero,
removing 160 GPU kernel launches and 160 CUDA runtime calls in each paired case.
Cache buffers now retain their addresses until growth or migration.

Model `.to(dtype=BF16)` also casts RoPE buffers. Compiled preparation explicitly
restores the configured cache dtype once, preserving the FP32 cache values
previously rebuilt every step. Ordinary eager calls retain their previous
precision behavior. An initial broader dtype correction changed eager greedy
choices in a local-model regression and was narrowed before completion. No
inverse-frequency precision change is included.

Baseline profiles load only the original `ensure_capacity` from commit
`cb3761f`; the remaining executor/runtime is shared. Candidate profiles were
collected with the working-tree fix, before dtype restoration was explicitly
scoped to compiled preparation. That scoping preserves the same measured
compiled path. The `git_commit` field records the base commit, not a committed
candidate change. Final real-engine handoff runs use the scoped implementation.

## Where overlap could help

The current worker already lets request handling run while the GPU executes.
Extra CUDA copy streams have very little raw transfer time to hide in this
GPU-resident workload. The next token depends on the previous token; simply
making `.tolist()` asynchronous cannot remove that dependency or the CPU's
completion/EOS decision.

The more plausible opportunity is CPU scheduler/stream work: dispatching a safe
next batch before formatting and emitting the preceding batch's text. This needs
immutable token snapshots and explicit ownership of cancellation, completion and
KV release. A second worker must not mutate live KV or convolution state while
the current worker owns it. Host bookkeeping and metadata preparation may also
be reduced, but copying a few bytes faster is unlikely to be material here.

`handoff_b*_json.json` and `handoff_b*_stream.json` measure real engine requests,
with normal EOS handling, two warmup waves and three measured waves capped at
64 output tokens. `scheduler_extra_s` measures the enclosing scheduler decode
call minus worker execution. It includes selection, worker handoff, completion
and stream emission; it is not solely thread handoff. It excludes time spent
outside that call, including some consumer/event-loop activity. Streaming means
the engine iterator, not HTTP/SSE serialization. End-to-end rates include prefill.

| Requests per wave | Output mode | Median worker step | Median scheduler extra | p95 scheduler extra | End-to-end output tokens/s |
|---|---|---:|---:|---:|---:|
| 1 | Non-streaming | 16.194 ms | 0.213 ms | 0.273 ms | 60.81 |
| 1 | Streaming | 16.317 ms | 0.215 ms | 0.287 ms | 60.46 |
| 256 | Non-streaming | 39.950 ms | 0.390 ms | 0.614 ms | 5,132.48 |
| 256 | Streaming | 39.905 ms | 2.151 ms | 3.033 ms | 4,916.05 |

These are three waves in one process per configuration. Each generated 64 tokens
per request without error. Non-streaming and streaming use separate processes;
their throughput difference is diagnostic, not a controlled estimate of one
operation's cost. The extra CPU time for batched streaming is the best measured
overlap candidate: roughly 2 ms beside a 40 ms worker step. Hiding it perfectly
would recover about 5% of that step's time, not yield a multiple of throughput.
Actual recoverable time is smaller because selection, completion and state
updates cannot all move freely across step boundaries. A single request's
scheduler overhead is roughly 1.3% of worker time.

No communication-overlap implementation or speculative decoding was added.

## Reproduce

Run with the inference environment and CUDA access, sequentially on an idle GPU:

```bash
PY=/home/pratik/pyenvs/inference/.venv/bin/python
"$PY" benchmarks/execution_timing/run_profiles.py
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 LFM3_FUSED_PAGED_ATTENTION=0 \
  "$PY" benchmarks/execution_timing/handoff.py --batch-size 256 --stream
```

Repeat the handoff command with batch size 1/256 and with/without `--stream`.
Profile scripts write raw Chrome traces under `/tmp/minivllm_execution_traces/`
and compact summaries/operator tables here. Unprefixed `b*_summary.json` files
are initial exploratory measurements; use `baseline_*` and `candidate_*` for
the paired comparison above. Compilation/loading is excluded after warmup.

Validation after the scoped fix: standard GPU suite 180 passed, 9 skipped;
local-checkpoint suite 3 passed; real-engine regression suite 4 passed.
The latter suites run in separate processes to fit the GPU memory budget.
