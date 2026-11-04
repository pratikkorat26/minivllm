# GPU execution overlap — November 3, 2026

This pass reduces CPU work between GPU forwards and overlaps streamed text
emission with device execution. It uses resident BF16 LFM3-1.7B weights and KV
on one RTX 4070 Laptop GPU, with PyTorch compilation and library kernels.
No speculative decoding or handwritten CUDA/Triton kernels are involved.

## Results and measurement boundaries

The final measurements are under `final/`; `summary.json` verifies source hashes,
warmup counters and output hashes before reporting a comparison. Rates count
aggregate generated output tokens and include prefill, decode, scheduling and
the streaming validation harness. They exclude model loading, compilation and
CUDA graph capture. These are engine measurements, not HTTP/SSE capacity.

<!-- final-measurements:start -->
| Requests per wave | Output cap | Mode | Aggregate output tokens/s | Change vs baseline |
|---|---:|---|---:|---:|
| 256 | 64 | Features disabled | 4,813 | — |
| 256 | 64 | Default S1 overlap | 5,016 | +4.2% |
| 256 | 64 | Exact greedy S8 | 5,250 | +9.1% |
| 128 | 512 | Features disabled | 2,791 | — |
| 128 | 512 | Default S1 overlap | 2,990 | +7.1% |
| 128 | 512 | Exact greedy S8 | 3,104 | +11.2% |

These runs measured 6,272 requests and 917,504 generated tokens; warmup is excluded.

CPU affinity experiment: 5,253 tokens/s (+0.06% versus unpinned S8). Affinity remains unchanged by the application.

The diagnostic trace records 73.40 ms of host text-emission work, with 69.44 ms (94.6%) overlapping actual GPU kernels. Profiled throughput is excluded from the table.
<!-- final-measurements:end -->

Every final case uses a fresh process. Warmup continues until two consecutive
complete request waves add neither a Dynamo graph nor recorded CUDA graph
inputs. Each measured wave checks those counters again and fails if either
changes. Ordinary two-wave warmup was insufficient: a third wave could still
capture another graph. Short outputs use five measured waves; 512-token outputs
use three. EOS remains enabled. Token and text hashes compare complete ordered
request results between modes, and each streamed text must equal full decoding
of its final token IDs.

The test laptop uses the supported `performance` platform profile on AC, four
OMP/MKL threads, a 2,048-token scheduling budget, and compatible prefill batching.
Performance mode is checked before model loading and measurement; a profile
change during measurement invalidates the run. Clocks and temperatures are not
locked, so percentages are workload-specific observations, not a hardware
maximum or confidence interval.

## Architectural changes

1. **Stream emission moves beside GPU execution.** After queuing forward,
   sampling and the sampled-ID copy, the worker notifies the event loop to emit
   the previous step's text. Pending emission records a token-prefix length and
   token ID. Prefixes stay immutable while the worker appends a newer suffix.
   At most one notification is retained per active stream.
2. **The sampled-ID buffer has an explicit fence.** A reusable pinned CPU buffer
   receives one asynchronous device copy. A CUDA event completes before the
   worker reads or reuses it. Slow-stream errors are retired after worker
   completion, so a callback cannot release KV pages during a live forward.
3. **Stable metadata stays on the GPU.** One bounded cache entry retains the
   block tables, length tensor and convolution slots for a compiled decode
   cohort. Request order, page mappings, padded dimensions, slots and expected
   lengths must match. Active lengths advance on the same stream; padded lengths
   stay zero. Cohort or capacity changes invalidate the entry.
4. **Text decoding consumes only the new suffix.** The tokenizer's native
   incremental decoder handles split UTF-8 characters. Completion performs one
   full decode to flush an unfinished tail. Generic tokenizers retain the prior
   fallback. This removes repeated full-history decoding as output grows.
5. **Optional greedy chunks reduce host barriers.** A settled cohort can queue
   several exact autoregressive steps before copying IDs and committing history
   in bulk. Every input comes from the actual preceding argmax. Intermediary GPU
   token views are unnecessary; only the last input is retained for the next
   chunk.

```mermaid
sequenceDiagram
    participant Loop as Scheduler / event loop
    participant Worker as Single execution worker
    participant GPU
    participant Client
    Loop->>Worker: Decode next step or bounded chunk
    Worker->>GPU: Forward, sample, pinned ID copy, event
    Worker-->>Loop: Device work queued
    par Independent host work
        Loop->>Client: Decode and emit previous immutable prefix
    and Device execution
        GPU->>GPU: Execute queued kernels
    end
    GPU-->>Worker: Copy event completes
    Worker->>Worker: Commit IDs and completion state
    Worker-->>Loop: Worker returns
    Loop->>Loop: Retire errors/cancellation; release pages safely
```

`decode_steps_per_round` defaults to one. Chunks require compilation, temperature
zero and repetition penalty 1. Other sampling uses the single-step path. Pending
prefill, waiting requests or another decode cohort also retain one-step
scheduling. Chunk size cannot exceed the shared scheduling budget, any request's
remaining output reservation, or its remaining context. EOS rows may execute
unused private positions until the bounded chunk ends, but output and history
stop at the first EOS.

Larger chunks emit text in larger bursts and check cancellation later. At batch
256, eight-step rounds take roughly 0.3 seconds in the warmed sweep. Default
single-step streaming retains finer cadence. Average time per output token does
not describe that burstiness. No default sampling or chunk-size change is needed
to obtain incremental decoding and streaming overlap.

## Power, temperature and affinity

Every final run records a `.gpu.csv` and a telemetry summary: GPU activity,
memory activity, power draw/limit, SM/memory clocks, temperature and active power
or thermal limiting reasons. GPU activity reports time with work present, not
the fraction of shader units or theoretical bandwidth in use. A low wattage
reading during model loading or graph compilation does not describe steady
decode.

Final S8 runs reached median GPU activity of 100%. Median load power was 100.6 W
for batch 256/64-token outputs and 95.6 W for batch 128/512-token outputs, against
an enforced 115 W limit. Memory clocks stayed at 8,001 MHz. Peak temperatures
were 81°C and 87°C respectively; no sampled power or thermal limiting reason was
active. The supported AC performance profile was selected through Omarchy; no
GPU overclock or power-limit increase was applied. Profiles can revert across
sessions, so reproduction must check the current setting.

The final runner also tests affinity to the fastest CPU cores detected from
sysfs. This is a benchmark experiment, not application-wide pinning. The runtime
keeps normal CPU affinity; topology-specific pinning should earn its complexity
through a repeatable measurement.

## Validation and reproduction

The final implementation passed 195 standard GPU tests (9 skipped), three local
checkpoint tests and four real-engine regressions. Lifecycle tests block the
worker while exercising cancellation and slow consumers, proving that pages stay
owned until worker completion. Metadata tests cover reuse, cohort reordering,
unexpected lengths and padded rows. Unicode tests cover incremental prefixes,
literal replacement characters and generation ending inside a multi-byte
character.

Eight diverse real requests, covering greedy and seeded history-dependent
sampling, were streamed in three configurations. All 24 token sequences, texts
and stop reasons matched. Separately, native incremental decoding matched 1,847
valid UTF-8 prefixes across 56 saved/text cases. These checks establish parity
for the tested cases; the model's [quality limitations](../quality_assessment/README.md)
remain relevant.

```bash
PY=/home/pratik/pyenvs/inference/.venv/bin/python
omarchy powerprofiles set ac performance
"$PY" benchmarks/execution_timing/run_final_overlap.py
"$PY" benchmarks/execution_timing/summarize_final_overlap.py
```

For the optional throughput setting in the server, use
`--decode-steps-per-round 8 --temperature 0 --repetition-penalty 1` together with
`--compile-model`. In Python, set the equivalent `EngineConfig` and
`SamplingConfig` fields. The four underlying optimizations have independent
`EngineConfig` switches for controlled comparisons.

`run_final_overlap.py` runs checkpoint suites in separate processes to fit 8 GB,
then tests baseline/default overlap/greedy chunks and longer output. Its final
profile is diagnostic: profiled rates are excluded from throughput claims.
`profile_full_b256_n64.trace_summary.json` quantifies CPU text-emission spans
overlapping actual GPU kernel intervals; raw traces live under
`/tmp/minivllm_overlap_traces/`.

`phase1/` contains exploratory runs before strict capture warmup and final
scheduler gating. `balanced/` records separate balanced-profile observations.
JSON files outside `final/` preserve the warmed intermediate sweep before the
last bulk-history cleanup. Use `final/` and `summary.json` for final claims.
