# Prefill shape scheduling experiment

The scheduler now defers an oldest prefill chunk that does not fit the remaining
token budget when more than one prefill request remains in the round. This avoids
some small, single-row partial forwards and lets compatible chunks batch in a
later round. If decode repeatedly leaves too little room, the next tick gives
prefill priority. A lone remaining request can still be truncated, and the
schedule is not globally fixed. Decode already uses power-of-two shape buckets.

## Measurements

Aggregate streaming output tokens/s, including prefill and scheduling, on the
RTX 4070 Laptop 8 GB, BF16, compiled, 2,048-token budget. The previous columns
are the recorded pre-change runs, not paired simultaneous measurements; clocks
and temperatures were not locked. S1/S8 denote decode steps per round. The
default backend is memory-efficient attention; cuDNN is separately opt-in.

| Workload | Previous default | New default | Previous cuDNN | New cuDNN |
|---|---:|---:|---:|---:|
| 256 requests, 64 tokens, S1 | 5,016 | 5,148 (+2.6%) | 5,896 | 6,044 (+2.5%) |
| 256 requests, 64 tokens, S8 | 5,250 | 5,767 (+9.8%) | 6,206 | 6,352 (+2.4%) |
| 128 requests, 512 tokens, S1 | 2,990 | 3,254 (+8.8%) | 3,710 | 3,798 (+2.4%) |
| 128 requests, 512 tokens, S8 | 3,104 | 3,339 (+7.6%) | 3,855 | 3,870 (+0.4%) |

The small differences in particular may be measurement noise. For the short
workload, Dynamo reported 5 to 4 unique graphs with the default backend and 6
to 4 with cuDNN. For the long workload it reported 5 in both versions; fewer
graphs are not guaranteed. All measured waves reported no compilation or
capture during measurement. The new raw waves, including aggregate GPU
telemetry, are in `final/`; the prior runs are in
[`overlap_optimization/final`](../overlap_optimization/final/) and
[`cudnn_attention/final`](../cudnn_attention/final/).

## Output and fairness checks

Changing cohorts can change BF16 reduction order and greedy near-ties. With the
default backend and the benchmark prompts, 253/256 short-output rows and
125/128 long-output rows matched their previous token sequences. Every new
sequence was already present among the prior run's sequences: the long prompt
first diverged at token indices 141 (two rows) and 316 (one row). These repeated
prompts are not independent samples. The 18-task compiled greedy quality audit
produced 18/18 identical response texts and retained 8/14 strict passes; see
[`quality_optimized_greedy.json`](quality_optimized_greedy.json) and the
[audit limitations](../quality_assessment/README.md). This is not proof of
identical outputs or quality for other inputs. cuDNN's separate nondeterminism
also remains unaffected.

Scheduler tests cover deferral and progress with busy decode; the full suite
passed 200 tests (9 skipped), plus 3 local-model and 4 real-engine tests.

## Reproduction

Use the [handoff harness](../execution_timing/handoff.py) with the same settings
as the [previous benchmark](../overlap_optimization/README.md), adding
`--cudnn-attention` for the opt-in backend. For example:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 LFM3_FUSED_PAGED_ATTENTION=0 \
  /home/pratik/pyenvs/inference/.venv/bin/python benchmarks/execution_timing/handoff.py \
  --batch-size 256 --output-tokens 64 --variant full --decode-steps 1 \
  --stream --require-platform-profile performance --repeats 5 \
  --output /tmp/opencode/prefill-shape.json
```

The 512-token prompt is recorded in each corresponding raw JSON file.
