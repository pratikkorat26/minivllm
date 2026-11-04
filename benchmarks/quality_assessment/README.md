# Practical output-quality audit — November 2, 2026

## Verdict

LFM3-1.7B produces useful answers on narrow grounded tasks and editable drafts.
The tested setup is unreliable for unattended SQL generation and strict-format
automation. Output control and task validation should take priority over further
throughput optimization.

The main failures reproduce in the checkpoint's official Transformers
implementation. This audit found no evidence that the throughput optimizations
are the primary cause of the poor answers. That is a limited finding on these
tasks, not proof that every execution path is correct.

## Setup and scope

- Native implementation: commit `cb3761f`.
- Checkpoint/reference revision: `ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7`.
- Real BF16 weights, RTX 4070 Laptop 8 GB, PyTorch 2.15.0+cu132,
  Transformers 5.22.1.
- 18 handwritten tasks across five configurations: 90 evaluated responses.
- 14 automatically checked task/format contracts, plus four manually reviewed
  writing, translation, summarization, and Python tasks.
- Maximum 256 generated tokens; explicit stop reasons recorded.
- Thinking disabled through the checkpoint's official template. Native formatting
  matches the official template on every task, including the multi-turn example.
- Native optimized mode enables compilation and compatible prefill batching and
  submits groups of up to four requests. Custom paged attention is disabled.
- Long-context probes contain 7,109 formatted input tokens, with a target record
  placed at the beginning, middle, and end. They are synthetic retrieval probes,
  not a multi-turn or 256K-context capability benchmark.
- Sampled configurations use seed 42 once per task. This does not establish
  reliability across seeds or an unbiased population-level accuracy estimate.

Strict grading counts a correct number with extra explanation as a failure when
the task requires only the number. Likewise, JSON inside Markdown fails a
JSON-only contract. **The following scores are not factual-answer accuracy.**

| Configuration | Strict contracts passed | Truncated responses |
|---|---:|---:|
| Native eager, greedy | 9/14 | 0 |
| Native current defaults: temperature 0.8, top-p 0.9, repetition penalty 1.1 | 8/14 | 1 |
| Native recommended general sampling: temperature 0.1, top-k 50, repetition penalty 1.1 | 8/14 | 0 |
| Native optimized, greedy | 8/14 | 0 |
| Official Transformers reference, greedy | 8/14 | 0 |

Optimized and reference execution have identical pass/fail outcomes on all 14
strict checks; 15 of their 18 response texts match exactly. Native eager matches
14 reference texts exactly. BF16 execution shapes can change near-tied choices:
the native eager long-end answer is terse, while optimized and reference answers
retrieve the same correct code with unwanted explanation and a reasoning tag.

## What was useful

- **Extraction:** returned a valid JSON object with Maya, quantity 3, and high
  priority, without invented fields.
- **Grounding:** answered `Lena` from supplied project facts and correctly returned
  `NOT PROVIDED` for an absent budget.
- **Classification and ordering:** produced the expected sentiment label and
  sorted list.
- **Long-context retrieval:** all five configurations recovered the correct target
  code at all three positions. Some added unwanted text despite the code-only
  instruction, so recovery and strict-format compliance are distinct results.
- **Writing:** the meeting email is a useful editable draft with the requested
  days and no invented reason.
- **Summarization:** preserves the core chronology and unresolved root cause,
  though it awkwardly says errors were rolled back rather than the deployment.
- **Translation:** preserves the meeting-date meaning, with slightly unidiomatic
  French phrasing (`au lundi` rather than `à lundi` in this context).
- **Simple Python:** the reviewed `dedupe` functions pass five bounded functional
  checks each after removing Markdown. The default sampled variant uses list
  membership with quadratic worst-case complexity; the others use a set.

## Concrete failures

### Output instructions are not reliable

The cafe arithmetic answer correctly calculates `18 * 7 - 42 = 84` but returns
an explanation despite a request for only the dollar amount. The train timetable
answer correctly finds `08:50`, with the same unwanted explanation.

Multi-turn recall correctly returns Rina and Osaka, but usually encloses the JSON
in Markdown. Python output is also fenced despite an explicit plain-code request.

### SQL can be substantively wrong

Given a table called `orders`, every baseline configuration generates a query
using `table_orders`. It fails against the supplied in-memory SQLite schema with
`no such table: table_orders`. The official reference produces the same error.

A follow-up with an explicit `CREATE TABLE orders ...` schema and stricter system
instructions fixes the identifier but drops `WHERE status = 'paid'` and emits a
stray Markdown fence. After removing the fence, its result includes pending and
cancelled orders: `(1,150), (3,99), (2,25)` instead of `(1,50), (2,25)`.
This is a semantic error, not just a formatting problem.

### Reasoning can leak or fail to finish

The current default sampled arithmetic response spends all 256 tokens debating
whether to print `84` or `$84` and hits the token limit. Other outputs contain
`</think>` despite thinking being disabled. Recommended lower-temperature sampling
avoids that truncation in this example but does not solve strict-format compliance.

The model author also notes that disabling thinking can be inconsistent, and
does not recommend this small checkpoint for agentic coding or complex multi-step
reasoning. See the [model card](https://huggingface.co/LiquidAI/LFM3-1.7B).

## Prompting follow-up

Three diagnostic retries use a strict system instruction and more explicit output
requirements. Multi-turn JSON becomes valid. SQL remains wrong as described above.
Asking for the arithmetic result as JSON first produces an unevaluated expression
and then visible reasoning before a final JSON object. The entire response is
invalid JSON. Prompting is helpful on some tasks but is not a sufficient guarantee.

## Serving capability gap

The current API has no wired `response_format`/JSON-schema constrained decoding
or tool-call protocol. Its message roles are system, user, and assistant. Extra
`response_format` and `tools` fields are currently ignored by the request schema;
they cannot be relied on to enforce output contracts. This was checked locally
without executing model-generated tools. Model-card tool capability should not
be interpreted as implemented tool support in this server.

## Recommended next steps

1. Select one intended workflow and build a held-out set of representative inputs
   with task-specific success criteria before continuing throughput comparisons.
2. For extraction, validate schemas and handle failures explicitly. Do not equate
   a natural-language JSON instruction with constrained decoding.
3. Treat SQL and generated code as proposals. Validate identifiers, query results,
   and function behavior against an isolated test environment before use.
4. Evaluate sampling and reasoning presentation on that workflow; current default
   temperature 0.8 is not the checkpoint author's recommended general preset.
5. Keep a quality regression suite alongside performance measurements. If the
   selected workflow needs stronger reasoning/code reliability, compare a more
   suitable checkpoint before spending more effort on throughput.

## Artifacts and reproduction

- `summary.json`: strict results, reference agreement, and manual review notes.
- `*_greedy.json`, `default_sampling.json`, `recommended_sampling.json`: complete
  prompts, responses, token IDs, stop reasons, and grades.
- `chat_template_comparison.json`: exact official/native template comparison.
- `reference_smoke.json`: initial six-prompt independent reference comparison.
- `prompting_probe.json`: diagnostic retries, including the SQL semantic result.
- `reviewed_python_checks.json`: bounded tests of the reviewed generated functions.
- `api_capability_check.json`: request-schema capability check.

With the existing environment, local weights, and cached official tokenizer/model
code, reproduce the main audit from the repository root:

```bash
python benchmarks/quality_assessment/evaluate.py
```

The native/reference processes run sequentially to fit within the GPU memory
limit. This overwrites the main audit JSON files. Run the prompting follow-up with
`python benchmarks/quality_assessment/prompt_probe.py`.

No production inference code was changed as part of this assessment.
