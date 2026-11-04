# Verification of the 8K prefill result

This reruns the committed LFM3-1.7B benchmark to check the prefill claim before
sharing it publicly. Source implementation: commit `cb3761f`.

## Method

- RTX 4070 Laptop 8 GB; BF16; PyTorch 2.15.0+cu132.
- One request at a time, 8,118 input tokens including the chat template.
- 2,048-token prefill chunks/budget, 16,384-token KV capacity, 64 output tokens.
- Greedy sampling, compilation enabled, custom paged attention disabled.
- Prefix caching disabled; each measured request must process all 8,118 tokens.
- Two inputs: the original repeated sentence and varied repository documentation.
- Three independent fresh processes per input; two warmup requests followed by
  ten measured requests in each process: 60 measured requests total.
- Model loading and warmup excluded. Compiler disk caches are retained, so this
  does not measure a completely cold compilation or startup.
- CUDA events measure worker-phase time. Request TTFT independently measures
  submit-to-first-token latency, including scheduling and tokenization.
- Input-token and decode-token counts are checked on every measured request.
- GPU clocks and power are not locked. These are engine measurements, not HTTP
  serving measurements or a comparison with another inference engine.

## Results

| Input | Measured requests | Prefill input tokens/s | Median TTFT | P95 TTFT | Decode output tokens/s |
|---|---:|---:|---:|---:|---:|
| Original repeated text | 30 | 9,295 | 0.880 s | 0.900 s | 50.9 |
| Varied documentation | 30 | 9,305 | 0.880 s | 0.899 s | 51.1 |

The six process-level prefill rates range from 9,269 to 9,324 input tokens/s.
All 60 measured requests completed successfully and processed exactly 8,118 input
tokens. Output hashes matched across the three processes for each input. The
original approximately 9,200 input tokens/s result reproduced on both text types.

Dividing input tokens by the complete request TTFT instead of worker-phase
GPU time gives approximately 9,194 input tokens/s for repeated text and 9,199
for documentation. The difference is about 1% and includes tokenization and
scheduling overhead. The TTFT result therefore supports the worker-phase rate.

A defensible public statement is:

> My PyTorch LFM3-1.7B engine processed 8K-token prompts at about 9,300 input
> tokens/s during warm prefill on an RTX 4070 Laptop 8 GB. Across 60 requests
> in six fresh processes, median time to first token was about 0.88 seconds;
> subsequent decoding was about 51 output tokens/s. BF16, torch.compile,
> 2,048-token chunks, no custom handwritten CUDA kernels; loading and warmup
> excluded.

`summary.json` aggregates all measured requests. Each `*_requests.json` contains
per-request TTFT, TPOT, phase times, token counts, and the source commit. Each
`*_engine.json` records full configuration, prompt text, warmup, output hashes,
and the original benchmark's aggregate measurements.

## Reproduce

Install the repo requirements and place the local model/tokenizer files in
`LFM3-1.7B/`, then run from the repository root:

```bash
python benchmarks/prefill_verification/verify.py
```

The runner executes six fresh GPU processes sequentially and overwrites this
folder's measurement JSON files. It uses the original committed 8K prompt as its
reference configuration. Documentation content is trimmed to exactly 8,118
formatted input tokens; future documentation revisions may change its wording.

The defensible claim is a warm, single-request prefill rate on this workload.
Generated-text decode speed is a separate measurement. These data do not establish
state-of-the-art performance, cross-engine superiority, or general model quality.
