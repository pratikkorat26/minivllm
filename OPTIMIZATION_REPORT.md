# MiniVLLM (`LiquidAI/LFM3-1.7B`) Optimization & Verification Report

## Environment & Methodology
- **GPU**: NVIDIA GeForce RTX 4070 Laptop GPU (`7.704 GB` VRAM, SM89), `power_profile=balanced`, SM clock `2445–2460 MHz`, temp `66–75 °C`, power `22–75 W`, no thermal throttling (`throttle=0x400`).
- **Software**: PyTorch `2.15.0+cu132`, Triton `3.9.0`, `torch.bfloat16`.
- **Benchmark Rigor (`rigorous_benchmark.json`)**: Same-config comparison between baseline commit `9c23667` (`0252408^`) and `d25090f` in clean `/tmp` git worktrees, using `torch.cuda.synchronize()`-timed TTFT/prefill/decode (`--sync-timing`), 2 warmup waves followed by 5 measured repeats (`mean ± stdev`, `min`).
- **Quality Rigor (`quality_eval.json`)**: Local evaluation via `eval_quality.py` (`seed=42`, no network) over 200 teacher-forced held-out samples (`44,554` tokens from `fuvty/tau-bench-synthetic` + `nvidia/Open-SWE-Traces`) and 200 greedy prompts (`max_new_tokens=96`: 80 function-calling, 60 coding/chat, 60 math/fact/JSON).

---

## Same-Config Performance Comparison (`9c23667` vs `d25090f`)

### 1. Default BF16 Eager (`decode_steps_per_round=1`, `max_batch_tokens=256`)

| Context | Baseline TTFT (s) | After TTFT (s) | Baseline Sync Prefill (tok/s) | After Sync Prefill (tok/s) | Baseline Decode (tok/s) | After Decode (tok/s) | Decode Speedup | VRAM (Base → After) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.085 ± 0.020 | 0.077 ± 0.022 | 6,454.3 ± 1,238.9 | 7,293.3 ± 1,841.4 | 24.1 ± 1.6 (min 21.7) | 38.4 ± 5.3 (min 30.8) | **1.59x** | 3.94G → 3.67G |
| 8192 | 1.175 ± 0.030 | 1.032 ± 0.143 | 6,757.0 ± 169.1 | 7,825.3 ± 980.3 | 29.5 ± 0.9 (min 28.5) | 41.6 ± 8.9 (min 29.0) | **1.41x** | 3.96G → 3.73G |
| 32768 | 5.577 ± 0.149 | 5.149 ± 0.289 | 5,638.8 ± 145.8 | 6,128.6 ± 321.8 | 27.4 ± 1.7 (min 24.7) | 40.7 ± 6.0 (min 30.2) | **1.49x** | 4.09G → 3.98G |

### 2. Tuned BF16 Eager (`decode_steps_per_round=8`, `max_batch_tokens=2048`)

| Context | Baseline TTFT (s) | After TTFT (s) | Baseline Sync Prefill (tok/s) | After Sync Prefill (tok/s) | Baseline Decode (tok/s) | After Decode (tok/s) | Decode Speedup | VRAM (Base → After) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.060 ± 0.001 | 0.051 ± 0.002 | 8,840.2 ± 134.6 | 10,448.3 ± 329.0 | 34.4 ± 3.9 (min 30.8) | 53.7 ± 2.7 (min 49.9) | **1.56x** | 3.96G → 3.68G |
| 8192 | 1.002 ± 0.002 | 0.807 ± 0.049 | 7,821.6 ± 19.5 | 9,795.6 ± 541.8 | 36.2 ± 2.7 (min 32.9) | 55.2 ± 10.2 (min 37.4) | **1.52x** | 4.08G → 3.81G |
| 32768 | 5.004 ± 0.168 | 3.744 ± 0.111 | 6,233.1 ± 213.1 | 8,374.1 ± 197.0 | 28.9 ± 2.4 (min 25.3) | 51.6 ± 6.7 (min 39.7) | **1.79x** | 4.36G → 4.04G |

*(Note: With synchronized timing, `d25090f` is faster than `9c23667` in prefill across all contexts; the apparent Phase 1 prefill drop in `after_lfm3.json` was an unsynchronized CPU-launch measurement artifact in `baseline_lfm3.json`.)*

### 3. Quantized, Compiled, Speculative & High-Concurrency Rows (`d25090f`, `mbt=2048`, `S8`)

| Workload / Mode | Context | TTFT (s) | Sync Prefill (tok/s) | Decode (tok/s) | Min Decode | Peak VRAM (GB) |
|---|---:|---:|---:|---:|---:|---:|
| `bf16` compiled (`S8`) | 512 | 5.074 ± 11.241* | 8,932.2 ± 5,069.5 | 53.2 ± 27.5 | 4.1* | 3.67 |
| `bf16` compiled (`S8`) | 8192 | 23.030 ± 35.341* | 5,656.3 ± 5,007.7 | 50.0 ± 25.8 | 3.8* | 3.88 |
| `int8` (`gs=32`) eager | 512 | 0.171 ± 0.007 | 3,035.4 ± 113.7 | 39.9 ± 4.7 | 33.6 | 2.70 |
| `int8` (`gs=32`) eager | 8192 | 1.285 ± 0.044 | 6,092.3 ± 198.8 | 50.2 ± 2.2 | 46.9 | 2.74 |
| `fp8` eager | 512 | 0.043 ± 0.001 | 12,230.1 ± 388.5 | 35.8 ± 6.4 | 30.0 | 2.35 |
| `fp8` eager | 8192 | 0.887 ± 0.037 | 8,857.7 ± 346.4 | 36.8 ± 3.6 | 31.3 | 2.54 |
| `int4_w4a16` eager | 512 | 0.180 ± 0.001 | 2,865.7 ± 11.2 | 39.4 ± 3.1 | 35.5 | 1.82 |
| `int4_w4a16` eager | 8192 | 1.377 ± 0.032 | 5,706.1 ± 134.9 | 34.8 ± 5.3 | 26.7 | 1.89 |
| `speculative` chat (off → on) | 128 | 0.029 → 0.026 | 1,644.8 → 1,843.4 | 46.7 ± 3.4 → 52.6 ± 5.7 | 40.7 → 47.7 | 3.65 |
| `speculative` JSON (off → on) | 128 | 0.035 → 0.035 | 9,743.9 → 9,736.8 | 62.0 ± 1.7 → 59.7 ± 3.3 | 59.2 → 55.0 | 3.67 |
| `batch=64` BF16 (`9c23667` → `d25090f`) | 128 | 0.582 ± 0.016 → 0.530 ± 0.017 | 14,082.6 → 15,457.6 | 1,269.5 ± 85.9 → **1,505.7 ± 80.7** (**1.19x**) | 1,144.0 → 1,409.1 | 4.08 → 3.92 |
| `batch=64` `fp8` (`d25090f`) | 128 | 0.600 ± 0.019 | 13,661.5 ± 422.0 | 1,166.0 ± 132.1 | 1,076.6 | 2.66 |

*\*Notes*:
- `bf16` compiled variance includes Inductor/CUDA-graph compilation on the first measured repeat (`min=4.1` / `3.8`).
- `fp8`, `int8`, and `int4_w4a16` with `compile_model=True` timed out (`>360s`) in Inductor compilation and are recorded as `timeout` in `rigorous_benchmark.json`.
- Speculative decoding gains are workload-dependent (**up to 1.13x** on the measured chat prompt, **0.96x** on the measured JSON prompt).

---

## Quality Evaluation (`quality_eval.json`)

Evaluated across `44,554` teacher-forced tokens (`200` samples) and `200` greedy prompts (`max_new_tokens=96`):

| Mode | NLL | PPL | KL vs BF16 | Top-1 Agree | Greedy Exact vs BF16 | Clean EOS | Func JSON Valid | Func Tool Match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `none` (BF16) | 3.7499 | 42.5162 | 0.0000 | 1.0000 | 200/200 (100.0%) | 180/200 (90.0%) | 98.75% | 86.25% |
| `none_speculative` | 3.7499 | 42.5162 | 0.0000 | 1.0000 | **200/200 (100.0%)** | 180/200 (90.0%) | 98.75% | 86.25% |
| `int8` (`gs=32`) | 3.7503 | 42.5337 | 0.0036 | 0.9738 | 140/200 (70.0%) | 182/200 (91.0%) | **98.75%** | **86.25%** |
| `fp8` | 3.7518 | 42.5971 | 0.0532 | 0.9091 | 98/200 (49.0%) | 179/200 (89.5%) | 97.50% | 85.00% |
| `int4_w4a16` | 3.8014 | 44.7644 | 0.2686 | 0.7903 | 53/200 (26.5%) | 161/200 (80.5%) | 100.00% | 57.50% |

- **Automated Test Gate (`RUN_REAL_ENGINE_TESTS=1 pytest -v`)**: **234 passed, 5 skipped** (`tests/test_lfm3_parity_gate.py` + unit/regression/architecture suites).
- **Speculative Decoding**: **200/200 (100%) bit-exact** with plain BF16 across all 200 prompts.
- **Mixed-Precision `int8` (`gs=32`)**: Preserves function-calling tool match (`86.25%`) and JSON validity (`98.75%`) with negligible perplexity delta (`+0.0175`, KL `0.0036`) and `2.70–2.74 GB` VRAM (`6/6` bit-exact on `tests/test_lfm3_parity_gate.py` prompts).
- **`int4_w4a16` (evaluated and removed)**: Cut VRAM to `1.82–1.89 GB` but showed clear quality degradation (`PPL 44.76`, `KL 0.2686`, `clean_eos 80.5%`, `func_tool_name_match 57.5%`), so it was removed from supported quantization modes.
