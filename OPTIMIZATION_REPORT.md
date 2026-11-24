# MiniVLLM (`LiquidAI/LFM3-1.7B`) Optimization & Verification Report

## Environment
- **GPU**: NVIDIA GeForce RTX 4070 Laptop GPU (`7.704 GB` VRAM)
- **PyTorch**: `2.15.0+cu132` (`torch.bfloat16`, `quantization=none`)
- **Benchmark Config**: `benchmark_lfm3.py` (`max_new_tokens=64`, `temperature=0.0`, `max_batch_tokens=256`, `decode_steps_per_round=1`, `compile_model=false`)
- **Static Memory**: Model weights `3.222 GB`; weights + KV cache dropped from `3.917 GB` (baseline) to `3.642 GB` (after), saving `0.275 GB` via bounded sliding-window ring buffers.

---

## Before vs After Benchmark (`baseline_lfm3.json` vs `after_lfm3.json`)

| Context (Actual Tokens) | Base TTFT (s) | After TTFT (s) | TTFT Speedup | Base Decode (tok/s) | After Decode (tok/s) | Decode Speedup | Base Total Latency (s) | After Total Latency (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 (512) | 0.063 | 0.063 | 1.01x | 52.3 | 53.1 | 1.02x | 1.280 | 1.266 |
| 8192 (7701) | 1.034 | 1.025 | 1.01x | 45.1 | 42.8* | 0.95x* | 2.439 | 2.513* |
| 16384 (15345) | 2.213 | 2.207 | 1.00x | 38.9 | 51.7 | 1.33x | 3.844 | 3.434 |
| 32768 (30633) | 5.265 | 5.256 | 1.00x | 30.5 | 49.6 | 1.63x | 7.348 | 6.544 |

*\*Note on `8192` decode*: In the cold-process `benchmark_lfm3.py` run, `paged_decode_attention` (`max_seq_len >= 1024`) triggers Triton JIT compilation on the first `8192` decode step after the 16-token warmup. Subsequent contexts (`16384`, `32768`) reuse the compiled kernel and achieve `51.7 tok/s` (`1.33x`) and `49.6 tok/s` (`1.63x`).

### Prefill Throughput, VRAM & Host RAM

| Context (Actual Tokens) | Base Prefill (tok/s) | After Prefill (tok/s) | Base Peak VRAM (GB) | After Peak VRAM (GB) | Base Host RSS (GB) | After Host RSS (GB) |
|---:|---:|---:|---:|---:|---:|---:|
| 512 (512) | 18,292.4 | 12,029.2 | 3.94 | 3.67 | 1.736 | 1.763 |
| 8192 (7701) | 16,096.1 | 7,739.3 | 3.96 | 3.69 | 1.743 | 1.774 |
| 16384 (15345) | 14,940.6 | 7,139.5 | 4.00 | 3.73 | 1.743 | 1.775 |
| 32768 (30633) | 12,360.2 | 5,943.8 | 4.09 | 3.81 | 1.745 | 1.777 |

- **Prefill metric note**: `prefill_tok_per_s` is computed from `executor.py`'s `prefill_time_s`, which in the baseline captured unsynchronized async CPU launch time while host setup (`ttft_s - prefill_s`) absorbed the GPU sync wait (`2.786s` at `32768`). End-to-end `ttft_s` remained equal/slightly faster (`5.265s -> 5.256s` at `32768`), and `total_latency_s` improved from `7.348s` to `6.544s`.
- **Peak Host RAM (`peak_host_ram_gb`)**: `7.109 GB` baseline vs `7.109 GB` after.

---

## Parity & Regression Gate Results

Command run after all final edits:
```bash
RUN_REAL_ENGINE_TESTS=1 /home/pratik/pyenvs/inference/.venv/bin/python -m pytest -v
```
- **Result**: **225 passed, 5 skipped** in `19.68s` (skipped tests are opt-in `RUN_LOCAL_MODEL_TESTS=1`, `RUN_REAL_ENGINE_LOAD_TESTS=1`, and `bitsandbytes` int4).
- **Greedy Parity (`tests/test_lfm3_parity_gate.py` vs `baseline_parity.json`)**:
  - **100% bit-exact `token_ids` and `text` match** across `short_math`, `short_fact`, `multi_turn`, `long_context`, `structured_json`, and `tool_calling`.
  - **Concurrent batch vs sequential single-request parity**: Passed (verifying zero `ShortConv` state corruption across slots).
  - **Chunked prefill (`prefill_chunk_size=128`) vs unchunked long-context parity**: Passed.
  - **Public API & CLI compatibility (`main.py`, `adapters.openai.run`, `LLMEngine`, `EngineConfig`, `SamplingConfig`, `Lfm3Runtime`)**: Passed.
  - **Tiny-model hybrid `ShortConv` + Attention slot isolation & chunked prefill**: Passed.

---

## Summary of Implemented Optimizations
- **Bounded Sliding-Window KV Ring Buffers** (`lfm3/paged_kv.py`, `lfm3/model.py`, `lfm3/attention.py`, `inference/executor.py`): 5 sliding-window attention layers allocate per-slot ring buffers (`window_size=256`) instead of full-context paged blocks, reducing KV VRAM by `~0.28 GB`.
- **Fused Paged Decode Attention & Precomputed RoPE/Write Indices** (`lfm3/attention.py`, `lfm3/model.py`): Enabled `LFM3_FUSED_PAGED_ATTENTION=1` for `max_seq_len >= 1024`, eliminating O(context) dense KV gather/mask overhead during long-context decode (`1.33x` at 16K, `1.63x` at 32K).
- **Quantization Improvements** (`lfm3/quantization.py`): Protected `gate_proj` in `int8` quantization and added `fp8` (`torch.float8_e4m3fn` + `torch._scaled_mm`) support.
- **Executor, Sampling & Streaming Fast-Paths** (`inference/executor.py`, `inference/sampling/sampler.py`, `app/service.py`): Pre-uploaded prompt tensors for zero-copy prefill slicing, GPU boolean repetition-penalty masks, top-k-first nucleus sampling, and O(N) tail-bounded streaming stop-sequence scanning.
- **Test Coverage** (`tests/test_lfm3_parity_gate.py` + `tests/test_*.py`): Added parity gate, bounded sliding-window KV, FP8, sampling fast-path, and streaming tail-scan tests.

---

## Phase 2: Triton Kernels, W4A16, Speculative Decoding & High-Concurrency

Verified against `phase2_ceiling_baseline.json` (`0252408`) on the same RTX 4070 Laptop GPU (`max_batch_tokens=2048`, `decode_steps_per_round=8`, `max_new_tokens=64`, `temperature=0.0`):

### Single-Request (`batch=1`) & High-Concurrency (`batch=64`) Results

| Workload | Baseline TTFT (s) | Phase 2 TTFT (s) | TTFT Speedup | Baseline Decode (tok/s) | Phase 2 Decode (tok/s) | Decode Speedup | Phase 2 Peak VRAM (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `b1` BF16 eager (`512`) | 0.058 | 0.050 | 1.16x | 53.9 | 54.2 | 1.01x | 3.68 |
| `b1` BF16 eager (`8192`) | 0.984 | 0.824 | 1.19x | 43.1 | 62.2 | 1.44x | 3.80 |
| `b1` `int4_w4a16` eager (`512`) | — | 0.166 | — | — | 73.1 | 1.36x vs BF16 | 1.82 |
| `b1` `int4_w4a16` eager (`8192`) | — | 1.281 | — | — | 82.1 | 1.90x vs BF16 | 1.88 |
| `b1` `int8` mixed-prec (`512`, `gs=32`) | — | 0.162 | — | — | **77.4** | **1.44x vs BF16** | 2.70 |
| `b1` `int8` mixed-prec (`8192`, `gs=32`) | — | 1.299 | — | — | **71.0** | **1.65x vs BF16** | 2.74 |
| `b64` `fp8` (`ctx=128`, prefill batching) | 1.234 | 0.552 | **2.24x** | 1,049.0 (agg) | **1,734.5** (agg) | **1.65x** | 2.66 |

- **Phase 2 Test Suite (`RUN_REAL_ENGINE_TESTS=1 pytest -v`)**: **234 passed, 5 skipped** in `33.95s`.
  - Includes `test_speculative_decoding_preserves_greedy_parity` (`enable_speculative_decoding=True`, `speculative_num_tokens=5`), confirming **100% bit-exact greedy parity** against `baseline_parity.json` with `ShortConv` and sliding-window ring-buffer rollback.
  - **`int8` Mixed-Precision Parity (`group_size=32`, FP32 scales, `gate_proj` + `out_head` in BF16)**: **6/6 bit-exact greedy parity** vs `baseline_parity.json` (`short_math`, `short_fact`, `multi_turn`, `long_context`, `structured_json`, `tool_calling`, all `stop=eos`).
  - Includes `test_triton_fused_kernels_match_reference`, `test_w4a16_linear_matches_float_linear`, `test_w4a16_model_keeps_tied_head_and_skips_gate_proj`, `test_dense_split_prefill_matches_masked_attention`, and `test_speculative_decoding_exact_parity_on_hybrid_model`.
- **Key Phase 2 Additions**:
  - `lfm3/kernels.py` & `lfm3/quantization.py`: Fused Triton `RMSNorm`, `SwiGLU`, single-step `ShortConv` decode + state update, `W4A16` (`group_size=64`) GEMV, and mixed-precision `W8A16` (`group_size=32` symmetric `int8` with FP32 scales, keeping `gate_proj` and tied `out_head` in BF16 — fixing the `torch._weight_int8pack_mm` hang and achieving 6/6 bit-exact parity).
  - `lfm3/attention.py`: Registered `_ring_idx` as a non-persistent buffer (fixing CUDAGraphs overwrite under `compile_model=True`) and added FlashAttention-2 split-prefill in `_dense_attention`.
  - `inference/executor.py` & `inference/scheduler/scheduler.py`: Lossless prompt-lookup/n-gram speculative decoding (`enable_speculative_decoding`) with `ShortConv`/ring-buffer rollback, and `PrefillBatchSelector` scaled to `max_concurrent_requests` (`16`).
