"""Benchmark script for MiniVLLM (LiquidAI/LFM3-1.7B).

Measures TTFT, prefill throughput, decode throughput, peak VRAM, and host RAM
across configurable context lengths, with graceful OOM handling and JSON
export/comparison support.
"""

import argparse
import asyncio
import gc
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import torch

from inference import EngineConfig, GenerateRequest, LLMEngine, SamplingConfig
from lfm3.quantization import QUANTIZATION_CHOICES
from runtime import DEFAULT_MODEL_ID, Lfm3Runtime, get_device


def _rss_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)


def _max_rss_gb() -> float:
    # ru_maxrss is in KiB on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)


def _gpu_telemetry() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 5:
            info["sm_clock_mhz"] = int(parts[0]) if parts[0].isdigit() else parts[0]
            info["mem_clock_mhz"] = int(parts[1]) if parts[1].isdigit() else parts[1]
            info["temp_c"] = int(parts[2]) if parts[2].isdigit() else parts[2]
            try:
                info["power_w"] = float(parts[3])
            except ValueError:
                info["power_w"] = parts[3]
            info["throttle_reasons"] = parts[4]
    except Exception as exc:
        info["nvidia_smi_error"] = str(exc)
    try:
        info["power_profile"] = subprocess.check_output(
            ["powerprofilesctl", "get"], text=True, timeout=2
        ).strip()
    except Exception:
        pass
    return info


def _build_exact_prompt(
    tokenizer: Any, target_tokens: int, prompt_type: str = "text"
) -> tuple[str, int]:
    """Construct a deterministic prompt of ~target_tokens for text, chat, or json."""
    if prompt_type == "chat":
        prompt = (
            "<|startoftext|><|im_start|>user\n"
            "Explain in clear technical detail how paged KV caching and sliding-window "
            "attention reduce memory bandwidth during autoregressive LLM decoding, "
            "and compare them to standard dense attention.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return prompt, len(tokenizer.encode(prompt))
    if prompt_type == "json":
        record = (
            '{"service": "payments-api", "region": "us-east-1", "status": "healthy", '
            '"latency_ms": 12.4, "retries": 0, "version": "2.4.1"}, '
        )
        prompt = (
            "<|startoftext|><|im_start|>user\n"
            "Repeat and extend this exact JSON array by adding 5 more identical-schema entries:\n"
            "[" + (record * 6).rstrip(", ") + "]\n"
            "<|im_end|>\n<|im_start|>assistant\n["
        )
        return prompt, len(tokenizer.encode(prompt))

    unit = "The quick brown fox jumps over the lazy dog in a quiet forest. "
    unit_ids = tokenizer.encode(unit)
    reps = max(1, (target_tokens // max(1, len(unit_ids))) + 4)
    text = unit * reps
    ids = tokenizer.encode(text)
    if len(ids) > target_tokens:
        ids = ids[:target_tokens]
    prompt = tokenizer.decode(ids)
    actual_len = len(tokenizer.encode(prompt))
    # Trim one token at a time if BPE re-encoding changed length slightly.
    while actual_len > target_tokens and len(ids) > 1:
        ids = ids[:-1]
        prompt = tokenizer.decode(ids)
        actual_len = len(tokenizer.encode(prompt))
    return prompt, actual_len


def _is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


class _SyncTimer:
    """Measures true GPU-synchronized prefill, TTFT, and decode time without touching engine code."""

    def __init__(self, executor: Any) -> None:
        self.executor = executor
        self._orig_prefill = executor.prefill_batch
        self._orig_decode = executor.decode
        self.prefill_s = 0.0
        self.decode_s = 0.0
        self.first_token_wall_s: Optional[float] = None
        self.request_start_s: float = 0.0

    def reset(self) -> None:
        self.prefill_s = 0.0
        self.decode_s = 0.0
        self.first_token_wall_s = None
        self.request_start_s = time.perf_counter()

    def install(self) -> None:
        orig_prefill = self._orig_prefill
        orig_decode = self._orig_decode

        def wrapped_prefill(*args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = orig_prefill(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            self.prefill_s += t1 - t0
            if out and self.first_token_wall_s is None:
                self.first_token_wall_s = t1
            return out

        def wrapped_decode(*args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = orig_decode(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.decode_s += time.perf_counter() - t0
            return out

        self.executor.prefill_batch = wrapped_prefill
        self.executor.decode = wrapped_decode

    def uninstall(self) -> None:
        self.executor.prefill_batch = self._orig_prefill
        self.executor.decode = self._orig_decode


async def _run_single_context(
    engine: LLMEngine,
    runtime: Lfm3Runtime,
    target_tokens: int,
    max_new_tokens: int,
    sampling: SamplingConfig,
    req_id: str,
    prompt_type: str = "text",
    sync_timer: Optional[_SyncTimer] = None,
) -> Dict[str, Any]:
    prompt, actual_tokens = _build_exact_prompt(runtime.tokenizer, target_tokens, prompt_type)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    if sync_timer is not None:
        sync_timer.reset()

    try:
        res = await engine.generate(
            GenerateRequest(
                request_id=req_id,
                prompt=prompt,
                sampling=sampling,
                max_new_tokens=max_new_tokens,
            )
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if res.stop_reason in ("error", "capacity_exceeded"):
            return {
                "target_context_tokens": target_tokens,
                "prompt_tokens": actual_tokens,
                "status": "oom" if "capacity" in (res.error_message or "").lower() else "error",
                "stop_reason": res.stop_reason,
                "error": res.error_message,
                "peak_vram_allocated_gb": (
                    torch.cuda.max_memory_allocated() / (1024**3)
                    if torch.cuda.is_available()
                    else 0.0
                ),
                "peak_vram_reserved_gb": (
                    torch.cuda.max_memory_reserved() / (1024**3)
                    if torch.cuda.is_available()
                    else 0.0
                ),
                "host_ram_rss_gb": _rss_gb(),
            }

        gen_tokens = len(res.token_ids)
        prefill_tps = actual_tokens / res.prefill_s if res.prefill_s > 0 else 0.0
        decode_tps = res.decode_steps / res.decode_s if res.decode_s > 0 else 0.0
        out: Dict[str, Any] = {
            "target_context_tokens": target_tokens,
            "prompt_tokens": actual_tokens,
            "generated_tokens": gen_tokens,
            "decode_steps": res.decode_steps,
            "prefill_steps": res.prefill_steps,
            "stop_reason": res.stop_reason,
            "status": "ok",
            "ttft_s": res.ttft_s,
            "prefill_s": res.prefill_s,
            "prefill_tok_per_s": prefill_tps,
            "decode_s": res.decode_s,
            "decode_tok_per_s": decode_tps,
            "tpot_s": res.tpot_s,
            "total_latency_s": res.total_latency_s,
            "peak_vram_allocated_gb": (
                torch.cuda.max_memory_allocated() / (1024**3)
                if torch.cuda.is_available()
                else 0.0
            ),
            "peak_vram_reserved_gb": (
                torch.cuda.max_memory_reserved() / (1024**3)
                if torch.cuda.is_available()
                else 0.0
            ),
            "host_ram_rss_gb": _rss_gb(),
        }
        if sync_timer is not None:
            sync_ttft = (
                sync_timer.first_token_wall_s - sync_timer.request_start_s
                if sync_timer.first_token_wall_s is not None
                else res.ttft_s
            )
            out["sync_ttft_s"] = sync_ttft
            out["sync_prefill_s"] = sync_timer.prefill_s
            out["sync_prefill_tok_per_s"] = (
                actual_tokens / sync_timer.prefill_s if sync_timer.prefill_s > 0 else 0.0
            )
            out["sync_decode_s"] = sync_timer.decode_s
            out["sync_decode_tok_per_s"] = (
                res.decode_steps / sync_timer.decode_s if sync_timer.decode_s > 0 else 0.0
            )
        return out
    except BaseException as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return {
            "target_context_tokens": target_tokens,
            "prompt_tokens": actual_tokens,
            "status": "oom" if _is_oom_error(exc) else "error",
            "error": f"{type(exc).__name__}: {exc}",
            "peak_vram_allocated_gb": (
                torch.cuda.max_memory_allocated() / (1024**3)
                if torch.cuda.is_available()
                else 0.0
            ),
            "peak_vram_reserved_gb": (
                torch.cuda.max_memory_reserved() / (1024**3)
                if torch.cuda.is_available()
                else 0.0
            ),
            "host_ram_rss_gb": _rss_gb(),
        }


async def _run_batch_context(
    engine: LLMEngine,
    runtime: Lfm3Runtime,
    target_tokens: int,
    max_new_tokens: int,
    sampling: SamplingConfig,
    batch_size: int,
    req_prefix: str,
    prompt_type: str = "text",
    sync_timer: Optional[_SyncTimer] = None,
) -> Dict[str, Any]:
    if batch_size == 1:
        return await _run_single_context(
            engine,
            runtime,
            target_tokens=target_tokens,
            max_new_tokens=max_new_tokens,
            sampling=sampling,
            req_id=req_prefix,
            prompt_type=prompt_type,
            sync_timer=sync_timer,
        )

    prompt, actual_tokens = _build_exact_prompt(runtime.tokenizer, target_tokens, prompt_type)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    if sync_timer is not None:
        sync_timer.reset()

    try:
        t0 = time.perf_counter()
        results = await asyncio.gather(
            *(
                engine.generate(
                    GenerateRequest(
                        request_id=f"{req_prefix}-{i}",
                        prompt=prompt,
                        sampling=sampling,
                        max_new_tokens=max_new_tokens,
                    )
                )
                for i in range(batch_size)
            )
        )
        wall_s = time.perf_counter() - t0
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        bad = [r for r in results if r.stop_reason in ("error", "capacity_exceeded")]
        if bad:
            return {
                "target_context_tokens": target_tokens,
                "prompt_tokens": actual_tokens,
                "batch_size": batch_size,
                "status": "oom" if "capacity" in (bad[0].error_message or "").lower() else "error",
                "stop_reason": bad[0].stop_reason,
                "error": bad[0].error_message,
                "peak_vram_allocated_gb": (
                    torch.cuda.max_memory_allocated() / (1024**3)
                    if torch.cuda.is_available()
                    else 0.0
                ),
                "host_ram_rss_gb": _rss_gb(),
            }

        total_gen = sum(len(r.token_ids) for r in results)
        total_decode_steps = sum(r.decode_steps for r in results)
        ttfts = [r.ttft_s for r in results if r.ttft_s is not None]
        mean_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
        max_ttft = max(ttfts) if ttfts else 0.0
        total_prompt_tokens = actual_tokens * batch_size
        return {
            "target_context_tokens": target_tokens,
            "prompt_tokens": actual_tokens,
            "batch_size": batch_size,
            "total_generated_tokens": total_gen,
            "total_decode_steps": total_decode_steps,
            "wall_s": wall_s,
            "status": "ok",
            "ttft_s": mean_ttft,
            "max_ttft_s": max_ttft,
            "throughput_total_tok_per_s": (total_prompt_tokens + total_gen) / wall_s if wall_s > 0 else 0.0,
            "throughput_gen_tok_per_s": total_gen / wall_s if wall_s > 0 else 0.0,
            "prefill_tok_per_s": total_prompt_tokens / mean_ttft if mean_ttft > 0 else 0.0,
            "decode_tok_per_s": total_gen / wall_s if wall_s > 0 else 0.0,
            "peak_vram_allocated_gb": (
                torch.cuda.max_memory_allocated() / (1024**3)
                if torch.cuda.is_available()
                else 0.0
            ),
            "peak_vram_reserved_gb": (
                torch.cuda.max_memory_reserved() / (1024**3)
                if torch.cuda.is_available()
                else 0.0
            ),
            "host_ram_rss_gb": _rss_gb(),
        }
    except BaseException as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return {
            "target_context_tokens": target_tokens,
            "prompt_tokens": actual_tokens,
            "batch_size": batch_size,
            "status": "oom" if _is_oom_error(exc) else "error",
            "error": f"{type(exc).__name__}: {exc}",
            "peak_vram_allocated_gb": (
                torch.cuda.max_memory_allocated() / (1024**3)
                if torch.cuda.is_available()
                else 0.0
            ),
            "host_ram_rss_gb": _rss_gb(),
        }


def _summarize_metric(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"mean": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": statistics.mean(vals),
        "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def _aggregate_repeats(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(runs) == 1:
        return runs[0]
    bad = [r for r in runs if r.get("status") != "ok"]
    if bad:
        return bad[0]
    base = dict(runs[0])
    metric_keys = [
        "ttft_s",
        "prefill_s",
        "prefill_tok_per_s",
        "decode_s",
        "decode_tok_per_s",
        "tpot_s",
        "total_latency_s",
        "peak_vram_allocated_gb",
        "sync_ttft_s",
        "sync_prefill_s",
        "sync_prefill_tok_per_s",
        "sync_decode_s",
        "sync_decode_tok_per_s",
        "wall_s",
        "throughput_gen_tok_per_s",
        "throughput_total_tok_per_s",
    ]
    stats: Dict[str, Dict[str, float]] = {}
    for k in metric_keys:
        if k in runs[0] and runs[0][k] is not None:
            vals = [float(r[k]) for r in runs if r.get(k) is not None]
            if vals:
                s = _summarize_metric(vals)
                stats[k] = s
                base[k] = s["mean"]
    base["repeats"] = len(runs)
    base["stats"] = stats
    base["raw_runs"] = runs
    return base


async def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    if args.fused_paged_attention:
        os.environ["LFM3_FUSED_PAGED_ATTENTION"] = "1"

    contexts = [int(c.strip()) for c in args.contexts.split(",") if c.strip()]
    batch_size = max(1, int(args.batch_size))
    max_kv = args.max_kv_cache_tokens or max(
        32_768, batch_size * (max(contexts) + args.max_new_tokens + 32)
    )

    device = get_device()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    total_vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )

    t_load0 = time.perf_counter()
    runtime = Lfm3Runtime(
        model_id=args.model_id,
        use_instruct_model=False,
        device=device,
        quantization=args.quantization,
        compile_model=args.compile_model,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_s = time.perf_counter() - t_load0
    weights_vram_gb = (
        torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    )

    sampling = SamplingConfig(
        temperature=args.temperature,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
    )
    config_kwargs: Dict[str, Any] = dict(
        use_instruct_model=False,
        max_new_tokens=args.max_new_tokens,
        sampling=sampling,
        max_kv_cache_tokens=max_kv,
        max_batch_tokens=args.max_batch_tokens,
        decode_batch_size=max(4, batch_size),
        max_concurrent_requests=max(16, batch_size),
        max_queue_size=max(128, batch_size * 2),
        enable_prefill_batching=args.enable_prefill_batching,
        enable_cudnn_attention=args.enable_cudnn_attention,
        decode_steps_per_round=args.decode_steps_per_round,
    )
    if args.enable_speculative:
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(EngineConfig)}
        if "enable_speculative_decoding" in field_names:
            config_kwargs["enable_speculative_decoding"] = True
        elif "enable_speculative" in field_names:
            config_kwargs["enable_speculative"] = True
    config = EngineConfig(**config_kwargs)
    engine = LLMEngine(runtime=runtime, config=config)
    sync_timer = _SyncTimer(engine.executor) if args.sync_timing else None
    if sync_timer is not None:
        sync_timer.install()
    kv_allocated_vram_gb = (
        torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    )
    gpu_telemetry_start = _gpu_telemetry()

    # Warmup waves
    if args.warmup_tokens > 0 and args.warmup_runs > 0:
        for w in range(args.warmup_runs):
            await _run_batch_context(
                engine,
                runtime,
                target_tokens=64,
                max_new_tokens=args.warmup_tokens,
                sampling=sampling,
                batch_size=min(batch_size, 4),
                req_prefix=f"warmup-{w}",
                prompt_type=args.prompt_type,
                sync_timer=None,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    results: List[Dict[str, Any]] = []
    for idx, ctx in enumerate(contexts):
        print(
            f"Benchmarking context={ctx}, batch_size={batch_size}, repeats={args.repeats}, max_new_tokens={args.max_new_tokens}...",
            flush=True,
        )
        runs: List[Dict[str, Any]] = []
        for rep in range(max(1, args.repeats)):
            r_entry = await _run_batch_context(
                engine,
                runtime,
                target_tokens=ctx,
                max_new_tokens=args.max_new_tokens,
                sampling=sampling,
                batch_size=batch_size,
                req_prefix=f"bench-{ctx}-{idx}-r{rep}",
                prompt_type=args.prompt_type,
                sync_timer=sync_timer,
            )
            runs.append(r_entry)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if r_entry.get("status") != "ok":
                break
        entry = _aggregate_repeats(runs)
        entry["gpu_telemetry"] = _gpu_telemetry()
        results.append(entry)
        if entry["status"] == "ok":
            print(
                f"  -> TTFT={entry['ttft_s']:.3f}s | "
                f"prefill={entry['prefill_tok_per_s']:.1f} tok/s | "
                f"decode={entry['decode_tok_per_s']:.1f} tok/s | "
                f"peak_vram={entry['peak_vram_allocated_gb']:.2f} GB",
                flush=True,
            )
        else:
            print(f"  -> {entry['status'].upper()}: {entry.get('error')}", flush=True)

    if sync_timer is not None:
        sync_timer.uninstall()
    gpu_telemetry_end = _gpu_telemetry()
    await engine.shutdown()

    return {
        "metadata": {
            "model_id": args.model_id,
            "device": str(device),
            "gpu_name": gpu_name,
            "total_vram_gb": round(total_vram_gb, 3),
            "torch_version": torch.__version__,
            "dtype": str(runtime.dtype),
            "quantization": args.quantization,
            "compile_model": runtime.compile_model,
            "enable_cudnn_attention": config.enable_cudnn_attention,
            "enable_prefill_batching": config.enable_prefill_batching,
            "enable_speculative": bool(args.enable_speculative),
            "fused_paged_attention": bool(args.fused_paged_attention),
            "sync_timing": bool(args.sync_timing),
            "prompt_type": args.prompt_type,
            "warmup_runs": args.warmup_runs,
            "repeats": args.repeats,
            "batch_size": batch_size,
            "max_kv_cache_tokens": max_kv,
            "max_batch_tokens": config.max_batch_tokens,
            "decode_steps_per_round": config.decode_steps_per_round,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "model_load_s": round(model_load_s, 3),
            "weights_vram_gb": round(weights_vram_gb, 3),
            "weights_plus_kv_vram_gb": round(kv_allocated_vram_gb, 3),
            "peak_host_ram_gb": round(_max_rss_gb(), 3),
            "gpu_telemetry_start": gpu_telemetry_start,
            "gpu_telemetry_end": gpu_telemetry_end,
        },
        "results": results,
    }


def _print_comparison(current: Dict[str, Any], baseline_path: str) -> None:
    baseline = json.loads(Path(baseline_path).read_text())
    base_by_ctx = {r["target_context_tokens"]: r for r in baseline.get("results", [])}

    print("\n=== Comparison vs Baseline ===")
    header = (
        f"{'Ctx':>7} | {'Base TTFT':>9} {'Curr TTFT':>9} {'Speedup':>7} | "
        f"{'Base Pref':>9} {'Curr Pref':>9} | "
        f"{'Base Dec':>9} {'Curr Dec':>9} {'Speedup':>7} | "
        f"{'Base VRAM':>9} {'Curr VRAM':>9}"
    )
    print(header)
    print("-" * len(header))

    for cur in current.get("results", []):
        ctx = cur["target_context_tokens"]
        base = base_by_ctx.get(ctx)
        if not base or base.get("status") != "ok" or cur.get("status") != "ok":
            b_stat = base.get("status", "n/a") if base else "n/a"
            c_stat = cur.get("status", "n/a")
            print(f"{ctx:>7} | base={b_stat}, curr={c_stat}")
            continue
        ttft_sp = base["ttft_s"] / cur["ttft_s"] if cur["ttft_s"] > 0 else 0.0
        dec_sp = (
            cur["decode_tok_per_s"] / base["decode_tok_per_s"]
            if base["decode_tok_per_s"] > 0
            else 0.0
        )
        print(
            f"{ctx:>7} | "
            f"{base['ttft_s']:>8.3f}s {cur['ttft_s']:>8.3f}s {ttft_sp:>6.2f}x | "
            f"{base['prefill_tok_per_s']:>9.1f} {cur['prefill_tok_per_s']:>9.1f} | "
            f"{base['decode_tok_per_s']:>9.1f} {cur['decode_tok_per_s']:>9.1f} {dec_sp:>6.2f}x | "
            f"{base['peak_vram_allocated_gb']:>8.2f}G {cur['peak_vram_allocated_gb']:>8.2f}G"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MiniVLLM LFM3-1.7B")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--quantization", choices=QUANTIZATION_CHOICES, default="none"
    )
    parser.add_argument(
        "--contexts",
        default="512,8192,16384,32768",
        help="Comma-separated prompt token lengths",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--prompt-type", choices=("text", "chat", "json"), default="text"
    )
    parser.add_argument("--sync-timing", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batch-tokens", type=int, default=256)
    parser.add_argument("--max-kv-cache-tokens", type=int, default=None)
    parser.add_argument("--decode-steps-per-round", type=int, default=1)
    parser.add_argument("--compile-model", "--compile", dest="compile_model", action="store_true")
    parser.add_argument("--enable-cudnn-attention", "--cudnn-attention", dest="enable_cudnn_attention", action="store_true")
    parser.add_argument("--enable-prefill-batching", action="store_true")
    parser.add_argument("--enable-speculative", action="store_true")
    parser.add_argument("--fused-paged-attention", action="store_true")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument(
        "--compare", type=str, default=None, help="Baseline JSON to compare against"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = asyncio.run(run_benchmark(args))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nSaved benchmark results to {args.output}")
    if args.compare:
        _print_comparison(report, args.compare)


if __name__ == "__main__":
    main()
