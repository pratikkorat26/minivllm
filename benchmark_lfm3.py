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
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import torch

from inference import EngineConfig, GenerateRequest, LLMEngine, SamplingConfig
from runtime import DEFAULT_MODEL_ID, Lfm3Runtime, get_device


def _rss_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)


def _max_rss_gb() -> float:
    # ru_maxrss is in KiB on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)


def _build_exact_prompt(tokenizer: Any, target_tokens: int) -> tuple[str, int]:
    """Construct a deterministic prompt that encodes to ~target_tokens."""
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


async def _run_single_context(
    engine: LLMEngine,
    runtime: Lfm3Runtime,
    target_tokens: int,
    max_new_tokens: int,
    sampling: SamplingConfig,
    req_id: str,
) -> Dict[str, Any]:
    prompt, actual_tokens = _build_exact_prompt(runtime.tokenizer, target_tokens)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

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
        return {
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


async def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    if args.fused_paged_attention:
        os.environ["LFM3_FUSED_PAGED_ATTENTION"] = "1"

    contexts = [int(c.strip()) for c in args.contexts.split(",") if c.strip()]
    max_kv = args.max_kv_cache_tokens or max(32_768, max(contexts) + args.max_new_tokens + 64)

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
        compile_model=args.compile,
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
    config = EngineConfig(
        use_instruct_model=False,
        max_new_tokens=args.max_new_tokens,
        sampling=sampling,
        max_kv_cache_tokens=max_kv,
        enable_cudnn_attention=args.cudnn_attention,
        decode_steps_per_round=args.decode_steps_per_round,
    )
    engine = LLMEngine(runtime=runtime, config=config)
    kv_allocated_vram_gb = (
        torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    )

    # Warmup
    if args.warmup_tokens > 0:
        await _run_single_context(
            engine,
            runtime,
            target_tokens=64,
            max_new_tokens=args.warmup_tokens,
            sampling=sampling,
            req_id="warmup",
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results: List[Dict[str, Any]] = []
    for idx, ctx in enumerate(contexts):
        print(f"Benchmarking context={ctx}, max_new_tokens={args.max_new_tokens}...", flush=True)
        entry = await _run_single_context(
            engine,
            runtime,
            target_tokens=ctx,
            max_new_tokens=args.max_new_tokens,
            sampling=sampling,
            req_id=f"bench-{ctx}-{idx}",
        )
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
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
            "fused_paged_attention": bool(args.fused_paged_attention),
            "max_kv_cache_tokens": max_kv,
            "max_batch_tokens": config.max_batch_tokens,
            "decode_steps_per_round": config.decode_steps_per_round,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "model_load_s": round(model_load_s, 3),
            "weights_vram_gb": round(weights_vram_gb, 3),
            "weights_plus_kv_vram_gb": round(kv_allocated_vram_gb, 3),
            "peak_host_ram_gb": round(_max_rss_gb(), 3),
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
        "--quantization", choices=("none", "int8", "int4"), default="none"
    )
    parser.add_argument(
        "--contexts",
        default="512,8192,16384,32768",
        help="Comma-separated prompt token lengths",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-kv-cache-tokens", type=int, default=None)
    parser.add_argument("--decode-steps-per-round", type=int, default=1)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--cudnn-attention", action="store_true")
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
