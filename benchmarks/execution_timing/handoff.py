"""Measure scheduler/worker handoff and streaming overhead on real requests."""

import argparse
import asyncio
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


async def run(args):
    import torch
    from inference import LLMEngine, EngineConfig, GenerateRequest, SamplingConfig
    from runtime import Lfm3Runtime
    from gpu_telemetry import GpuTelemetry, platform_profile

    def check_power_profile():
        if (
            args.require_platform_profile
            and platform_profile() != args.require_platform_profile
        ):
            raise RuntimeError(
                f"Expected platform profile {args.require_platform_profile!r}, got {platform_profile()!r}"
            )

    check_power_profile()

    source_hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            ROOT / "inference/executor.py",
            ROOT / "inference/scheduler/scheduler.py",
            ROOT / "runtime/provider.py",
            ROOT / "lfm3/rope.py",
        )
    }

    sampling = SamplingConfig(temperature=0, top_p=1, top_k=0, repetition_penalty=1)
    runtime = Lfm3Runtime(device=torch.device("cuda"), compile_model=True)
    engine = LLMEngine(
        runtime,
        EngineConfig(
            use_instruct_model=True,
            max_new_tokens=args.output_tokens,
            max_batch_tokens=2048,
            max_kv_cache_tokens=max(
                16384, args.batch_size * (args.output_tokens + 128)
            ),
            max_concurrent_requests=args.batch_size,
            max_queue_size=max(128, args.batch_size),
            decode_batch_size=args.batch_size,
            decode_steps_per_round=args.decode_steps,
            enable_prefill_batching=True,
            enable_cudnn_attention=args.cudnn_attention,
            enable_stream_overlap=args.variant in ("full", "stream"),
            enable_decode_metadata_cache=args.variant in ("full", "metadata"),
            enable_pinned_token_copy=args.variant in ("full", "pinned"),
            enable_incremental_detokenization=args.variant in ("full", "incremental"),
            sampling=sampling,
        ),
    )
    worker_rows = []
    scheduler_rows = []
    profiler = None
    profiling = False
    telemetry = None
    original_decode = engine.executor.decode
    original_scheduled_decode = engine.scheduler._run_decode
    original_result = engine.executor.result
    finished_results = {}

    def capture_result(request):
        result = original_result(request)
        finished_results[str(result.request_id)] = result
        return result

    engine.executor.result = capture_result

    def decode(requests, **kwargs):
        began = time.perf_counter()
        result = original_decode(requests, **kwargs)
        worker_rows.append(
            dict(elapsed_s=time.perf_counter() - began, tokens=len(result))
        )
        return result

    async def scheduled_decode():
        before = len(worker_rows)
        began = time.perf_counter()
        count = await original_scheduled_decode()
        elapsed = time.perf_counter() - began
        if len(worker_rows) > before:
            worker = worker_rows[-1]["elapsed_s"]
            scheduler_rows.append(
                dict(
                    elapsed_s=elapsed,
                    worker_s=worker,
                    extra_s=max(0, elapsed - worker),
                    tokens=count,
                )
            )
        return count

    engine.executor.decode = decode
    engine.scheduler._run_decode = scheduled_decode
    original_flush = engine.scheduler._flush_stream_tokens

    def flush(*args, **kwargs):
        if profiling and engine.scheduler._pending_stream_tokens:
            with torch.profiler.record_function("scheduler.stream_flush"):
                return original_flush(*args, **kwargs)
        return original_flush(*args, **kwargs)

    engine.scheduler._flush_stream_tokens = flush

    async def one(index, label):
        request = GenerateRequest(
            f"{label}-{index}", args.prompt, max_new_tokens=args.output_tokens
        )
        if args.stream:
            events = [event async for event in engine.stream(request)]
            assert events[-1].kind == "done" and not events[-1].error_message
            result = finished_results.pop(request.request_id)
            assert "".join(e.text for e in events) == result.text
            assert events[-1].generated_token_count == len(result.token_ids)
        else:
            result = await engine.generate(request)
            finished_results.pop(request.request_id)
        if result.error_message:
            raise RuntimeError(result.error_message)
        return dict(
            count=len(result.token_ids),
            token_hash=hashlib.sha256(
                json.dumps(result.token_ids).encode()
            ).hexdigest(),
            text_hash=hashlib.sha256(result.text.encode()).hexdigest(),
            ttft_s=result.ttft_s,
            tpot_s=result.tpot_s,
        )

    async def wave(label):
        return await asyncio.gather(*(one(i, label) for i in range(args.batch_size)))

    def compile_counters():
        counters = torch._dynamo.utils.counters
        return dict(
            unique_graphs=counters["stats"]["unique_graphs"],
            cudagraph_recorded_non_static_inputs=counters["inductor"][
                "cudagraph_recorded_non_static_inputs"
            ],
        )

    try:
        warmup_rows, quiet_waves = [], 0
        for repeat in range(max(10, args.warmups + 4)):
            before = compile_counters()
            await wave(f"warmup{repeat}")
            torch.cuda.synchronize()
            after = compile_counters()
            quiet_waves = quiet_waves + 1 if before == after else 0
            warmup_rows.append(dict(before=before, after=after))
            print(
                f"WARMUP {repeat + 1}: quiet_waves={quiet_waves}, counters={after}",
                flush=True,
            )
            if repeat + 1 >= args.warmups and quiet_waves >= 2:
                break
        else:
            raise RuntimeError("Compilation/capture did not settle during warmup")
        worker_rows.clear()
        scheduler_rows.clear()
        torch.cuda.synchronize()
        check_power_profile()
        profile_before = platform_profile()
        telemetry_path = (
            args.output or ROOT / "benchmarks/execution_timing/handoff.json"
        ).with_suffix(".gpu.csv")
        telemetry = GpuTelemetry(telemetry_path)
        if args.profile:
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            )
            profiler.start()
            profiling = True
        began = time.perf_counter()
        counts = []
        waves = []
        for repeat in range(args.repeats):
            compile_before = compile_counters()
            start = time.perf_counter()
            results = await wave(f"run{repeat}")
            duration = time.perf_counter() - start
            compile_after = compile_counters()
            if compile_after != compile_before and not args.profile:
                raise RuntimeError("Compilation/capture occurred during measurement")
            counts.extend(r["count"] for r in results)
            waves.append(
                dict(
                    elapsed_s=duration,
                    generated_tokens=sum(r["count"] for r in results),
                    token_hash=hashlib.sha256(
                        json.dumps([r["token_hash"] for r in results]).encode()
                    ).hexdigest(),
                    text_hash=hashlib.sha256(
                        json.dumps([r["text_hash"] for r in results]).encode()
                    ).hexdigest(),
                    median_ttft_s=statistics.median(r["ttft_s"] for r in results),
                    median_tpot_s=statistics.median(r["tpot_s"] for r in results),
                    compile_before=compile_before,
                    compile_after=compile_after,
                )
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - began
        power_report = telemetry.stop()
        telemetry = None
        profile_after = platform_profile()
        if profile_after != profile_before:
            raise RuntimeError("Platform power profile changed during measurement")
        if profiler is not None:
            profiling = False
            profiler.stop()
            trace_path = Path("/tmp/minivllm_overlap_traces") / (
                args.output.stem + ".json"
            )
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(trace_path))
            profiler = None
            from overlap_trace_summary import summarize_overlap

            trace_report = summarize_overlap(trace_path)
            args.output.with_suffix(".trace_summary.json").write_text(
                json.dumps(trace_report, indent=2) + "\n"
            )
        report = dict(
            device=torch.cuda.get_device_name(),
            cpu_affinity=sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            platform_profile=profile_before,
            gpu_telemetry=power_report,
            torch=torch.__version__,
            config=asdict(engine.config),
            source_sha256=source_hashes,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            batch_size=args.batch_size,
            variant=args.variant,
            features={
                name: getattr(engine.config, name)
                for name in (
                    "enable_stream_overlap",
                    "enable_decode_metadata_cache",
                    "enable_pinned_token_copy",
                    "enable_incremental_detokenization",
                )
            },
            profile_perturbed=args.profile,
            output_token_cap=args.output_tokens,
            decode_steps_per_round=args.decode_steps,
            prompt=args.prompt,
            waves=waves,
            warmup_waves=warmup_rows,
            no_compilation_or_capture_during_measurement=all(
                w["compile_before"] == w["compile_after"] for w in waves
            ),
            stream=args.stream,
            requests=len(counts),
            generated_tokens=sum(counts),
            end_to_end_tokens_per_s=sum(counts) / elapsed,
            decode_steps=len(scheduler_rows),
            worker_decode_s=sum(r["elapsed_s"] for r in worker_rows),
            scheduler_extra_s=sum(r["extra_s"] for r in scheduler_rows),
            median_worker_step_ms=1000
            * statistics.median(r["worker_s"] for r in scheduler_rows),
            median_scheduler_extra_ms=1000
            * statistics.median(r["extra_s"] for r in scheduler_rows),
            p95_scheduler_extra_ms=1000
            * statistics.quantiles(
                [r["extra_s"] for r in scheduler_rows], n=100, method="inclusive"
            )[94],
            steps=scheduler_rows,
        )
        path = args.output or (
            ROOT
            / "benchmarks/execution_timing"
            / f"handoff_b{args.batch_size}_{'stream' if args.stream else 'json'}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: v
                    for k, v in report.items()
                    if k
                    not in (
                        "steps",
                        "waves",
                        "warmup_waves",
                        "source_sha256",
                        "config",
                        "gpu_telemetry",
                    )
                }
            ),
            flush=True,
        )
    finally:
        if telemetry is not None:
            telemetry.stop()
        if profiler is not None:
            profiler.stop()
        await engine.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--cudnn-attention",
        action="store_true",
        help="Opt in to cuDNN SDPA priority; not bitwise reproducible at some shapes.",
    )
    parser.add_argument("--stream", action="store_true")
    parser.add_argument(
        "--variant",
        choices=("baseline", "pinned", "metadata", "stream", "incremental", "full"),
        default="full",
    )
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--decode-steps", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--prompt", default="The purpose of a KV cache is")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-platform-profile")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Diagnostic trace; invalidates throughput comparison",
    )
    args = parser.parse_args()
    if args.profile and args.output is None:
        parser.error("--profile requires --output")
    asyncio.run(run(args))
