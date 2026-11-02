"""Measure steady-state LFM3 engine output throughput under concurrent load."""

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from inference import EngineConfig, GenerateRequest, LLMEngine, SamplingConfig
from runtime import Lfm3Runtime, get_device


PROMPTS = (
    "The purpose of a KV cache is",
    "Continuous batching improves GPU utilization because",
    "Explain how inference memory allocation works.",
    "A useful engineering tradeoff in serving is",
)


class PhaseTimer:
    """Measure worker phases without synchronizing the GPU after every chunk."""

    def __init__(self, device):
        self.device = device
        self.samples = []

    def measure(self, phase, operation, requests, *args):
        before = sum(r.prompt_cursor for r in requests)
        if self.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start, end = None, None
        started = time.perf_counter()
        result = operation(requests, *args)
        wall_s = time.perf_counter() - started
        if end is not None:
            end.record()
        tokens = (
            sum(r.prompt_cursor for r in requests) - before
            if phase == "prefill"
            else len(result)
        )
        self.samples.append((phase, tokens, start, end, wall_s))
        return result

    def report(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        report = {}
        for phase in ("prefill", "decode"):
            samples = [row for row in self.samples if row[0] == phase]
            tokens = sum(row[1] for row in samples)
            elapsed_s = sum(
                row[2].elapsed_time(row[3]) / 1000 if row[2] is not None else row[4]
                for row in samples
            )
            report[phase] = {
                "tokens": tokens,
                "steps": len(samples),
                "device_elapsed_s": elapsed_s,
                "worker_wall_s": sum(row[4] for row in samples),
                "tokens_per_s": tokens / elapsed_s if elapsed_s else 0,
                "mean_step_ms": elapsed_s * 1000 / len(samples) if samples else 0,
            }
        return report


async def _wave(
    engine: LLMEngine,
    *,
    concurrency: int,
    max_new_tokens: int,
    wave_id: str,
    prompt: str | None = None,
):
    return await asyncio.gather(
        *[
            engine.generate(
                GenerateRequest(
                    request_id=f"{wave_id}-{index}",
                    prompt=prompt or PROMPTS[index % len(PROMPTS)],
                    max_new_tokens=max_new_tokens,
                )
            )
            for index in range(concurrency)
        ]
    )


async def run(args: argparse.Namespace) -> None:
    if args.concurrency <= 0 or args.decode_batch_size <= 0:
        raise SystemExit("concurrency and decode-batch-size must be positive")
    if args.max_batch_tokens < args.decode_batch_size:
        raise SystemExit("max-batch-tokens must be >= decode-batch-size")
    if args.repeats <= 0 or args.max_new_tokens <= 0:
        raise SystemExit("repeats and max-new-tokens must be positive")

    device = get_device()
    if args.fused_paged_attention:
        os.environ["LFM3_FUSED_PAGED_ATTENTION"] = "1"
    runtime = Lfm3Runtime(
        use_instruct_model=True,
        device=device,
        quantization=args.quantization,
        compile_model=args.compile,
    )
    engine = LLMEngine(
        runtime,
        EngineConfig(
            use_instruct_model=True,
            max_new_tokens=args.max_new_tokens,
            max_kv_cache_tokens=args.max_kv_cache_tokens,
            max_concurrent_requests=args.concurrency,
            max_queue_size=args.max_queue_size or max(128, args.concurrency),
            decode_batch_size=args.decode_batch_size,
            max_batch_tokens=args.max_batch_tokens,
            enable_prefill_batching=args.batch_prefill,
            sampling=SamplingConfig(
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            ),
        ),
    )

    phase_timer = PhaseTimer(device) if args.phase_timing else None
    if phase_timer is not None:
        engine.executor.prefill_batch = partial(
            phase_timer.measure, "prefill", engine.executor.prefill_batch
        )
        engine.executor.decode = partial(
            phase_timer.measure, "decode", engine.executor.decode
        )

    try:
        warmup_started = time.perf_counter()
        await engine.generate(
            GenerateRequest("warmup-single", args.prompt or PROMPTS[0])
        )
        warmup_results = await _wave(
            engine,
            concurrency=args.concurrency,
            max_new_tokens=args.max_new_tokens,
            wave_id="warmup-wave",
            prompt=args.prompt,
        )
        warmup_s = time.perf_counter() - warmup_started
        if any(result.error_message for result in warmup_results):
            raise RuntimeError("warmup request failed; reduce concurrency or KV budget")

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        if phase_timer is not None:
            phase_timer.samples.clear()

        measured = []
        waves = []
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for repeat in range(args.repeats):
            wave_started = time.perf_counter()
            results = await _wave(
                engine,
                concurrency=args.concurrency,
                max_new_tokens=args.max_new_tokens,
                wave_id=f"run-{repeat}",
                prompt=args.prompt,
            )
            waves.append(
                {
                    "elapsed_s": time.perf_counter() - wave_started,
                    "generated_tokens": sum(len(r.token_ids) for r in results),
                }
            )
            measured.extend(results)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - started

        failures = [result for result in measured if result.error_message]
        if failures:
            raise RuntimeError(
                f"{len(failures)} measured requests failed: {failures[0].error_message}"
            )
        generated_tokens = sum(len(result.token_ids) for result in measured)
        phases = phase_timer.report() if phase_timer is not None else None
        print(f"model={runtime.model_id}")
        device_name = torch.cuda.get_device_name() if device.type == "cuda" else device
        print(f"device={device_name}")
        print(f"quantization={runtime.quantization} compile={args.compile}")
        print(
            "fused_paged_attention="
            f"{os.environ.get('LFM3_FUSED_PAGED_ATTENTION', '0') == '1'}"
        )
        print(
            f"decode_batch_size={args.decode_batch_size} "
            f"concurrency={args.concurrency} max_new_tokens={args.max_new_tokens}"
        )
        print(f"warmup_s={warmup_s:.3f} (excluded from steady-state throughput)")
        print(
            f"measured_requests={len(measured)} generated_tokens={generated_tokens} "
            f"elapsed_s={elapsed_s:.3f} "
            f"aggregate_tokens_per_s={generated_tokens / elapsed_s:.2f} "
            f"tokens_per_s_per_request="
            f"{generated_tokens / elapsed_s / args.concurrency:.2f}"
        )
        if device.type == "cuda":
            print(
                f"peak_allocated_gib={torch.cuda.max_memory_allocated() / 1024**3:.2f}"
            )
        if measured:
            print(f"sample_output={measured[0].text!r}")
        if phases is not None:
            print(f"phase_timings={json.dumps(phases)}")
        if args.output:
            report = {
                "model": runtime.model_id,
                "device": str(device_name),
                "torch_version": torch.__version__,
                "config": {
                    key: value for key, value in vars(args).items() if key != "output"
                },
                "warmup_s": warmup_s,
                "elapsed_s": elapsed_s,
                "generated_tokens": generated_tokens,
                "aggregate_tokens_per_s": generated_tokens / elapsed_s,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3
                if device.type == "cuda"
                else None,
                "waves": waves,
                "token_hash": hashlib.sha256(
                    json.dumps([r.token_ids for r in measured]).encode()
                ).hexdigest(),
                "sample_output": measured[0].text if measured else "",
                "phases": phases,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        await engine.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--decode-batch-size", type=int, default=32)
    parser.add_argument("--max-batch-tokens", type=int, default=256)
    parser.add_argument("--max-kv-cache-tokens", type=int, default=16_384)
    parser.add_argument("--max-queue-size", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--phase-timing",
        action="store_true",
        help="Measure prefill input and decode output tokens/s using CUDA events",
    )
    parser.add_argument(
        "--batch-prefill",
        action="store_true",
        help="Batch compatible equal-length prefill chunks",
    )
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1)
    parser.add_argument("--prompt", help="Use the same prompt for every request")
    parser.add_argument(
        "--output",
        type=Path,
        help="Save configuration and raw wave measurements as JSON",
    )
    parser.add_argument(
        "--quantization", choices=("none", "int8", "int4"), default="none"
    )
    parser.add_argument(
        "--fused-paged-attention",
        action="store_true",
        help="Use experimental direct-page Triton attention for CUDA decode",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable experimental torch.compile (first-run warmup can be slow)",
    )
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
