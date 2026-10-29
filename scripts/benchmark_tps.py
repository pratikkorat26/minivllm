"""Measure steady-state LFM3 engine output throughput under concurrent load."""

import argparse
import asyncio
import os
import sys
import time
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


async def _wave(
    engine: LLMEngine,
    *,
    concurrency: int,
    max_new_tokens: int,
    wave_id: str,
):
    return await asyncio.gather(
        *[
            engine.generate(
                GenerateRequest(
                    request_id=f"{wave_id}-{index}",
                    prompt=PROMPTS[index % len(PROMPTS)],
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
            sampling=SamplingConfig(
                temperature=0,
                top_p=1,
                top_k=0,
                repetition_penalty=1,
            ),
        ),
    )

    try:
        warmup_started = time.perf_counter()
        await engine.generate(GenerateRequest("warmup-single", PROMPTS[0]))
        warmup_results = await _wave(
            engine,
            concurrency=args.concurrency,
            max_new_tokens=args.max_new_tokens,
            wave_id="warmup-wave",
        )
        warmup_s = time.perf_counter() - warmup_started
        if any(result.error_message for result in warmup_results):
            raise RuntimeError("warmup request failed; reduce concurrency or KV budget")

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        measured = []
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for repeat in range(args.repeats):
            results = await _wave(
                engine,
                concurrency=args.concurrency,
                max_new_tokens=args.max_new_tokens,
                wave_id=f"run-{repeat}",
            )
            measured.extend(results)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - started

        failures = [result for result in measured if result.error_message]
        if failures:
            raise RuntimeError(f"{len(failures)} measured requests failed")
        generated_tokens = sum(len(result.token_ids) for result in measured)
        print(f"model={runtime.model_id}")
        device_name = (
            torch.cuda.get_device_name() if device.type == "cuda" else device
        )
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
                f"peak_allocated_gib="
                f"{torch.cuda.max_memory_allocated() / 1024**3:.2f}"
            )
        if measured:
            print(f"sample_output={measured[0].text!r}")
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
