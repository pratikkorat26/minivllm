"""Reproducible closed-loop engine benchmark; saves raw client-observed latencies."""

import argparse
import asyncio
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from inference import EngineConfig, GenerateRequest, LLMEngine, SamplingConfig
from runtime import GemmaRuntime, get_device


async def run(args):
    runtime = GemmaRuntime(use_instruct_model=args.instruct, device=get_device())
    engine = LLMEngine(
        runtime,
        EngineConfig(
            use_instruct_model=args.instruct,
            max_new_tokens=args.tokens,
            max_kv_cache_tokens=8192,
            sampling=SamplingConfig(
                temperature=0, top_p=1, top_k=0, repetition_penalty=1
            ),
        ),
    )
    prompts = [
        "The purpose of a KV cache is",
        "Continuous batching improves GPU utilization because",
        "Explain how inference memory allocation works.",
        "A useful engineering tradeoff in serving is",
    ]
    await engine.generate(GenerateRequest("warmup", prompts[0]))
    rows = []

    async def one(index):
        times, ids = [], []
        token_count = 0
        start = time.perf_counter()
        async for event in engine.stream(
            GenerateRequest(str(index), prompts[index % len(prompts)])
        ):
            if event.kind == "text":
                times.append(time.perf_counter())
                ids.append(event.token_id)
            elif event.error_message:
                raise RuntimeError(event.error_message)
            elif event.kind == "done":
                token_count = event.generated_token_count
        end = time.perf_counter()
        return dict(
            request=index,
            tokens=token_count,
            content_events=len(ids),
            ttft_ms=(times[0] - start) * 1000 if times else None,
            tpot_ms=(times[-1] - times[0]) * 1000 / (token_count - 1)
            if token_count > 1 and times
            else None,
            latency_ms=(end - start) * 1000,
            token_hash=hashlib.sha256(str(ids).encode()).hexdigest(),
        )

    try:
        for concurrency in args.concurrency:
            if runtime.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            requests = []
            for repeat in range(args.repeats):
                requests.extend(
                    await asyncio.gather(*(one(i) for i in range(concurrency)))
                )
            elapsed = time.perf_counter() - start
            rows.append(
                dict(
                    concurrency=concurrency,
                    elapsed_s=elapsed,
                    output_tokens_per_s=sum(r["tokens"] for r in requests) / elapsed,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated()
                    if runtime.device.type == "cuda"
                    else None,
                    requests=requests,
                )
            )
    finally:
        await engine.shutdown()
    result = dict(
        kind="real_model_closed_loop",
        model=runtime.repo_id,
        torch=torch.__version__,
        python=platform.python_version(),
        platform=platform.platform(),
        dtype=str(runtime.dtype),
        engine_config=asdict(engine.config),
        prompts=prompts,
        warmup_requests=1,
        device=torch.cuda.get_device_name() if runtime.device.type == "cuda" else "cpu",
        max_new_tokens=args.tokens,
        repeats=args.repeats,
        rows=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
    for row in rows:
        print({k: v for k, v in row.items() if k != "requests"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instruct", action="store_true")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args()
    if min(args.tokens, args.repeats, *args.concurrency) <= 0:
        parser.error("tokens, repeats and concurrency must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
