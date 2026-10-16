#!/usr/bin/env python3
"""Single-request decode throughput microbenchmark."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gemma3.tokenizer import apply_chat_template
from inference import EngineConfig, GenerateRequest, LLMEngine, SamplingConfig
from runtime import GemmaRuntime, get_device, preferred_dtype


async def run_once(
    engine: LLMEngine, prompt: str, max_new_tokens: int, prompt_tokens: int
) -> dict:
    result = await engine.generate(
        GenerateRequest(
            request_id="bench",
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            sampling=SamplingConfig(
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
            ),
        )
    )
    return {
        "stop_reason": result.stop_reason,
        "error": result.error_message,
        "text": result.text.strip(),
        "new_tokens": len(result.token_ids),
        "prefill_s": result.prefill_s,
        "decode_s": result.decode_s,
        "total_s": result.total_latency_s,
        "prefill_steps": result.prefill_steps,
        "decode_steps": result.decode_steps,
        "prefill_tok_s": (prompt_tokens / result.prefill_s)
        if result.prefill_s > 0
        else 0.0,
        "decode_tok_s": (len(result.token_ids) / result.decode_s)
        if result.decode_s > 0
        else 0.0,
        "prompt_tokens": prompt_tokens,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="Say hello in one short sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()

    device = get_device()
    dtype = preferred_dtype(device)
    print(f"device={device} dtype={dtype}")

    runtime = GemmaRuntime(choose_model="270m", use_instruct_model=True, device=device)
    prompt_tokens = len(runtime.tokenizer.encode(apply_chat_template(args.prompt)))
    engine = LLMEngine(
        runtime=runtime,
        config=EngineConfig(
            choose_model="270m",
            use_instruct_model=True,
            max_new_tokens=args.max_new_tokens,
            sampling=SamplingConfig(
                temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.0
            ),
        ),
    )

    for _ in range(max(0, args.warmup)):
        await run_once(engine, args.prompt, args.max_new_tokens, prompt_tokens)

    wall_start = time.perf_counter()
    rows = []
    for _ in range(max(1, args.runs)):
        rows.append(
            await run_once(engine, args.prompt, args.max_new_tokens, prompt_tokens)
        )
    wall_s = time.perf_counter() - wall_start
    await engine.shutdown()

    def avg(key: str) -> float:
        return sum(row[key] for row in rows) / len(rows)

    sample = rows[-1]
    print(f"runs={len(rows)} warmup={args.warmup}")
    print(f"prompt_tokens≈{sample['prompt_tokens']} new_tokens≈{sample['new_tokens']}")
    print(
        f"avg_prefill_s={avg('prefill_s'):.4f} avg_decode_s={avg('decode_s'):.4f} avg_total_s={avg('total_s'):.4f}"
    )
    print(
        f"avg_prefill_tok_s={avg('prefill_tok_s'):.1f} avg_decode_tok_s={avg('decode_tok_s'):.1f}"
    )
    print(f"wall_s={wall_s:.4f}")
    print(f"stop_reason={sample['stop_reason']} error={sample['error']}")
    print(f"text={sample['text']!r}")


if __name__ == "__main__":
    asyncio.run(main())
