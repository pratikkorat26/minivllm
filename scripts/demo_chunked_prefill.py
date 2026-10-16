import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import EngineConfig, GenerateRequest, LLMEngine, SamplingConfig
from runtime import GemmaRuntime, get_device


def build_engine(
    runtime: GemmaRuntime, *, prefill_chunk_size: Optional[int]
) -> LLMEngine:
    return LLMEngine(
        runtime=runtime,
        config=EngineConfig(
            choose_model="270m",
            use_instruct_model=False,
            max_new_tokens=32,
            prefill_chunk_size=prefill_chunk_size,
            sampling=SamplingConfig(
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
            ),
        ),
    )


async def run_case(
    runtime: GemmaRuntime, *, prompt: str, prefill_chunk_size: Optional[int]
) -> dict:
    engine = build_engine(runtime, prefill_chunk_size=prefill_chunk_size)
    started = time.perf_counter()
    result = await engine.generate(GenerateRequest("case", prompt))
    elapsed_s = time.perf_counter() - started
    tok_per_s = 0.0 if elapsed_s <= 0 else len(result.token_ids) / elapsed_s

    print(f"\nprefill_chunk_size={prefill_chunk_size}")
    print(f"elapsed_s={elapsed_s:.4f}")
    print(f"prefill_steps={result.prefill_steps}")
    print(f"decode_steps={result.decode_steps}")
    print(f"stop_reason={result.stop_reason}")
    print(f"generated_tokens={len(result.token_ids)}")
    print(f"end_to_end_tok_per_s={tok_per_s:.2f}")
    print(f"text_preview={result.text[:200]!r}")
    await engine.shutdown()
    return {
        "prefill_chunk_size": prefill_chunk_size,
        "elapsed_s": elapsed_s,
        "prefill_steps": result.prefill_steps,
        "decode_steps": result.decode_steps,
        "generated_tokens": len(result.token_ids),
        "tok_per_s": tok_per_s,
    }


async def benchmark_case(
    runtime: GemmaRuntime,
    *,
    prompt: str,
    prefill_chunk_size: Optional[int],
    warmup_runs: int,
    benchmark_runs: int,
) -> dict:
    for _ in range(max(0, warmup_runs)):
        engine = build_engine(runtime, prefill_chunk_size=prefill_chunk_size)
        await engine.generate(GenerateRequest("warmup", prompt))
        await engine.shutdown()

    elapsed_values = []
    last_result = None
    for _ in range(max(1, benchmark_runs)):
        engine = build_engine(runtime, prefill_chunk_size=prefill_chunk_size)
        started = time.perf_counter()
        result = await engine.generate(GenerateRequest("benchmark", prompt))
        elapsed_s = time.perf_counter() - started
        elapsed_values.append(elapsed_s)
        last_result = result
        await engine.shutdown()

    avg_elapsed_s = sum(elapsed_values) / len(elapsed_values)
    avg_tok_per_s = (
        0.0 if avg_elapsed_s <= 0 else len(last_result.token_ids) / avg_elapsed_s
    )
    return {
        "prefill_chunk_size": prefill_chunk_size,
        "avg_elapsed_s": avg_elapsed_s,
        "prefill_steps": last_result.prefill_steps,
        "decode_steps": last_result.decode_steps,
        "generated_tokens": len(last_result.token_ids),
        "avg_tok_per_s": avg_tok_per_s,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show chunked-prefill behavior on one prompt"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Prefill chunk size for the chunked run",
    )
    parser.add_argument(
        "--repeat", type=int, default=40, help="How many times to repeat the seed text"
    )
    parser.add_argument(
        "--warmup-runs", type=int, default=1, help="Warmup runs before timing"
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=3,
        help="Number of timed runs for each case",
    )
    parser.add_argument(
        "--prompt",
        default="Explain KV cache reuse in one short paragraph. ",
        help="Seed prompt text to repeat",
    )
    args = parser.parse_args()

    prompt = args.prompt * max(1, args.repeat)
    runtime = GemmaRuntime(
        choose_model="270m",
        use_instruct_model=True,
        device=get_device(),
    )

    print(f"prompt_chars={len(prompt)}")
    await run_case(runtime, prompt=prompt, prefill_chunk_size=None)
    await run_case(runtime, prompt=prompt, prefill_chunk_size=args.chunk_size)

    print("\nbenchmarking...")
    base_bench = await benchmark_case(
        runtime,
        prompt=prompt,
        prefill_chunk_size=None,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
    )
    chunked_bench = await benchmark_case(
        runtime,
        prompt=prompt,
        prefill_chunk_size=args.chunk_size,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
    )

    print("\nbenchmark_summary")
    print(
        "no_chunk "
        f"avg_elapsed_s={base_bench['avg_elapsed_s']:.4f} "
        f"prefill_steps={base_bench['prefill_steps']} "
        f"avg_tok_per_s={base_bench['avg_tok_per_s']:.2f}"
    )
    print(
        "chunked "
        f"avg_elapsed_s={chunked_bench['avg_elapsed_s']:.4f} "
        f"prefill_steps={chunked_bench['prefill_steps']} "
        f"avg_tok_per_s={chunked_bench['avg_tok_per_s']:.2f}"
    )
    if chunked_bench["avg_elapsed_s"] > 0:
        ratio = base_bench["avg_elapsed_s"] / chunked_bench["avg_elapsed_s"]
        print(f"speed_ratio_no_chunk_over_chunked={ratio:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
