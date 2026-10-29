import argparse
import asyncio
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import EngineConfig, GenerateRequest, LLMEngine, SamplingConfig
from runtime import Lfm3Runtime, get_device


def build_engine(
    runtime: Lfm3Runtime, *, prefill_chunk_size: Optional[int], max_new_tokens: int
) -> LLMEngine:
    return LLMEngine(
        runtime=runtime,
        config=EngineConfig(
            use_instruct_model=False,
            max_new_tokens=max_new_tokens,
            prefill_chunk_size=prefill_chunk_size,
            sampling=SamplingConfig(
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
            ),
        ),
    )


def make_prompts(*, num_prompts: int, repeat: int) -> list[str]:
    seeds = [
        "Explain how KV cache helps LLM inference in production. ",
        "Describe the tradeoff between latency and throughput in serving. ",
        "Summarize why batching helps decoder-only inference. ",
        "Explain continuous batching in simple terms. ",
        "Describe how paged KV cache improves memory utilization. ",
        "Explain what chunked prefill is and why it can help fairness. ",
    ]
    prompts = []
    for idx in range(num_prompts):
        seed = seeds[idx % len(seeds)]
        prompts.append((seed + f"Request {idx}. ") * repeat)
    return prompts


def summarize_results(results, *, elapsed_s: float, label: str) -> None:
    successful = [result for result in results if result.error_message is None]
    print(f"\n{label}")
    print(f"elapsed_s={elapsed_s:.4f}")
    print(f"completed={len(successful)}/{len(results)}")

    if not successful:
        return

    avg_queue = mean(result.queue_wait_s for result in successful)
    avg_prefill = mean(result.prefill_s for result in successful)
    avg_decode = mean(result.decode_s for result in successful)
    avg_total = mean(result.total_latency_s for result in successful)
    avg_prefill_steps = mean(result.prefill_steps for result in successful)
    avg_decode_steps = mean(result.decode_steps for result in successful)
    avg_tokens = mean(len(result.token_ids) for result in successful)

    print(
        "summary "
        f"avg_queue_wait_s={avg_queue:.4f} "
        f"avg_prefill_s={avg_prefill:.4f} "
        f"avg_decode_s={avg_decode:.4f} "
        f"avg_total_s={avg_total:.4f} "
        f"avg_prefill_steps={avg_prefill_steps:.2f} "
        f"avg_decode_steps={avg_decode_steps:.2f} "
        f"avg_generated_tokens={avg_tokens:.2f}"
    )

    print("per_request")
    for result in results:
        print(
            f"request_id={result.request_id} "
            f"status={result.stop_reason} "
            f"queue_wait_s={result.queue_wait_s:.4f} "
            f"prefill_s={result.prefill_s:.4f} "
            f"decode_s={result.decode_s:.4f} "
            f"total_s={result.total_latency_s:.4f} "
            f"prefill_steps={result.prefill_steps} "
            f"decode_steps={result.decode_steps} "
            f"generated_tokens={len(result.token_ids)}"
        )


async def run_case(
    runtime: Lfm3Runtime,
    *,
    prompts: list[str],
    prefill_chunk_size: Optional[int],
    max_new_tokens: int,
    warmup_runs: int,
    benchmark_runs: int,
    label: str,
) -> float:
    for _ in range(max(0, warmup_runs)):
        engine = build_engine(
            runtime,
            prefill_chunk_size=prefill_chunk_size,
            max_new_tokens=max_new_tokens,
        )
        await asyncio.gather(
            *[
                engine.generate(GenerateRequest(str(index), prompt))
                for index, prompt in enumerate(prompts)
            ]
        )
        await engine.shutdown()

    elapsed_values = []
    last_results = None
    for _ in range(max(1, benchmark_runs)):
        engine = build_engine(
            runtime,
            prefill_chunk_size=prefill_chunk_size,
            max_new_tokens=max_new_tokens,
        )
        started = time.perf_counter()
        results = await asyncio.gather(
            *[
                engine.generate(GenerateRequest(str(index), prompt))
                for index, prompt in enumerate(prompts)
            ]
        )
        elapsed_s = time.perf_counter() - started
        elapsed_values.append(elapsed_s)
        last_results = results
        await engine.shutdown()

    avg_elapsed_s = sum(elapsed_values) / len(elapsed_values)
    summarize_results(last_results, elapsed_s=avg_elapsed_s, label=label)
    return avg_elapsed_s


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark long concurrent prompts with and without chunked prefill"
    )
    parser.add_argument(
        "--num-prompts", type=int, default=4, help="Number of concurrent long prompts"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=40,
        help="How many times to repeat each long prompt seed",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="Chunk size for the chunked-prefill run",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=32, help="Generated tokens per request"
    )
    parser.add_argument(
        "--warmup-runs", type=int, default=1, help="Warmup runs before timing"
    )
    parser.add_argument(
        "--benchmark-runs", type=int, default=3, help="Number of timed runs per case"
    )
    parser.add_argument(
        "--quantization", choices=("none", "int8", "int4"), default="none"
    )
    args = parser.parse_args()

    prompts = make_prompts(
        num_prompts=max(1, args.num_prompts), repeat=max(1, args.repeat)
    )
    runtime = Lfm3Runtime(
        use_instruct_model=True,
        device=get_device(),
        quantization=args.quantization,
    )

    print(f"num_prompts={len(prompts)}")
    print(f"prompt_chars_first={len(prompts[0])}")

    no_chunk_elapsed = await run_case(
        runtime,
        prompts=prompts,
        prefill_chunk_size=None,
        max_new_tokens=args.max_new_tokens,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
        label="no_chunk",
    )
    chunked_elapsed = await run_case(
        runtime,
        prompts=prompts,
        prefill_chunk_size=args.chunk_size,
        max_new_tokens=args.max_new_tokens,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
        label="chunked",
    )

    print("\ncomparison")
    print(f"no_chunk_avg_elapsed_s={no_chunk_elapsed:.4f}")
    print(f"chunked_avg_elapsed_s={chunked_elapsed:.4f}")
    if chunked_elapsed > 0:
        print(
            f"speed_ratio_no_chunk_over_chunked={no_chunk_elapsed / chunked_elapsed:.3f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
