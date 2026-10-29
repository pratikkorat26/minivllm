import argparse
import asyncio

import torch

from inference import EngineConfig, GenerateRequest, LLMEngine, SamplingConfig
from runtime import DEFAULT_MODEL_ID, Lfm3Runtime, get_device


async def _main(args) -> None:
    sampling = SamplingConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )
    runtime = Lfm3Runtime(
        model_id=args.model_id,
        use_instruct_model=args.instruct,
        device=get_device(),
        quantization=args.quantization,
    )
    engine = LLMEngine(
        runtime=runtime,
        config=EngineConfig(
            use_instruct_model=args.instruct,
            max_new_tokens=args.max_new_tokens,
            sampling=sampling,
        ),
    )

    print(f"Prompt: {args.prompt}\n")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    async for event in engine.stream(
        GenerateRequest("cli", args.prompt, sampling, args.max_new_tokens)
    ):
        if event.kind == "text":
            print(event.text, end="", flush=True)
    await engine.shutdown()
    print()

    if torch.cuda.is_available():
        print(f"\nGPU memory used: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with LFM3-1.7B")
    parser.add_argument(
        "--prompt",
        default="Give me a short introduction to large language models.",
        help="Input prompt",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=180,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (0 = greedy)",
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling"
    )
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument(
        "--repetition-penalty", type=float, default=1.1, help="Repetition penalty"
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face repo id or local model directory name",
    )
    parser.add_argument(
        "--quantization",
        choices=("none", "int8", "int4"),
        default="none",
        help=(
            "Weight-only quantization; int8 is recommended, int4 requires CUDA, "
            "bitsandbytes and is experimental"
        ),
    )
    parser.add_argument(
        "--no-instruct",
        action="store_false",
        dest="instruct",
        help="Send the prompt verbatim instead of applying the chat template",
    )
    args = parser.parse_args()

    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
