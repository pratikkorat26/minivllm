"""Compare single-pass and split-KV in the real engine on synthetic long chats.

Decode-worker throughput excludes prefill and scheduler idle time; request
throughput includes both. Compilation and full-wave warmup are excluded.
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from inference import EngineConfig, GenerateRequest, LLMEngine, SamplingConfig
from runtime import Lfm3Runtime, get_device


def make_prompt(tokenizer, target_tokens):
    prefix = "<|startoftext|><|im_start|>user\n"
    suffix = (
        "\nSummarize the text above in detail, using at least five sentences."
        "<|im_end|>\n<|im_start|>assistant\n<think></think>"
    )
    sentence = (
        "Inference engines schedule requests and store attention keys and values. "
    )
    overhead = len(tokenizer.encode(prefix)) + len(tokenizer.encode(suffix))
    body_ids = tokenizer.encode(sentence * (target_tokens // 8 + 1))
    return prefix + tokenizer.decode(body_ids[:target_tokens - overhead]) + suffix


async def run(args):
    os.environ["LFM3_FUSED_PAGED_ATTENTION"] = "1"
    runtime = Lfm3Runtime(device=get_device(), compile_model=not args.eager)
    if runtime.device.type != "cuda":
        raise SystemExit("CUDA required")
    report = {
        "kind": "real_engine_split_kv_comparison",
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "compile": runtime.compile_model,
        "concurrency": args.concurrency,
        "max_new_tokens": args.max_new_tokens,
        "repeats": args.repeats,
        "warmup_waves": 2,
        "notes": [
            "Synthetic repeated-text chat; BF16 greedy decoding.",
            "No HTTP; clocks/power not locked; run each mode in a fresh process.",
            "Decode-worker throughput excludes prefill and scheduler idle time."
        ],
        "runs": [],
    }
    for context in args.contexts:
        prompt = make_prompt(runtime.tokenizer, context)
        prompt_tokens = len(runtime.tokenizer.encode(prompt))
        # Pre-size persistent RoPE before compilation sees any input shapes.
        rope_bound = 1 << (prompt_tokens + args.max_new_tokens - 1).bit_length()
        runtime.model.rope_sliding.ensure_capacity(rope_bound, device=runtime.device)
        runtime.model.rope_full.ensure_capacity(rope_bound, device=runtime.device)
        # Allow room for block rounding and the full output reservation.
        kv_budget = args.concurrency * (prompt_tokens + args.max_new_tokens + 32)
        engine = LLMEngine(runtime, EngineConfig(
            use_instruct_model=False,
            max_new_tokens=args.max_new_tokens,
            max_concurrent_requests=args.concurrency,
            max_queue_size=max(128, args.concurrency),
            decode_batch_size=args.concurrency,
            max_kv_cache_tokens=kv_budget,
            max_batch_tokens=256,
            sampling=SamplingConfig(
                temperature=0, top_p=1, top_k=0, repetition_penalty=1,
            ),
        ))
        original_decode = engine.executor.decode
        decode_seconds = 0.0
        decode_tokens = 0

        def timed_decode(requests):
            nonlocal decode_seconds, decode_tokens
            before = sum(len(r.generated_ids) for r in requests)
            started = time.perf_counter()
            result = original_decode(requests)
            # Executor already synchronizes when copying sampled IDs to CPU.
            decode_seconds += time.perf_counter() - started
            decode_tokens += sum(len(r.generated_ids) for r in requests) - before
            return result

        engine.executor.decode = timed_decode

        async def wave(wave_id, engine=engine, prompt=prompt):
            results = await asyncio.gather(*[
                engine.generate(GenerateRequest(f"{wave_id}-{i}", prompt))
                for i in range(args.concurrency)
            ])
            errors = [r.error_message for r in results if r.error_message]
            if errors:
                raise RuntimeError(str(errors))
            return results

        try:
            modes = {"single_pass": 1, "auto_split_kv": None}
            for name in [args.mode]:
                splits = modes[name]
                for block in runtime.model.blocks:
                    if block.is_attention:
                        block.self_attn.paged_attention_num_splits = splits
                warmup_started = time.perf_counter()
                for warmup in range(2):
                    await wave(f"warm-{name}-{warmup}")
                torch.cuda.synchronize()
                warmup_s = time.perf_counter() - warmup_started
                decode_seconds = 0.0
                decode_tokens = 0
                measured = []
                wave_seconds = []
                started = time.perf_counter()
                for repeat in range(args.repeats):
                    wave_started = time.perf_counter()
                    measured.extend(await wave(f"{name}-{repeat}"))
                    wave_seconds.append(time.perf_counter() - wave_started)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                tokens = sum(len(r.token_ids) for r in measured)
                token_hash = hashlib.sha256(json.dumps(
                    [r.token_ids for r in measured]
                ).encode()).hexdigest()
                row = {
                    "mode": name,
                    "prompt_tokens": prompt_tokens,
                    "warmup_s": warmup_s,
                    "generated_tokens": tokens,
                    "elapsed_s": elapsed,
                    "wave_seconds": wave_seconds,
                    "request_output_tokens_per_s": tokens / elapsed,
                    "decode_tokens": decode_tokens,
                    "decode_worker_s": decode_seconds,
                    "decode_worker_tokens_per_s": decode_tokens / decode_seconds,
                    "token_ids_sha256": token_hash,
                    "stop_reasons": [r.stop_reason for r in measured],
                }
                report["runs"].append(row)
                print(json.dumps(row), flush=True)
        finally:
            await engine.shutdown()
        # Drop the old cache buffers/compiled graph references before next shape.
        del engine

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", nargs="+", type=int, default=[8192])
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--mode", choices=["single_pass", "auto_split_kv"],
                        default="auto_split_kv")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.concurrency, args.max_new_tokens, args.repeats) <= 0:
        parser.error("concurrency, output length and repeats must be positive")
    if min(args.contexts) < 128:
        parser.error("contexts must be at least 128 tokens")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
