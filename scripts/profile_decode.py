"""Profile one fixed-batch LFM3 decode forward on the local GPU."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.profiler import ProfilerActivity, profile

from runtime import Lfm3Runtime, get_device


def _shape_bucket(length: int) -> int:
    return 1 << (max(1, length) - 1).bit_length()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--prompt-tokens", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--quantization", choices=("none", "int8", "int4"), default="none"
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.steps, args.prompt_tokens, args.block_size) <= 0:
        parser.error("batch size, steps, prompt tokens and block size must be positive")
    device = get_device()
    if device.type != "cuda":
        raise SystemExit("decode profiling requires CUDA")
    runtime = Lfm3Runtime(
        use_instruct_model=True,
        device=device,
        quantization=args.quantization,
        compile_model=args.compile,
    )
    model = runtime.model
    batch_size = args.batch_size
    block_size = args.block_size
    reserved_tokens = args.prompt_tokens + args.steps + 4
    if args.compile:
        reserved_tokens = _shape_bucket(reserved_tokens)
    pages_per_request = (reserved_tokens + block_size - 1) // block_size
    block_tables = torch.arange(
        batch_size * pages_per_request, device=device
    ).reshape(batch_size, pages_per_request)
    kv_caches = model.init_paged_kv_caches(
        num_blocks=batch_size * pages_per_request,
        block_size=block_size,
        device=device,
    )
    conv_states = model.init_conv_state_caches(
        max_slots=batch_size,
        device=device,
        dtype=runtime.dtype,
    )
    if args.compile:
        for cache in kv_caches:
            if cache is not None:
                torch._dynamo.mark_static_address(cache.k_blocks)
                torch._dynamo.mark_static_address(cache.v_blocks)
        for state in conv_states:
            torch._dynamo.mark_static_address(state)
    conv_slots = torch.arange(batch_size, device=device)
    prompt_ids = torch.randint(
        0, 128_000, (batch_size, args.prompt_tokens), device=device
    )
    next_ids = torch.randint(0, 128_000, (batch_size, 1), device=device)

    def decode(cursor: int):
        kv_lens = torch.full((batch_size,), cursor, device=device, dtype=torch.long)
        max_seq_len = cursor + 1
        if args.compile:
            max_seq_len = _shape_bucket(max_seq_len)
        return model(
            next_ids,
            block_tables=block_tables,
            kv_lens=kv_lens,
            paged_kv_caches=kv_caches,
            conv_state_caches=conv_states,
            conv_slots=conv_slots,
            max_seq_len=max_seq_len,
        )

    with torch.inference_mode():
        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()
        model(
            prompt_ids,
            block_tables=block_tables,
            kv_lens=torch.zeros(batch_size, device=device, dtype=torch.long),
            paged_kv_caches=kv_caches,
            conv_state_caches=conv_states,
            conv_slots=conv_slots,
            max_seq_len=(
                _shape_bucket(args.prompt_tokens)
                if args.compile
                else args.prompt_tokens
            ),
        )
        cursor = args.prompt_tokens
        for _ in range(4):
            if args.compile:
                torch.compiler.cudagraph_mark_step_begin()
            decode(cursor)
            cursor += 1
        torch.cuda.synchronize()

        started = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        started.record()
        for _ in range(args.steps):
            if args.compile:
                torch.compiler.cudagraph_mark_step_begin()
            decode(cursor)
            cursor += 1
        ended.record()
        ended.synchronize()
        elapsed_s = started.elapsed_time(ended) / 1000

        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            for _ in range(4):
                if args.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                decode(cursor)
                cursor += 1
            torch.cuda.synchronize()

    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(args.trace))
    print(
        f"model={runtime.model_id} quantization={runtime.quantization} "
        f"compile={args.compile} batch={batch_size}"
    )
    print(
        f"model_only_tokens_per_s={batch_size * args.steps / elapsed_s:.2f} "
        f"step_latency_ms={elapsed_s * 1000 / args.steps:.3f}"
    )
    print("Top CUDA operators:")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
    print("Top CPU operators:")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))


if __name__ == "__main__":
    main()
