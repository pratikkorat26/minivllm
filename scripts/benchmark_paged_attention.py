"""Compare one-layer dense-gather SDPA with direct-page Triton decode.

This isolates attention on synthetic KV, not end-to-end model throughput.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lfm3.attention import GroupedQueryAttention
from lfm3.paged_attention import paged_decode_attention
from lfm3.paged_kv import PagedKVCache


def latency_ms(fn, repeats, cuda_graphs=False):
    for _ in range(5):
        fn()
    if cuda_graphs:
        graph = torch.cuda.CUDAGraph()
        outputs = []  # Keep graph-owned output storage alive through replay.
        with torch.cuda.graph(graph):
            outputs.append(fn())
        fn = graph.replay
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--contexts", nargs="+", type=int, default=[128, 512, 2048])
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--splits", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--cuda-graphs", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.repeats, *args.contexts) <= 0:
        parser.error("batch, repeats and contexts must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA required")
    if any(not 1 <= splits <= 32 for splits in args.splits):
        parser.error("splits must be between 1 and 32")

    torch.manual_seed(23)
    page = 16
    for context in args.contexts:
        width = (context + page - 1) // page
        cache = PagedKVCache.empty(
            num_blocks=args.batch_size * width, num_kv_groups=4,
            block_size=page, head_dim=128, device=torch.device("cuda"),
            dtype=torch.bfloat16,
        )
        cache.k_blocks.normal_()
        cache.v_blocks.normal_()
        tables = torch.randperm(args.batch_size * width, device="cuda").reshape(
            args.batch_size, width
        )
        lengths = torch.full((args.batch_size,), context - 1, device="cuda")
        q = torch.randn(args.batch_size, 16, 1, 128, device="cuda",
                        dtype=torch.bfloat16)
        for window in (None, 256):
            attention = GroupedQueryAttention(2048, 16, 4, 128,
                                              sliding_window=window)

            def dense():
                return attention._dense_attention(q, tables, lengths, cache, context)

            dense_ms = latency_ms(dense, args.repeats, args.cuda_graphs)
            for splits in args.splits:
                def fused():
                    return paged_decode_attention(
                        q, cache=cache, block_tables=tables, kv_lens=lengths,
                        scale=128**-0.5, sliding_window=window,
                        max_seq_len=context, num_splits=splits,
                    )

                torch.testing.assert_close(fused(), dense(), atol=0.004, rtol=0.02)
                fused_ms = latency_ms(fused, args.repeats, args.cuda_graphs)
                print(
                    f"batch={args.batch_size} context={context} window={window} "
                    f"splits={splits} cuda_graphs={args.cuda_graphs} "
                    f"dense_ms={dense_ms:.3f} "
                    f"fused_ms={fused_ms:.3f} "
                    f"speedup={dense_ms / fused_ms:.2f}x", flush=True,
                )


if __name__ == "__main__":
    main()
