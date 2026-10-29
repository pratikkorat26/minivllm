"""Synthetic LFM3-shaped attention microbenchmark, not a model benchmark."""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from lfm3.attention import GroupedQueryAttention
from lfm3.paged_kv import PagedKVCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    torch.manual_seed(42)
    rows = []
    for batch in (1, 4):
        for context in (128, 512):
            att = (
                GroupedQueryAttention(2048, 16, 4, 128, dtype=dtype)
                .to(device)
                .eval()
            )
            blocks = (context + 31 + 15) // 16
            cache = PagedKVCache.empty(
                num_blocks=blocks * batch,
                num_kv_groups=4,
                block_size=16,
                head_dim=128,
                device=device,
                dtype=dtype,
            )
            cache.k_blocks.normal_()
            cache.v_blocks.normal_()
            tables = torch.arange(blocks * batch, device=device).reshape(batch, blocks)
            for query in (1, 32):
                x = torch.randn(batch, query, 2048, device=device, dtype=dtype)
                lengths = torch.full(
                    (batch,), context - 1, device=device, dtype=torch.long
                )

                def step():
                    return att(
                        x, block_tables=tables, kv_lens=lengths, paged_kv_cache=cache
                    )

                with torch.inference_mode():
                    for _ in range(10):
                        step()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(args.iterations):
                        step()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                rows.append(
                    dict(
                        batch=batch,
                        context=context,
                        query=query,
                        latency_ms=(time.perf_counter() - start)
                        * 1000
                        / args.iterations,
                    )
                )
    result = dict(
        kind="synthetic_attention",
        torch=torch.__version__,
        python=platform.python_version(),
        device=torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        iterations=args.iterations,
        rows=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
