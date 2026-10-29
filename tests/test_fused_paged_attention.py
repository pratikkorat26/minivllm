"""Direct page reads agree with independent dense attention on ragged contexts."""

import pytest
import torch
import torch.nn.functional as F

from lfm3.paged_attention import paged_decode_attention
from lfm3.paged_kv import PagedKVCache


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("window", [None, 31, 256])
@pytest.mark.parametrize("page_size", [4, 16])
@pytest.mark.parametrize("num_splits", [None, 1, 3, 8])
def test_fused_paged_decode(dtype, window, page_size, num_splits):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("triton")
    torch.manual_seed(7)
    lengths = [1, 17, 129, 2049]
    batch, heads, groups, dim = 4, 4, 2, 32
    width = (max(lengths) + page_size - 1) // page_size
    cache = PagedKVCache.empty(
        num_blocks=batch * width, num_kv_groups=groups, block_size=page_size,
        head_dim=dim, device=torch.device("cuda"), dtype=dtype,
    )
    cache.k_blocks.normal_()
    cache.v_blocks.normal_()
    tables = torch.randperm(batch * width, device="cuda").reshape(batch, width)
    # Invalid unused pages must never be dereferenced by the fused kernel.
    for row, length in enumerate(lengths):
        tables[row, (length + page_size - 1) // page_size:] = -1
    q = torch.randn(batch, heads, 1, dim, device="cuda", dtype=dtype)
    kv_lens = torch.tensor(lengths, device="cuda") - 1
    actual = paged_decode_attention(
        q, cache=cache, block_tables=tables, kv_lens=kv_lens,
        scale=dim**-0.5, sliding_window=window,
        num_splits=num_splits,
    )
    for row, length in enumerate(lengths):
        positions = torch.arange(max(0, length - window) if window else 0,
                                 length, device="cuda")
        pages = tables[row, positions // page_size]
        keys = cache.k_blocks[pages, :, positions % page_size, :].transpose(0, 1)
        values = cache.v_blocks[pages, :, positions % page_size, :].transpose(0, 1)
        expected = F.scaled_dot_product_attention(
            q[row:row + 1].float(), keys[None].float(), values[None].float(),
            enable_gqa=True,
        )
        torch.testing.assert_close(
            actual[row:row + 1].float(), expected,
            atol=0.004 if dtype == torch.bfloat16 else 2e-5,
            rtol=0.01 if dtype == torch.bfloat16 else 2e-5,
        )


@pytest.mark.parametrize("num_splits", [1, 3])
def test_cuda_graph_replay_reads_updated_metadata_and_pages(num_splits):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("triton")
    torch.manual_seed(29)
    cache = PagedKVCache.empty(
        num_blocks=4, num_kv_groups=1, block_size=16, head_dim=32,
        device=torch.device("cuda"), dtype=torch.bfloat16,
    )
    cache.k_blocks.normal_()
    cache.v_blocks.normal_()
    q = torch.randn(1, 2, 1, 32, device="cuda", dtype=torch.bfloat16)
    tables = torch.tensor([[2, 0]], device="cuda")
    lengths = torch.tensor([3], device="cuda")

    def run():
        return paged_decode_attention(
            q, cache=cache, block_tables=tables, kv_lens=lengths,
            scale=32**-0.5, sliding_window=None,
            num_splits=num_splits,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()

    for length in (4, 17, 31):
        lengths.fill_(length - 1)
        tables.copy_(torch.tensor([[3, 1]], device="cuda"))
        cache.k_blocks.normal_()
        cache.v_blocks.normal_()
        q.normal_()
        graph.replay()
        positions = torch.arange(length, device="cuda")
        pages = tables[0, positions // 16]
        keys = cache.k_blocks[pages, :, positions % 16].transpose(0, 1)[None]
        values = cache.v_blocks[pages, :, positions % 16].transpose(0, 1)[None]
        expected = F.scaled_dot_product_attention(
            q.float(), keys.float(), values.float(), enable_gqa=True
        )
        torch.testing.assert_close(output.float(), expected, atol=0.004, rtol=0.01)


def test_split_kv_compiles_and_replays():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("triton")
    torch.manual_seed(31)
    cache = PagedKVCache.empty(
        num_blocks=8, num_kv_groups=1, block_size=16, head_dim=32,
        device=torch.device("cuda"), dtype=torch.bfloat16,
    )
    cache.k_blocks.normal_()
    cache.v_blocks.normal_()
    tables = torch.arange(8, device="cuda").reshape(1, 8)
    lengths = torch.tensor([126], device="cuda")

    def run(q):
        return paged_decode_attention(
            q, cache=cache, block_tables=tables, kv_lens=lengths,
            scale=32**-0.5, sliding_window=None, num_splits=3,
        )

    compiled = torch.compile(run, fullgraph=True, mode="reduce-overhead")
    for length in (127, 65, 1, 128):
        lengths.fill_(length - 1)
        q = torch.randn(1, 2, 1, 32, device="cuda", dtype=torch.bfloat16)
        torch.compiler.cudagraph_mark_step_begin()
        actual = compiled(q).clone()
        torch.testing.assert_close(actual, run(q), atol=0.004, rtol=0.01)
