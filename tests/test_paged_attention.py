import pytest
import torch

from lfm3.attention import GroupedQueryAttention
from lfm3.paged_kv import PagedKVCache
from lfm3.rope import RotaryEmbedding, apply_rope_single


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("query_len", [1, 3, 9])
def test_window_gather_matches_full_history_for_ragged_chunks(device, query_len):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(21)
    window, context = 4, 32  # Context includes padding for compiled shapes.
    attention = GroupedQueryAttention(16, 4, 2, 4, sliding_window=window).to(device)
    cache = PagedKVCache.empty(
        num_blocks=24,
        num_kv_groups=2,
        block_size=4,
        head_dim=4,
        device=torch.device(device),
        dtype=torch.float32,
    )
    cache.k_blocks.normal_()
    cache.v_blocks.normal_()
    tables = torch.randperm(24, device=device).reshape(3, 8)
    lengths = torch.tensor([0, 5, 17], device=device)
    q = torch.randn(3, 4, query_len, 4, device=device)

    actual = attention._dense_attention(q, tables, lengths, cache, context)
    keys, values = attention._gather_batch_kv(
        block_tables=tables,
        max_seq_len=context,
        paged_kv_cache=cache,
    )
    scores = q @ keys.repeat_interleave(2, dim=1).transpose(-1, -2)
    scores *= attention.scale
    q_positions = (
        lengths[:, None, None] + torch.arange(query_len, device=device)[None, :, None]
    )
    k_positions = torch.arange(context, device=device)[None, None, :]
    blocked = (k_positions > q_positions) | (k_positions <= q_positions - window)
    scores.masked_fill_(blocked[:, None], float("-inf"))
    expected = scores.softmax(-1) @ values.repeat_interleave(2, dim=1)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("groups", [1, 2])
def test_physical_page_layout(groups):
    attention = GroupedQueryAttention(16, 4, groups, 4)
    cache = PagedKVCache.empty(
        num_blocks=5,
        num_kv_groups=groups,
        block_size=2,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    keys = torch.arange(2 * groups * 3 * 4).reshape(2, groups, 3, 4).float()
    tables = torch.tensor([[3, 1], [4, 0]])
    lengths = torch.zeros(2, dtype=torch.long)
    attention._append_to_paged_cache(
        k_new=keys,
        v_new=-keys,
        block_tables=tables,
        kv_lens=lengths,
        paged_kv_cache=cache,
    )
    for row in range(2):
        for token in range(3):
            block = tables[row, token // 2]
            torch.testing.assert_close(
                cache.k_blocks[block, :, token % 2], keys[row, :, token]
            )
            torch.testing.assert_close(
                cache.v_blocks[block, :, token % 2], -keys[row, :, token]
            )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("window", [None, 3])
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("fused", [False, True])
def test_ragged_decode_matches_independent_dense_attention(
    device, window, groups, fused
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(12)
    device = torch.device(device)
    attention = (
        GroupedQueryAttention(
            16,
            4,
            groups,
            4,
            rope=RotaryEmbedding(4, max_position_embeddings=32),
            sliding_window=window,
        )
        .to(device)
        .eval()
    )
    cache = PagedKVCache.empty(
        num_blocks=10,
        num_kv_groups=groups,
        block_size=2,
        head_dim=4,
        device=device,
        dtype=torch.float32,
    )
    attention.use_fused_paged_attention = fused
    tables = torch.tensor([[3, 1, 2, 0, 4], [9, 6, 8, 5, 7]], device=device)
    histories = [torch.randn(1, length, 16, device=device) for length in (4, 7)]
    with torch.inference_mode():
        for row, history in enumerate(histories):
            attention(
                history,
                block_tables=tables[row : row + 1],
                kv_lens=torch.zeros(1, device=device, dtype=torch.long),
                paged_kv_cache=cache,
            )
        inputs = torch.randn(2, 1, 16, device=device)
        actual = attention(
            inputs,
            block_tables=tables,
            kv_lens=torch.tensor([4, 7], device=device),
            paged_kv_cache=cache,
            max_seq_len=8,
        )
        expected = []
        for row, history in enumerate(histories):
            x = torch.cat([history, inputs[row : row + 1]], dim=1)
            qw, kw, vw = attention.qkv_proj.weight.split(attention.qkv_sizes, dim=0)
            q = attention.q_norm(
                torch.nn.functional.linear(x[:, -1:], qw)
                .view(1, 1, 4, 4)
                .transpose(1, 2)
            )
            k = attention.k_norm(
                torch.nn.functional.linear(x, kw).view(1, -1, groups, 4).transpose(1, 2)
            )
            v = torch.nn.functional.linear(x, vw).view(1, -1, groups, 4).transpose(1, 2)
            cos, sin = attention.rope.get_cos_sin(
                x.shape[1], device=device, dtype=torch.float32
            )
            q = apply_rope_single(q.float(), cos[-1:], sin[-1:]).to(x.dtype)
            k = apply_rope_single(k.float(), cos, sin).to(x.dtype)
            k = k.repeat_interleave(4 // groups, dim=1)
            v = v.repeat_interleave(4 // groups, dim=1)
            scores = (q @ k.transpose(-1, -2)) * attention.scale
            if window:
                scores[..., :-window] = float("-inf")
            out = torch.softmax(scores, dim=-1) @ v
            gate = torch.sigmoid(attention.gate_proj(x[:, -1:])).unsqueeze(-1)
            out = (out.transpose(1, 2) * gate).reshape(1, 1, 16)
            expected.append(attention.o_proj(out))
        torch.testing.assert_close(actual, torch.cat(expected), atol=2e-6, rtol=2e-5)
