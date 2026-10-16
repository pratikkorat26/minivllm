import pytest
import torch

from gemma3.attention import GroupedQueryAttention
from gemma3.paged_kv import PagedKVCache
from gemma3.rope import RotaryEmbedding, apply_rope_single


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
def test_ragged_decode_matches_independent_dense_attention(device, window, groups):
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
            q = attention.q_proj(x[:, -1:]).view(1, 1, 4, 4).transpose(1, 2)
            k = attention.k_proj(x).view(1, -1, groups, 4).transpose(1, 2)
            v = attention.v_proj(x).view(1, -1, groups, 4).transpose(1, 2)
            cos, sin = attention.rope.get_cos_sin(
                x.shape[1], device=device, dtype=x.dtype
            )
            q = apply_rope_single(q, cos[-1:], sin[-1:])
            k = apply_rope_single(k, cos, sin).repeat_interleave(4 // groups, dim=1)
            v = v.repeat_interleave(4 // groups, dim=1)
            scores = (q @ k.transpose(-1, -2)) * attention.scale
            if window:
                scores[..., :-window] = float("-inf")
            out = torch.softmax(scores, dim=-1) @ v
            expected.append(attention.out_proj(out.transpose(1, 2).reshape(1, 1, 16)))
        torch.testing.assert_close(actual, torch.cat(expected), atol=2e-6, rtol=2e-5)
