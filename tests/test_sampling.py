import torch
import pytest

from lfm3.attention import RMSNorm
from inference.sampling import apply_repetition_penalty_, sample_next_token


@pytest.mark.parametrize("tensor_ids", [False, True])
def test_batched_repetition_penalty_applies_once_per_token(tensor_ids):
    logits = torch.tensor([[2.0, -4.0, 6.0, 8.0], [-2.0, 4.0, -6.0, 8.0]])
    ids = [[0, 0, 1], [1, 2, 2]]
    if tensor_ids:
        ids = torch.tensor(ids)
    actual = apply_repetition_penalty_(logits, ids, 2.0)
    torch.testing.assert_close(
        actual, torch.tensor([[1.0, -8.0, 6.0, 8.0], [-2.0, 2.0, -12.0, 8.0]])
    )


def test_repetition_penalty_handles_empty_and_ragged_histories():
    logits = torch.tensor([[2.0, -4.0, 6.0], [-2.0, 4.0, -6.0]])
    actual = apply_repetition_penalty_(logits, [set(), {0, 2}], 2.0)
    torch.testing.assert_close(
        actual, torch.tensor([[2.0, -4.0, 6.0], [-4.0, 4.0, -12.0]])
    )


def test_bfloat16_norm_applies_unit_weight_in_float32():
    norm = RMSNorm(4).to(dtype=torch.bfloat16)
    norm.weight.data.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.bfloat16))
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.bfloat16)
    normalized = (
        x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + norm.eps)
    ).to(x.dtype)
    expected = norm.weight * normalized
    torch.testing.assert_close(norm(x), expected, atol=0, rtol=0)


def test_top_k_and_nucleus_mask_excluded_tokens():
    logits = torch.tensor([[8.0, 0.0, -2.0]])
    for top_k, top_p in [(1, 1), (0, 0.5)]:
        token = sample_next_token(logits, temperature=1, top_p=top_p, top_k=top_k)
        assert token.item() == 0


def test_seeded_sampling_is_repeatable_and_repetition_penalty_respects_sign():
    logits = torch.tensor([[2.0, -2.0, 3.0]])
    penalized = apply_repetition_penalty_(logits.clone(), [{0, 1}], 2)
    torch.testing.assert_close(penalized, torch.tensor([[1.0, -4.0, 3.0]]))
    kwargs = dict(temperature=0.8, top_p=0.9, top_k=2)
    first = sample_next_token(
        logits, generator=torch.Generator().manual_seed(12), **kwargs
    )
    second = sample_next_token(
        logits, generator=torch.Generator().manual_seed(12), **kwargs
    )
    assert torch.equal(first, second)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_repetition_penalty_bool_mask_fast_path(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    logits = torch.tensor([[2.0, -4.0, 6.0, 8.0], [-2.0, 4.0, -6.0, 8.0]], device=device)
    mask = torch.tensor(
        [[True, True, False, False], [False, True, True, False]],
        dtype=torch.bool,
        device=device,
    )
    actual = apply_repetition_penalty_(logits, mask, 2.0)
    expected = torch.tensor(
        [[1.0, -8.0, 6.0, 8.0], [-2.0, 2.0, -12.0, 8.0]], device=device
    )
    torch.testing.assert_close(actual, expected)


def test_top_k_with_top_p_filters_nucleus_within_top_k():
    # Top-2 are tokens 0 and 1; with top_p=0.6 only token 0 (~0.88 prob) survives.
    logits = torch.tensor([[4.0, 2.0, -1.0, -5.0]])
    gen = torch.Generator().manual_seed(0)
    sampled = {
        sample_next_token(logits, temperature=1.0, top_k=2, top_p=0.6, generator=gen).item()
        for _ in range(10)
    }
    assert sampled == {0}


def test_top_p_without_top_k_excludes_tail_tokens():
    # Probabilities: token 0 ~0.468, token 1 ~0.468, token 2 ~0.063, token 3 ~0.001.
    logits = torch.tensor([[2.0, 2.0, 0.0, -4.0]])
    gen = torch.Generator().manual_seed(7)
    sampled = {
        sample_next_token(logits, temperature=1.0, top_k=0, top_p=0.9, generator=gen).item()
        for _ in range(25)
    }
    assert sampled.issubset({0, 1})
    assert len(sampled) == 2
