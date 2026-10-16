import torch

from gemma3.attention import RMSNorm
from inference.sampling import apply_repetition_penalty_, sample_next_token


def test_bfloat16_norm_adds_scale_in_float32():
    norm = RMSNorm(4).to(dtype=torch.bfloat16)
    norm.scale.data.fill_(0.01)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.bfloat16)
    normalized = x.float() * torch.rsqrt(
        x.float().square().mean(-1, keepdim=True) + norm.eps
    )
    expected = (normalized * (1 + norm.scale.float())).to(x.dtype)
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
