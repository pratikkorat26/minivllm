"""Packed projections must preserve the original separate-linear math."""

import pytest
import torch
import torch.nn.functional as F

from lfm3.attention import GroupedQueryAttention
from lfm3.feedforward import FeedForward
from lfm3.quantization import quantize_model_int8


@pytest.mark.parametrize("shape", [(1, 1, 16), (3, 7, 16)])
def test_qkv_matches_separate_projections(shape):
    torch.manual_seed(17)
    attention = GroupedQueryAttention(16, 4, 2, 4)
    x = torch.randn(shape)
    weights = attention.qkv_proj.weight.split(attention.qkv_sizes, dim=0)
    expected = tuple(F.linear(x, weight) for weight in weights)
    actual = attention.qkv_proj(x).split(attention.qkv_sizes, dim=-1)
    for packed, separate in zip(actual, expected):
        torch.testing.assert_close(packed, separate)


@pytest.mark.parametrize("shape", [(1, 1, 16), (3, 7, 16)])
@pytest.mark.parametrize("quantized", [False, True])
def test_gate_up_matches_separate_projections(shape, quantized):
    torch.manual_seed(19)
    ffn = FeedForward(16, 32)
    x = torch.randn(shape)
    gate_weight, up_weight = ffn.gate_up_proj.weight.detach().chunk(2, dim=0)
    expected = ffn.w2(F.silu(F.linear(x, gate_weight)) * F.linear(x, up_weight))
    if quantized:
        quantize_model_int8(ffn)
    actual = ffn(x)
    torch.testing.assert_close(
        actual, expected, atol=0.01 if quantized else 1e-6,
        rtol=0.02 if quantized else 1e-5,
    )
