import pytest
import torch
from torch import nn

from lfm3.model import Lfm3Model
from lfm3.quantization import Int8Linear, quantize_model


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_int8_linear_matches_float_linear(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(0)
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    linear = nn.Linear(256, 64, bias=True).to(device=device, dtype=dtype)
    with torch.no_grad():
        linear.weight.normal_(0.0, 0.02)
        linear.bias.normal_(0.0, 0.02)
    quantized = Int8Linear.from_linear(linear)
    x = torch.randn(3, 5, 256, device=device, dtype=dtype)
    expected = linear(x)
    actual = quantized(x)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)


def _tiny_config():
    return {
        "vocab_size": 64,
        "context_length": 32,
        "emb_dim": 32,
        "n_heads": 4,
        "n_kv_groups": 2,
        "n_layers": 2,
        "hidden_dim": 64,
        "head_dim": 8,
        "norm_eps": 1e-5,
        "conv_kernel_size": 3,
        "conv_bias": False,
        "sliding_window": 4,
        "layer_types": ["conv", "sliding_attention"],
        "rope_theta_full": 1_000_000.0,
        "rope_theta_sliding": 10_000.0,
        "partial_rotary_factor_full": 0.5,
        "partial_rotary_factor_sliding": 1.0,
        "tie_word_embeddings": True,
        "dtype": torch.float32,
    }


def test_int8_model_keeps_tied_head_and_runs():
    torch.manual_seed(0)
    model = Lfm3Model(_tiny_config()).eval()
    model = quantize_model(model, "int8")
    assert model.out_head.weight is model.tok_emb.weight

    ids = torch.tensor([[1, 2, 3, 4]])
    cache = model.init_paged_kv_caches(num_blocks=4, block_size=2)
    with torch.inference_mode():
        logits = model(
            ids,
            block_tables=torch.tensor([[0, 1]]),
            kv_lens=torch.tensor([0]),
            paged_kv_caches=cache,
        )
    assert logits.shape == (1, 1, 64)
    assert torch.isfinite(logits).all()


def test_unknown_quantization_is_rejected():
    model = Lfm3Model(_tiny_config())
    with pytest.raises(ValueError, match="unknown quantization"):
        quantize_model(model, "int3")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_int4_model_runs():
    pytest.importorskip("bitsandbytes")
    torch.manual_seed(0)
    model = Lfm3Model(_tiny_config()).to(device="cuda").eval()
    model = quantize_model(model, "int4", compute_dtype=torch.float32)
    assert model.out_head.weight is model.tok_emb.weight

    ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    cache = model.init_paged_kv_caches(num_blocks=4, block_size=2, device="cuda")
    with torch.inference_mode():
        logits = model(
            ids,
            block_tables=torch.tensor([[0, 1]], device="cuda"),
            kv_lens=torch.tensor([0], device="cuda"),
            paged_kv_caches=cache,
        )
    assert logits.shape == (1, 1, 64)
    assert torch.isfinite(logits).all()
