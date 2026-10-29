"""Independent Transformers reference for a random tiny LFM3 model.

The LFM3 architecture ships as Hugging Face remote code, so this test needs
the reference implementation to be available (cached or downloadable). It is
skipped in offline environments instead of downloading silently.
"""

import pytest
import torch

from lfm3.model import Lfm3Model
from lfm3.utilities import load_weights_into_lfm3

MODEL_ID = "LiquidAI/LFM3-1.7B"
LAYER_TYPES = ["conv", "sliding_attention", "full_attention", "conv"]


def _load_reference_classes():
    pytest.importorskip("transformers")
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    try:
        config_cls = type(
            AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
        )
        model_cls = get_class_from_dynamic_module(
            "modeling_lfm3.Lfm3ForCausalLM", MODEL_ID
        )
    except Exception as exc:  # pragma: no cover - offline/network dependent
        pytest.skip(f"LFM3 Transformers reference unavailable: {exc}")
    return config_cls, model_cls


def test_chunked_model_logits_match_transformers():
    config_cls, model_cls = _load_reference_classes()

    torch.manual_seed(42)
    hf_config = config_cls(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        sliding_window=4,
        conv_L_cache=3,
        layer_types=LAYER_TYPES,
        norm_eps=1e-5,
        tie_word_embeddings=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_sandwich_norm=True,
        conv_bias=False,
    )
    hf_config._attn_implementation = "eager"
    reference = model_cls(hf_config).eval()

    config = dict(
        vocab_size=64,
        context_length=64,
        emb_dim=32,
        n_heads=4,
        n_kv_groups=2,
        n_layers=4,
        hidden_dim=64,
        head_dim=8,
        norm_eps=1e-5,
        conv_kernel_size=3,
        conv_bias=False,
        sliding_window=4,
        layer_types=LAYER_TYPES,
        rope_theta_full=1_000_000.0,
        rope_theta_sliding=10_000.0,
        partial_rotary_factor_full=0.5,
        partial_rotary_factor_sliding=1.0,
        tie_word_embeddings=True,
        dtype=torch.float32,
    )
    model = Lfm3Model(config).eval()
    load_weights_into_lfm3(model, reference.state_dict())

    ids = torch.tensor([[2, 12, 7, 34, 9, 13, 5]])
    tables = torch.tensor([[3, 1, 5, 0]])
    with torch.inference_mode():
        expected = reference(ids, use_cache=False).logits

        cache = model.init_paged_kv_caches(num_blocks=8, block_size=2)
        full = model(
            ids,
            block_tables=tables,
            kv_lens=torch.tensor([0]),
            paged_kv_caches=cache,
            logits_last_only=False,
        )
        torch.testing.assert_close(full, expected, atol=3e-6, rtol=3e-5)

        cache = model.init_paged_kv_caches(num_blocks=8, block_size=2)
        conv_state = model.init_conv_state_caches(
            max_slots=1, device=torch.device("cpu"), dtype=torch.float32
        )
        cursor = 0
        for size in (2, 3, 1, 1):
            logits = model(
                ids[:, cursor : cursor + size],
                block_tables=tables,
                kv_lens=torch.tensor([cursor]),
                paged_kv_caches=cache,
                conv_state_caches=conv_state,
                conv_slots=torch.tensor([0]),
                logits_last_only=False,
            )
            torch.testing.assert_close(
                logits,
                expected[:, cursor : cursor + size],
                atol=3e-6,
                rtol=3e-5,
            )
            cursor += size


def test_conv_state_slots_are_independent():
    torch.manual_seed(0)
    config = dict(
        vocab_size=32,
        context_length=16,
        emb_dim=8,
        n_heads=2,
        n_kv_groups=1,
        n_layers=2,
        hidden_dim=16,
        head_dim=4,
        norm_eps=1e-5,
        conv_kernel_size=3,
        conv_bias=False,
        sliding_window=4,
        layer_types=["conv", "conv"],
        rope_theta_full=10_000.0,
        rope_theta_sliding=10_000.0,
        partial_rotary_factor_full=1.0,
        partial_rotary_factor_sliding=1.0,
        tie_word_embeddings=True,
        dtype=torch.float32,
    )
    model = Lfm3Model(config).eval()
    left_tables = torch.tensor([[0, 1]])
    right_tables = torch.tensor([[2, 3]])
    left = torch.tensor([[1, 2, 3, 4]])
    right = torch.tensor([[9, 8, 7, 6]])

    cache = model.init_paged_kv_caches(num_blocks=8, block_size=2)
    with torch.inference_mode():
        expected_left = model(
            left,
            block_tables=left_tables,
            kv_lens=torch.tensor([0]),
            paged_kv_caches=cache,
            logits_last_only=False,
        )
        expected_right = model(
            right,
            block_tables=right_tables,
            kv_lens=torch.tensor([0]),
            paged_kv_caches=cache,
            logits_last_only=False,
        )

        cache = model.init_paged_kv_caches(num_blocks=8, block_size=2)
        conv_state = model.init_conv_state_caches(
            max_slots=3, device=torch.device("cpu"), dtype=torch.float32
        )
        left_first = model(
            left[:, :2],
            block_tables=left_tables,
            kv_lens=torch.tensor([0]),
            paged_kv_caches=cache,
            conv_state_caches=conv_state,
            conv_slots=torch.tensor([0]),
            logits_last_only=False,
        )
        right_all = model(
            right,
            block_tables=right_tables,
            kv_lens=torch.tensor([0]),
            paged_kv_caches=cache,
            conv_state_caches=conv_state,
            conv_slots=torch.tensor([2]),
            logits_last_only=False,
        )
        left_rest = model(
            left[:, 2:],
            block_tables=left_tables,
            kv_lens=torch.tensor([2]),
            paged_kv_caches=cache,
            conv_state_caches=conv_state,
            conv_slots=torch.tensor([0]),
            logits_last_only=False,
        )
    torch.testing.assert_close(
        torch.cat([left_first, left_rest], dim=1),
        expected_left,
        atol=3e-6,
        rtol=3e-5,
    )
    torch.testing.assert_close(right_all, expected_right, atol=3e-6, rtol=3e-5)
