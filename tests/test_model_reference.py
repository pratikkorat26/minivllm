"""Independent Transformers implementation, random tiny weights; no download."""

import pytest
import torch

from gemma3.model import Gemma3Model
from gemma3.utilities import load_weights_into_gemma


def test_chunked_model_logits_match_transformers():
    pytest.importorskip("transformers")
    from transformers import Gemma3TextConfig, Gemma3ForCausalLM

    torch.manual_seed(42)
    config = dict(
        vocab_size=64,
        context_length=32,
        emb_dim=32,
        n_heads=4,
        n_layers=2,
        hidden_dim=64,
        head_dim=8,
        qk_norm=True,
        n_kv_groups=2,
        rope_local_base=10000.0,
        rope_base=1000000.0,
        sliding_window=4,
        layer_types=["sliding_attention", "full_attention"],
        dtype=torch.float32,
        query_pre_attn_scalar=8,
    )
    hf_config = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        sliding_window=4,
        layer_types=config["layer_types"],
        query_pre_attn_scalar=8,
        sliding_window_pattern=2,
        attn_implementation="eager",
    )
    reference = Gemma3ForCausalLM(hf_config).eval()
    model = Gemma3Model(config).eval()
    load_weights_into_gemma(model, config, reference.state_dict())
    cache = model.init_paged_kv_caches(num_blocks=8, block_size=2)
    ids = torch.tensor([[2, 12, 7, 34, 9, 13, 5]])
    tables = torch.tensor([[3, 1, 5, 0]])
    with torch.inference_mode():
        expected = reference(ids).logits
        full = model(
            ids,
            block_tables=tables,
            kv_lens=torch.tensor([0]),
            paged_kv_caches=cache,
            logits_last_only=False,
        )
        torch.testing.assert_close(full, expected, atol=3e-6, rtol=3e-5)
        cursor = 0
        for size in (2, 3, 1, 1):
            logits = model(
                ids[:, cursor : cursor + size],
                block_tables=tables,
                kv_lens=torch.tensor([cursor]),
                paged_kv_caches=cache,
                logits_last_only=False,
            )
            torch.testing.assert_close(
                logits, expected[:, cursor : cursor + size], atol=3e-6, rtol=3e-5
            )
            cursor += size
