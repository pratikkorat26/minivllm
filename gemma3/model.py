import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional

from gemma3.attention import RMSNorm, GroupedQueryAttention
from gemma3.feedforward import FeedForward
from gemma3.paged_kv import PagedKVCache
from gemma3.rope import RotaryEmbedding


GEMMA3_CONFIG_270M = {
    "vocab_size": 262_144,
    "context_length": 32_768,
    "emb_dim": 640,
    "n_heads": 4,
    "n_layers": 18,
    "hidden_dim": 2048,
    "head_dim": 256,
    "qk_norm": True,
    "n_kv_groups": 1,
    "rope_local_base": 10_000.0,
    "rope_base": 1_000_000.0,
    "sliding_window": 512,
    "layer_types": [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ],
    "dtype": torch.bfloat16,
    "query_pre_attn_scalar": 256,
}


class TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        emb_dim: int,
        hidden_dim: int,
        n_heads: int,
        n_kv_groups: int,
        head_dim: int,
        rope: RotaryEmbedding,
        sliding_window: Optional[int],
        qk_norm: bool,
        query_pre_attn_scalar: Optional[float],
        dtype: Optional[torch.dtype],
    ):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_model=emb_dim,
            num_heads=n_heads,
            num_kv_groups=n_kv_groups,
            head_dim=head_dim,
            rope=rope,
            sliding_window=sliding_window,
            qk_norm=qk_norm,
            query_pre_attn_scalar=query_pre_attn_scalar,
            dtype=dtype,
        )
        self.ff = FeedForward(emb_dim=emb_dim, hidden_dim=hidden_dim, dtype=dtype)
        self.input_layernorm = RMSNorm(emb_dim)
        self.post_attention_layernorm = RMSNorm(emb_dim)
        self.pre_feedforward_layernorm = RMSNorm(emb_dim)
        self.post_feedforward_layernorm = RMSNorm(emb_dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        block_tables: torch.Tensor,
        kv_lens: torch.Tensor,
        paged_kv_cache: PagedKVCache,
        max_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        h = self.input_layernorm(x)
        att_out = self.att(
            h,
            block_tables=block_tables,
            kv_lens=kv_lens,
            paged_kv_cache=paged_kv_cache,
            max_seq_len=max_seq_len,
        )
        x = x + self.post_attention_layernorm(att_out)
        h = self.pre_feedforward_layernorm(x)
        ff_out = self.ff(h)
        x = x + self.post_feedforward_layernorm(ff_out)
        return x


class Gemma3Model(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        vocab_size = config["vocab_size"]
        context_length = config["context_length"]
        emb_dim = config["emb_dim"]
        n_heads = config["n_heads"]
        n_layers = config["n_layers"]
        hidden_dim = config["hidden_dim"]
        head_dim = config["head_dim"]
        qk_norm = config["qk_norm"]
        n_kv_groups = config["n_kv_groups"]
        rope_local_base = config["rope_local_base"]
        rope_base = config["rope_base"]
        sliding_window = config["sliding_window"]
        layer_types = config["layer_types"]
        dtype = config.get("dtype")
        query_pre_attn_scalar = config.get("query_pre_attn_scalar")

        if len(layer_types) != n_layers:
            raise ValueError("layer_types length must equal n_layers")

        self.context_length = context_length
        self.embed_scale = math.sqrt(emb_dim)
        self.tok_emb = nn.Embedding(vocab_size, emb_dim, dtype=dtype)

        rope_local = RotaryEmbedding(
            dim=head_dim, theta=rope_local_base, max_position_embeddings=context_length
        )
        rope_global = RotaryEmbedding(
            dim=head_dim, theta=rope_base, max_position_embeddings=context_length
        )

        blocks: List[TransformerBlock] = []
        for layer_type in layer_types:
            rope = rope_local if layer_type == "sliding_attention" else rope_global
            layer_window = sliding_window if layer_type == "sliding_attention" else None
            blocks.append(
                TransformerBlock(
                    emb_dim=emb_dim,
                    hidden_dim=hidden_dim,
                    n_heads=n_heads,
                    n_kv_groups=n_kv_groups,
                    head_dim=head_dim,
                    rope=rope,
                    sliding_window=layer_window,
                    qk_norm=qk_norm,
                    query_pre_attn_scalar=query_pre_attn_scalar,
                    dtype=dtype,
                )
            )

        self.blocks = nn.ModuleList(blocks)
        self.final_norm = RMSNorm(emb_dim)
        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False, dtype=dtype)
        self.num_kv_groups = n_kv_groups
        self.head_dim = head_dim
        self.dtype = dtype or self.out_head.weight.dtype

    def init_paged_kv_caches(
        self,
        *,
        num_blocks: int,
        block_size: int,
        device: Optional[torch.device] = None,
    ) -> List[PagedKVCache]:
        cache_device = device or self.tok_emb.weight.device
        return [
            PagedKVCache.empty(
                num_blocks=num_blocks,
                num_kv_groups=self.num_kv_groups,
                block_size=block_size,
                head_dim=self.head_dim,
                device=cache_device,
                dtype=self.dtype,
            )
            for _ in self.blocks
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        block_tables: torch.Tensor,
        kv_lens: torch.Tensor,
        paged_kv_caches: List[PagedKVCache],
        compute_logits: bool = True,
        logits_last_only: bool = True,
        max_seq_len: Optional[int] = None,
    ):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B, T]")
        if input_ids.shape[1] > self.context_length:
            raise ValueError("input sequence length exceeds context_length")
        if paged_kv_caches is not None and len(paged_kv_caches) != len(self.blocks):
            raise ValueError(
                "paged_kv_caches must have one entry per transformer block"
            )

        if max_seq_len is None:
            max_seq_len = int(kv_lens.max().item()) + input_ids.shape[1]
        x = self.tok_emb(input_ids) * torch.tensor(
            self.embed_scale, dtype=self.tok_emb.weight.dtype, device=input_ids.device
        )

        for i, block in enumerate(self.blocks):
            x = block(
                x,
                block_tables=block_tables,
                kv_lens=kv_lens,
                paged_kv_cache=paged_kv_caches[i],
                max_seq_len=max_seq_len,
            )

        x = self.final_norm(x)
        if not compute_logits:
            return x
        if logits_last_only:
            x = x[:, -1:, :]
        return self.out_head(x)


def build_gemma3_270m() -> Gemma3Model:
    return Gemma3Model(GEMMA3_CONFIG_270M)
