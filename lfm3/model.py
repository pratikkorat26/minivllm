from typing import Dict, List, Optional

import torch
import torch.nn as nn

from lfm3.attention import GroupedQueryAttention, RMSNorm
from lfm3.conv import ShortConv
from lfm3.feedforward import FeedForward
from lfm3.paged_kv import PagedKVCache
from lfm3.rope import RotaryEmbedding


LFM3_CONFIG_1_7B = {
    "vocab_size": 128_000,
    "context_length": 262_144,
    "emb_dim": 2_048,
    "n_heads": 16,
    "n_kv_groups": 4,
    "n_layers": 24,
    "hidden_dim": 7_680,
    "head_dim": 128,
    "norm_eps": 1e-5,
    "conv_kernel_size": 7,
    "conv_bias": False,
    "sliding_window": 256,
    "layer_types": [
        "conv",
        "conv",
        "full_attention",
        "conv",
        "sliding_attention",
        "conv",
        "full_attention",
        "conv",
        "sliding_attention",
        "conv",
        "full_attention",
        "conv",
        "sliding_attention",
        "conv",
        "full_attention",
        "conv",
        "sliding_attention",
        "conv",
        "full_attention",
        "conv",
        "sliding_attention",
        "conv",
        "full_attention",
        "conv",
    ],
    "rope_theta_full": 1_000_000.0,
    "rope_theta_sliding": 10_000.0,
    "partial_rotary_factor_full": 0.5,
    "partial_rotary_factor_sliding": 1.0,
    "tie_word_embeddings": True,
    "dtype": torch.bfloat16,
}


class TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        layer_type: str,
        emb_dim: int,
        hidden_dim: int,
        n_heads: int,
        n_kv_groups: int,
        head_dim: int,
        rope_sliding: RotaryEmbedding,
        rope_full: RotaryEmbedding,
        sliding_window: int,
        norm_eps: float,
        conv_kernel_size: int,
        conv_bias: bool,
        dtype: Optional[torch.dtype],
    ):
        super().__init__()
        self.layer_type = layer_type
        self.is_attention = layer_type in ("sliding_attention", "full_attention")

        if self.is_attention:
            rope = rope_full if layer_type == "full_attention" else rope_sliding
            window = None if layer_type == "full_attention" else sliding_window
            self.self_attn = GroupedQueryAttention(
                d_model=emb_dim,
                num_heads=n_heads,
                num_kv_groups=n_kv_groups,
                head_dim=head_dim,
                rope=rope,
                sliding_window=window,
                norm_eps=norm_eps,
                dtype=dtype,
            )
        else:
            self.conv = ShortConv(
                hidden_size=emb_dim,
                kernel_size=conv_kernel_size,
                bias=conv_bias,
                dtype=dtype,
            )

        self.feed_forward = FeedForward(
            emb_dim=emb_dim, hidden_dim=hidden_dim, dtype=dtype
        )
        self.operator_norm = RMSNorm(emb_dim, eps=norm_eps)
        self.post_operator_norm = RMSNorm(emb_dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(emb_dim, eps=norm_eps)
        self.post_ffn_norm = RMSNorm(emb_dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        *,
        block_tables: torch.Tensor,
        kv_lens: torch.Tensor,
        paged_kv_cache: Optional[PagedKVCache] = None,
        conv_state: Optional[torch.Tensor] = None,
        conv_slots: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        residual = x
        h = self.operator_norm(x)
        if self.is_attention:
            h = self.self_attn(
                h,
                block_tables=block_tables,
                kv_lens=kv_lens,
                paged_kv_cache=paged_kv_cache,
                max_seq_len=max_seq_len,
            )
        else:
            h = self.conv(h, conv_state=conv_state, slots=conv_slots)
        h = self.post_operator_norm(h)
        x = residual + h

        ff = self.feed_forward(self.ffn_norm(x))
        ff = self.post_ffn_norm(ff)
        return x + ff


class Lfm3Model(nn.Module):
    # Reusing paged KV prefixes would also require restoring the short-conv
    # recurrent state at the cache boundary; not implemented yet.
    supports_prefix_cache = False

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
        norm_eps = config["norm_eps"]
        n_kv_groups = config["n_kv_groups"]
        conv_kernel_size = config["conv_kernel_size"]
        conv_bias = config.get("conv_bias", False)
        sliding_window = config["sliding_window"]
        layer_types = config["layer_types"]
        dtype = config.get("dtype")

        if len(layer_types) != n_layers:
            raise ValueError("layer_types length must equal n_layers")
        if any(
            layer_type not in ("conv", "sliding_attention", "full_attention")
            for layer_type in layer_types
        ):
            raise ValueError("unsupported layer type")

        self.context_length = context_length
        self.layer_types = list(layer_types)
        self.conv_layer_indices = [
            i for i, layer_type in enumerate(layer_types) if layer_type == "conv"
        ]

        self.tok_emb = nn.Embedding(vocab_size, emb_dim, dtype=dtype)

        rope_sliding = RotaryEmbedding(
            dim=head_dim,
            theta=config["rope_theta_sliding"],
            max_position_embeddings=context_length,
            partial_rotary_factor=config.get("partial_rotary_factor_sliding", 1.0),
        )
        rope_full = RotaryEmbedding(
            dim=head_dim,
            theta=config["rope_theta_full"],
            max_position_embeddings=context_length,
            partial_rotary_factor=config.get("partial_rotary_factor_full", 1.0),
        )

        blocks: List[TransformerBlock] = []
        for layer_type in layer_types:
            blocks.append(
                TransformerBlock(
                    layer_type=layer_type,
                    emb_dim=emb_dim,
                    hidden_dim=hidden_dim,
                    n_heads=n_heads,
                    n_kv_groups=n_kv_groups,
                    head_dim=head_dim,
                    rope_sliding=rope_sliding,
                    rope_full=rope_full,
                    sliding_window=sliding_window,
                    norm_eps=norm_eps,
                    conv_kernel_size=conv_kernel_size,
                    conv_bias=conv_bias,
                    dtype=dtype,
                )
            )

        self.blocks = nn.ModuleList(blocks)
        self.embedding_norm = RMSNorm(emb_dim, eps=norm_eps)
        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False, dtype=dtype)
        if config.get("tie_word_embeddings", True):
            self.out_head.weight = self.tok_emb.weight

        self.num_kv_groups = n_kv_groups
        self.head_dim = head_dim
        self.kernel_size = conv_kernel_size
        self.dtype = dtype or self.out_head.weight.dtype

        self.rope_sliding = rope_sliding
        self.rope_full = rope_full

    def init_paged_kv_caches(
        self,
        *,
        num_blocks: int,
        block_size: int,
        device: Optional[torch.device] = None,
    ) -> List[Optional[PagedKVCache]]:
        cache_device = device or self.tok_emb.weight.device
        caches: List[Optional[PagedKVCache]] = []
        for layer_type in self.layer_types:
            if layer_type == "conv":
                caches.append(None)
                continue
            caches.append(
                PagedKVCache.empty(
                    num_blocks=num_blocks,
                    num_kv_groups=self.num_kv_groups,
                    block_size=block_size,
                    head_dim=self.head_dim,
                    device=cache_device,
                    dtype=self.dtype,
                )
            )
        return caches

    def init_conv_state_caches(
        self,
        *,
        max_slots: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> List[torch.Tensor]:
        cache_device = device or self.tok_emb.weight.device
        cache_dtype = dtype or self.dtype
        state_len = self.kernel_size - 1
        return [
            torch.zeros(
                max_slots,
                self.config["emb_dim"],
                state_len,
                device=cache_device,
                dtype=cache_dtype,
            )
            for _ in self.conv_layer_indices
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        block_tables: torch.Tensor,
        kv_lens: torch.Tensor,
        paged_kv_caches: List[Optional[PagedKVCache]],
        conv_state_caches: Optional[List[torch.Tensor]] = None,
        conv_slots: Optional[torch.Tensor] = None,
        compute_logits: bool = True,
        logits_last_only: bool = True,
        max_seq_len: Optional[int] = None,
    ):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B, T]")
        if input_ids.shape[1] > self.context_length:
            raise ValueError("input sequence length exceeds context_length")
        if paged_kv_caches is not None and len(paged_kv_caches) != len(self.blocks):
            raise ValueError("paged_kv_caches must have one entry per transformer block")
        if conv_state_caches is not None and len(conv_state_caches) != len(
            self.conv_layer_indices
        ):
            raise ValueError("conv_state_caches must have one entry per conv layer")

        if max_seq_len is None:
            max_seq_len = int(kv_lens.max().item()) + input_ids.shape[1]
        self.rope_sliding.ensure_capacity(max_seq_len, device=input_ids.device)
        self.rope_full.ensure_capacity(max_seq_len, device=input_ids.device)

        x = self.tok_emb(input_ids)

        conv_index = 0
        for i, block in enumerate(self.blocks):
            if block.is_attention:
                x = block(
                    x,
                    block_tables=block_tables,
                    kv_lens=kv_lens,
                    paged_kv_cache=paged_kv_caches[i],
                    max_seq_len=max_seq_len,
                )
            else:
                conv_state = (
                    None
                    if conv_state_caches is None
                    else conv_state_caches[conv_index]
                )
                x = block(
                    x,
                    block_tables=block_tables,
                    kv_lens=kv_lens,
                    conv_state=conv_state,
                    conv_slots=conv_slots,
                    max_seq_len=max_seq_len,
                )
                conv_index += 1

        x = self.embedding_norm(x)
        if not compute_logits:
            return x
        if logits_last_only:
            x = x[:, -1:, :]
        return self.out_head(x)


def build_lfm3_1_7b() -> Lfm3Model:
    return Lfm3Model(LFM3_CONFIG_1_7B)
