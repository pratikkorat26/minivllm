import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from lfm3.paged_kv import PagedKVCache
from lfm3.paged_attention import can_use_paged_decode, paged_decode_attention
from lfm3.rope import rotate_half


class RMSNorm(nn.Module):
    """LFM3 RMSNorm (unit-offset weight, computed in float32)."""

    def __init__(self, emb_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_norm = (x_f * torch.rsqrt(var + self.eps)).to(orig_dtype)
        return self.weight * x_norm


class GroupedQueryAttention(nn.Module):
    """Paged GQA for LFM3 attention layers.

    Key differences from Gemma 3 attention: standard RMSNorm weights on Q/K,
    a sigmoid output gate per head, and per-layer-type RoPE (partial on full
    attention, full on sliding attention).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_groups: int,
        head_dim: int,
        rope=None,
        sliding_window: Optional[int] = None,
        norm_eps: float = 1e-5,
        dtype=None,
    ):
        super().__init__()
        if num_heads % num_kv_groups != 0:
            raise ValueError("num_heads must be divisible by num_kv_groups")

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.sliding_window = sliding_window
        self.rope = rope
        self.use_fused_paged_attention = (
            os.environ.get("LFM3_FUSED_PAGED_ATTENTION", "0") == "1"
        )
        # None selects the context-dependent heuristic; 1 forces single-pass.
        self.paged_attention_num_splits = None

        self.qkv_sizes = (
            num_heads * head_dim,
            num_kv_groups * head_dim,
            num_kv_groups * head_dim,
        )
        self.qkv_proj = nn.Linear(
            d_model, sum(self.qkv_sizes), bias=False, dtype=dtype
        )
        self.o_proj = nn.Linear(head_dim * num_heads, d_model, bias=False, dtype=dtype)
        self.gate_proj = nn.Linear(d_model, num_heads, bias=False, dtype=dtype)

        self.q_norm = RMSNorm(head_dim, eps=norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=norm_eps)

    def _append_to_paged_cache(
        self,
        *,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        block_tables: torch.Tensor,
        kv_lens: torch.Tensor,
        paged_kv_cache: PagedKVCache,
    ) -> None:
        # Storage is [physical block, KV head, token in block, head dim].
        batch_size, groups, query_len, dim = k_new.shape
        positions = kv_lens[:, None] + torch.arange(query_len, device=k_new.device)
        ids = block_tables.gather(1, positions // paged_kv_cache.block_size)
        offsets = positions % paged_kv_cache.block_size
        paged_kv_cache.k_blocks[ids.reshape(-1), :, offsets.reshape(-1), :] = (
            k_new.transpose(1, 2).reshape(-1, groups, dim)
        )
        paged_kv_cache.v_blocks[ids.reshape(-1), :, offsets.reshape(-1), :] = (
            v_new.transpose(1, 2).reshape(-1, groups, dim)
        )

    def _gather_batch_kv(
        self,
        *,
        block_tables: torch.Tensor,
        max_seq_len: int,
        paged_kv_cache: PagedKVCache,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dense SDPA fallback: returns [batch, KV heads, tokens, dim]."""
        positions = torch.arange(max_seq_len, device=block_tables.device)
        ids = block_tables[:, positions // paged_kv_cache.block_size].clamp_min(0)
        offsets = positions % paged_kv_cache.block_size
        keys = paged_kv_cache.k_blocks[ids, :, offsets, :].transpose(1, 2)
        values = paged_kv_cache.v_blocks[ids, :, offsets, :].transpose(1, 2)
        return keys, values

    def _build_attn_mask(
        self,
        *,
        q_len: int,
        max_seq_len: int,
        seq_lens: torch.Tensor,
        kv_lens: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return additive mask ``[B, 1, q_len, max_seq_len]``."""
        batch_size = seq_lens.shape[0]
        q_pos = kv_lens.unsqueeze(1) + torch.arange(
            q_len, device=device, dtype=torch.long
        ).unsqueeze(0)
        k_pos = torch.arange(max_seq_len, device=device, dtype=torch.long).view(
            1, 1, max_seq_len
        )
        q_pos = q_pos.unsqueeze(-1)

        blocked = k_pos > q_pos
        if self.sliding_window is not None:
            blocked = blocked | (k_pos < q_pos - self.sliding_window + 1)
        blocked = blocked | (k_pos >= seq_lens.view(batch_size, 1, 1))

        mask = torch.zeros(
            (batch_size, q_len, max_seq_len), device=device, dtype=dtype
        )
        mask = mask.masked_fill(blocked, float("-inf"))
        return mask.unsqueeze(1)

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        kv_lens: torch.Tensor,
        max_seq_len: int,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rope is None:
            return q, k
        # The model grows the cache before the layer loop; this is a cheap check.
        self.rope.ensure_capacity(max_seq_len, device=q.device)
        positions = kv_lens[:, None] + torch.arange(q.shape[2], device=q.device)
        cos = self.rope.cos_cached[positions].float().unsqueeze(1)
        sin = self.rope.sin_cached[positions].float().unsqueeze(1)
        q_f = q.float()
        k_f = k.float()
        q_rot = (q_f * cos) + (rotate_half(q_f) * sin)
        k_rot = (k_f * cos) + (rotate_half(k_f) * sin)
        return q_rot.to(dtype), k_rot.to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        *,
        block_tables: torch.Tensor,
        kv_lens: torch.Tensor,
        paged_kv_cache: PagedKVCache,
        max_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size, q_len, _ = x.shape

        # One GEMM for three independent projections of the same input.
        q, k, v = self.qkv_proj(x).split(self.qkv_sizes, dim=-1)
        q = q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, q_len, self.num_kv_groups, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, q_len, self.num_kv_groups, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if max_seq_len is None:
            max_seq_len = int(kv_lens.max().item()) + q_len
        q, k = self._apply_rope(q, k, kv_lens, max_seq_len, x.dtype)

        self._append_to_paged_cache(
            k_new=k,
            v_new=v,
            block_tables=block_tables,
            kv_lens=kv_lens,
            paged_kv_cache=paged_kv_cache,
        )

        if self.use_fused_paged_attention and can_use_paged_decode(q):
            out = paged_decode_attention(
                q,
                cache=paged_kv_cache,
                block_tables=block_tables,
                kv_lens=kv_lens,
                scale=self.scale,
                sliding_window=self.sliding_window,
                max_seq_len=max_seq_len,
                num_splits=self.paged_attention_num_splits,
            )
        else:
            out = self._dense_attention(
                q, block_tables, kv_lens, paged_kv_cache, max_seq_len
            )

        out = out.transpose(1, 2).reshape(
            batch_size, q_len, self.num_heads, self.head_dim
        )
        gate = torch.sigmoid(self.gate_proj(x)).unsqueeze(-1)
        out = (out * gate).reshape(batch_size, q_len, -1)
        return self.o_proj(out)

    def _dense_attention(
        self, q, block_tables, kv_lens, paged_kv_cache, max_seq_len
    ):
        """Portable prefill/decode fallback using dense gathered KV and SDPA."""
        q_len = q.shape[2]
        seq_lens = kv_lens + q_len
        k_cache, v_cache = self._gather_batch_kv(
            block_tables=block_tables,
            max_seq_len=max_seq_len,
            paged_kv_cache=paged_kv_cache,
        )

        attn_mask = self._build_attn_mask(
            q_len=q_len,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            kv_lens=kv_lens,
            dtype=q.dtype,
            device=q.device,
        )
        return F.scaled_dot_product_attention(
            q,
            k_cache,
            v_cache,
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=self.scale,
            enable_gqa=True,
        )
