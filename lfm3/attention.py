import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from lfm3.paged_kv import PagedKVCache
from lfm3.paged_attention import can_use_paged_decode, paged_decode_attention
from lfm3.rope import rotate_half


def prefer_cudnn_attention(enabled: bool = True) -> bool:
    """Give cuDNN's fused SDPA kernel priority over the default backends.

    With an additive attention mask the default dispatcher prefers the
    memory-efficient kernel, even though cuDNN handles that mask several times
    faster on Ampere and newer GPUs.  The fused kernel reduces in a different
    order, so BF16 near-ties in greedy decoding can resolve differently, and
    repeated calls are not bitwise reproducible at some batch/context shapes.
    The option is therefore off by default; see the benchmark notes.  The
    setting is process-wide: choose it before constructing the first executor,
    because a later ``False`` cannot undo an installed priority.  Returns
    ``True`` when the priority was installed.
    """
    if not enabled or not torch.cuda.is_available():
        return False
    try:
        from torch.nn.attention import SDPBackend, _sdpa_kernel
    except ImportError:
        return False
    try:
        _sdpa_kernel(
            [
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ],
            set_priority=True,
        )
    except (AttributeError, RuntimeError, TypeError):
        return False
    # The backend choice is baked into Inductor's generated graph.  Compile
    # artifacts written before this preference existed would otherwise be
    # reused under the same graph key, so keep cuDNN-built artifacts in their
    # own cache directory.  This is process-wide, like the backend priority.
    try:
        from torch._inductor.runtime.cache_dir_utils import cache_dir

        base = cache_dir()
        suffix = "minivllm-cudnn-attention"
        if not base.endswith(suffix):
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(base, suffix)
    except (ImportError, OSError, RuntimeError):
        pass
    return True


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
            os.environ.get("LFM3_FUSED_PAGED_ATTENTION", "1") == "1"
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
        write_indices: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> None:
        # Storage is [physical block, KV head, token in block, head dim].
        batch_size, groups, query_len, dim = k_new.shape
        if write_indices is not None:
            flat_ids, flat_offsets = write_indices
        else:
            positions = kv_lens[:, None] + torch.arange(query_len, device=k_new.device)
            flat_ids = block_tables.gather(1, positions // paged_kv_cache.block_size).reshape(-1)
            flat_offsets = (positions % paged_kv_cache.block_size).reshape(-1)
        paged_kv_cache.k_blocks[flat_ids, :, flat_offsets, :] = (
            k_new.transpose(1, 2).reshape(-1, groups, dim)
        )
        paged_kv_cache.v_blocks[flat_ids, :, flat_offsets, :] = (
            v_new.transpose(1, 2).reshape(-1, groups, dim)
        )

    def _append_to_sliding_ring(
        self,
        *,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        kv_lens: torch.Tensor,
        paged_kv_cache: PagedKVCache,
        slots: torch.Tensor,
    ) -> None:
        """Write only the latest ``min(q_len, window)`` tokens into per-slot ring buffers."""
        W = paged_kv_cache.sliding_window
        q_len = k_new.shape[2]
        if q_len == 1:
            pos = kv_lens % W
            paged_kv_cache.k_blocks[slots, :, pos, :] = k_new[:, :, 0, :]
            paged_kv_cache.v_blocks[slots, :, pos, :] = v_new[:, :, 0, :]
            return
        write_len = min(q_len, W)
        src_start = q_len - write_len
        positions = (kv_lens[:, None] + torch.arange(src_start, q_len, device=k_new.device)) % W
        paged_kv_cache.k_blocks[slots[:, None], :, positions, :] = (
            k_new[:, :, src_start:, :].transpose(1, 2)
        )
        paged_kv_cache.v_blocks[slots[:, None], :, positions, :] = (
            v_new[:, :, src_start:, :].transpose(1, 2)
        )

    def _gather_batch_kv(
        self,
        *,
        block_tables: torch.Tensor,
        max_seq_len: int,
        paged_kv_cache: PagedKVCache,
        key_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dense SDPA fallback: returns [batch, KV heads, tokens, dim]."""
        if key_positions is None:
            positions = torch.arange(max_seq_len, device=block_tables.device)
            ids = block_tables[:, positions // paged_kv_cache.block_size]
        else:
            # A short row can extend past its valid keys; the mask excludes
            # these lanes. Clamp the gather to the allocated table range.
            positions = key_positions.clamp(max=max_seq_len - 1)
            ids = block_tables.gather(1, positions // paged_kv_cache.block_size)
        ids = ids.clamp_min(0)
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
        key_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return an additive mask ``[B, 1, q_len, gathered_keys]``."""
        batch_size = seq_lens.shape[0]
        q_pos = kv_lens.unsqueeze(1) + torch.arange(
            q_len, device=device, dtype=torch.long
        ).unsqueeze(0)
        if key_positions is None:
            k_pos = torch.arange(max_seq_len, device=device).view(1, 1, -1)
        else:
            k_pos = key_positions.unsqueeze(1)
        q_pos = q_pos.unsqueeze(-1)

        blocked = k_pos > q_pos
        if self.sliding_window is not None:
            blocked = blocked | (k_pos < q_pos - self.sliding_window + 1)
        blocked = blocked | (k_pos >= seq_lens.view(batch_size, 1, 1))

        mask = torch.zeros(
            (batch_size, q_len, k_pos.shape[-1]), device=device, dtype=dtype
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
        rope_cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rope is None:
            return q, k
        if rope_cos_sin is not None:
            cos, sin = rope_cos_sin
        else:
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
        slots: Optional[torch.Tensor] = None,
        rope_cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        write_indices: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
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
        q, k = self._apply_rope(q, k, kv_lens, max_seq_len, x.dtype, rope_cos_sin=rope_cos_sin)

        if paged_kv_cache.sliding_window is not None:
            if slots is None:
                slots = torch.arange(batch_size, device=x.device)
            out = self._sliding_ring_attention(
                q, k, v, kv_lens=kv_lens, paged_kv_cache=paged_kv_cache,
                slots=slots, max_seq_len=max_seq_len,
            )
        else:
            self._append_to_paged_cache(
                k_new=k,
                v_new=v,
                block_tables=block_tables,
                kv_lens=kv_lens,
                paged_kv_cache=paged_kv_cache,
                write_indices=write_indices,
            )

            if (
                q_len >= 64
                and max_seq_len == q_len
                and (self.sliding_window is None or q_len <= self.sliding_window)
            ):
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=0.0,
                    is_causal=True, scale=self.scale, enable_gqa=True,
                )
            elif (
                self.use_fused_paged_attention
                and can_use_paged_decode(q)
                and max_seq_len >= 1024
            ):
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

    def _sliding_ring_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        kv_lens: torch.Tensor,
        paged_kv_cache: PagedKVCache,
        slots: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        """Attention over bounded per-slot sliding-window ring buffers."""
        batch_size, _, q_len, _ = q.shape
        W = paged_kv_cache.sliding_window
        seq_lens = kv_lens + q_len

        if q_len == 1:
            self._append_to_sliding_ring(
                k_new=k, v_new=v, kv_lens=kv_lens,
                paged_kv_cache=paged_kv_cache, slots=slots,
            )
            k_ring = paged_kv_cache.k_blocks[slots]
            v_ring = paged_kv_cache.v_blocks[slots]
            ring_idx = getattr(self, "_ring_idx", None)
            if ring_idx is None or ring_idx.device != q.device or ring_idx.shape[-1] != W:
                ring_idx = torch.arange(W, device=q.device).view(1, 1, W)
                self._ring_idx = ring_idx
            blocked = ring_idx >= seq_lens.view(batch_size, 1, 1)
            mask = torch.zeros((batch_size, 1, 1, W), device=q.device, dtype=q.dtype)
            mask = mask.masked_fill(blocked.unsqueeze(1), float("-inf"))
            return F.scaled_dot_product_attention(
                q, k_ring, v_ring, attn_mask=mask, dropout_p=0.0,
                scale=self.scale, enable_gqa=True,
            )

        # Prefill (q_len > 1): read up to W-1 prior tokens before overwriting ring.
        if max_seq_len > q_len:
            prev_width = min(max_seq_len - q_len, W - 1)
            prev_pos = kv_lens[:, None] - prev_width + torch.arange(prev_width, device=q.device)
            ring_idx = prev_pos.clamp_min(0) % W
            prev_k = paged_kv_cache.k_blocks[slots[:, None], :, ring_idx, :].transpose(1, 2)
            prev_v = paged_kv_cache.v_blocks[slots[:, None], :, ring_idx, :].transpose(1, 2)
            k_all = torch.cat([prev_k, k], dim=2)
            v_all = torch.cat([prev_v, v], dim=2)
            new_pos = kv_lens[:, None] + torch.arange(q_len, device=q.device)
            k_pos = torch.cat([prev_pos, new_pos], dim=1).unsqueeze(1)
        else:
            k_all, v_all = k, v
            k_pos = torch.arange(q_len, device=q.device).view(1, 1, q_len)

        self._append_to_sliding_ring(
            k_new=k, v_new=v, kv_lens=kv_lens,
            paged_kv_cache=paged_kv_cache, slots=slots,
        )

        if max_seq_len == q_len and q_len <= W and q_len >= 64:
            return F.scaled_dot_product_attention(
                q, k_all, v_all, attn_mask=None, dropout_p=0.0,
                is_causal=True, scale=self.scale, enable_gqa=True,
            )

        q_pos = (kv_lens.unsqueeze(1) + torch.arange(q_len, device=q.device).unsqueeze(0)).unsqueeze(-1)
        blocked = (k_pos > q_pos) | (k_pos < q_pos - W + 1) | (k_pos < 0) | (k_pos >= seq_lens.view(batch_size, 1, 1))
        mask = torch.zeros((batch_size, 1, q_len, k_all.shape[2]), device=q.device, dtype=q.dtype)
        mask = mask.masked_fill(blocked.unsqueeze(1), float("-inf"))
        return F.scaled_dot_product_attention(
            q, k_all, v_all, attn_mask=mask, dropout_p=0.0,
            scale=self.scale, enable_gqa=True,
        )

    def _dense_attention(
        self, q, block_tables, kv_lens, paged_kv_cache, max_seq_len
    ):
        """Portable prefill/decode fallback using dense gathered KV and SDPA."""
        q_len = q.shape[2]
        seq_lens = kv_lens + q_len
        key_positions = None
        if self.sliding_window is not None and max_seq_len > self.sliding_window:
            # Gather only the union of keys visible to this chunk. Each row
            # starts at its own window, so ragged decode remains bounded too.
            width = min(max_seq_len, self.sliding_window + q_len - 1)
            starts = (kv_lens - self.sliding_window + 1).clamp_min(0)
            key_positions = starts[:, None] + torch.arange(width, device=q.device)
        k_cache, v_cache = self._gather_batch_kv(
            block_tables=block_tables,
            max_seq_len=max_seq_len,
            paged_kv_cache=paged_kv_cache,
            key_positions=key_positions,
        )

        attn_mask = self._build_attn_mask(
            q_len=q_len,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            kv_lens=kv_lens,
            dtype=q.dtype,
            device=q.device,
            key_positions=key_positions,
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
