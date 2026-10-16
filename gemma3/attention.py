import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from gemma3.rope import rotate_half
from gemma3.paged_kv import PagedKVCache


class RMSNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()
        var = torch.mean(x_f * x_f, dim=-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(var + self.eps)
        out = x_norm * (1.0 + self.scale.float())
        if self.shift is not None:
            out = out + self.shift
        return out.to(orig_dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_groups: int,
        head_dim: int,
        rope=None,
        sliding_window: Optional[int] = None,
        qk_norm: bool = False,
        query_pre_attn_scalar: Optional[float] = None,
        dtype=None,
    ):
        super().__init__()
        if num_heads % num_kv_groups != 0:
            raise ValueError("num_heads must be divisible by num_kv_groups")

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        self.head_dim = head_dim
        scale_base = (
            float(query_pre_attn_scalar)
            if query_pre_attn_scalar is not None
            else float(head_dim)
        )
        self.scale = scale_base**-0.5
        self.sliding_window = sliding_window
        self.rope = rope

        self.q_proj = nn.Linear(d_model, num_heads * head_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(
            d_model, num_kv_groups * head_dim, bias=False, dtype=dtype
        )
        self.v_proj = nn.Linear(
            d_model, num_kv_groups * head_dim, bias=False, dtype=dtype
        )
        self.out_proj = nn.Linear(
            num_heads * head_dim, d_model, bias=False, dtype=dtype
        )

        self.q_norm = RMSNorm(head_dim) if qk_norm else None
        self.k_norm = RMSNorm(head_dim) if qk_norm else None

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
        # Index physical pages directly; viewing this as head-major is invalid.
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
        seq_lens: torch.Tensor,
        max_seq_len: int,
        paged_kv_cache: PagedKVCache,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized dense SDPA fallback: returns [batch, KV heads, tokens, dim].

        Invalid right padding reads page zero, then the attention mask excludes it.
        Live block tables are validated by the CPU executor, outside layer loops.
        """
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
        """Return additive mask ``[B, 1, q_len, max_seq_len]`` (0 keep, -inf block)."""
        batch_size = seq_lens.shape[0]
        q_pos = kv_lens.unsqueeze(1) + torch.arange(
            q_len, device=device, dtype=torch.long
        ).unsqueeze(0)
        k_pos = torch.arange(max_seq_len, device=device, dtype=torch.long).view(
            1, 1, max_seq_len
        )
        q_pos = q_pos.unsqueeze(-1)  # [B, q_len, 1]

        blocked = k_pos > q_pos
        if self.sliding_window is not None:
            blocked = blocked | (k_pos < q_pos - self.sliding_window + 1)
        # Pad keys beyond each sequence's true length.
        blocked = blocked | (k_pos >= seq_lens.view(batch_size, 1, 1))

        mask = torch.zeros((batch_size, q_len, max_seq_len), device=device, dtype=dtype)
        mask = mask.masked_fill(blocked, float("-inf"))
        return mask.unsqueeze(1)  # [B, 1, q_len, S]

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        kv_lens: torch.Tensor,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rope is None:
            return q, k
        positions = kv_lens[:, None] + torch.arange(q.shape[2], device=q.device)
        cos = self.rope.cos_cached[positions].to(dtype=dtype).unsqueeze(1)
        sin = self.rope.sin_cached[positions].to(dtype=dtype).unsqueeze(1)
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin

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
        device = x.device

        q = (
            self.q_proj(x)
            .view(batch_size, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, q_len, self.num_kv_groups, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, q_len, self.num_kv_groups, self.head_dim)
            .transpose(1, 2)
        )

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k = self._apply_rope(q, k, kv_lens, x.dtype)

        self._append_to_paged_cache(
            k_new=k,
            v_new=v,
            block_tables=block_tables,
            kv_lens=kv_lens,
            paged_kv_cache=paged_kv_cache,
        )

        seq_lens = kv_lens + q_len
        if max_seq_len is None:
            max_seq_len = int(seq_lens.max().item())
        k_cache, v_cache = self._gather_batch_kv(
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            paged_kv_cache=paged_kv_cache,
        )

        attn_mask = self._build_attn_mask(
            q_len=q_len,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            kv_lens=kv_lens,
            dtype=q.dtype,
            device=device,
        )
        out = F.scaled_dot_product_attention(
            q,
            k_cache,
            v_cache,
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=self.scale,
            enable_gqa=True,
        )
        out = out.transpose(1, 2).reshape(
            batch_size, q_len, self.num_heads * self.head_dim
        )
        return self.out_proj(out)
