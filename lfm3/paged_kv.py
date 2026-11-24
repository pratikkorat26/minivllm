from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class PagedKVCache:
    k_blocks: torch.Tensor
    v_blocks: torch.Tensor
    block_size: int
    sliding_window: Optional[int] = None

    @classmethod
    def empty(
        cls,
        *,
        num_blocks: int,
        num_kv_groups: int,
        block_size: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "PagedKVCache":
        k = torch.zeros((num_blocks, num_kv_groups, block_size, head_dim), device=device, dtype=dtype)
        v = torch.zeros((num_blocks, num_kv_groups, block_size, head_dim), device=device, dtype=dtype)
        return cls(k_blocks=k, v_blocks=v, block_size=block_size)

    @classmethod
    def empty_sliding(
        cls,
        *,
        max_slots: int,
        num_kv_groups: int,
        window_size: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "PagedKVCache":
        """Allocate bounded per-slot ring buffers for sliding-window attention."""
        k = torch.zeros((max_slots, num_kv_groups, window_size, head_dim), device=device, dtype=dtype)
        v = torch.zeros((max_slots, num_kv_groups, window_size, head_dim), device=device, dtype=dtype)
        return cls(k_blocks=k, v_blocks=v, block_size=window_size, sliding_window=window_size)
