from dataclasses import dataclass

import torch


@dataclass
class PagedKVCache:
    k_blocks: torch.Tensor
    v_blocks: torch.Tensor
    block_size: int

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
