# Rotary position embeddings for LFM3.
#
# LFM3 uses per-layer-type RoPE. Full-attention layers rotate only the
# first `partial_rotary_factor * head_dim` channels, with the remaining
# inverse frequencies set to zero (the "zero-tail" partial rotation used by
# the reference implementation). Sliding-attention layers rotate the whole
# head. The cache grows lazily because the advertised context is 262K.

import torch
import torch.nn as nn
from typing import Optional, Tuple


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # x: [..., D]
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_single(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply RoPE to `x` of shape [B, H, T, D] with cos/sin [T, D]."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    """Lazily grown RoPE cache with optional zero-tail partial rotation."""

    def __init__(
        self,
        dim: int,
        theta: float = 10000.0,
        max_position_embeddings: int = 2048,
        partial_rotary_factor: float = 1.0,
        cache_dtype: torch.dtype = torch.float32,
        growth_factor: float = 1.25,
        initial_cache_len: int = 2048,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE dim must be even")
        if not 0.0 < partial_rotary_factor <= 1.0:
            raise ValueError("partial_rotary_factor must be in (0, 1]")

        self.dim = dim
        self.theta = float(theta)
        self.max_position_embeddings = int(max_position_embeddings)
        self.partial_rotary_factor = float(partial_rotary_factor)
        self.cache_dtype = cache_dtype
        self.growth_factor = float(growth_factor)

        # Match the reference: frequencies follow the *full* head-dimension
        # progression, then the unrotated tail is zeroed out.
        rope_angles = int(self.partial_rotary_factor * dim) // 2
        inv_freq = 1.0 / (
            self.theta
            ** (
                torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32)
                / float(dim)
            )
        )
        if inv_freq.numel() < dim // 2:
            inv_freq = torch.cat(
                [
                    inv_freq,
                    torch.zeros(dim // 2 - inv_freq.numel(), dtype=torch.float32),
                ]
            )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.register_buffer(
            "cos_cached", torch.empty(0, dtype=self.cache_dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", torch.empty(0, dtype=self.cache_dtype), persistent=False
        )
        self.max_seq_len_cached = 0
        self._build_cache(
            max(1, min(initial_cache_len, self.max_position_embeddings)),
            device=self.inv_freq.device,
        )

    @torch.no_grad()
    def _build_cache(
        self, seq_len: int, device: Optional[torch.device] = None
    ) -> None:
        if device is None:
            device = self.inv_freq.device
        inv = self.inv_freq.to(device=device)
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv)
        angles = torch.cat([freqs, freqs], dim=-1)
        self.cos_cached = angles.cos().to(dtype=self.cache_dtype)
        self.sin_cached = angles.sin().to(dtype=self.cache_dtype)
        self.max_seq_len_cached = seq_len

    @torch.no_grad()
    def ensure_capacity(
        self, seq_len: int, device: Optional[torch.device] = None
    ) -> None:
        if device is None:
            device = self.inv_freq.device
        if self.cos_cached.device != device:
            self._build_cache(max(seq_len, self.max_seq_len_cached), device=device)
        if seq_len > self.max_seq_len_cached:
            new_len = max(seq_len, int(self.max_seq_len_cached * self.growth_factor) + 1)
            self._build_cache(new_len, device=device)

    def get_cos_sin(
        self,
        seq_len: int,
        offset: int = 0,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        if offset < 0:
            raise ValueError("offset must be >= 0")
        self.ensure_capacity(offset + seq_len, device=device)
        cos = self.cos_cached[offset : offset + seq_len].to(dtype=dtype)
        sin = self.sin_cached[offset : offset + seq_len].to(dtype=dtype)
        return cos, sin
