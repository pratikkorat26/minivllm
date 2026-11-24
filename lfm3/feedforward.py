import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from lfm3.kernels import triton as _triton, triton_swiglu


class FeedForward(nn.Module):
    """SwiGLU MLP with gate (w1) and up (w3) computed in one GEMM."""

    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int,
        *,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.gate_up_proj = nn.Linear(
            emb_dim, 2 * hidden_dim, bias=False, dtype=dtype
        )
        self.w2 = nn.Linear(hidden_dim, emb_dim, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        if _triton is not None and x.is_cuda and not torch.compiler.is_compiling():
            return self.w2(triton_swiglu(gate_up))
        gate, up = gate_up.chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)
