import torch
import torch.nn as nn
from typing import Optional


class ShortConv(nn.Module):
    """LFM3 gated short convolution.

    The recurrent state is the last ``kernel_size - 1`` gated inputs
    (``B * x``) for every channel. It is carried across prefill chunks and
    decode steps, so no attention KV cache exists for these layers.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 7,
        *,
        bias: bool = False,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.bias = bias
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=0,
            dtype=dtype,
        )
        self.in_proj = nn.Linear(
            hidden_size, 3 * hidden_size, bias=bias, dtype=dtype
        )
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        *,
        conv_state: Optional[torch.Tensor] = None,
        slots: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        bcx = self.in_proj(x).transpose(1, 2)
        gate_b, gate_c, gated_x = bcx.chunk(3, dim=1)
        gated = gate_b * gated_x

        state_len = self.kernel_size - 1
        if conv_state is None:
            state = gated.new_zeros(batch_size, self.hidden_size, state_len)
        else:
            if slots is None:
                slots = torch.arange(batch_size, device=x.device)
            state = conv_state[slots]

        conv_input = torch.cat([state, gated], dim=-1)
        if seq_len == 1 and torch.compiler.is_compiling():
            # One output per channel is a short dot product. Expressing
            # it directly lets torch.compile fuse this with the surrounding
            # gates instead of calling a general depthwise convolution.
            conv_out = (
                conv_input.float() * self.conv.weight[:, 0, :].float()
            ).sum(dim=-1, keepdim=True)
            if self.conv.bias is not None:
                conv_out = conv_out + self.conv.bias.float()[None, :, None]
            conv_out = conv_out.to(gated.dtype)
        else:
            conv_out = self.conv(conv_input)
        if conv_state is not None:
            conv_state[slots] = conv_input[..., -state_len:]

        y = gate_c * conv_out
        return self.out_proj(y.transpose(1, 2))
