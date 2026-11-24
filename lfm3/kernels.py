"""Fused Triton kernels for LFM3-1.7B eager & quantized execution."""

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU fallback
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _rms_norm_kernel(
        X,
        W,
        Out,
        N: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X + row * N + cols, mask=mask, other=0.0)
        x_f = x.to(tl.float32)
        var = tl.sum(x_f * x_f, axis=0) / N
        rstd = tl.rsqrt(var + EPS)
        x_norm = (x_f * rstd).to(x.dtype)
        w = tl.load(W + cols, mask=mask, other=0.0).to(x.dtype)
        tl.store(Out + row * N + cols, x_norm * w, mask=mask)

    @triton.jit
    def _swiglu_kernel(
        GateUp,
        Out,
        H: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        col_block = tl.program_id(1)
        cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = cols < H
        base = row * (2 * H)
        gate = tl.load(GateUp + base + cols, mask=mask, other=0.0)
        up = tl.load(GateUp + base + H + cols, mask=mask, other=0.0)
        g_f = gate.to(tl.float32)
        silu_g = (g_f * tl.sigmoid(g_f)).to(gate.dtype)
        tl.store(Out + row * H + cols, silu_g * up, mask=mask)

    @triton.jit
    def _short_conv_decode_kernel(
        BCX,
        State,
        Slots,
        W,
        Bias,
        Out,
        H: tl.constexpr,
        KERNEL_SIZE: tl.constexpr,
        STATE_LEN: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        row = tl.program_id(0)
        ch_block = tl.program_id(1)
        cols = ch_block * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = cols < H

        slot = tl.load(Slots + row)
        bcx_base = row * (3 * H)
        b = tl.load(BCX + bcx_base + cols, mask=mask, other=0.0)
        c = tl.load(BCX + bcx_base + H + cols, mask=mask, other=0.0)
        x_in = tl.load(BCX + bcx_base + 2 * H + cols, mask=mask, other=0.0)
        gated = (b.to(tl.float32) * x_in.to(tl.float32)).to(b.dtype)

        state_base = (slot * H + cols) * STATE_LEN
        w_base = cols * KERNEL_SIZE
        acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

        for k in tl.static_range(STATE_LEN):
            s_k = tl.load(State + state_base + k, mask=mask, other=0.0)
            w_k = tl.load(W + w_base + k, mask=mask, other=0.0)
            acc += s_k.to(tl.float32) * w_k.to(tl.float32)
            if k > 0:
                tl.store(State + state_base + (k - 1), s_k, mask=mask)

        tl.store(State + state_base + (STATE_LEN - 1), gated, mask=mask)
        w_last = tl.load(W + w_base + STATE_LEN, mask=mask, other=0.0)
        acc += gated.to(tl.float32) * w_last.to(tl.float32)
        if HAS_BIAS:
            acc += tl.load(Bias + cols, mask=mask, other=0.0).to(tl.float32)

        y = acc.to(b.dtype) * c
        tl.store(Out + row * H + cols, y, mask=mask)

    @triton.jit
    def _w4a16_gemv_kernel(
        X,
        QWeight,
        Scales,
        Zeros,
        Out,
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        n_block = tl.program_id(1)
        n_offs = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N

        HALF_G: tl.constexpr = GROUP_SIZE // 2
        K_HALF: tl.constexpr = K // 2
        NUM_GROUPS: tl.constexpr = K // GROUP_SIZE

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        g_offs = tl.arange(0, HALF_G)

        for g in range(NUM_GROUPS):
            s = tl.load(Scales + n_offs * NUM_GROUPS + g, mask=n_mask, other=0.0).to(tl.float32)
            z = tl.load(Zeros + n_offs * NUM_GROUPS + g, mask=n_mask, other=0.0).to(tl.float32)

            qw_ptrs = QWeight + n_offs[:, None] * K_HALF + g * HALF_G + g_offs[None, :]
            qw = tl.load(qw_ptrs, mask=n_mask[:, None], other=0)
            lo = (qw & 0x0F).to(tl.float32)
            hi = (qw >> 4).to(tl.float32)

            x_base = X + row * K + g * GROUP_SIZE
            x_even = tl.load(x_base + g_offs * 2).to(tl.float32)
            x_odd = tl.load(x_base + g_offs * 2 + 1).to(tl.float32)

            w_sum_x = tl.sum(lo * x_even[None, :] + hi * x_odd[None, :], axis=1)
            x_sum = tl.sum(x_even + x_odd, axis=0)
            acc += s * w_sum_x + z * x_sum

        tl.store(Out + row * N + n_offs, acc, mask=n_mask)


def triton_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    orig_shape = x.shape
    N = orig_shape[-1]
    x_2d = x.reshape(-1, N)
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    M = x_2d.shape[0]
    out = torch.empty_like(x_2d)
    BLOCK_N = triton.next_power_of_2(N)
    _rms_norm_kernel[(M,)](x_2d, weight, out, N=N, EPS=eps, BLOCK_N=BLOCK_N)
    return out.reshape(orig_shape)


def triton_swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    orig_shape = gate_up.shape
    H = orig_shape[-1] // 2
    x_2d = gate_up.reshape(-1, 2 * H)
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    M = x_2d.shape[0]
    out = torch.empty((M, H), device=gate_up.device, dtype=gate_up.dtype)
    BLOCK_N = 1024
    grid = (M, triton.cdiv(H, BLOCK_N))
    _swiglu_kernel[grid](x_2d, out, H=H, BLOCK_N=BLOCK_N)
    return out.reshape(*orig_shape[:-1], H)


def triton_short_conv_decode(
    bcx: torch.Tensor,
    conv_state: torch.Tensor,
    slots: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    B = bcx.shape[0]
    H = bcx.shape[-1] // 3
    kernel_size = weight.shape[-1]
    state_len = kernel_size - 1
    if not bcx.is_contiguous():
        bcx = bcx.contiguous()
    out = torch.empty((B, H), device=bcx.device, dtype=bcx.dtype)
    BLOCK_C = 128
    grid = (B, triton.cdiv(H, BLOCK_C))
    _short_conv_decode_kernel[grid](
        bcx,
        conv_state,
        slots,
        weight,
        bias if bias is not None else bcx,
        out,
        H=H,
        KERNEL_SIZE=kernel_size,
        STATE_LEN=state_len,
        HAS_BIAS=bias is not None,
        BLOCK_C=BLOCK_C,
    )
    return out.unsqueeze(1)


def triton_w4a16_gemv(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    M, K = x.shape
    N = qweight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    if N <= 2048:
        BLOCK_N, num_warps = 16, 4
    else:
        BLOCK_N, num_warps = 32, 2
    grid = (M, triton.cdiv(N, BLOCK_N))
    _w4a16_gemv_kernel[grid](
        x,
        qweight,
        scales,
        zeros,
        out,
        M,
        N=N,
        K=K,
        GROUP_SIZE=group_size,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    return out


if triton is not None:

    @triton.jit
    def _w8a16_gemv_kernel(
        X,
        QWeight,
        Scales,
        Out,
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        GROUPS_PER_ITER: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        n_block = tl.program_id(1)
        n_offs = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N

        NUM_GROUPS: tl.constexpr = K // GROUP_SIZE
        STEP_K: tl.constexpr = GROUP_SIZE * GROUPS_PER_ITER
        k_offs = tl.arange(0, STEP_K)
        g_sub = tl.arange(0, GROUPS_PER_ITER)
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for g0 in range(0, NUM_GROUPS, GROUPS_PER_ITER):
            s = tl.load(
                Scales + n_offs[:, None] * NUM_GROUPS + (g0 + g_sub[None, :]),
                mask=n_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            x_k = tl.load(X + row * K + g0 * GROUP_SIZE + k_offs).to(tl.float32)
            qw = tl.load(
                QWeight + n_offs[:, None] * K + g0 * GROUP_SIZE + k_offs[None, :],
                mask=n_mask[:, None],
                other=0,
            ).to(tl.float32)
            prod = tl.reshape(qw * x_k[None, :], (BLOCK_N, GROUPS_PER_ITER, GROUP_SIZE))
            acc += tl.sum(s * tl.sum(prod, axis=2), axis=1)

        tl.store(Out + row * N + n_offs, acc, mask=n_mask)


def triton_w8a16_gemv(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    M, K = x.shape
    N = qweight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    num_groups = K // group_size
    groups_per_iter = 4 if (group_size == 32 and num_groups % 4 == 0) else 1
    BLOCK_N = 16 if N <= 2048 else 32
    num_warps = 4 if N <= 2048 else 2
    grid = (M, triton.cdiv(N, BLOCK_N))
    _w8a16_gemv_kernel[grid](
        x,
        qweight,
        scales,
        out,
        M,
        N=N,
        K=K,
        GROUP_SIZE=group_size,
        GROUPS_PER_ITER=groups_per_iter,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    return out
