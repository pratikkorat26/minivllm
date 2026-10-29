"""Optional Triton decode attention over physical KV pages.

Programs handle one request/query head and optionally one context partition.
Online softmax keeps temporary storage bounded to a tile; split-KV uses a second
kernel to merge FP32 partial numerators, denominators and softmax maxima.
Prefill and platforms without Triton continue to use PyTorch SDPA.
"""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU and native Windows installations need no Triton.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _decode_attention(
        Q, K, V, Tables, Lengths, Out, Maxima, Denominators,
        Q_BATCH: tl.constexpr, Q_HEAD: tl.constexpr,
        TABLE_WIDTH: tl.constexpr,
        HEADS: tl.constexpr, KV_HEADS: tl.constexpr,
        DIM: tl.constexpr, PAGE: tl.constexpr,
        SCALE: tl.constexpr, WINDOW: tl.constexpr,
        TILE: tl.constexpr, PAD_DIM: tl.constexpr,
        SPLITS: tl.constexpr, PARTITION_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        head = tl.program_id(1)
        partition = tl.program_id(2)
        kv_head = head // (HEADS // KV_HEADS)
        length = tl.load(Lengths + row) + 1  # Includes the newly appended token.
        start = tl.maximum(0, length - WINDOW) if WINDOW > 0 else 0
        if SPLITS > 1:
            start += partition * PARTITION_SIZE
            end = tl.minimum(start + PARTITION_SIZE, length)
        else:
            end = length
        dims = tl.arange(0, PAD_DIM)
        q = tl.load(
            Q + row * Q_BATCH + head * Q_HEAD + dims, dims < DIM, 0
        ).to(tl.float32)
        maximum = tl.full((), -float("inf"), tl.float32)
        denominator = tl.full((), 0, tl.float32)
        numerator = tl.full((PAD_DIM,), 0, tl.float32)

        for tile_start in range(start, end, TILE):
            positions = tile_start + tl.arange(0, TILE)
            valid = positions < end
            pages = tl.load(
                Tables + row * TABLE_WIDTH + positions // PAGE, valid, 0
            )
            offsets = (
                (pages * KV_HEADS + kv_head) * PAGE + positions % PAGE
            )[:, None] * DIM + dims[None, :]
            mask = valid[:, None] & (dims[None, :] < DIM)
            keys = tl.load(K + offsets, mask, 0).to(tl.float32)
            scores = tl.sum(keys * q[None, :], axis=1) * SCALE
            scores = tl.where(valid, scores, -float("inf"))
            next_maximum = tl.maximum(maximum, tl.max(scores, axis=0))
            correction = tl.exp(maximum - next_maximum)
            probabilities = tl.exp(scores - next_maximum)
            values = tl.load(V + offsets, mask, 0).to(tl.float32)
            numerator = numerator * correction + tl.sum(
                probabilities[:, None] * values, axis=0
            )
            denominator = denominator * correction + tl.sum(probabilities, axis=0)
            maximum = next_maximum

        if SPLITS > 1:
            partial = (row * HEADS + head) * SPLITS + partition
            tl.store(Out + partial * DIM + dims, numerator, dims < DIM)
            tl.store(Maxima + partial, maximum)
            tl.store(Denominators + partial, denominator)
        else:
            tl.store(
                Out + (row * HEADS + head) * DIM + dims,
                numerator / denominator,
                dims < DIM,
            )

    @triton.jit
    def _merge_partitions(
        Numerators, Maxima, Denominators, Out,
        HEADS: tl.constexpr, DIM: tl.constexpr, SPLITS: tl.constexpr,
        PAD_DIM: tl.constexpr, PAD_SPLITS: tl.constexpr,
    ):
        row = tl.program_id(0)
        head = tl.program_id(1)
        partitions = tl.arange(0, PAD_SPLITS)
        dims = tl.arange(0, PAD_DIM)
        base = (row * HEADS + head) * SPLITS
        maxima = tl.load(Maxima + base + partitions, partitions < SPLITS,
                         -float("inf"))
        denominators = tl.load(Denominators + base + partitions,
                               partitions < SPLITS, 0)
        # Empty partitions (common for short ragged rows) contribute zero.
        maximum = tl.max(maxima, axis=0)
        corrections = tl.where(denominators > 0, tl.exp(maxima - maximum), 0)
        numerators = tl.load(
            Numerators + (base + partitions[:, None]) * DIM + dims[None, :],
            (partitions[:, None] < SPLITS) & (dims[None, :] < DIM), 0,
        )
        numerator = tl.sum(numerators * corrections[:, None], axis=0)
        denominator = tl.sum(denominators * corrections, axis=0)
        tl.store(Out + (row * HEADS + head) * DIM + dims,
                 numerator / denominator, dims < DIM)


def can_use_paged_decode(q: torch.Tensor) -> bool:
    return (
        triton is not None
        and q.is_cuda
        and q.shape[2] == 1
        and q.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and q.stride(-1) == 1
        and q.shape[-1] <= 256
    )


def paged_decode_attention(
    q: torch.Tensor,
    *,
    cache,
    block_tables: torch.Tensor,
    kv_lens: torch.Tensor,
    scale: float,
    sliding_window: int | None,
    max_seq_len: int | None = None,
    num_splits: int | None = None,
) -> torch.Tensor:
    """Read already-appended contiguous KV pages; return ``[B, H, 1, D]``.

    The caller owns validation/allocation: lengths are nonnegative, table entries
    for live tokens are valid, and physical cache pages are uniquely owned for
    writes. Padded decode lanes must use their own scratch pages.
    """
    batch, heads, _, dim = q.shape
    context_bound = max_seq_len or block_tables.shape[1] * cache.block_size
    if sliding_window is not None:
        context_bound = min(context_bound, sliding_window)
    if num_splits is None:
        # Short windows do not amortize a second launch. Small batches benefit
        # from more partitions to expose enough parallel work on long contexts.
        tokens_per_partition = 512 if batch * heads < 128 else 1024
        num_splits = (
            min(16, triton.cdiv(context_bound, tokens_per_partition))
            if context_bound > 1024
            else 1
        )
    if not 1 <= num_splits <= 32:
        raise ValueError("num_splits must be between 1 and 32")
    partition_size = triton.cdiv(context_bound, num_splits * 64) * 64
    out = torch.empty((batch, heads, 1, dim), device=q.device, dtype=q.dtype)
    partials = maxima = denominators = None
    if num_splits > 1:
        partials = torch.empty((batch, heads, num_splits, dim), device=q.device,
                               dtype=torch.float32)
        maxima = torch.empty((batch, heads, num_splits), device=q.device,
                             dtype=torch.float32)
        denominators = torch.empty_like(maxima)
    _decode_attention[(batch, heads, num_splits)](
        q, cache.k_blocks, cache.v_blocks, block_tables, kv_lens,
        partials if num_splits > 1 else out, maxima, denominators,
        Q_BATCH=q.stride(0), Q_HEAD=q.stride(1),
        TABLE_WIDTH=block_tables.shape[1], HEADS=heads,
        KV_HEADS=cache.k_blocks.shape[1], DIM=dim, PAGE=cache.block_size,
        SCALE=scale, WINDOW=sliding_window or 0,
        TILE=64, PAD_DIM=triton.next_power_of_2(dim),
        SPLITS=num_splits, PARTITION_SIZE=partition_size,
    )
    if num_splits > 1:
        _merge_partitions[(batch, heads)](
            partials, maxima, denominators, out,
            HEADS=heads, DIM=dim, SPLITS=num_splits,
            PAD_DIM=triton.next_power_of_2(dim),
            PAD_SPLITS=triton.next_power_of_2(num_splits),
        )
    return out
