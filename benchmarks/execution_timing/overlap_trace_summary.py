"""Quantify host stream-emission spans that overlap actual GPU kernels."""

from collections import Counter
import json


def summarize_overlap(path):
    events = json.loads(path.read_text())["traceEvents"]
    kernels = [
        (e["ts"], e["ts"] + e["dur"])
        for e in events
        if e.get("ph") == "X" and e.get("cat") == "kernel"
    ]
    flushes = [
        e
        for e in events
        if e.get("ph") == "X" and e.get("name") == "scheduler.stream_flush"
    ]
    merged = []
    for start, end in sorted(kernels):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    overlap_us = sum(
        max(0, min(e["ts"] + e["dur"], end) - max(e["ts"], start))
        for e in flushes
        for start, end in merged
    )
    flush_us = sum(e["dur"] for e in flushes)
    copies = Counter(
        e["name"] for e in events if e.get("ph") == "X" and e.get("cat") == "gpu_memcpy"
    )
    return dict(
        trace_path=str(path),
        cpu_stream_flushes=len(flushes),
        cpu_stream_flush_ms=flush_us / 1000,
        cpu_stream_flush_overlapping_gpu_kernels_ms=overlap_us / 1000,
        overlap_fraction=overlap_us / flush_us if flush_us else None,
        gpu_kernel_union_ms=sum(end - start for start, end in merged) / 1000,
        copies=dict(copies),
        caveat="Profiler perturbs execution. Kernel intervals are merged; overlap duration is not throughput gain.",
    )
