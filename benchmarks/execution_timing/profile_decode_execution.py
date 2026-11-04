"""Profile fixed-work decode, including executor metadata and sampling.

EOS is disabled only in these bounded microbenchmarks so every variant executes
identical numbers of steps. These rates are not end-to-end serving benchmarks.
"""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/execution_timing"
sys.path.insert(0, str(ROOT))


def summarize_trace(path):
    trace = json.loads(path.read_text())
    events = trace["traceEvents"]
    gpu = [
        e
        for e in events
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    ]
    kernels = [e for e in gpu if e.get("cat") == "kernel"]
    copies = [e for e in gpu if e.get("cat") == "gpu_memcpy"]
    categories = Counter(e.get("cat") for e in events if e.get("ph") == "X")
    intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in gpu)
    busy = 0.0
    last = None
    gaps = []
    for start, end in intervals:
        if last is None:
            busy += end - start
            last = end
        elif start > last:
            gaps.append(start - last)
            busy += end - start
            last = end
        elif end > last:
            busy += end - last
            last = end
    span = (
        (max(e for _, e in intervals) - min(s for s, _ in intervals))
        if intervals
        else 0
    )
    overlap = 0.0
    for copy in copies:
        a, b = copy["ts"], copy["ts"] + copy["dur"]
        # Kernels in this engine are serialized. Clip each intersection.
        overlap += sum(
            max(0, min(b, k["ts"] + k["dur"]) - max(a, k["ts"])) for k in kernels
        )
    cpu = Counter()
    counts = Counter()
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver"):
            cpu[e["name"]] += e["dur"]
            counts[e["name"]] += 1
    return dict(
        categories=dict(categories),
        gpu_activity_span_ms=span / 1000,
        gpu_busy_ms=busy / 1000,
        gpu_idle_gap_ms=sum(gaps) / 1000,
        gpu_busy_fraction=busy / span if span else None,
        gpu_kernel_ms=sum(e["dur"] for e in kernels) / 1000,
        gpu_copy_ms=sum(e["dur"] for e in copies) / 1000,
        copy_kernel_overlap_ms=overlap / 1000,
        copies=[
            dict(name=e["name"], duration_us=e["dur"], args=e.get("args", {}))
            for e in copies
        ],
        runtime_calls=[
            dict(name=name, total_ms=duration / 1000, calls=counts[name])
            for name, duration in cpu.most_common(16)
        ],
        largest_gpu_gaps_us=sorted(gaps, reverse=True)[:20],
    )


def run(args):
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function
    from inference import LLMEngine, EngineConfig, SamplingConfig
    from runtime import Lfm3Runtime
    from lfm3.rope import RotaryEmbedding

    if args.rope_cache == "baseline":
        # Use the unchanged committed implementation, with the same runtime,
        # weights and executor, to isolate the cache fix in paired measurements.
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        baseline_path = args.trace_dir / "baseline_rope.py"
        baseline_path.write_bytes(
            subprocess.check_output(["git", "show", "cb3761f:lfm3/rope.py"], cwd=ROOT)
        )
        spec = importlib.util.spec_from_file_location(
            "execution_timing_baseline_rope", baseline_path
        )
        baseline_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline_module
        spec.loader.exec_module(baseline_module)

        def baseline_ensure(self, seq_len, device=None, *, restore_cache_dtype=False):
            return baseline_module.RotaryEmbedding.ensure_capacity(
                self, seq_len, device
            )

        RotaryEmbedding.ensure_capacity = baseline_ensure

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    sampling = SamplingConfig(temperature=0, top_p=1, top_k=0, repetition_penalty=1)
    runtime = Lfm3Runtime(device=torch.device("cuda"), compile_model=True)
    budget = max(16384, args.batch_size * 128)
    engine = LLMEngine(
        runtime,
        EngineConfig(
            use_instruct_model=True,
            max_new_tokens=96,
            max_batch_tokens=2048,
            max_kv_cache_tokens=budget,
            max_concurrent_requests=args.batch_size,
            decode_batch_size=args.batch_size,
            sampling=sampling,
        ),
    )
    executor = engine.executor
    prompt = "The purpose of a KV cache is"
    if args.context == "long":
        prompt = json.loads(
            (ROOT / "benchmarks/software_phases_long_budget2048.json").read_text()
        )["config"]["prompt"]

    def prepare(label):
        requests = []
        for i in range(args.batch_size):
            request = executor.build_request(
                f"{label}-{i}", prompt, sampling, 96, time.perf_counter()
            )
            request.eos_token_id = None
            executor.initialize(request)
            requests.append(request)
        pending = list(requests)
        while pending:
            length = int(pending[0].current_input.shape[1])
            count = max(1, 2048 // length)
            selected = pending[:count]
            pending = pending[count:]
            executor.prefill_batch(selected, 2048)
            pending.extend(
                r for r in selected if r.prompt_cursor < len(r.prompt_token_ids)
            )
        return requests

    def release(requests):
        for request in requests:
            executor.finish(request, "max_new_tokens")

    warm = prepare("warm")
    for _ in range(66):
        executor.decode(warm)
    torch.cuda.synchronize()
    release(warm)
    timings = []
    hashes = []
    for repeat in range(args.repeats):
        requests = prepare(f"measured-{repeat}")
        torch.cuda.synchronize()
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        began = time.perf_counter()
        for _ in range(args.steps):
            executor.decode(requests)
        end.record()
        end.synchronize()
        wall = time.perf_counter() - began
        timings.append(
            dict(
                wall_s=wall,
                device_envelope_s=start.elapsed_time(end) / 1000,
                tokens=args.steps * args.batch_size,
                tokens_per_s=args.steps * args.batch_size / wall,
            )
        )
        hashes.append(
            hashlib.sha256(
                json.dumps([r.generated_ids for r in requests]).encode()
            ).hexdigest()
        )
        release(requests)
    requests = prepare("trace")
    torch.cuda.synchronize()
    original_forward = executor._forward
    original_sample = executor._sample

    def forward(*a, **kw):
        with record_function("executor.forward_including_metadata"):
            return original_forward(*a, **kw)

    def sample(*a, **kw):
        with record_function("executor.sampling"):
            return original_sample(*a, **kw)

    executor._forward = forward
    executor._sample = sample
    label = f"{args.rope_cache}_b{args.batch_size}_{args.context}"
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.trace_dir / f"{label}_trace.json"
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
    ) as prof:
        for _ in range(args.trace_steps):
            with record_function("executor.decode_step"):
                executor.decode(requests)
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace_path))
    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25)
    (OUT / f"{label}_operators.txt").write_text(
        "\n".join(line.rstrip() for line in table.splitlines()).rstrip() + "\n"
    )
    report = dict(
        kind="fixed_work_executor_decode_profile",
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        batch_size=args.batch_size,
        context=args.context,
        prompt_tokens=len(requests[0].prompt_token_ids),
        steps=args.steps,
        trace_steps=args.trace_steps,
        repeats=args.repeats,
        eos_disabled_for_fixed_work=True,
        timings=timings,
        median_tokens_per_s=statistics.median(r["tokens_per_s"] for r in timings),
        identical_token_hashes=len(set(hashes)) == 1,
        token_hashes=hashes,
        rope_cache_variant=args.rope_cache,
        trace_path=str(trace_path),
        trace=summarize_trace(trace_path),
    )
    (OUT / f"{label}_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "trace"}), flush=True)
    print(
        json.dumps(
            {
                k: v
                for k, v in report["trace"].items()
                if k not in ("copies", "runtime_calls")
            },
            indent=2,
        ),
        flush=True,
    )
    print(table, flush=True)
    release(requests)
    engine.scheduler._worker.shutdown(wait=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context", choices=("short", "long"), default="short")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--trace-steps", type=int, default=16)
    parser.add_argument(
        "--rope-cache", choices=("baseline", "candidate"), default="candidate"
    )
    parser.add_argument(
        "--trace-dir", type=Path, default=Path("/tmp/minivllm_execution_traces")
    )
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    run(args)
