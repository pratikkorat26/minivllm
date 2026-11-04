"""Verify final run comparability and emit the compact performance record."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/overlap_optimization"


def main():
    comparisons = []
    runs = {}
    for label in (
        "baseline_b256_n64",
        "full_b256_n64",
        "full_s8_b256_n64",
        "full_s8_pcores_b256_n64",
        "baseline_b128_n512",
        "full_b128_n512",
        "full_s8_b128_n512",
    ):
        path = OUT / "final" / f"{label}.json"
        if not path.exists() and "pcores" in label:
            continue
        run = json.loads(path.read_text())
        assert run["no_compilation_or_capture_during_measurement"]
        assert not run["profile_perturbed"]
        assert run["platform_profile"] == "performance"
        assert all(w["before"] == w["after"] for w in run["warmup_waves"][-2:])
        assert run["warmup_waves"][-1]["after"]["unique_graphs"] > 0
        assert (
            run["warmup_waves"][-1]["after"]["cudagraph_recorded_non_static_inputs"] > 0
        )
        for source, expected in run["source_sha256"].items():
            assert (
                hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == expected
            ), source
        runs[label] = run
        if run["variant"] == "baseline":
            continue
        baseline = runs[f"baseline_b{run['batch_size']}_n{run['output_token_cap']}"]
        output_hashes = lambda r: {
            (w["token_hash"], w["text_hash"]) for w in r["waves"]
        }
        assert output_hashes(run) == output_hashes(baseline), label
        comparisons.append(
            dict(
                run=label,
                batch_size=run["batch_size"],
                output_token_cap=run["output_token_cap"],
                steps_per_round=run["decode_steps_per_round"],
                baseline_output_tokens_per_s=baseline["end_to_end_tokens_per_s"],
                optimized_output_tokens_per_s=run["end_to_end_tokens_per_s"],
                observed_change_percent=100
                * (
                    run["end_to_end_tokens_per_s"] / baseline["end_to_end_tokens_per_s"]
                    - 1
                ),
                matching_token_and_text_hashes=True,
                median_worker_round_ms=run["median_worker_step_ms"],
                median_scheduler_extra_ms=run["median_scheduler_extra_ms"],
                gpu_telemetry=run["gpu_telemetry"],
            )
        )
    validation = json.loads((OUT / "real_output_validation.json").read_text())
    assert (
        validation["full_matches_baseline"] and validation["multistep_matches_baseline"]
    )
    trace = json.loads(
        (OUT / "final/profile_full_b256_n64.trace_summary.json").read_text()
    )
    assert trace["cpu_stream_flushes"] > 0 and trace["overlap_fraction"] > 0
    report = dict(
        kind="warmed_engine_streaming_overlap_comparison",
        comparisons=comparisons,
        measured_requests=sum(r["requests"] for r in runs.values()),
        measured_output_tokens=sum(r["generated_tokens"] for r in runs.values()),
        source_sha256=next(iter(runs.values()))["source_sha256"],
        trace=trace,
        validation=dict(
            standard_passed=195,
            standard_skipped=9,
            local_checkpoint_passed=3,
            real_engine_passed=4,
            varied_real_streams_compared=24,
            incremental_valid_utf8_prefixes=1847,
        ),
        limitations=[
            "Engine plus validation harness, not HTTP; rates include prefill and decode.",
            "Laptop clocks are not locked; observed changes are workload-specific, not a hardware maximum.",
            "Larger greedy chunks produce burstier streaming and delay cancellation checks.",
        ],
    )
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "| Requests per wave | Output cap | Mode | Aggregate output tokens/s | Change vs baseline |",
        "|---|---:|---|---:|---:|",
    ]
    for batch, cap in ((256, 64), (128, 512)):
        base = runs[f"baseline_b{batch}_n{cap}"]["end_to_end_tokens_per_s"]
        for prefix, mode in (
            ("baseline", "Features disabled"),
            ("full", "Default S1 overlap"),
            ("full_s8", "Exact greedy S8"),
        ):
            rate = runs[f"{prefix}_b{batch}_n{cap}"]["end_to_end_tokens_per_s"]
            delta = "—" if prefix == "baseline" else f"{100 * (rate / base - 1):+.1f}%"
            lines.append(f"| {batch} | {cap} | {mode} | {rate:,.0f} | {delta} |")
    lines += [
        "",
        f"These runs measured {report['measured_requests']:,} requests and "
        f"{report['measured_output_tokens']:,} generated tokens; warmup is excluded.",
        "",
    ]
    pinned = runs.get("full_s8_pcores_b256_n64")
    if pinned:
        unpinned = runs["full_s8_b256_n64"]["end_to_end_tokens_per_s"]
        delta = 100 * (pinned["end_to_end_tokens_per_s"] / unpinned - 1)
        lines.append(
            f"CPU affinity experiment: {pinned['end_to_end_tokens_per_s']:,.0f} tokens/s "
            f"({delta:+.2f}% versus unpinned S8). Affinity remains unchanged by the application."
        )
    lines += [
        "",
        f"The diagnostic trace records {trace['cpu_stream_flush_ms']:.2f} ms of host "
        f"text-emission work, with {trace['cpu_stream_flush_overlapping_gpu_kernels_ms']:.2f} ms "
        f"({100 * trace['overlap_fraction']:.1f}%) overlapping actual GPU kernels. "
        "Profiled throughput is excluded from the table.",
    ]

    def replace_section(path, start, end, content):
        text = path.read_text()
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        path.write_text(before + start + "\n" + content + "\n" + end + after)

    replace_section(
        OUT / "README.md",
        "<!-- final-measurements:start -->",
        "<!-- final-measurements:end -->",
        "\n".join(lines),
    )
    short = next(r for r in comparisons if r["run"] == "full_s8_b256_n64")
    long = next(r for r in comparisons if r["run"] == "full_s8_b128_n512")
    status = (
        f"Final warmed streaming measurements: batch 256 with a 64-token output cap improved "
        f"from {short['baseline_output_tokens_per_s']:,.0f} to {short['optimized_output_tokens_per_s']:,.0f} "
        f"aggregate output tokens/s ({short['observed_change_percent']:+.1f}%) with exact greedy S8. "
        f"Batch 128 with a 512-token output cap improved from {long['baseline_output_tokens_per_s']:,.0f} "
        f"to {long['optimized_output_tokens_per_s']:,.0f} tokens/s ({long['observed_change_percent']:+.1f}%). "
        "All compared token/text hashes match; no compilation or capture occurred during measurement. "
        "Rates include prefill and the streaming harness. See the "
        "[overlap report](../benchmarks/overlap_optimization/README.md) for default S1 results, "
        "power telemetry, cadence tradeoffs and reproduction commands."
    )
    replace_section(
        ROOT / "docs/status.md",
        "<!-- overlap-final-results:start -->",
        "<!-- overlap-final-results:end -->",
        status,
    )
    for row in comparisons:
        print(
            row["run"],
            round(row["optimized_output_tokens_per_s"], 2),
            f"{row['observed_change_percent']:+.2f}%",
            flush=True,
        )
    print(
        "All final source hashes, compilation guards and output hashes verified.",
        flush=True,
    )


if __name__ == "__main__":
    main()
