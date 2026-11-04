"""Recheck the published 8K prefill workload in independent fresh processes."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks" / "prefill_verification"
sys.path.insert(0, str(ROOT))


def child(case, run_number):
    from argparse import Namespace
    import torch
    from scripts import benchmark_tps as benchmark
    from runtime.provider import Lfm3Tokenizer
    from lfm3.tokenizer import apply_chat_template

    source = json.loads(
        (ROOT / "benchmarks/software_phases_long_budget2048.json").read_text()
    )
    config = source["config"].copy()
    tokenizer = Lfm3Tokenizer(str(ROOT / "LFM3-1.7B/tokenizer.json"))
    if case == "documentation":
        content = "\n\n".join(
            (ROOT / path).read_text()
            for path in ("docs/study.md", "docs/architecture.md", "README.md")
        )
        prefix = "Engineering reference material:\n\n"
        suffix = "\n\nSummarize the main architectural decisions and trade-offs in five sentences."
        tokens = tokenizer.encode(content)
        budget = 8118 - len(tokenizer.encode(apply_chat_template(prefix + suffix)))
        for _ in range(8):
            prompt = prefix + tokenizer.decode(tokens[:budget]) + suffix
            actual = len(tokenizer.encode(apply_chat_template(prompt)))
            if actual == 8118:
                break
            budget += 8118 - actual
        config["prompt"] = prompt
    prompt_tokens = len(tokenizer.encode(apply_chat_template(config["prompt"])))
    assert prompt_tokens == 8118, prompt_tokens
    config.update(repeats=10, phase_timing=True, batch_prefill=False)
    args = Namespace(**config, output=OUT / f"{case}_run{run_number}_engine.json")
    timers = []
    samples = []
    original_timer = benchmark.PhaseTimer
    original_wave = benchmark._wave

    class RecordingTimer(original_timer):
        def __init__(self, device):
            super().__init__(device)
            timers.append(self)

    async def wave(*values, **kwargs):
        timer = timers[0]
        start_index = len(timer.samples)
        wall_started = time.perf_counter()
        results = await original_wave(*values, **kwargs)
        elapsed = time.perf_counter() - wall_started
        if kwargs["wave_id"].startswith("run-"):
            all_samples = timer.samples
            timer.samples = all_samples[start_index:]
            try:
                phases = timer.report()
            finally:
                timer.samples = all_samples
            result = results[0]
            if result.error_message:
                raise RuntimeError(result.error_message)
            assert phases["prefill"]["tokens"] == prompt_tokens
            assert phases["decode"]["tokens"] == len(result.token_ids) - 1
            samples.append(
                dict(
                    wave=kwargs["wave_id"],
                    prompt_tokens=prompt_tokens,
                    generated_tokens=len(result.token_ids),
                    elapsed_s=elapsed,
                    ttft_s=result.ttft_s,
                    tpot_s=result.tpot_s,
                    input_tokens_per_ttft_s=prompt_tokens / result.ttft_s,
                    phases=phases,
                )
            )
        return results

    benchmark.PhaseTimer = RecordingTimer
    benchmark._wave = wave
    started = datetime.now(timezone.utc).isoformat()
    asyncio.run(benchmark.run(args))
    engine_report = json.loads(args.output.read_text())
    report = dict(
        case=case,
        run=run_number,
        started_at_utc=started,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        device=torch.cuda.get_device_name(),
        torch_version=torch.__version__,
        prompt_tokens=prompt_tokens,
        measured_requests=len(samples),
        warmup_s=engine_report["warmup_s"],
        token_hash=engine_report["token_hash"],
        samples=samples,
    )
    (OUT / f"{case}_run{run_number}_requests.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        "VERIFICATION",
        case,
        run_number,
        "prefill_tokens_per_s=",
        round(engine_report["phases"]["prefill"]["tokens_per_s"], 2),
        "median_ttft_s=",
        round(statistics.median(row["ttft_s"] for row in samples), 4),
        flush=True,
    )


def parent():
    OUT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", LFM3_FUSED_PAGED_ATTENTION="0")
    for run_number in range(1, 4):
        for case in ("repeated", "documentation"):
            print("START", case, run_number, flush=True)
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--case",
                    case,
                    "--run-number",
                    str(run_number),
                ],
                cwd=ROOT,
                env=env,
                check=True,
            )
    summary = {}
    for case in ("repeated", "documentation"):
        reports = [
            json.loads((OUT / f"{case}_run{i}_requests.json").read_text())
            for i in range(1, 4)
        ]
        samples = [row for report in reports for row in report["samples"]]
        total_tokens = sum(row["prompt_tokens"] for row in samples)
        phase_seconds = sum(
            row["phases"]["prefill"]["device_elapsed_s"] for row in samples
        )
        run_rates = [
            sum(row["prompt_tokens"] for row in report["samples"])
            / sum(
                row["phases"]["prefill"]["device_elapsed_s"]
                for row in report["samples"]
            )
            for report in reports
        ]
        summary[case] = dict(
            requests=len(samples),
            fresh_processes=len(reports),
            prompt_tokens=8118,
            prefill_input_tokens_per_s=total_tokens / phase_seconds,
            process_prefill_rates=run_rates,
            per_request_prefill_min=min(
                row["phases"]["prefill"]["tokens_per_s"] for row in samples
            ),
            per_request_prefill_median=statistics.median(
                row["phases"]["prefill"]["tokens_per_s"] for row in samples
            ),
            per_request_prefill_max=max(
                row["phases"]["prefill"]["tokens_per_s"] for row in samples
            ),
            median_ttft_s=statistics.median(row["ttft_s"] for row in samples),
            p95_ttft_s=statistics.quantiles(
                [row["ttft_s"] for row in samples], n=100, method="inclusive"
            )[94],
            prompt_tokens_over_total_ttft=total_tokens
            / sum(row["ttft_s"] for row in samples),
            decode_output_tokens_per_s=sum(
                row["phases"]["decode"]["tokens"] for row in samples
            )
            / sum(row["phases"]["decode"]["device_elapsed_s"] for row in samples),
            token_hashes_match=len({report["token_hash"] for report in reports}) == 1,
        )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("FINAL_VERIFICATION", json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("repeated", "documentation"))
    parser.add_argument("--run-number", type=int)
    args = parser.parse_args()
    child(args.case, args.run_number) if args.case else parent()
