"""Validate local checkpoints, then measure the frozen overlap implementation."""

import os
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/overlap_optimization/final"
LONG_PROMPT = "Write a numbered list of 200 practical tips for writing reliable software. Explain each tip in two sentences. Start directly with item 1 and continue through item 200."


def performance_cores():
    frequencies = {}
    for cpu in os.sched_getaffinity(0):
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq")
        if path.exists():
            frequencies[cpu] = int(path.read_text())
    if not frequencies:
        return []
    threshold = max(frequencies.values()) * 0.95
    return [cpu for cpu, frequency in frequencies.items() if frequency >= threshold]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    if Path("/sys/firmware/acpi/platform_profile").read_text().strip() != "performance":
        raise SystemExit(
            "Select the performance platform profile before this benchmark."
        )
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        LFM3_FUSED_PAGED_ATTENTION="0",
        RUN_LOCAL_MODEL_TESTS="0",
        RUN_REAL_ENGINE_TESTS="0",
    )
    for flag, path in (
        ("RUN_LOCAL_MODEL_TESTS", "tests/test_model_local.py"),
        ("RUN_REAL_ENGINE_TESTS", "tests/test_llmengine_regression_real.py"),
    ):
        if args.skip_validation:
            continue
        test_env = dict(env, **{flag: "1"})
        subprocess.run(
            [sys.executable, "-m", "pytest", "-x", "-q", path],
            cwd=ROOT,
            env=test_env,
            check=True,
        )

    cases = [
        (256, 64, "baseline", 1, "baseline_b256_n64", False),
        (256, 64, "full", 1, "full_b256_n64", False),
        (256, 64, "full", 8, "full_s8_b256_n64", False),
        (256, 64, "full", 8, "full_s8_pcores_b256_n64", True),
        (128, 512, "baseline", 1, "baseline_b128_n512", False),
        (128, 512, "full", 1, "full_b128_n512", False),
        (128, 512, "full", 8, "full_s8_b128_n512", False),
    ]
    for batch, tokens, variant, steps, label, pin in cases:
        command = [
            sys.executable,
            str(ROOT / "benchmarks/execution_timing/handoff.py"),
            "--batch-size",
            str(batch),
            "--output-tokens",
            str(tokens),
            "--variant",
            variant,
            "--decode-steps",
            str(steps),
            "--stream",
            "--require-platform-profile",
            "performance",
            "--repeats",
            "5" if tokens == 64 else "3",
            "--output",
            str(OUT / f"{label}.json"),
        ]
        if tokens > 64:
            command += ["--prompt", LONG_PROMPT]
        if pin:
            cores = performance_cores()
            if not cores:
                continue
            command = ["taskset", "--cpu-list", ",".join(map(str, cores))] + command
        print("FINAL BENCHMARK", label, flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/execution_timing/handoff.py"),
            "--batch-size",
            "256",
            "--variant",
            "full",
            "--stream",
            "--repeats",
            "1",
            "--profile",
            "--require-platform-profile",
            "performance",
            "--output",
            str(OUT / "profile_full_b256_n64.json"),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
