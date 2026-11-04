"""Run feature ablations sequentially; never share the single GPU across cases."""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/overlap_optimization"

if __name__ == "__main__":
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", LFM3_FUSED_PAGED_ATTENTION="0")
    for batch, tokens, variant, steps, repeat, label in (
        (256, 64, "baseline", 1, 5, "baseline_b256_n64_a"),
        (256, 64, "full", 1, 5, "full_b256_n64_a"),
        (256, 64, "full", 8, 5, "full_s8_b256_n64_a"),
        (256, 64, "full", 1, 3, "full_b256_n64_b"),
        (256, 64, "baseline", 1, 3, "baseline_b256_n64_b"),
        (1, 64, "full", 1, 5, "full_b1_n64"),
        (1, 64, "baseline", 1, 5, "baseline_b1_n64"),
        (128, 512, "baseline", 1, 3, "baseline_b128_n512"),
        (128, 512, "full", 1, 3, "full_b128_n512"),
        (128, 512, "incremental", 1, 3, "incremental_b128_n512"),
    ):
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
            "--repeats",
            str(repeat),
            "--stream",
            "--output",
            str(OUT / f"{label}.json"),
        ]
        if tokens > 64:
            command += [
                "--prompt",
                "Write a numbered list of 200 practical tips for writing reliable software. Explain each tip in two sentences. Start directly with item 1 and continue through item 200.",
            ]
        print("BENCHMARK", label, flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
