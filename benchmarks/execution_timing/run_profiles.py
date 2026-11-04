"""Run executor profiling cases sequentially on the single GPU."""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", LFM3_FUSED_PAGED_ATTENTION="0")
    for batch, context, variant in (
        (1, "short", "baseline"),
        (1, "short", "candidate"),
        (256, "short", "candidate"),
        (256, "short", "baseline"),
        (1, "long", "baseline"),
        (1, "long", "candidate"),
    ):
        print("PROFILE", batch, context, variant, flush=True)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "benchmarks/execution_timing/profile_decode_execution.py"),
                "--batch-size",
                str(batch),
                "--context",
                context,
                "--rope-cache",
                variant,
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )
