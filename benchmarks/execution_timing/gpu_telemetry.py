"""Read-only NVML samples alongside a benchmark; never change device settings."""

import csv
from pathlib import Path
import statistics
import subprocess

FIELDS = (
    "timestamp",
    "pstate",
    "utilization.gpu",
    "utilization.memory",
    "power.draw.average",
    "power.draw.instant",
    "enforced.power.limit",
    "clocks.sm",
    "clocks.mem",
    "temperature.gpu",
    "clocks_event_reasons.sw_power_cap",
    "clocks_event_reasons.hw_power_brake_slowdown",
    "clocks_event_reasons.sw_thermal_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown",
)


def platform_profile():
    path = Path("/sys/firmware/acpi/platform_profile")
    return path.read_text().strip() if path.exists() else None


class GpuTelemetry:
    def __init__(self, path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w")
        self.file.write(",".join(FIELDS) + "\n")
        self.file.flush()
        self.process = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=" + ",".join(FIELDS),
                "--format=csv,noheader,nounits",
                "--loop-ms=500",
            ],
            stdout=self.file,
            stderr=subprocess.PIPE,
            text=True,
        )

    def stop(self):
        self.process.terminate()
        try:
            _, error = self.process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            _, error = self.process.communicate()
        self.file.close()
        with self.path.open() as source:
            rows = [
                {k: v.strip() for k, v in row.items()} for row in csv.DictReader(source)
            ]
        busy = [r for r in rows if float(r["utilization.gpu"]) >= 80]

        def describe(field, selected):
            values = []
            for row in selected:
                try:
                    values.append(float(row[field]))
                except ValueError:
                    pass
            return (
                dict(min=min(values), median=statistics.median(values), max=max(values))
                if values
                else None
            )

        return dict(
            csv_path=str(self.path),
            samples=len(rows),
            busy_samples=len(busy),
            busy_threshold_gpu_utilization_percent=80,
            all_samples={f: describe(f, rows) for f in FIELDS[2:10]},
            busy_samples_summary={f: describe(f, busy) for f in FIELDS[2:10]},
            throttle_active_samples={
                f: sum(r[f] == "Active" for r in rows) for f in FIELDS[10:]
            },
            error=error.strip() or None,
        )
