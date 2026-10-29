"""Render saved measurements and a clearly labeled illustrative KV study figure."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(directory / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("benchmarks"))
    parser.add_argument("--output", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    before = read(args.results / "attention_baseline.json")
    after = read(args.results / "attention_vectorized.json")
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(before["rows"]))
    ax.bar(
        x - 0.18,
        [r["latency_ms"] for r in before["rows"]],
        0.36,
        label="Baseline",
        color="#718096",
    )
    ax.bar(
        x + 0.18,
        [r["latency_ms"] for r in after["rows"]],
        0.36,
        label="Vectorized",
        color="#16827c",
    )
    ax.set_xticks(
        x, [f"B={r['batch']}, S={r['context']}\nQ={r['query']}" for r in before["rows"]]
    )
    ax.set_ylabel("Wall time per attention forward (ms)")
    ax.set_title(
        "Historical Gemma 270M synthetic BF16 attention · 50 timed iterations"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    save(fig, args.output, "attention")

    baseline = read(args.results / "engine_baseline.json")
    improved = read(args.results / "engine_improved.json")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for result, label, color in [
        (baseline, "Baseline", "#718096"),
        (improved, "Improved", "#16827c"),
    ]:
        rows = result["rows"]
        concurrency = [r["concurrency"] for r in rows]
        axes[0].plot(
            concurrency,
            [r["output_tokens_per_s"] for r in rows],
            "o-",
            label=label,
            color=color,
        )
        for ax, metric in zip(axes[1:], ["ttft_ms", "tpot_ms"]):
            values = [
                [q[metric] for q in r["requests"] if q[metric] is not None]
                for r in rows
            ]
            ax.plot(
                concurrency,
                [np.median(v) for v in values],
                "o-",
                label=label + " median",
                color=color,
            )
            ax.plot(
                concurrency,
                [np.percentile(v, 95) for v in values],
                "o--",
                label=label + " p95",
                color=color,
                alpha=0.65,
            )
    for ax, title, ylabel in zip(
        axes,
        ["Output throughput", "Time to first text", "Time per output token"],
        ["tokens/s", "ms", "ms"],
    ):
        ax.set_title(title)
        ax.set_xlabel("Concurrent requests")
        ax.set_ylabel(ylabel)
        ax.set_xticks([1, 2, 4, 8])
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Historical Gemma 270M base · RTX 4070 Laptop · BF16 · 16-token cap",
        y=1.04,
    )
    save(fig, args.output, "engine")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Illustrative page mapping: no fabricated benchmark numbers.
    for x, label in enumerate(
        [
            "shared prefix\npage 3",
            "shared prefix\npage 7",
            "A private\npage 1",
            "B private\npage 5",
        ]
    ):
        color = "#97d9cf" if x < 2 else "#f2c27b"
        axes[0].add_patch(plt.Rectangle((x * 1.15, 0.4), 1, 0.6, color=color))
        axes[0].text(x * 1.15 + 0.5, 0.7, label, ha="center", va="center")
    axes[0].text(0, 1.65, "Request A: [3, 7, 1]\nRequest B: [3, 7, 5]", fontsize=12)
    axes[0].text(
        0,
        -0.1,
        "Pages 3 and 7: refcount = 3\n(cache + A + B); private pages: refcount = 1",
    )
    axes[0].set_xlim(-0.1, 4.6)
    axes[0].set_ylim(-0.4, 2.1)
    axes[0].axis("off")
    axes[0].set_title("Logical block tables → shared physical pages")
    tokens = np.arange(0, 32769, 256)
    # LFM3 layout: 11 attention layers * K/V * 4 KV heads * 128 dim * BF16 bytes.
    mib = tokens * 11 * 2 * 4 * 128 * 2 / 1024**2
    axes[1].plot(tokens, mib, color="#16827c")
    axes[1].set_xlabel("Physical KV token slots (all requests combined)")
    axes[1].set_ylabel("KV storage (MiB)")
    axes[1].grid(alpha=0.2)
    axes[1].set_title("LFM3-1.7B BF16 KV capacity (analytical)")
    fig.suptitle(
        "Study diagrams · mapping is illustrative; memory curve is a formula", y=1.04
    )
    save(fig, args.output, "kv_study")
    print(f"Wrote figures to {args.output}")


if __name__ == "__main__":
    main()
