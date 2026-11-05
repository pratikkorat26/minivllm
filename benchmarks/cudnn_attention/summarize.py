"""Build the compact cuDNN benchmark summary from the recorded waves."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/cudnn_attention"

CASES = [
    ("batch_256_64_s1", "full_b256_n64", "full_b256_n64"),
    ("batch_256_64_s8", "full_s8_b256_n64", "full_s8_b256_n64"),
    ("batch_128_512_s1", "full_b128_n512", "full_b128_n512"),
    ("batch_128_512_s8", "full_s8_b128_n512", "full_s8_b128_n512"),
]


def load(path):
    return json.loads(path.read_text())


def case_record(label, baseline_path, cudnn_path):
    base = load(ROOT / "benchmarks/overlap_optimization/final" / f"{baseline_path}.json")
    cudnn = load(OUT / "final" / f"{cudnn_path}.json")
    return dict(
        label=label,
        batch_size=cudnn["batch_size"],
        output_token_cap=cudnn["output_token_cap"],
        decode_steps_per_round=cudnn["decode_steps_per_round"],
        baseline_tokens_per_s=round(base["end_to_end_tokens_per_s"], 1),
        cudnn_tokens_per_s=round(cudnn["end_to_end_tokens_per_s"], 1),
        change_percent=round(
            100 * (cudnn["end_to_end_tokens_per_s"] / base["end_to_end_tokens_per_s"] - 1),
            1,
        ),
        baseline_wave_token_hashes=[w["token_hash"][:12] for w in base["waves"]],
        cudnn_wave_token_hashes=[w["token_hash"][:12] for w in cudnn["waves"]],
        requests=cudnn["requests"],
        generated_tokens=cudnn["generated_tokens"],
        peak_cuda_allocated_bytes=cudnn["peak_cuda_allocated_bytes"],
        source_sha256=cudnn["source_sha256"],
    )


summary = dict(
    date="2026-11-04",
    device="NVIDIA GeForce RTX 4070 Laptop GPU",
    torch="2.15.0+cu132",
    torch_profile="performance",
    workload="BF16 LFM3-1.7B, PyTorch compilation, 2048-token scheduling budget, compatible prefill batching, custom paged attention disabled",
    option=dict(
        name="enable_cudnn_attention",
        default=False,
        enable="--enable-cudnn-attention / LFM3_API_ENABLE_CUDNN_ATTENTION=true / --cudnn-attention in handoff.py",
        reason="cuDNN is not bitwise reproducible at some batch/context shapes",
    ),
    cases=[case_record(*c) for c in CASES],
    validation=dict(
        determinism="determinism.json",
        token_divergence="token_divergence.json",
        quality_optimized="quality_optimized_greedy.json",
        quality_eager="quality_eager_greedy.json",
    ),
)
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary["cases"], indent=1))
