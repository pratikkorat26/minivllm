"""Local quality evaluation harness for LFM3-1.7B quantization & speculative decoding.

Uses only local weights (LFM3-1.7B/) and local cached datasets in
~/.cache/huggingface/datasets/ (no network access).
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from app.prompting import messages_to_lfm3_prompt
from app.types import ChatMessage
from inference import EngineConfig, GenerateRequest, LLMEngine, SamplingConfig
from runtime import Lfm3Runtime, get_device

PYARROW_ARCHIVE = "/home/pratik/.cache/uv/archive-v0/nTn5UwdktRqOb2h9"
if Path(PYARROW_ARCHIVE).exists() and PYARROW_ARCHIVE not in sys.path:
    sys.path.insert(0, PYARROW_ARCHIVE)


def _load_arrow_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.ipc as ipc

    with open(path, "rb") as f:
        table = ipc.open_stream(f).read_all()
    cols = table.column_names
    return [{c: table[c][i].as_py() for c in cols} for i in range(table.num_rows)]


def _find_local_datasets() -> dict[str, list[dict[str, Any]]]:
    hf_datasets = Path.home() / ".cache" / "huggingface" / "datasets"
    tau_paths = sorted(hf_datasets.glob("fuvty___tau-bench-synthetic/**/tau-bench-synthetic-train.arrow"))
    swe_paths = sorted(hf_datasets.glob("json/**/json-train.arrow"))

    tau_rows: list[dict[str, Any]] = []
    for p in tau_paths:
        tau_rows.extend(_load_arrow_rows(p))

    swe_rows: list[dict[str, Any]] = []
    for p in swe_paths:
        rows = _load_arrow_rows(p)
        if rows and "repo" in rows[0] and "prompt" in rows[0] and "completion" in rows[0]:
            swe_rows.extend(rows)

    return {"tau_bench_synthetic": tau_rows, "open_swe_traces": swe_rows}


def _build_teacher_forced_samples(
    datasets: dict[str, list[dict[str, Any]]],
    tokenizer: Any,
    seed: int = 42,
    n_tau: int = 100,
    n_swe: int = 100,
    max_tokens: int = 256,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    tau_rows = list(datasets["tau_bench_synthetic"])
    swe_rows = list(datasets["open_swe_traces"])
    rng.shuffle(tau_rows)
    rng.shuffle(swe_rows)

    samples: list[dict[str, Any]] = []
    for row in tau_rows[:n_tau]:
        scenario = json.loads(row["user_scenario"])
        instr = scenario.get("instructions", {})
        user_msg = (
            f"Domain: {row['domain']}\n"
            f"Request: {instr.get('reason_for_call', '')}\n"
            f"Context: {instr.get('known_info', '')}"
        )
        criteria = json.loads(row["evaluation_criteria"])
        actions = [
            {"name": a["name"], "arguments": a["arguments"]}
            for a in criteria.get("actions", [])
        ]
        prompt = messages_to_lfm3_prompt(
            [
                ChatMessage(role="user", content=user_msg),
                ChatMessage(role="assistant", content=json.dumps(actions)),
            ],
            enable_thinking=False,
        )
        ids = tokenizer.encode(prompt)[:max_tokens]
        if len(ids) >= 16:
            samples.append({"source": "tau_bench_synthetic", "id": row["id"], "token_ids": ids})

    for row in swe_rows[:n_swe]:
        user_msg = row["prompt"][:1000]
        asst_msg = row["completion"][:500]
        prompt = messages_to_lfm3_prompt(
            [
                ChatMessage(role="user", content=user_msg),
                ChatMessage(role="assistant", content=asst_msg),
            ],
            enable_thinking=False,
        )
        ids = tokenizer.encode(prompt)[:max_tokens]
        if len(ids) >= 16:
            samples.append({"source": "open_swe_traces", "id": row["instance_id"], "token_ids": ids})

    meta = {
        "seed": seed,
        "total_samples": len(samples),
        "max_tokens_per_sample": max_tokens,
        "sources": {
            "fuvty/tau-bench-synthetic": sum(1 for s in samples if s["source"] == "tau_bench_synthetic"),
            "nvidia/Open-SWE-Traces": sum(1 for s in samples if s["source"] == "open_swe_traces"),
        },
        "note": (
            "Salesforce/xlam-function-calling-60k and Salesforce/APIGen-MT-5k only had README.md "
            "in ~/.cache/huggingface/hub (no local data shards); held-out set uses cached "
            "fuvty/tau-bench-synthetic and nvidia/Open-SWE-Traces."
        ),
    }
    return samples, meta


def _build_generation_prompts(
    datasets: dict[str, list[dict[str, Any]]],
    seed: int = 42,
    n_func: int = 80,
    n_swe: int = 60,
    n_math_fact: int = 60,
) -> list[dict[str, Any]]:
    rng = random.Random(seed + 1)
    tau_rows = list(datasets["tau_bench_synthetic"])
    swe_rows = list(datasets["open_swe_traces"])
    rng.shuffle(tau_rows)
    rng.shuffle(swe_rows)

    prompts: list[dict[str, Any]] = []

    # 1. Function-calling / JSON tool-call prompts from tau-bench-synthetic
    for row in tau_rows[:n_func]:
        scenario = json.loads(row["user_scenario"])
        instr = scenario.get("instructions", {})
        criteria = json.loads(row["evaluation_criteria"])
        actions = criteria.get("actions", [])
        expected_names = [a["name"] for a in actions]
        first_action = actions[0] if actions else {}
        arg_keys = list((first_action.get("arguments") or {}).keys())
        user_msg = (
            f"You are a {row['domain']} assistant. Convert this user request into the FIRST tool call "
            f"as a single valid JSON object with keys \"name\" and \"arguments\" "
            f"(allowed tool name: \"{first_action.get('name', '')}\", argument keys: {arg_keys}). "
            f"Output ONLY raw JSON, no markdown.\n\n"
            f"User: {instr.get('reason_for_call', '')}\n"
            f"Details: {instr.get('known_info', '')}"
        )
        prompts.append({
            "id": f"func_{row['id']}",
            "category": "function_calling",
            "prompt": user_msg,
            "expected_tool_names": expected_names,
            "first_expected_name": first_action.get("name"),
        })

    # 2. Coding / SWE chat prompts from Open-SWE-Traces
    for row in swe_rows[:n_swe]:
        issue_snippet = row["prompt"][:600].strip()
        user_msg = (
            f"Given this repository issue snippet ({row['repo']}), suggest the first shell command "
            f"or diagnosis step in 1-2 concise sentences:\n\n{issue_snippet}"
        )
        prompts.append({
            "id": f"swe_{row['instance_id']}",
            "category": "coding_chat",
            "prompt": user_msg,
        })

    # 3. Math / fact / structured JSON prompts (deterministic seed)
    facts = [
        ("What is the capital of France? Answer in one word.", "Paris"),
        ("What is the chemical symbol for gold? Answer in one word.", "Au"),
        ("How many continents are on Earth? Answer with just the number.", "7"),
        ("What is the boiling point of water at sea level in Celsius? Answer with just the number.", "100"),
        ("Who wrote Romeo and Juliet? Answer with just the name.", "Shakespeare"),
        ("What planet is known as the Red Planet? Answer in one word.", "Mars"),
        ("What is the largest ocean on Earth? Answer in two words.", "Pacific Ocean"),
        ("What is the square root of 144? Answer with just the number.", "12"),
        ("How many bits are in a byte? Answer with just the number.", "8"),
        ("What gas do plants primarily absorb from the atmosphere? Answer in two words.", "Carbon dioxide"),
    ]
    for i in range(n_math_fact):
        if i < 25:
            a = rng.randint(12, 99)
            b = rng.randint(12, 99)
            user_msg = f"What is {a} * {b}? Answer with just the number."
        elif i < 45:
            a = rng.randint(100, 999)
            b = rng.randint(100, 999)
            user_msg = f"What is {a} + {b}? Answer with just the number."
        elif i < 55:
            q, _ = facts[i - 45]
            user_msg = q
        else:
            val = rng.randint(1, 100)
            user_msg = (
                f'Return a valid JSON object with keys "id" (integer {i}) and '
                f'"value" (integer {val}). Output ONLY raw JSON.'
            )
        prompts.append({
            "id": f"math_fact_{i}",
            "category": "math_fact",
            "prompt": user_msg,
        })

    return prompts


def _extract_json(text: str) -> Any | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


@torch.inference_mode()
def _eval_teacher_forced(
    model: torch.nn.Module,
    samples: list[dict[str, Any]],
    device: torch.device,
    ref_model: torch.nn.Module | None = None,
) -> dict[str, float]:
    total_nll = 0.0
    total_tokens = 0
    total_kl = 0.0
    total_top1_match = 0

    for sample in samples:
        ids = torch.tensor(sample["token_ids"], dtype=torch.long, device=device).unsqueeze(0)
        seq_len = ids.shape[1]
        num_blocks = (seq_len + 15) // 16 + 1
        caches = model.init_paged_kv_caches(num_blocks=num_blocks, block_size=16, device=device, max_slots=1)
        conv_caches = model.init_conv_state_caches(max_slots=1, device=device, dtype=model.dtype)
        tables = torch.arange(num_blocks, dtype=torch.long, device=device).unsqueeze(0)
        kv_lens = torch.zeros(1, dtype=torch.long, device=device)
        slots = torch.zeros(1, dtype=torch.long, device=device)

        logits = model(
            ids,
            block_tables=tables,
            kv_lens=kv_lens,
            paged_kv_caches=caches,
            conv_state_caches=conv_caches,
            conv_slots=slots,
            logits_last_only=False,
        )[0, :-1, :].float()
        targets = ids[0, 1:]
        n_tok = targets.numel()

        log_p = F.log_softmax(logits, dim=-1)
        nll = F.nll_loss(log_p, targets, reduction="sum").item()
        total_nll += nll
        total_tokens += n_tok

        if ref_model is not None:
            ref_caches = ref_model.init_paged_kv_caches(
                num_blocks=num_blocks, block_size=16, device=device, max_slots=1
            )
            ref_conv = ref_model.init_conv_state_caches(
                max_slots=1, device=device, dtype=ref_model.dtype
            )
            ref_logits = ref_model(
                ids,
                block_tables=tables,
                kv_lens=kv_lens,
                paged_kv_caches=ref_caches,
                conv_state_caches=ref_conv,
                conv_slots=slots,
                logits_last_only=False,
            )[0, :-1, :].float()
            ref_log_p = F.log_softmax(ref_logits, dim=-1)
            ref_p = ref_log_p.exp()
            kl = (ref_p * (ref_log_p - log_p)).sum(dim=-1).sum().item()
            total_kl += kl
            top1_match = (logits.argmax(dim=-1) == ref_logits.argmax(dim=-1)).sum().item()
            total_top1_match += top1_match

    mean_nll = total_nll / max(1, total_tokens)
    out = {
        "tokens_evaluated": total_tokens,
        "mean_nll": round(mean_nll, 6),
        "perplexity": round(math.exp(min(mean_nll, 20.0)), 4),
    }
    if ref_model is not None:
        out["mean_kl_vs_bf16"] = round(total_kl / max(1, total_tokens), 6)
        out["top1_agreement_vs_bf16"] = round(total_top1_match / max(1, total_tokens), 6)
    else:
        out["mean_kl_vs_bf16"] = 0.0
        out["top1_agreement_vs_bf16"] = 1.0
    return out


async def _run_generation_eval(
    runtime: Lfm3Runtime,
    prompts: list[dict[str, Any]],
    *,
    enable_speculative: bool = False,
    max_new_tokens: int = 96,
) -> list[dict[str, Any]]:
    config = EngineConfig(
        use_instruct_model=True,
        max_new_tokens=max_new_tokens,
        max_batch_tokens=2048,
        decode_batch_size=1,
        decode_steps_per_round=8,
        enable_speculative_decoding=enable_speculative,
        speculative_num_tokens=5,
        sampling=SamplingConfig(temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.0),
    )
    engine = LLMEngine(runtime, config)
    results = []
    try:
        for item in prompts:
            res = await engine.generate(GenerateRequest(item["id"], item["prompt"]))
            results.append({
                "id": item["id"],
                "category": item["category"],
                "token_ids": res.token_ids,
                "text": res.text,
                "stop_reason": res.stop_reason,
                "error": res.error_message,
            })
    finally:
        await engine.shutdown()
    return results


def _score_generations(
    outputs: list[dict[str, Any]],
    prompts: list[dict[str, Any]],
    bf16_outputs: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    n = len(outputs)
    clean_eos = sum(1 for o in outputs if o["stop_reason"] == "eos")
    errors = sum(1 for o in outputs if o["error"] is not None)

    func_total = 0
    json_valid = 0
    tool_name_match = 0
    for out, p in zip(outputs, prompts):
        if p["category"] == "function_calling":
            func_total += 1
            parsed = _extract_json(out["text"])
            if isinstance(parsed, dict):
                json_valid += 1
                if parsed.get("name") == p.get("first_expected_name"):
                    tool_name_match += 1

    exact_match = n
    if bf16_outputs is not None:
        exact_match = sum(
            1 for o, ref in zip(outputs, bf16_outputs) if o["token_ids"] == ref["token_ids"]
        )

    return {
        "prompts_evaluated": n,
        "exact_match_vs_bf16_rate": round(exact_match / max(1, n), 4),
        "exact_match_vs_bf16_count": f"{exact_match}/{n}",
        "clean_eos_rate": round(clean_eos / max(1, n), 4),
        "clean_eos_count": f"{clean_eos}/{n}",
        "func_prompts": func_total,
        "func_json_valid_rate": round(json_valid / max(1, func_total), 4),
        "func_tool_name_match_rate": round(tool_name_match / max(1, func_total), 4),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LFM3-1.7B quality across quantization & spec modes.")
    parser.add_argument("--output", default="quality_eval.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tf-samples", type=int, default=200)
    parser.add_argument("--gen-prompts", type=int, default=200)
    args = parser.parse_args()

    device = get_device()
    datasets = _find_local_datasets()

    # Load BF16 reference runtime first
    t0 = time.perf_counter()
    bf16_runtime = Lfm3Runtime(use_instruct_model=True, device=device, quantization="none", compile_model=False)
    tf_samples, tf_meta = _build_teacher_forced_samples(
        datasets,
        bf16_runtime.tokenizer,
        seed=args.seed,
        n_tau=args.tf_samples // 2,
        n_swe=args.tf_samples - args.tf_samples // 2,
    )
    gen_prompts = _build_generation_prompts(datasets, seed=args.seed)
    if args.gen_prompts < len(gen_prompts):
        gen_prompts = gen_prompts[: args.gen_prompts]

    report: dict[str, Any] = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "teacher_forced_dataset": tf_meta,
        "generation_prompts": {
            "total": len(gen_prompts),
            "function_calling": sum(1 for p in gen_prompts if p["category"] == "function_calling"),
            "coding_chat": sum(1 for p in gen_prompts if p["category"] == "coding_chat"),
            "math_fact": sum(1 for p in gen_prompts if p["category"] == "math_fact"),
        },
        "modes": {},
    }

    # 1. BF16 reference
    print("[eval] Running none (BF16 reference)...", flush=True)
    bf16_tf = _eval_teacher_forced(bf16_runtime.model, tf_samples, device, ref_model=None)
    bf16_gen_outputs = asyncio.run(_run_generation_eval(bf16_runtime, gen_prompts, enable_speculative=False))
    bf16_gen_score = _score_generations(bf16_gen_outputs, gen_prompts, bf16_outputs=None)
    report["modes"]["none"] = {**bf16_tf, **bf16_gen_score}
    print("[eval] none:", report["modes"]["none"], flush=True)

    # 2. BF16 + speculative decoding
    print("[eval] Running none + speculative_decoding...", flush=True)
    spec_gen_outputs = asyncio.run(_run_generation_eval(bf16_runtime, gen_prompts, enable_speculative=True))
    spec_gen_score = _score_generations(spec_gen_outputs, gen_prompts, bf16_outputs=bf16_gen_outputs)
    report["modes"]["none_speculative"] = {**bf16_tf, **spec_gen_score}
    print("[eval] none_speculative:", report["modes"]["none_speculative"], flush=True)

    # 3. Quantized modes (keep bf16_runtime.model on GPU for exact KL & top-1)
    for mode in ("int8", "fp8", "int4_w4a16"):
        print(f"[eval] Running {mode}...", flush=True)
        q_runtime = Lfm3Runtime(use_instruct_model=True, device=device, quantization=mode, compile_model=False)
        q_tf = _eval_teacher_forced(q_runtime.model, tf_samples, device, ref_model=bf16_runtime.model)
        q_gen_outputs = asyncio.run(_run_generation_eval(q_runtime, gen_prompts, enable_speculative=False))
        q_gen_score = _score_generations(q_gen_outputs, gen_prompts, bf16_outputs=bf16_gen_outputs)
        report["modes"][mode] = {**q_tf, **q_gen_score}
        print(f"[eval] {mode}:", report["modes"][mode], flush=True)
        del q_runtime
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report["elapsed_s"] = round(time.perf_counter() - t0, 2)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[eval] Saved {args.output} in {report['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
