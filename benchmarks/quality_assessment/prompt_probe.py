"""Check whether stricter application prompts repair observed output failures."""
# ruff: noqa: E402

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from benchmarks.quality_assessment.evaluate import grade, prompt_text
from inference import LLMEngine, EngineConfig, SamplingConfig, GenerateRequest
from runtime import Lfm3Runtime
import torch


async def run():
    runtime = Lfm3Runtime(device=torch.device("cuda"), compile_model=False)
    engine = LLMEngine(
        runtime,
        EngineConfig(
            use_instruct_model=False,
            max_kv_cache_tokens=2048,
            max_new_tokens=256,
            sampling=SamplingConfig(
                temperature=0.1, top_p=1, top_k=50, repetition_penalty=1.1, seed=42
            ),
        ),
    )
    system = "You are a precise data-processing assistant. Follow the requested output format exactly. Do not include Markdown fences, explanations, or reasoning. Use schema identifiers verbatim."
    cases = [
        dict(
            id="sql_schema",
            check="sql",
            expected=None,
            messages=[
                dict(role="system", content=system),
                dict(
                    role="user",
                    content="SQLite schema:\nCREATE TABLE orders (customer_id INTEGER, status TEXT, amount INTEGER);\nReturn only a SELECT query for total paid amount per customer, with columns customer_id and total, ordered by total descending. Use the exact table name orders.",
                ),
            ],
        ),
        dict(
            id="math_json",
            check="json",
            expected=dict(amount=84),
            messages=[
                dict(role="system", content=system),
                dict(
                    role="user",
                    content="A cafe sold 18 sandwiches at $7 each and paid $42 for ingredients. Return only JSON with a single integer field amount containing revenue minus ingredient cost.",
                ),
            ],
        ),
        dict(
            id="multi_turn_strict",
            check="json",
            expected=dict(name="Rina", destination="Osaka"),
            messages=[
                dict(role="system", content=system),
                dict(
                    role="user", content="My name is Rina and I am traveling to Osaka."
                ),
                dict(
                    role="assistant",
                    content="Understood, Rina. Your destination is Osaka.",
                ),
                dict(
                    role="user",
                    content="Return only a JSON object with name and destination. No code fences.",
                ),
            ],
        ),
    ]
    rows = []
    try:
        for case in cases:
            result = await engine.generate(
                GenerateRequest(case["id"], prompt_text(case), max_new_tokens=256)
            )
            row = dict(
                id=case["id"],
                messages=case["messages"],
                text=result.text,
                stop_reason=result.stop_reason,
                grade=grade(case, result.text),
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        await engine.shutdown()
    Path("benchmarks/quality_assessment/prompting_probe.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )


if __name__ == "__main__":
    asyncio.run(run())
