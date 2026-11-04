"""Small practical quality audit; task checks are not a general accuracy benchmark."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/quality_assessment"
REVISION = "ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7"
sys.path.insert(0, str(ROOT))


def cases():
    def case(key, prompt, check="manual", expected=None, category="instruction"):
        return dict(
            id=key,
            category=category,
            messages=[dict(role="user", content=prompt)],
            check=check,
            expected=expected,
        )

    rows = [
        case(
            "arithmetic",
            "A cafe sold 18 sandwiches at $7 each and paid $42 for ingredients. What is revenue minus ingredient cost? Answer with only the dollar amount.",
            "money",
            84,
            "arithmetic",
        ),
        case(
            "discount",
            "An item costs $80 before a 25% discount. What is the price after the discount? Reply only with the dollar amount.",
            "money",
            60,
            "arithmetic",
        ),
        case(
            "json",
            "Return only a JSON object with keys name, quantity, priority. Order: Maya ordered 3 notebooks. Priority: high.",
            "json",
            dict(name="Maya", quantity=3, priority="high"),
            "extraction",
        ),
        case(
            "abstain",
            "Answer using only this context. Project Cedar is owned by Lena and is due on June 14. What is its budget? If the context does not say, reply exactly: NOT PROVIDED",
            "exact",
            "NOT PROVIDED",
            "grounding",
        ),
        case(
            "sentiment",
            "Classify this review as positive, negative, or neutral. Review: Delivery arrived two days late and the box was damaged. Reply with only the lowercase label.",
            "exact",
            "negative",
            "classification",
        ),
        case(
            "sorting",
            "Return these words in lowercase alphabetical order: Zebra, apple, Mango. Return only a comma-separated list.",
            "list",
            ["apple", "mango", "zebra"],
        ),
        case(
            "schedule",
            "A train departs at 08:10 and every 20 minutes after that. What is its first departure after 08:45? Reply only in HH:MM format.",
            "exact",
            "08:50",
            "reasoning",
        ),
        case(
            "owner",
            "Use only this context: Project Cedar is owned by Lena. Its deadline is June 14 and it is blocked pending security review. Who owns Cedar? Reply only with the name.",
            "exact",
            "Lena",
            "grounding",
        ),
        case(
            "deduction",
            "There are three boxes: red, blue, and green. The key is not in the red box and is not in the green box. Which box contains the key? Reply with only one lowercase word.",
            "exact",
            "blue",
            "reasoning",
        ),
        case(
            "sql",
            "Table orders has columns customer_id, status, amount. Write SQL returning customer_id and SUM(amount) as total for paid orders, grouped by customer_id, ordered by total descending. Return only SQL.",
            "sql",
            None,
            "code",
        ),
        case(
            "email",
            "Write a polite two-sentence email declining a Tuesday meeting and offering Thursday instead. Do not invent a reason. Return only the email.",
            category="writing",
        ),
        case(
            "summary",
            "Summarize this incident in one sentence: A deployment at 10:00 increased checkout errors. The team rolled back at 10:12. Error rates returned to normal at 10:15. The root cause is still under investigation.",
            category="summarization",
        ),
        case(
            "translation",
            "Translate into French. Return only the translation: The meeting has been moved to Monday.",
            category="translation",
        ),
        case(
            "python",
            "Write only a Python function named dedupe(items) that removes duplicates from a list of hashable items while preserving their original order. Do not use imports. Do not include Markdown or explanation.",
            category="code",
        ),
    ]
    rows.append(
        dict(
            id="multi_turn",
            category="conversation",
            check="json",
            expected=dict(name="Rina", destination="Osaka"),
            messages=[
                dict(
                    role="user", content="My name is Rina and I am traveling to Osaka."
                ),
                dict(
                    role="assistant",
                    content="Understood, Rina. Your destination is Osaka.",
                ),
                dict(
                    role="user",
                    content="What are my name and destination? Return only JSON with keys name and destination.",
                ),
            ],
        )
    )
    filler = [
        f"Record {i:04}: project P{i:04} is assigned to team {i % 17}; capacity is {10 + i % 83} units; region is zone {i % 9}; status is active."
        for i in range(220)
    ]
    needle = "IMPORTANT RECORD: Project Juniper has recovery code K4T9-W2MN. This is the authoritative recovery code for Project Juniper."
    for location, index in [("start", 0), ("middle", 110), ("end", 220)]:
        records = filler.copy()
        records.insert(index, needle)
        prompt = (
            "Read these records and answer using only their contents.\n\n"
            + "\n".join(records)
            + "\n\nWhat is the recovery code for Project Juniper? Reply with only the code."
        )
        rows.append(
            case("long_" + location, prompt, "exact", "K4T9-W2MN", "long_context")
        )
    return rows


def strip_fence(text):
    match = re.fullmatch(
        r"```(?:sql|python)?\s*\n?(.*?)\n?```", text.strip(), flags=re.S | re.I
    )
    return match.group(1).strip() if match else text.strip()


def grade(case, text):
    check = case["check"]
    expected = case["expected"]
    value = text.strip()
    if check == "manual":
        return dict(automatic=False, needs_review=True)
    if check == "exact":
        passed = value == expected
    elif check == "money":
        passed = value in {
            str(expected),
            f"${expected}",
            f"{expected}.00",
            f"${expected}.00",
        }
    elif check == "json":
        try:
            passed = json.loads(value) == expected
        except json.JSONDecodeError:
            passed = False
    elif check == "list":
        passed = [x.strip() for x in value.split(",")] == expected
    elif check == "sql":
        db = sqlite3.connect(":memory:")
        try:
            db.execute(
                "CREATE TABLE orders(customer_id INTEGER,status TEXT,amount INTEGER)"
            )
            db.executemany(
                "INSERT INTO orders VALUES (?,?,?)",
                [
                    (1, "paid", 20),
                    (1, "paid", 30),
                    (1, "pending", 100),
                    (2, "paid", 15),
                    (2, "paid", 10),
                    (3, "cancelled", 99),
                ],
            )
            allowed = {
                sqlite3.SQLITE_SELECT,
                sqlite3.SQLITE_READ,
                sqlite3.SQLITE_FUNCTION,
            }
            db.set_authorizer(
                lambda action, *args: (
                    sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY
                )
            )
            sql = strip_fence(value)
            result = db.execute(sql).fetchall()
            correct = result == [(1, 50), (2, 25)]
            return dict(
                automatic=True,
                pass_strict=correct and "```" not in value,
                query_correct=correct,
                format_correct="```" not in value,
                result=result,
            )
        except sqlite3.Error as exc:
            return dict(
                automatic=True,
                pass_strict=False,
                query_correct=False,
                error=str(exc),
                format_correct="```" not in value,
            )
        finally:
            db.close()
    return dict(automatic=True, pass_strict=passed)


def prompt_text(case):
    from app.prompting import messages_to_lfm3_prompt
    from app.types import ChatMessage

    return messages_to_lfm3_prompt(
        [ChatMessage(**m) for m in case["messages"]], enable_thinking=False
    )


def metadata(mode):
    import torch
    import transformers

    return dict(
        mode=mode,
        checkpoint_revision=REVISION,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        transformers=transformers.__version__,
        started_utc=datetime.now(timezone.utc).isoformat(),
        max_new_tokens=256,
    )


async def native(mode):
    import torch
    from inference import LLMEngine, EngineConfig, GenerateRequest, SamplingConfig
    from runtime import Lfm3Runtime
    from transformers import AutoTokenizer

    data = cases()
    optimized = mode == "optimized_greedy"
    runtime = Lfm3Runtime(device=torch.device("cuda"), compile_model=optimized)
    if mode == "default_sampling":
        sampling = SamplingConfig(seed=42)
    elif mode == "recommended_sampling":
        sampling = SamplingConfig(
            temperature=0.1, top_p=1, top_k=50, repetition_penalty=1.1, seed=42
        )
    else:
        sampling = SamplingConfig(temperature=0, top_p=1, top_k=0, repetition_penalty=1)
    engine = LLMEngine(
        runtime,
        EngineConfig(
            use_instruct_model=False,
            max_new_tokens=256,
            max_batch_tokens=2048,
            max_kv_cache_tokens=32768,
            max_concurrent_requests=4,
            decode_batch_size=4,
            enable_prefill_batching=optimized,
            sampling=sampling,
        ),
    )
    official = AutoTokenizer.from_pretrained(
        "LiquidAI/LFM3-1.7B",
        revision=REVISION,
        local_files_only=True,
        trust_remote_code=True,
    )
    report = metadata(mode)
    report["sampling"] = dict(vars(sampling))
    report["rows"] = []

    async def run_one(case):
        prompt = prompt_text(case)
        expected = official.apply_chat_template(
            case["messages"],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        result = await engine.generate(
            GenerateRequest(case["id"], prompt, max_new_tokens=256)
        )
        if result.error_message:
            raise RuntimeError(result.error_message)
        row = dict(
            id=case["id"],
            category=case["category"],
            prompt_tokens=len(runtime.tokenizer.encode(prompt)),
            messages=case["messages"],
            template_matches_official=prompt == expected,
            text=result.text,
            token_ids=result.token_ids,
            stop_reason=result.stop_reason,
            latency_s=result.total_latency_s,
            grade=grade(case, result.text),
        )
        print(
            json.dumps(
                dict(
                    mode=mode,
                    id=row["id"],
                    tokens=len(result.token_ids),
                    stop=row["stop_reason"],
                    grade=row["grade"],
                    text=row["text"],
                )
            ),
            flush=True,
        )
        return row

    try:
        if optimized:
            for offset in range(0, len(data), 4):
                report["rows"].extend(
                    await asyncio.gather(
                        *(run_one(c) for c in data[offset : offset + 4])
                    )
                )
        else:
            for case in data:
                report["rows"].append(await run_one(case))
    finally:
        await engine.shutdown()
    OUT.mkdir(exist_ok=True)
    (OUT / f"{mode}.json").write_text(json.dumps(report, indent=2) + "\n")


def reference():
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from safetensors.torch import load_file

    identifier = "LiquidAI/LFM3-1.7B"
    config = AutoConfig.from_pretrained(
        identifier, revision=REVISION, trust_remote_code=True, local_files_only=True
    )
    model_cls = get_class_from_dynamic_module(
        "modeling_lfm3.Lfm3ForCausalLM",
        identifier,
        revision=REVISION,
        local_files_only=True,
    )
    model = model_cls(config).to(dtype=torch.bfloat16)
    state = load_file(str(ROOT / "LFM3-1.7B/model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected and all(k.endswith("lm_head.weight") for k in missing)
    del state
    model = model.to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(
        identifier, revision=REVISION, trust_remote_code=True, local_files_only=True
    )
    report = metadata("reference_greedy")
    report["rows"] = []
    report["attention_implementation"] = model.config._attn_implementation
    with torch.inference_mode():
        for case in cases():
            prompt = prompt_text(case)
            inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(
                "cuda"
            )
            started = time.perf_counter()
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=None,
                top_k=None,
                top_p=None,
                repetition_penalty=1.0,
                eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"),
                pad_token_id=tok.convert_tokens_to_ids("<|pad|>"),
                use_cache=True,
            )
            ids = output[0, inputs["input_ids"].shape[1] :].tolist()
            text = tok.decode(ids, skip_special_tokens=True)
            row = dict(
                id=case["id"],
                category=case["category"],
                prompt_tokens=inputs["input_ids"].shape[1],
                text=text,
                token_ids=ids,
                latency_s=time.perf_counter() - started,
                stop_reason="eos"
                if ids[-1] == tok.convert_tokens_to_ids("<|im_end|>")
                else "max_new_tokens",
                grade=grade(case, text),
            )
            report["rows"].append(row)
            print(
                json.dumps(
                    dict(
                        mode="reference_greedy",
                        id=case["id"],
                        tokens=len(ids),
                        grade=row["grade"],
                        text=text,
                    )
                ),
                flush=True,
            )
    (OUT / "reference_greedy.json").write_text(json.dumps(report, indent=2) + "\n")


def parent():
    OUT.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", LFM3_FUSED_PAGED_ATTENTION="0")
    for mode in (
        "eager_greedy",
        "default_sampling",
        "recommended_sampling",
        "optimized_greedy",
        "reference_greedy",
    ):
        subprocess.run(
            [sys.executable, __file__, "--mode", mode], cwd=ROOT, env=env, check=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "eager_greedy",
            "default_sampling",
            "recommended_sampling",
            "optimized_greedy",
            "reference_greedy",
        ),
    )
    args = parser.parse_args()
    if args.mode == "reference_greedy":
        reference()
    elif args.mode:
        asyncio.run(native(args.mode))
    else:
        parent()
