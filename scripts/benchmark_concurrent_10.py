import asyncio
import sys
import time
from statistics import mean
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import EngineConfig, GenerateRequest, LLMEngine, SamplingConfig
from runtime import GemmaRuntime, get_device

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "pandas is required for dataframe benchmark output. Install with: python3 -m pip install pandas"
    ) from exc


async def main() -> None:
    runtime = GemmaRuntime(
        choose_model="270m",
        use_instruct_model=True,
        device=get_device(),
    )
    engine = LLMEngine(
        runtime=runtime,
        config=EngineConfig(
            choose_model="270m",
            use_instruct_model=True,
            max_new_tokens=32,
            sampling=SamplingConfig(
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
            ),
        ),
    )

    prompts = [
        "Explain quantum entanglement to a 12-year-old in one sentence.",
        "Write one concise SQL query to find duplicate emails in a users table.",
        "Give one practical tip for reducing anxiety before a job interview.",
        "Summarize why the Roman Empire declined in one sentence.",
        "Describe how photosynthesis works in one short sentence.",
        "Write a one-line product pitch for a low-sugar sports drink.",
        "Suggest one respectful opener for a difficult feedback conversation.",
        "Explain inflation in plain English in one sentence.",
        "Give one cybersecurity tip to avoid phishing attacks.",
        "Write one creative sentence that starts a sci-fi story on Mars.",
    ]
    start = time.perf_counter()
    results = await asyncio.gather(
        *[
            engine.generate(GenerateRequest(str(index), prompt))
            for index, prompt in enumerate(prompts)
        ]
    )
    elapsed_s = time.perf_counter() - start

    ok = 0
    rows = []
    for result in results:
        if result.error_message is None:
            ok += 1
        prompt = prompts[int(result.request_id)]
        rows.append(
            {
                "request_id": result.request_id,
                "status": result.stop_reason,
                "error": result.error_message is not None,
                "tokens": len(result.token_ids),
                "queue_wait_s": round(result.queue_wait_s, 4),
                "prefill_s": round(result.prefill_s, 4),
                "decode_s": round(result.decode_s, 4),
                "total_s": round(result.total_latency_s, 4),
                "tok_per_s": round(result.model_tokens_per_s, 2),
                "prefill_steps": result.prefill_steps,
                "decode_steps": result.decode_steps,
                "prompt": prompt,
                "text": result.text.strip(),
            }
        )

    df = pd.DataFrame(rows).sort_values("request_id").reset_index(drop=True)
    print("\nbenchmark_dataframe")
    print(df.to_string(index=False))

    successful = [r for r in results if r.error_message is None]
    if successful:
        avg_queue = mean(r.queue_wait_s for r in successful)
        avg_prefill = mean(r.prefill_s for r in successful)
        avg_decode = mean(r.decode_s for r in successful)
        avg_total = mean(r.total_latency_s for r in successful)
        avg_tps = mean(r.model_tokens_per_s for r in successful)
        print(
            "baseline_summary "
            f"avg_queue_wait_s={avg_queue:.4f} avg_prefill_s={avg_prefill:.4f} "
            f"avg_decode_s={avg_decode:.4f} avg_total_s={avg_total:.4f} avg_tok_per_s={avg_tps:.2f}"
        )

    print(f"\ncompleted={ok}/10 elapsed_s={elapsed_s:.3f}")
    await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
