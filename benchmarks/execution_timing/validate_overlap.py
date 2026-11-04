"""Compare real streamed outputs across optimization modes and sampling paths."""

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


async def main():
    import torch
    from inference import EngineConfig, GenerateRequest, LLMEngine, SamplingConfig
    from runtime import Lfm3Runtime

    greedy = SamplingConfig(temperature=0, top_p=1, top_k=0, repetition_penalty=1)
    runtime = Lfm3Runtime(device=torch.device("cuda"), compile_model=True)
    engine = LLMEngine(
        runtime,
        EngineConfig(
            use_instruct_model=True,
            max_new_tokens=64,
            decode_batch_size=4,
            max_concurrent_requests=4,
            max_batch_tokens=2048,
            max_kv_cache_tokens=4096,
            sampling=greedy,
            enable_prefill_batching=True,
        ),
    )
    results = {}
    original_result = engine.executor.result
    completed = {}

    def capture(request):
        result = original_result(request)
        completed[str(result.request_id)] = result
        return result

    engine.executor.result = capture
    prompts = (
        ("Reply with only OK.", 16),
        ("Explain the purpose of a KV cache in two sentences.", 64),
        ("Translate 'good morning' into French. Only give the translation.", 32),
        ("Write Python code to remove duplicates while preserving order.", 48),
    )

    async def one(index, prompt, cap, sampling):
        request = GenerateRequest(str(index), prompt, sampling, cap)
        events = [e async for e in engine.stream(request)]
        result = completed.pop(str(index))
        assert not result.error_message, result.error_message
        assert "".join(e.text for e in events) == result.text
        assert events[-1].generated_token_count == len(result.token_ids)
        return dict(
            token_ids=result.token_ids, text=result.text, stop_reason=result.stop_reason
        )

    try:
        for mode, enabled, steps in (
            ("baseline", False, 1),
            ("full", True, 1),
            ("multistep", True, 8),
        ):
            config = replace(
                engine.config,
                enable_stream_overlap=enabled,
                enable_decode_metadata_cache=enabled,
                enable_pinned_token_copy=enabled,
                enable_incremental_detokenization=enabled,
                decode_steps_per_round=steps,
            )
            engine.config = engine.executor.config = engine.scheduler.config = config
            engine.scheduler._overlap_stream_emission = enabled
            engine.executor._decode_metadata = None
            rows = []
            for sampling in (greedy, SamplingConfig(seed=42)):
                rows.extend(
                    await asyncio.gather(
                        *(
                            one(i, prompt, cap, sampling)
                            for i, (prompt, cap) in enumerate(prompts)
                        )
                    )
                )
            results[mode] = rows
            print(mode, "completed", len(rows), "streamed outputs", flush=True)
            assert engine.stats().kv_blocks_used == 0
        report = dict(
            full_matches_baseline=results["baseline"] == results["full"],
            multistep_matches_baseline=results["baseline"] == results["multistep"],
            results=results,
        )
        output = ROOT / "benchmarks/overlap_optimization/real_output_validation.json"
        output.write_text(json.dumps(report, indent=2) + "\n")
        assert report["full_matches_baseline"]
        assert report["multistep_matches_baseline"]
        print(
            "All baseline/full/multistep token IDs, text and stop reasons match.",
            flush=True,
        )
    finally:
        await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
