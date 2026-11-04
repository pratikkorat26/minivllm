"""Isolate host barriers from scheduler cohort changes using fixed-work decode."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from inference import EngineConfig, LLMEngine, SamplingConfig
    from runtime import Lfm3Runtime

    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    runtime = Lfm3Runtime(device=torch.device("cuda"), compile_model=True)
    engine = LLMEngine(runtime, EngineConfig(
        max_new_tokens=96, max_concurrent_requests=batch, decode_batch_size=batch,
        max_kv_cache_tokens=max(16384, batch * 128), max_batch_tokens=2048,
        sampling=SamplingConfig(temperature=0, top_p=1, top_k=0, repetition_penalty=1),
    ))
    executor = engine.executor

    def run(steps):
        requests = [executor.build_request(str(i), "The purpose of a KV cache is", None, 96, 0)
                    for i in range(batch)]
        for request in requests:
            executor.initialize(request)
            request.eos_token_id = None
        for start in range(0, batch, 107):
            executor.prefill_batch(requests[start:start + 107], 2048)
        torch.cuda.synchronize()
        began = time.perf_counter()
        remaining = 63
        while remaining:
            executor.config = replace(executor.config, decode_steps_per_round=min(steps, remaining))
            executor.decode(requests)
            remaining -= executor.last_decode_step_count
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - began
        ids = [r.generated_ids for r in requests]
        for request in requests:
            executor.finish(request, "max_new_tokens")
        return dict(steps=steps, elapsed_s=elapsed, tokens_per_s=batch * 63 / elapsed,
                    hash=hashlib.sha256(json.dumps(ids).encode()).hexdigest(), sample=ids[0])

    try:
        run(1)
        run(8)
        rows = [run(steps) for steps in (1, 8, 8, 1, 2, 4)]
        report = dict(kind="fixed_batch_non_speculative_decode", batch_size=batch,
                      eos_disabled_for_fixed_work=True, rows=rows)
        output = ROOT / f"benchmarks/overlap_optimization/fixed_chunks_b{batch}.json"
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        executor.capacity.clear_prefix_cache()
        engine.scheduler._worker.shutdown(wait=True)


if __name__ == "__main__":
    main()
