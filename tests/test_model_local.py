"""Opt-in local base checkpoint checks; never download assets in tests."""

import asyncio
import os
from pathlib import Path

import pytest
import torch

from inference import EngineConfig, GenerateRequest, LLMEngine, SamplingConfig
from runtime import Lfm3Runtime

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LOCAL_MODEL_TESTS") != "1"
    or not Path("LFM3-1.7B/model.safetensors").exists(),
    reason="set RUN_LOCAL_MODEL_TESTS=1 with local LFM3-1.7B checkpoint",
)


@pytest.fixture(scope="module")
def runtime():
    return Lfm3Runtime(
        use_instruct_model=False,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )


def settings(**kwargs):
    return EngineConfig(
        use_instruct_model=False,
        max_new_tokens=8,
        max_kv_cache_tokens=1024,
        sampling=SamplingConfig(temperature=0, top_p=1, top_k=0, repetition_penalty=1),
        **kwargs,
    )


def test_ragged_batch_and_stream_match_serial(runtime):
    async def run():
        engine = LLMEngine(runtime, settings())
        prompts = [
            "The purpose of a KV cache is",
            "A useful engineering tradeoff is",
            "Batching is",
        ]
        try:
            serial = [
                await engine.generate(GenerateRequest(str(i), p))
                for i, p in enumerate(prompts)
            ]
            batched = await asyncio.gather(
                *(
                    engine.generate(GenerateRequest(str(i), p))
                    for i, p in enumerate(prompts)
                )
            )
            assert [r.token_ids for r in serial] == [r.token_ids for r in batched]
            assert all(r.error_message is None for r in batched)
            events = [
                event
                async for event in engine.stream(GenerateRequest("stream", prompts[0]))
            ]
            assert "".join(event.text for event in events) == serial[0].text
            assert serial[0].ttft_s > 0 and serial[0].tpot_s > 0
        finally:
            await engine.shutdown()

    asyncio.run(run())


def test_chunked_prefill_outputs_match(runtime):
    async def run():
        plain = LLMEngine(runtime, settings(kv_block_size=4))
        chunked = LLMEngine(
            runtime,
            settings(kv_block_size=4, prefill_chunk_size=2),
        )
        prompt = "The purpose of a KV cache in an inference engine is"
        try:
            expected = await plain.generate(GenerateRequest("plain", prompt))
            for i in range(4):
                result = await chunked.generate(GenerateRequest(str(i), prompt))
                assert result.token_ids == expected.token_ids
                assert result.error_message is None
        finally:
            await plain.shutdown()
            await chunked.shutdown()
        assert chunked.stats().kv_blocks_used == 0

    asyncio.run(run())


def test_batched_prefill_logits_are_close_to_serial(runtime):
    # These raw prompts have equal token lengths, including a near-tied top
    # candidate. Compare logits because BF16 GEMM shapes can change argmax ties.
    engine = LLMEngine(runtime, settings())
    requests = [
        engine.executor.build_request(str(i), prompt, None, 8, 0)
        for i, prompt in enumerate(
            (
                "Explain batching in one concise sentence.",
                "Write one line about prompt templating.",
                "Describe top-k sampling in plain words.",
            )
        )
    ]
    try:
        for request in requests:
            engine.executor.initialize(request)
        expected = [engine.executor._forward([r]).clone() for r in requests]
        for request in requests:
            engine.executor.finish(request, "max_new_tokens")
            engine.executor.initialize(request)
        actual = engine.executor._forward(requests)
        tolerance = 0.5 if runtime.dtype == torch.bfloat16 else 1e-3
        for row, reference in enumerate(expected):
            assert (actual[row : row + 1] - reference).abs().max().item() < tolerance
            chosen = actual[row, -1].argmax()
            assert (reference.max() - reference[0, -1, chosen]).item() < tolerance
    finally:
        for request in requests:
            engine.executor.finish(request, "max_new_tokens")
        asyncio.run(engine.shutdown())
