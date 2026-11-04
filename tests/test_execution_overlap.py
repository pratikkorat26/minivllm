"""Exercise overlap ownership with a deterministic model and blocked worker."""

import asyncio
import threading

import pytest
import torch

from inference import GenerateRequest, LLMEngine
from test_round_robin import FakeRuntime, config


@pytest.mark.parametrize("slow_consumer", [False, True])
def test_stream_flush_never_reclaims_inflight_kv(slow_consumer):
    async def run():
        engine = LLMEngine(FakeRuntime(), config(max_new_tokens=5))
        engine.scheduler._overlap_stream_emission = True
        queued, release = threading.Event(), threading.Event()
        original_copy = engine.executor._copy_token_ids

        def copy(tokens, notify=None):
            result = original_copy(tokens, notify)
            if notify is not None:
                queued.set()
                if not release.wait(5):
                    raise RuntimeError("test did not release worker")
            return result

        engine.executor._copy_token_ids = copy
        try:
            online = await engine.scheduler.submit(
                GenerateRequest("x", "1"), stream=True
            )
            if slow_consumer:
                online.events = asyncio.Queue(maxsize=2)
            assert await asyncio.to_thread(queued.wait, 5)
            # Give the queued event-loop callback an opportunity to run.
            await asyncio.sleep(0)
            assert not online.result.done()
            assert engine.stats().kv_blocks_used > 0
            if slow_consumer:
                assert (
                    engine.scheduler._stream_errors["x"] == "stream consumer too slow"
                )
            else:
                events = [online.events.get_nowait(), online.events.get_nowait()]
                assert [e.generated_token_count for e in events] == [1, 2]
                assert "".join(e.text for e in events) == "<11><21>"
                await engine.abort("x")
            assert not online.result.done()
            release.set()
            result = await asyncio.wait_for(asyncio.shield(online.result), 5)
            assert result.stop_reason == ("error" if slow_consumer else "cancelled")
            assert engine.stats().kv_blocks_used == 0
        finally:
            release.set()
            await engine.shutdown()

    asyncio.run(run())


def test_deferred_stream_matches_serial_text_and_token_counts():
    async def run():
        engine = LLMEngine(FakeRuntime(), config(max_new_tokens=8))
        engine.scheduler._overlap_stream_emission = True
        try:
            expected = await engine.generate(GenerateRequest("plain", "1"))
            events = [e async for e in engine.stream(GenerateRequest("stream", "1"))]
            assert "".join(e.text for e in events) == expected.text
            assert [e.generated_token_count for e in events[:-1]] == list(range(1, 9))
            assert events[-1].kind == "done"
            assert not engine.scheduler._pending_stream_tokens
        finally:
            await engine.shutdown()

    asyncio.run(run())


@pytest.mark.parametrize("batch_size", [2, 3])
def test_decode_metadata_reuse_invalidation_and_lengths(batch_size):
    async def run():
        runtime = FakeRuntime()
        runtime.compile_model = True  # Exercise buckets without compiling fake kernels.
        engine = LLMEngine(runtime, config(max_new_tokens=10, max_kv_cache_tokens=64))
        executor = engine.executor
        requests = [
            executor.build_request(str(i), str(i + 1), None, 10, 0)
            for i in range(batch_size)
        ]
        try:
            for request in requests:
                executor.initialize(request)
                executor.prefill(request)
            executor.decode(requests)
            first = executor._decode_metadata
            executor.decode(requests)
            assert executor._decode_metadata is first
            padding = [0] * (len(first.lengths) - batch_size)
            assert first.lengths.tolist() == [2] * batch_size + padding
            executor.decode(list(reversed(requests)))
            second = executor._decode_metadata
            assert second is not first
            assert second.lengths.tolist() == [3] * batch_size + padding
            # An unexpected length change must force reconstruction.
            requests[0].live_kv_tokens = 1
            runtime.model.strict_cache_check = False
            executor.decode(list(reversed(requests)))
            assert executor._decode_metadata is not second
            assert (
                executor._decode_metadata.lengths.tolist()
                == [4] * (batch_size - 1) + [1] + padding
            )
        finally:
            for request in requests:
                executor.finish(request, "max_new_tokens")
            await engine.shutdown()

    asyncio.run(run())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_pinned_copy_matches_device_tokens_and_reuses_completed_storage():
    engine = LLMEngine(FakeRuntime(), config())
    executor = engine.executor
    notifications = []
    try:
        for size in (1, 257, 3):
            values = torch.arange(size, device="cuda", dtype=torch.long).reshape(-1, 1)
            ids = executor._copy_token_ids(values, lambda: notifications.append(size))
            assert ids == list(range(size))
            assert executor._token_copy_buffer.is_pinned()
        assert notifications == [1, 257, 3]
        assert executor._token_copy_buffer.numel() == 512
    finally:
        asyncio.run(engine.shutdown())


@pytest.mark.parametrize("eos", [999, 21, 31])
def test_exact_multistep_matches_single_step_and_respects_eos(eos):
    async def run():
        async def generate(steps):
            runtime = FakeRuntime()
            runtime.compile_model = True
            runtime.tokenizer.eos_token_id = eos
            engine = LLMEngine(
                runtime,
                config(
                    max_new_tokens=9,
                    decode_steps_per_round=steps,
                    max_batch_tokens=8,
                    max_kv_cache_tokens=64,
                ),
            )
            try:
                results = await asyncio.gather(
                    *(
                        engine.generate(GenerateRequest(str(i), str(i + 1)))
                        for i in range(3)
                    )
                )
                assert engine.stats().kv_blocks_used == 0
                assert engine.scheduler.last_scheduled_tokens <= 8
                return [(r.token_ids, r.text, r.stop_reason) for r in results]
            finally:
                await engine.shutdown()

        assert await generate(8) == await generate(1)

    asyncio.run(run())


def test_multistep_falls_back_for_history_dependent_sampling():
    from inference import SamplingConfig

    async def run():
        runtime = FakeRuntime()
        runtime.compile_model = True
        engine = LLMEngine(
            runtime,
            config(
                max_new_tokens=6,
                decode_steps_per_round=8,
                sampling=SamplingConfig(temperature=0, repetition_penalty=1.1),
            ),
        )
        try:
            result = await engine.generate(GenerateRequest("x", "1"))
            assert result.error_message is None
            assert engine.executor.last_decode_step_count == 1
        finally:
            await engine.shutdown()

    asyncio.run(run())
