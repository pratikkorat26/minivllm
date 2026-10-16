import asyncio
from contextlib import aclosing
import threading

import pytest

from inference import GenerateRequest, LLMEngine, SamplingConfig
from inference.scheduler import DecodeBatchSelector
from inference.types import RequestState
from test_round_robin import FakeRuntime, config


def test_selector_batches_ragged_requests_and_serves_oldest_sampling_cohort():
    from collections import deque

    states = [
        RequestState.from_prompt(
            str(i),
            [1] * (i + 1),
            SamplingConfig(temperature=0 if i == 0 else 1),
            4,
            999,
            0,
        )
        for i in range(4)
    ]
    queue = deque(states)
    selector = DecodeBatchSelector(max_batch_size=4, selection_window=4)
    assert selector.select(queue) == states[:1]
    assert selector.select(queue) == states[1:]


def test_token_budget_config_rejects_invalid_limits():
    with pytest.raises(ValueError):
        config(max_batch_tokens=0)
    with pytest.raises(ValueError):
        config(prefill_chunk_size=-1)


def test_long_prefill_does_not_starve_decode():
    async def run():
        runtime = FakeRuntime(strict_cache_check=False)
        engine = LLMEngine(runtime, config(max_batch_tokens=2, max_new_tokens=4))
        try:
            results = await asyncio.gather(
                engine.generate(GenerateRequest("short", "1")),
                engine.generate(GenerateRequest("long", " ".join(["99"] * 12))),
            )
            assert all(result.error_message is None for result in results)
            assert runtime.model.call_last_tokens[:3] == [1, 99, 11]
            assert max(runtime.model.call_seq_lens) <= 2
        finally:
            await engine.shutdown()

    asyncio.run(run())


def test_full_decode_budget_does_not_starve_prefill():
    async def run():
        runtime = FakeRuntime(strict_cache_check=False)
        engine = LLMEngine(runtime, config(max_batch_tokens=1, max_new_tokens=4))
        try:
            results = await asyncio.gather(
                engine.generate(GenerateRequest("one", "1")),
                engine.generate(GenerateRequest("two", "2 3")),
            )
            assert all(result.error_message is None for result in results)
            assert runtime.model.call_last_tokens.index(
                3
            ) < runtime.model.call_last_tokens.index(31)
        finally:
            await engine.shutdown()

    asyncio.run(run())


def test_initialization_error_releases_allocated_blocks():
    async def run():
        engine = LLMEngine(FakeRuntime(), config())
        original = engine.executor.initialize

        def fail(state):
            original(state)
            raise RuntimeError("initialization failed")

        engine.executor.initialize = fail
        try:
            result = await engine.generate(GenerateRequest("bad", "1"))
            assert result.stop_reason == "error"
            assert engine.stats().kv_blocks_used == 0
            engine.executor.initialize = original
            assert (await engine.generate(GenerateRequest("good", "2"))).token_ids == [
                12,
                22,
            ]
        finally:
            await engine.shutdown()

    asyncio.run(run())


@pytest.mark.parametrize("limit", [64, 100])
def test_slow_stream_is_bounded_and_releases_request(limit):
    async def run():
        engine = LLMEngine(FakeRuntime(), config(max_new_tokens=limit))
        try:
            online = await engine.scheduler.submit(
                GenerateRequest("slow", "1"), stream=True
            )
            result = await online.result
            assert result.error_message == "stream consumer too slow"
            assert online.events.qsize() == 64
            assert engine.stats().kv_blocks_used == 0
        finally:
            await engine.shutdown()

    asyncio.run(run())


def test_stream_close_and_shutdown_join_execution():
    async def run():
        engine = LLMEngine(
            FakeRuntime(), config(max_new_tokens=100, enable_prefix_cache=True)
        )
        async with aclosing(engine.stream(GenerateRequest("close", "1 2"))) as events:
            await anext(events)
        await engine.shutdown()
        assert engine.stats().active_requests == 0
        assert engine.stats().kv_blocks_used == 0

    asyncio.run(run())


def test_kv_admission_reserves_decode_growth():
    async def run():
        engine = LLMEngine(FakeRuntime(), config(num_kv_blocks=2, max_new_tokens=2))
        try:
            results = await asyncio.gather(
                *(engine.generate(GenerateRequest(str(i), str(i))) for i in (1, 2))
            )
            assert [r.token_ids for r in results] == [[11, 21], [12, 22]]
            assert engine.stats().kv_blocks_used == 0
        finally:
            await engine.shutdown()

    asyncio.run(run())


def test_event_loop_progresses_while_worker_executes():
    async def run():
        runtime = FakeRuntime()
        original = runtime.model
        entered, release = threading.Event(), threading.Event()

        class BlockingModel:
            context_length = original.context_length

            def init_paged_kv_caches(self, **kwargs):
                return original.init_paged_kv_caches(**kwargs)

            def __call__(self, *args, **kwargs):
                entered.set()
                assert release.wait(timeout=2), "event loop failed to release worker"
                return original(*args, **kwargs)

        runtime.model = BlockingModel()
        engine = LLMEngine(runtime, config())
        task = asyncio.create_task(engine.generate(GenerateRequest("worker", "1")))
        try:
            for _ in range(100):
                if entered.is_set():
                    break
                await asyncio.sleep(0.001)
            assert entered.is_set()
            release.set()
            assert (await task).error_message is None
        finally:
            release.set()
            await engine.shutdown()

    asyncio.run(run())
