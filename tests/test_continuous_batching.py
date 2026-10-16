import asyncio
import time
import unittest

from inference import GenerateRequest, LLMEngine
from test_round_robin import FakeRuntime, config


class SlowRuntime(FakeRuntime):
    def __init__(self, sleep_s=0.01, **kwargs):
        super().__init__(**kwargs)
        model = self.model

        class SlowModel:
            context_length = model.context_length

            def init_paged_kv_caches(self, **values):
                return model.init_paged_kv_caches(**values)

            def __call__(self, *args, **values):
                time.sleep(sleep_s)
                return model(*args, **values)

        self.model = SlowModel()


class ContinuousBatchingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        if hasattr(self, "engine"):
            await self.engine.shutdown()

    async def test_public_request_api_and_no_legacy_helpers(self):
        self.engine = LLMEngine(FakeRuntime(), config())
        result = await self.engine.generate(GenerateRequest("req", "1"))
        self.assertEqual(result.token_ids, [11, 21])
        self.assertFalse(hasattr(self.engine, "generate_many"))
        self.assertFalse(hasattr(self.engine, "generate_stream_events"))

    async def test_stream_emits_text_then_done(self):
        self.engine = LLMEngine(FakeRuntime(), config())
        events = [
            event async for event in self.engine.stream(GenerateRequest("stream", "1"))
        ]
        self.assertEqual([event.kind for event in events], ["text", "text", "done"])

    async def test_queue_full_and_timeout(self):
        self.engine = LLMEngine(
            SlowRuntime(sleep_s=0.02),
            config(
                max_concurrent_requests=1, max_queue_size=1, request_timeout_s=0.001
            ),
        )
        timed_out = await self.engine.generate(
            GenerateRequest("slow", "1", max_new_tokens=8)
        )
        rejected = await self.engine.generate(GenerateRequest("second", "2"))
        self.assertEqual(timed_out.stop_reason, "timeout")
        self.assertIn(rejected.stop_reason, {"timeout", "queue_full"})

    async def test_abort_is_idempotent(self):
        self.engine = LLMEngine(SlowRuntime(sleep_s=0.02), config())
        task = asyncio.create_task(
            self.engine.generate(GenerateRequest("cancel", "1", max_new_tokens=8))
        )
        await asyncio.sleep(0)
        await self.engine.abort("cancel")
        await self.engine.abort("cancel")
        self.assertEqual((await task).stop_reason, "cancelled")

    async def test_deadline_applies_to_a_final_inflight_step(self):
        self.engine = LLMEngine(
            SlowRuntime(sleep_s=0.02), config(request_timeout_s=0.001)
        )
        result = await self.engine.generate(
            GenerateRequest("deadline", "1", max_new_tokens=1)
        )
        self.assertEqual(result.stop_reason, "timeout")
        self.assertEqual(self.engine.stats().kv_blocks_used, 0)

    async def test_cancelled_caller_releases_blocks(self):
        self.engine = LLMEngine(FakeRuntime(), config())
        task = asyncio.create_task(
            self.engine.generate(GenerateRequest("caller", "1", max_new_tokens=100))
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        online = self.engine.scheduler.known["caller"]
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        result = await asyncio.wait_for(asyncio.shield(online.result), timeout=2)
        self.assertEqual(result.stop_reason, "cancelled")
        self.assertEqual(self.engine.stats().active_requests, 0)
        self.assertEqual(self.engine.stats().kv_blocks_used, 0)

    async def test_duplicate_active_request_id_is_rejected(self):
        self.engine = LLMEngine(SlowRuntime(sleep_s=0.01), config())
        first = asyncio.create_task(
            self.engine.generate(GenerateRequest("same", "1", max_new_tokens=4))
        )
        await asyncio.sleep(0)
        with self.assertRaisesRegex(ValueError, "duplicate request_id"):
            await self.engine.generate(GenerateRequest("same", "2"))
        await first

    async def test_shutdown_completes_pending_request(self):
        self.engine = LLMEngine(SlowRuntime(sleep_s=0.01), config())
        pending = asyncio.create_task(
            self.engine.generate(GenerateRequest("pending", "1", max_new_tokens=8))
        )
        await asyncio.sleep(0)
        await self.engine.shutdown()
        self.assertEqual((await pending).stop_reason, "cancelled")

    async def test_capacity_is_reused_by_waiting_request(self):
        self.engine = LLMEngine(
            FakeRuntime(), config(max_concurrent_requests=1, num_kv_blocks=2)
        )
        first, second = await asyncio.gather(
            self.engine.generate(GenerateRequest("first", "1")),
            self.engine.generate(GenerateRequest("second", "2")),
        )
        self.assertEqual(first.stop_reason, "max_new_tokens")
        self.assertEqual(second.stop_reason, "max_new_tokens")
        self.assertEqual(self.engine.stats().kv_blocks_used, 0)

    async def test_decode_respects_batch_token_budget(self):
        self.engine = LLMEngine(FakeRuntime(), config(max_batch_tokens=1))
        await asyncio.gather(
            self.engine.generate(GenerateRequest("one", "1")),
            self.engine.generate(GenerateRequest("two", "2")),
        )
        self.assertEqual(self.engine.stats().decode_batch_size, 1)
