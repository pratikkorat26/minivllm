import unittest

from inference import GenerateRequest, LLMEngine
from test_round_robin import FakeRuntime, config


class PrefixCachingIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        if hasattr(self, "engine"): await self.engine.shutdown()
    async def test_prefix_cache_reuses_completed_prompt_blocks(self):
        self.engine = LLMEngine(FakeRuntime(strict_cache_check=False), config(enable_prefix_cache=True, max_prefix_cache_entries=4))
        await self.engine.generate(GenerateRequest("one", "1 2"))
        first = self.engine.stats()
        await self.engine.generate(GenerateRequest("two", "1 2"))
        second = self.engine.stats()
        self.assertGreater(first.kv_blocks_used, 0)
        self.assertGreaterEqual(second.prefix_cache_hits, 1)
    async def test_prefix_cache_disabled_by_default(self):
        self.engine = LLMEngine(FakeRuntime(), config())
        await self.engine.generate(GenerateRequest("one", "1"))
        self.assertFalse(self.engine.stats().prefix_cache_enabled)
