import asyncio
import unittest
from dataclasses import dataclass
from typing import Optional

import torch

from inference import EngineConfig, LLMEngine, SamplingConfig
from lfm3.paged_kv import PagedKVCache
from inference import GenerateRequest
from inference.scheduler import DecodeBatchSelector
from inference.types import RequestState


class FakeTokenizer:
    eos_token_id = 999
    def encode(self, text): return [int(part) for part in text.split()]
    def decode(self, ids, skip_special_tokens=False): return "".join(f"<{item}>" for item in ids)


class FakePagedModel:
    def __init__(self, fail_on_token=None, *, strict_cache_check=True):
        self.context_length, self.fail_on_token, self.strict_cache_check = 1024, fail_on_token, strict_cache_check
        self.call_last_tokens, self.call_seq_lens = [], []
    def init_paged_kv_caches(self, *, num_blocks, block_size, device):
        return [PagedKVCache.empty(num_blocks=num_blocks, num_kv_groups=1, block_size=block_size, head_dim=1, device=device, dtype=torch.float32)]
    def __call__(self, input_ids, *, block_tables=None, kv_lens=None, paged_kv_caches=None, **kwargs):
        if block_tables is None or kv_lens is None or paged_kv_caches is None: raise RuntimeError("paged KV inputs are required")
        cache, batch_size, seq_len = paged_kv_caches[0], *input_ids.shape
        self.call_last_tokens.extend(int(input_ids[i, -1]) for i in range(batch_size)); self.call_seq_lens.extend([seq_len] * batch_size)
        logits = torch.full((batch_size, seq_len, 4096), -1e9)
        for row in range(batch_size):
            token = int(input_ids[row, -1])
            if token == self.fail_on_token: raise RuntimeError(f"forced failure on token {token}")
            kv_len = int(kv_lens[row])
            if self.strict_cache_check and kv_len > 0:
                previous_position = kv_len - 1
                previous_block = int(block_tables[row, previous_position // cache.block_size])
                previous_token = int(cache.k_blocks[previous_block, 0, previous_position % cache.block_size, 0])
                if previous_token != token - 10:
                    raise RuntimeError(f"KV cache leakage detected: cache={previous_token}, expected={token - 10}")
            for offset in range(seq_len):
                position = kv_len + offset; block = int(block_tables[row, position // cache.block_size])
                cache.k_blocks[block, 0, position % cache.block_size, 0] = float(int(input_ids[row, offset]))
                cache.v_blocks[block, 0, position % cache.block_size, 0] = float(int(input_ids[row, offset]))
            logits[row, -1, token + 10] = 0
        return logits


class FakeRuntime:
    def __init__(self, fail_on_token=None, *, strict_cache_check=True):
        self.device = torch.device("cpu"); self.tokenizer = FakeTokenizer(); self.model = FakePagedModel(fail_on_token, strict_cache_check=strict_cache_check)


def config(**overrides):
    values = dict(use_instruct_model=False, max_new_tokens=2, decode_batch_size=4, decode_selection_window=8, max_concurrent_requests=4, max_queue_size=8, kv_block_size=1, sampling=SamplingConfig(temperature=0, top_p=1, top_k=0, repetition_penalty=1))
    values.update(overrides); return EngineConfig(**values)


class AsyncEngineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        if hasattr(self, "engine"): await self.engine.shutdown()
    async def test_batches_concurrent_requests(self):
        self.engine = LLMEngine(FakeRuntime(), config())
        results = await asyncio.gather(*(self.engine.generate(GenerateRequest(str(i), str(i))) for i in (1, 2, 3)))
        self.assertEqual([result.token_ids for result in results], [[11, 21], [12, 22], [13, 23]])
        self.assertGreaterEqual(self.engine.stats().decode_batch_size, 3)
    async def test_stream_and_capacity_error(self):
        self.engine = LLMEngine(FakeRuntime(), config(num_kv_blocks=1))
        events = [event async for event in self.engine.stream(GenerateRequest("x", "1 2"))]
        self.assertEqual(events[-1].stop_reason, "capacity_exceeded")
    async def test_error_isolation(self):
        self.engine = LLMEngine(FakeRuntime(fail_on_token=2), config())
        good, bad = await asyncio.gather(self.engine.generate(GenerateRequest("good", "1")), self.engine.generate(GenerateRequest("bad", "2")))
        self.assertEqual(good.stop_reason, "max_new_tokens"); self.assertEqual(bad.stop_reason, "error")

    async def test_chunked_prefill_uses_multiple_model_steps(self):
        self.engine = LLMEngine(FakeRuntime(strict_cache_check=False), config(prefill_chunk_size=1))
        result = await self.engine.generate(GenerateRequest("chunked", "1 2 3"))
        self.assertEqual(result.prefill_steps, 3)
        self.assertEqual(self.engine.runtime.model.call_seq_lens[:3], [1, 1, 1])


class DecodePolicyTests(unittest.TestCase):
    def test_selects_largest_compatible_cohort(self):
        sampling = SamplingConfig(temperature=0, top_p=1, top_k=0, repetition_penalty=1)
        requests = [RequestState.from_prompt(str(i), [i], sampling, 1, 999, 0) for i in range(3)]
        for request in requests: request.all_token_ids = [1]
        selected = DecodeBatchSelector(max_batch_size=2, selection_window=3).select(__import__("collections").deque(requests))
        self.assertEqual([request.request_id for request in selected], ["0", "1"])
