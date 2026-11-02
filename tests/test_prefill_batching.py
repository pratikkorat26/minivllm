"""Equal-length prefill batching preserves budgets and request isolation."""

import asyncio

from inference import GenerateRequest, LLMEngine, SamplingConfig
from test_round_robin import FakeRuntime, config


def batched_runtime():
    runtime = FakeRuntime(strict_cache_check=False)
    model = runtime.model
    calls = []

    class RecordingModel:
        supports_batched_prefill = True
        context_length = model.context_length

        def init_paged_kv_caches(self, **kwargs):
            return model.init_paged_kv_caches(**kwargs)

        def __call__(self, inputs, **kwargs):
            calls.append((tuple(inputs.shape), kwargs["kv_lens"].tolist()))
            return model(inputs, **kwargs)

    runtime.model = RecordingModel()
    return runtime, calls


def test_equal_length_prompts_share_one_prefill_forward():
    async def run():
        runtime, calls = batched_runtime()
        engine = LLMEngine(
            runtime, config(max_batch_tokens=6, enable_prefill_batching=True)
        )
        try:
            results = await asyncio.gather(
                *(
                    engine.generate(GenerateRequest(str(i), prompt))
                    for i, prompt in enumerate(("1 2", "3 4", "5 6"))
                )
            )
            assert [r.token_ids for r in results] == [[12, 22], [14, 24], [16, 26]]
            assert calls[0] == ((3, 2), [0, 0, 0])
            assert engine.stats().prefill_batch_size == 3
            assert engine.stats().kv_blocks_used == 0
        finally:
            await engine.shutdown()

    asyncio.run(run())


def test_chunked_prefill_batches_respect_budget_and_keep_ragged_positions():
    async def run():
        runtime, calls = batched_runtime()
        engine = LLMEngine(
            runtime,
            config(
                max_batch_tokens=6,
                prefill_chunk_size=3,
                max_new_tokens=3,
                enable_prefill_batching=True,
            ),
        )
        prompts = ("1 2 3 4 5", "10 11 12 13 14 15 16 17", "20 21", "30 31 32 33")
        try:
            results = await asyncio.gather(
                *(
                    engine.generate(GenerateRequest(str(i), prompt))
                    for i, prompt in enumerate(prompts)
                )
            )
            for prompt, result in zip(prompts, results):
                last = int(prompt.split()[-1])
                assert result.error_message is None
                assert result.token_ids == [last + 10, last + 20, last + 30]
            assert any(batch > 1 and length > 1 for (batch, length), _ in calls)
            assert all(batch * length <= 6 for (batch, length), _ in calls)
            assert engine.stats().kv_blocks_used == 0
        finally:
            await engine.shutdown()

    asyncio.run(run())


def test_different_sampling_cohorts_are_not_combined_during_prefill():
    async def run():
        runtime, calls = batched_runtime()
        engine = LLMEngine(
            runtime, config(max_batch_tokens=8, enable_prefill_batching=True)
        )
        try:
            results = await asyncio.gather(
                *(
                    engine.generate(
                        GenerateRequest(
                            str(i),
                            "1 2",
                            sampling=SamplingConfig(
                                temperature=0,
                                top_p=1,
                                top_k=0,
                                repetition_penalty=1,
                                seed=i,
                            ),
                        )
                    )
                    for i in range(3)
                )
            )
            assert all(r.token_ids == [12, 22] for r in results)
            assert all(batch == 1 for (batch, length), _ in calls if length == 2)
        finally:
            await engine.shutdown()

    asyncio.run(run())


def test_model_ragged_prefill_logits_match_serial_full_sequences():
    import torch

    from lfm3.model import LFM3_CONFIG_1_7B, Lfm3Model

    torch.manual_seed(47)
    model = Lfm3Model(
        dict(
            LFM3_CONFIG_1_7B,
            vocab_size=64,
            context_length=32,
            emb_dim=32,
            hidden_dim=64,
            n_heads=4,
            n_kv_groups=2,
            head_dim=8,
            n_layers=3,
            layer_types=["conv", "sliding_attention", "full_attention"],
            conv_kernel_size=3,
            sliding_window=4,
            dtype=torch.float32,
        )
    ).eval()
    tables = torch.randperm(24).reshape(3, 8)
    prefixes = [0, 2, 4]
    sequences = [torch.randint(0, 64, (1, length + 4)) for length in prefixes]
    cache = model.init_paged_kv_caches(num_blocks=24, block_size=2)
    state = model.init_conv_state_caches(max_slots=5)
    slots = torch.tensor([3, 0, 4])
    with torch.inference_mode():
        expected = []
        for row, (prefix, ids) in enumerate(zip(prefixes, sequences)):
            reference_cache = model.init_paged_kv_caches(num_blocks=24, block_size=2)
            expected.append(
                model(
                    ids,
                    block_tables=tables[row : row + 1],
                    kv_lens=torch.tensor([0]),
                    paged_kv_caches=reference_cache,
                    logits_last_only=False,
                )[:, -4:]
            )
            if prefix:
                model(
                    ids[:, :prefix],
                    block_tables=tables[row : row + 1],
                    kv_lens=torch.tensor([0]),
                    paged_kv_caches=cache,
                    conv_state_caches=state,
                    conv_slots=slots[row : row + 1],
                )
        actual = model(
            torch.cat([ids[:, -4:] for ids in sequences]),
            block_tables=tables,
            kv_lens=torch.tensor(prefixes),
            paged_kv_caches=cache,
            conv_state_caches=state,
            conv_slots=slots,
            logits_last_only=False,
        )
    torch.testing.assert_close(actual, torch.cat(expected), atol=3e-6, rtol=3e-5)
