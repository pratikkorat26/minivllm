"""Parity, API compatibility, and hybrid-state integrity gate for LFM3-1.7B."""

import asyncio
import inspect
import json
import os
import re
import sys
import unittest
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, TESTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import main as cli_main
from adapters.openai import run as openai_run
from app.prompting import messages_to_lfm3_prompt
from app.types import ChatMessage
from inference import (
    EngineConfig,
    GenerateRequest,
    GenerateResult,
    LLMEngine,
    SamplingConfig,
    StreamEvent,
)
from lfm3.model import Lfm3Model
from real_engine_test_utils import get_shared_runtime, real_engine_skip_reason
from runtime import Lfm3Runtime

PARITY_BASELINE_PATH = Path(
    os.environ.get("LFM3_PARITY_BASELINE_PATH", str(PROJECT_ROOT / "baseline_parity.json"))
)


def _greedy_sampling() -> SamplingConfig:
    return SamplingConfig(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
    )


def _build_parity_prompts() -> list[dict]:
    multi_turn_prompt = messages_to_lfm3_prompt(
        [
            ChatMessage(role="user", content="Remember the code word: BLUE-FALCON-42."),
            ChatMessage(role="assistant", content="Understood, the code word is BLUE-FALCON-42."),
            ChatMessage(role="user", content="What is the code word I just gave you? Reply concisely."),
        ],
        enable_thinking=False,
    )

    filler_lines = [
        f"Log entry {i:03d}: sensor subsystem nominal, temperature 24.5C, pressure 101.3kPa."
        for i in range(60)
    ]
    long_context_body = (
        "\n".join(filler_lines[:10])
        + "\nCritical fact: The secret vault combination is 849201.\n"
        + "\n".join(filler_lines[10:])
        + "\nQuestion: What is the secret vault combination? Answer with just the digits."
    )

    return [
        {
            "name": "short_math",
            "prompt": "What is 17 * 23? Answer with just the number.",
            "instruct": True,
            "max_new_tokens": 24,
        },
        {
            "name": "short_fact",
            "prompt": "Name the capital of France in one word.",
            "instruct": True,
            "max_new_tokens": 16,
        },
        {
            "name": "multi_turn",
            "prompt": multi_turn_prompt,
            "instruct": False,
            "max_new_tokens": 32,
        },
        {
            "name": "long_context",
            "prompt": long_context_body,
            "instruct": True,
            "max_new_tokens": 24,
        },
        {
            "name": "structured_json",
            "prompt": (
                'Return a valid JSON object with keys "name" (string "alpha") and '
                '"count" (integer 3). Output only raw JSON with no markdown.'
            ),
            "instruct": True,
            "max_new_tokens": 48,
        },
        {
            "name": "tool_calling",
            "prompt": (
                "You have a tool named get_weather(city: str). "
                'Output only a valid JSON object with keys "tool" and "arguments" '
                'to check the weather in Tokyo.'
            ),
            "instruct": True,
            "max_new_tokens": 64,
        },
    ]


def _extract_json_object(text: str) -> dict:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise AssertionError(f"No JSON object found in output: {text!r}")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise AssertionError(f"Expected JSON object, got {type(parsed).__name__}: {text!r}")
    return parsed


class TestPublicAPICompatibility(unittest.TestCase):
    """Verify CLI entry points and public engine/runtime signatures remain intact."""

    def test_main_cli_signature_and_help(self):
        sig = inspect.signature(cli_main.main)
        self.assertEqual(len(sig.parameters), 0)
        self.assertTrue(inspect.iscoroutinefunction(cli_main._main))

        old_argv = sys.argv
        try:
            sys.argv = ["main.py", "--help"]
            with self.assertRaises(SystemExit) as ctx:
                cli_main.main()
            self.assertEqual(ctx.exception.code, 0)
        finally:
            sys.argv = old_argv

    def test_openai_run_signature(self):
        sig = inspect.signature(openai_run.main)
        self.assertEqual(
            list(sig.parameters.keys()),
            ["argv", "env", "runner", "app_factory", "service_cls"],
        )

    def test_llm_engine_and_config_signatures(self):
        init_params = list(inspect.signature(LLMEngine.__init__).parameters.keys())
        self.assertEqual(init_params, ["self", "runtime", "config"])

        self.assertTrue(inspect.iscoroutinefunction(LLMEngine.generate))
        self.assertTrue(inspect.isasyncgenfunction(LLMEngine.stream))
        self.assertTrue(inspect.iscoroutinefunction(LLMEngine.abort))
        self.assertTrue(inspect.iscoroutinefunction(LLMEngine.shutdown))
        self.assertTrue(callable(LLMEngine.stats))

        runtime_params = list(inspect.signature(Lfm3Runtime.__init__).parameters.keys())
        for expected in (
            "model_id",
            "use_instruct_model",
            "device",
            "local_dir",
            "quantization",
            "compile_model",
        ):
            self.assertIn(expected, runtime_params)

        cfg = EngineConfig()
        self.assertIsInstance(cfg.sampling, SamplingConfig)
        req = GenerateRequest(request_id="r1", prompt="hello")
        self.assertEqual(req.request_id, "r1")
        res = GenerateResult(request_id="r1", text="hi", token_ids=[1], stop_reason="eos")
        self.assertEqual(res.stop_reason, "eos")
        ev = StreamEvent(kind="text", text="hi")
        self.assertEqual(ev.kind, "text")


class TestHybridConvStateIntegrity(unittest.TestCase):
    """Verify ShortConv + Attention hybrid state isolation and chunked prefill parity."""

    @staticmethod
    def _tiny_model() -> Lfm3Model:
        torch.manual_seed(42)
        config = {
            "vocab_size": 64,
            "context_length": 128,
            "emb_dim": 32,
            "n_heads": 4,
            "n_kv_groups": 2,
            "n_layers": 4,
            "hidden_dim": 64,
            "head_dim": 8,
            "norm_eps": 1e-5,
            "conv_kernel_size": 7,
            "conv_bias": False,
            "sliding_window": 16,
            "layer_types": ["conv", "sliding_attention", "conv", "full_attention"],
            "rope_theta_full": 10_000.0,
            "rope_theta_sliding": 10_000.0,
            "partial_rotary_factor_full": 0.5,
            "partial_rotary_factor_sliding": 1.0,
            "tie_word_embeddings": True,
            "dtype": torch.float32,
        }
        return Lfm3Model(config).eval()

    def test_conv_slot_isolation_and_chunked_prefill_parity(self):
        model = self._tiny_model()
        tokens = torch.randint(0, 64, (1, 16), dtype=torch.long)
        decode_tokens = torch.randint(0, 64, (1, 3), dtype=torch.long)

        # Full prefill + 3 decode steps on slot 0 (19 tokens -> 2 blocks of 16)
        kv_full = model.init_paged_kv_caches(num_blocks=4, block_size=16)
        conv_full = model.init_conv_state_caches(max_slots=4)
        bt_full = torch.tensor([[0, 1]], dtype=torch.long)
        with torch.inference_mode():
            logits_full = model(
                tokens,
                block_tables=bt_full,
                kv_lens=torch.tensor([0], dtype=torch.long),
                paged_kv_caches=kv_full,
                conv_state_caches=conv_full,
                conv_slots=torch.tensor([0], dtype=torch.long),
            )
            for step in range(3):
                logits_full = model(
                    decode_tokens[:, step : step + 1],
                    block_tables=bt_full,
                    kv_lens=torch.tensor([16 + step], dtype=torch.long),
                    paged_kv_caches=kv_full,
                    conv_state_caches=conv_full,
                    conv_slots=torch.tensor([0], dtype=torch.long),
                )

        # Chunked prefill (8 + 8) + 3 decode steps on slot 2 (with slot 1 dirty)
        kv_chunked = model.init_paged_kv_caches(num_blocks=4, block_size=16)
        conv_chunked = model.init_conv_state_caches(max_slots=4)
        for c in conv_chunked:
            c[1].fill_(999.0)
        bt_chunked = torch.tensor([[2, 3]], dtype=torch.long)
        with torch.inference_mode():
            model(
                tokens[:, :8],
                block_tables=bt_chunked,
                kv_lens=torch.tensor([0], dtype=torch.long),
                paged_kv_caches=kv_chunked,
                conv_state_caches=conv_chunked,
                conv_slots=torch.tensor([2], dtype=torch.long),
            )
            model(
                tokens[:, 8:],
                block_tables=bt_chunked,
                kv_lens=torch.tensor([8], dtype=torch.long),
                paged_kv_caches=kv_chunked,
                conv_state_caches=conv_chunked,
                conv_slots=torch.tensor([2], dtype=torch.long),
            )
            for step in range(3):
                logits_chunked = model(
                    decode_tokens[:, step : step + 1],
                    block_tables=bt_chunked,
                    kv_lens=torch.tensor([16 + step], dtype=torch.long),
                    paged_kv_caches=kv_chunked,
                    conv_state_caches=conv_chunked,
                    conv_slots=torch.tensor([2], dtype=torch.long),
                )

        torch.testing.assert_close(logits_chunked, logits_full, atol=1e-5, rtol=1e-5)
        for state_a, state_b in zip(conv_full, conv_chunked):
            torch.testing.assert_close(state_b[2], state_a[0], atol=1e-5, rtol=1e-5)
            # Dirty neighbor slot 1 must remain untouched
            self.assertTrue(torch.all(state_b[1] == 999.0).item())


@unittest.skipIf(bool(real_engine_skip_reason()), real_engine_skip_reason())
class TestLfm3RealParityGate(unittest.TestCase):
    """Real-model greedy parity, batch invariance, chunked prefill, and JSON/tool checks."""

    @classmethod
    def setUpClass(cls):
        cls.loop = asyncio.new_event_loop()
        cls.runtime = get_shared_runtime()
        cls.engine_instruct = LLMEngine(
            runtime=cls.runtime,
            config=EngineConfig(
                use_instruct_model=True,
                max_new_tokens=64,
                decode_batch_size=4,
                sampling=_greedy_sampling(),
            ),
        )
        cls.engine_raw = LLMEngine(
            runtime=cls.runtime,
            config=EngineConfig(
                use_instruct_model=False,
                max_new_tokens=64,
                decode_batch_size=4,
                sampling=_greedy_sampling(),
            ),
        )

    @classmethod
    def tearDownClass(cls):
        cls.loop.run_until_complete(cls.engine_instruct.shutdown())
        cls.loop.run_until_complete(cls.engine_raw.shutdown())
        cls.loop.close()

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    def _generate_case(self, case: dict, *, engine: LLMEngine | None = None) -> GenerateResult:
        target_engine = engine or (self.engine_instruct if case["instruct"] else self.engine_raw)
        req = GenerateRequest(
            request_id=f"parity-{case['name']}",
            prompt=case["prompt"],
            sampling=_greedy_sampling(),
            max_new_tokens=case["max_new_tokens"],
        )
        return self._run(target_engine.generate(req))

    def test_greedy_parity_against_baseline_and_sanity(self):
        cases = _build_parity_prompts()
        results = {case["name"]: self._generate_case(case) for case in cases}

        for name, res in results.items():
            self.assertIsNone(res.error_message, f"{name} failed: {res.error_message}")
            self.assertGreater(len(res.token_ids), 0, f"{name} produced 0 tokens")

        if os.environ.get("LFM3_CAPTURE_PARITY_BASELINE", "0") == "1":
            payload = {
                name: {
                    "token_ids": res.token_ids,
                    "text": res.text,
                    "stop_reason": res.stop_reason,
                }
                for name, res in results.items()
            }
            PARITY_BASELINE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        elif PARITY_BASELINE_PATH.exists():
            baseline = json.loads(PARITY_BASELINE_PATH.read_text(encoding="utf-8"))
            for name, res in results.items():
                self.assertIn(name, baseline, f"Missing {name} in {PARITY_BASELINE_PATH}")
                self.assertEqual(
                    res.token_ids,
                    baseline[name]["token_ids"],
                    f"Greedy token_ids mismatch for {name}:\ngot:      {res.text!r}\nexpected: {baseline[name]['text']!r}",
                )

        # Structured JSON sanity check
        json_obj = _extract_json_object(results["structured_json"].text)
        self.assertEqual(json_obj.get("name"), "alpha")
        self.assertEqual(json_obj.get("count"), 3)

        # Tool-calling sanity check
        tool_obj = _extract_json_object(results["tool_calling"].text)
        self.assertIn("tool", tool_obj)
        self.assertIn("arguments", tool_obj)
        self.assertIn("get_weather", str(tool_obj["tool"]))

    def test_concurrent_batch_preserves_conv_state_parity(self):
        cases = [c for c in _build_parity_prompts() if c["instruct"]][:4]
        singles = [self._generate_case(c) for c in cases]

        async def run_batch():
            return await asyncio.gather(
                *[
                    self.engine_instruct.generate(
                        GenerateRequest(
                            request_id=f"batch-{c['name']}",
                            prompt=c["prompt"],
                            sampling=_greedy_sampling(),
                            max_new_tokens=c["max_new_tokens"],
                        )
                    )
                    for c in cases
                ]
            )

        batched = self._run(run_batch())
        for case, single, batch in zip(cases, singles, batched):
            self.assertEqual(
                batch.token_ids,
                single.token_ids,
                f"Concurrent batch mismatch for {case['name']}",
            )

    def test_chunked_prefill_long_context_parity(self):
        long_case = next(c for c in _build_parity_prompts() if c["name"] == "long_context")
        unchunked = self._generate_case(long_case)

        engine_chunked = LLMEngine(
            runtime=self.runtime,
            config=EngineConfig(
                use_instruct_model=True,
                max_new_tokens=long_case["max_new_tokens"],
                decode_batch_size=4,
                prefill_chunk_size=128,
                sampling=_greedy_sampling(),
            ),
        )
        try:
            chunked = self._generate_case(long_case, engine=engine_chunked)
        finally:
            self._run(engine_chunked.shutdown())

        self.assertEqual(chunked.token_ids, unchunked.token_ids)
