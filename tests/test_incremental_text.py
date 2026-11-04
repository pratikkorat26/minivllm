"""Incremental streaming must preserve full-prefix decoding and UTF-8 tails."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from inference import EngineConfig, SamplingConfig
from inference.executor import PagedModelExecutor
from inference.types import RequestState
from runtime.provider import Lfm3Tokenizer


@pytest.mark.skipif(
    not Path("LFM3-1.7B/tokenizer.json").exists(), reason="local tokenizer unavailable"
)
@pytest.mark.parametrize("chunk_size", [1, 4, 8])
def test_incremental_prefixes_and_final_tail_match_full_decode(chunk_size):
    tokenizer = Lfm3Tokenizer("LFM3-1.7B/tokenizer.json")
    executor = PagedModelExecutor.__new__(PagedModelExecutor)
    executor.config = EngineConfig()
    executor.runtime = SimpleNamespace(tokenizer=tokenizer)
    for text in (
        " Hello café 👩🏽‍💻 中文 日本語 مرحبا\n\n```python\nprint(42)\n```",
        "Literal replacement character: \ufffd then more text.",
        "<|im_start|>assistant\n<think>hello</think>\n<|im_end|>",
    ):
        ids = tokenizer.encode(text)
        # Also end generation inside a split multi-byte character.
        for length in range(1, len(ids) + 1):
            request = RequestState.from_prompt("x", [1], SamplingConfig(), 128, None, 0)
            request.status = "active"
            output = ""
            for start in range(0, length, chunk_size):
                request.generated_ids.extend(
                    ids[start : min(start + chunk_size, length)]
                )
                output += executor.token_text(request)
                expected = tokenizer.decode(request.generated_ids)
                if not expected.endswith("\ufffd"):
                    assert output == expected
            request.status = "finished"
            output += executor.token_text(request)
            assert output == tokenizer.decode(ids[:length])


@pytest.mark.skipif(
    not Path("LFM3-1.7B/tokenizer.json").exists(), reason="local tokenizer unavailable"
)
def test_incremental_snapshot_does_not_consume_worker_appended_suffix():
    tokenizer = Lfm3Tokenizer("LFM3-1.7B/tokenizer.json")
    executor = PagedModelExecutor.__new__(PagedModelExecutor)
    executor.config = EngineConfig()
    executor.runtime = SimpleNamespace(tokenizer=tokenizer)
    ids = tokenizer.encode("The GPU is computing while text is emitted.")
    request = RequestState.from_prompt("x", [1], SamplingConfig(), 128, None, 0)
    request.status = "finished"  # Newer worker state must not affect the old snapshot.
    request.generated_ids = ids
    output = executor.token_text(request, token_count=3, finished=False)
    assert output == tokenizer.decode(ids[:3])
    assert request.streaming_text.consumed_tokens == 3
    output += executor.token_text(request)
    assert output == tokenizer.decode(ids)
