import io
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.benchmark_serving import request_one


def test_http_benchmark_uses_usage_rather_than_chunk_count():
    events = [
        dict(choices=[dict(delta=dict(content="several tokens"))]),
        dict(choices=[], usage=dict(completion_tokens=5)),
    ]
    body = (
        "".join("data: " + json.dumps(event) + "\n\n" for event in events)
        + "data: [DONE]\n\n"
    )
    args = SimpleNamespace(
        url="http://localhost:8000", model="LiquidAI/LFM3-1.7B", tokens=5
    )
    with patch("urllib.request.urlopen", return_value=io.BytesIO(body.encode())):
        result = request_one(args, 0)
    assert result["completion_tokens"] == 5
    assert result["content_chunks"] == 1
    assert result["tpot_ms"] is None  # One chunk cannot resolve token spacing.


def test_http_benchmark_rejects_missing_usage():
    args = SimpleNamespace(
        url="http://localhost:8000", model="LiquidAI/LFM3-1.7B", tokens=5
    )
    with patch("urllib.request.urlopen", return_value=io.BytesIO(b"data: [DONE]\n\n")):
        with pytest.raises(RuntimeError, match="usage"):
            request_one(args, 0)
