"""Compilation settings are bounded and restored after each model call."""

import pytest
import torch

from runtime import provider


def test_compile_specialization_budget_is_scoped_to_model_call(monkeypatch):
    observed = []

    class Model(torch.nn.Module):
        def forward(self, value):
            observed.append(torch._dynamo.config.cache_size_limit)
            if value == "fail":
                raise RuntimeError("model failed")
            return value

    def compile_forward(forward, **kwargs):
        assert kwargs == {"mode": "reduce-overhead"}
        return forward

    monkeypatch.setattr(provider, "build_lfm3_1_7b", Model)
    monkeypatch.setattr(provider, "download_weights", lambda **kwargs: {})
    monkeypatch.setattr(provider, "load_weights_into_lfm3", lambda *args: None)
    monkeypatch.setattr(provider, "resolve_tokenizer_path", lambda **kwargs: "unused")
    monkeypatch.setattr(provider, "Lfm3Tokenizer", lambda path: object())
    monkeypatch.setattr(torch, "compile", compile_forward)
    original_limit = torch._dynamo.config.cache_size_limit
    runtime = provider.Lfm3Runtime(device=torch.device("cuda"), compile_model=True)
    assert runtime.model("ok") == "ok"
    assert torch._dynamo.config.cache_size_limit == original_limit
    with pytest.raises(RuntimeError, match="model failed"):
        runtime.model("fail")
    assert observed == [32, 32]
    assert torch._dynamo.config.cache_size_limit == original_limit
