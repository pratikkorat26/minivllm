"""Backend-priority selection for dense SDPA."""

import torch

from inference.config import EngineConfig
from lfm3.attention import prefer_cudnn_attention


def test_disabled_preference_does_not_install_priority():
    assert prefer_cudnn_attention(False) is False


def test_preference_is_a_noop_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert prefer_cudnn_attention(True) is False


def test_engine_config_keeps_cudnn_attention_opt_in():
    assert EngineConfig().enable_cudnn_attention is False
