"""Persistent RoPE growth must happen before a captured model invocation."""

from types import SimpleNamespace
import copy

import pytest
import torch

from inference.executor import PagedModelExecutor
from lfm3.rope import RotaryEmbedding


def test_rope_restores_cache_dtype_once_after_module_conversion():
    rope = RotaryEmbedding(16, initial_cache_len=32).to(dtype=torch.bfloat16)
    expected = copy.deepcopy(rope)
    expected._build_cache(32, device=torch.device("cpu"))
    rope.ensure_capacity(8)
    assert rope.cos_cached.dtype == torch.bfloat16
    rope.ensure_capacity(8, restore_cache_dtype=True)
    assert rope.cos_cached.dtype == rope.cache_dtype == torch.float32
    torch.testing.assert_close(rope.cos_cached, expected.cos_cached, atol=0, rtol=0)
    torch.testing.assert_close(rope.sin_cached, expected.sin_cached, atol=0, rtol=0)
    pointers = rope.cos_cached.data_ptr(), rope.sin_cached.data_ptr()
    rope.ensure_capacity(16, device=torch.device("cpu:0"))
    assert (rope.cos_cached.data_ptr(), rope.sin_cached.data_ptr()) == pointers


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_rope_cuda_alias_reuses_cache_and_preserves_rebuild_values(monkeypatch):
    device = torch.device("cuda", torch.cuda.current_device())
    rope = RotaryEmbedding(128, initial_cache_len=32).to(
        device=device, dtype=torch.bfloat16
    )
    expected = copy.deepcopy(rope)
    original = rope._build_cache
    builds = []

    def build(length, device=None):
        builds.append(length)
        return original(length, device=device)

    monkeypatch.setattr(rope, "_build_cache", build)
    rope.ensure_capacity(8, device=torch.device("cuda"), restore_cache_dtype=True)
    expected._build_cache(32, device=device)
    torch.testing.assert_close(rope.cos_cached, expected.cos_cached, atol=0, rtol=0)
    torch.testing.assert_close(rope.sin_cached, expected.sin_cached, atol=0, rtol=0)
    pointers = rope.cos_cached.data_ptr(), rope.sin_cached.data_ptr()
    for requested in (torch.device("cuda"), device, torch.device("cuda")):
        rope.ensure_capacity(16, device=requested)
        assert (rope.cos_cached.data_ptr(), rope.sin_cached.data_ptr()) == pointers
    assert builds == [32]
    rope.ensure_capacity(33, device=torch.device("cuda"))
    expected._build_cache(rope.max_seq_len_cached, device=device)
    torch.testing.assert_close(rope.cos_cached, expected.cos_cached, atol=0, rtol=0)
    torch.testing.assert_close(rope.sin_cached, expected.sin_cached, atol=0, rtol=0)
    assert len(builds) == 2


def test_compiled_forward_prepares_rope_before_graph_step(monkeypatch):
    calls = []

    class Rope:
        def ensure_capacity(self, length, *, device, restore_cache_dtype):
            assert restore_cache_dtype
            calls.append(("grow_rope", length))

    class Model:
        rope_sliding = Rope()
        rope_full = Rope()

        def __call__(self, inputs, **kwargs):
            calls.append(("forward", kwargs["max_seq_len"]))
            return torch.zeros(1, 1, 8)

    monkeypatch.setattr(
        torch.compiler,
        "cudagraph_mark_step_begin",
        lambda: calls.append(("step_boundary", None)),
    )
    executor = PagedModelExecutor.__new__(PagedModelExecutor)
    executor.runtime = SimpleNamespace(model=Model(), device=torch.device("cpu"))
    executor.capacity = SimpleNamespace(block_size=16)
    executor.conv_state_caches = None
    executor.paged_kv_caches = []
    request = SimpleNamespace(
        current_input=torch.tensor([[1]]),
        live_kv_tokens=2048,
        block_table=list(range(129)),
    )
    executor._forward([request], use_compiled=True)
    assert calls == [
        ("grow_rope", 4096),
        ("grow_rope", 4096),
        ("step_boundary", None),
        ("forward", 4096),
    ]
    assert request.live_kv_tokens == 2049
