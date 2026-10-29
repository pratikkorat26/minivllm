"""Persistent RoPE growth must happen before a captured model invocation."""

from types import SimpleNamespace

import torch

from inference.executor import PagedModelExecutor


def test_compiled_forward_prepares_rope_before_graph_step(monkeypatch):
    calls = []

    class Rope:
        def ensure_capacity(self, length, *, device):
            calls.append(("grow_rope", length))

    class Model:
        rope_sliding = Rope()
        rope_full = Rope()

        def __call__(self, inputs, **kwargs):
            calls.append(("forward", kwargs["max_seq_len"]))
            return torch.zeros(1, 1, 8)

    monkeypatch.setattr(
        torch.compiler, "cudagraph_mark_step_begin",
        lambda: calls.append(("step_boundary", None)),
    )
    executor = PagedModelExecutor.__new__(PagedModelExecutor)
    executor.runtime = SimpleNamespace(model=Model(), device=torch.device("cpu"))
    executor.capacity = SimpleNamespace(block_size=16)
    executor.conv_state_caches = None
    executor.paged_kv_caches = []
    request = SimpleNamespace(
        current_input=torch.tensor([[1]]), live_kv_tokens=2048,
        block_table=list(range(129)),
    )
    executor._forward([request], use_compiled=True)
    assert calls == [
        ("grow_rope", 4096), ("grow_rope", 4096),
        ("step_boundary", None), ("forward", 4096),
    ]
    assert request.live_kv_tokens == 2049
