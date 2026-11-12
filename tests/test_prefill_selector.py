from collections import deque
from concurrent.futures import Future

import torch

from inference.config import SamplingConfig
from inference.scheduler import PrefillBatchSelector
from inference.scheduler.scheduler import ScheduledRequest
from inference.types import RequestState


def row(name, *, length=2, prompt_length=2, sampling=None, done=False):
    state = RequestState.from_prompt(
        name, list(range(prompt_length)), sampling or SamplingConfig(), 2, None, 0
    )
    state.current_input = torch.zeros((1, length), dtype=torch.long)
    future = Future()
    if done:
        future.set_result(None)
    return ScheduledRequest(request=state, stream=False, result=future)


def ids(rows):
    return [r.request_id for r in rows]


def test_prefill_selection_preserves_deferred_order_and_round_boundary():
    different_sampling = SamplingConfig(seed=7)
    queue = deque([
        row("anchor"),
        row("sampling", sampling=different_sampling),
        row("compatible"),
        row("unfinished", prompt_length=4),
        row("length", length=1),
        row("next_round"),
    ])
    selected = PrefillBatchSelector(max_batch_size=4).select(
        queue, budget=8, remaining=5, enabled=True
    )
    assert ids(selected) == ["anchor", "compatible"]
    assert ids(queue) == ["sampling", "unfinished", "length", "next_round"]


def test_prefill_selection_obeys_budget_and_batch_cap():
    for budget, cap, expected in [(4, 4, 2), (8, 2, 2), (1, 4, 1)]:
        queue = deque(row(str(i)) for i in range(4))
        selected = PrefillBatchSelector(max_batch_size=cap).select(
            queue, budget=budget, remaining=4, enabled=True
        )
        assert ids(selected) == [str(i) for i in range(expected)]
        assert ids(queue) == [str(i) for i in range(expected, 4)]


def test_prefill_selection_handles_disabled_and_completed_rows():
    selector = PrefillBatchSelector(max_batch_size=4)
    for enabled, done in [(False, False), (True, True)]:
        queue = deque([row("first", done=done), row("second")])
        assert ids(selector.select(queue, budget=4, remaining=2, enabled=enabled)) == [
            "first"
        ]
        assert ids(queue) == ["second"]

    queue = deque([row("first"), row("done", done=True), row("last")])
    assert ids(selector.select(queue, budget=6, remaining=3, enabled=True)) == [
        "first", "last"
    ]
    assert ids(queue) == ["done"]
