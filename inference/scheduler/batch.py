from typing import TYPE_CHECKING, Deque, List

from inference.types import RequestState

if TYPE_CHECKING:
    from .scheduler import ScheduledRequest


class PrefillBatchSelector:
    """Select compatible chunks from one prefill round, preserving queue order."""

    def __init__(self, *, max_batch_size: int):
        self.max_batch_size = max_batch_size

    def select(
        self,
        prefill_queue: "Deque[ScheduledRequest]",
        *,
        budget: int,
        remaining: int,
        enabled: bool,
    ) -> "List[ScheduledRequest]":
        online = prefill_queue.popleft()
        selected = [online]
        if online.result.done() or not enabled:
            return selected
        request = online.request
        if request.current_input is None:
            return selected
        chunk_len = min(budget, int(request.current_input.shape[1]))
        finishes = request.prompt_cursor + chunk_len >= len(request.prompt_token_ids)
        limit = min(self.max_batch_size, budget // chunk_len)
        if limit <= 1:
            return selected
        deferred = []
        for _ in range(min(remaining - 1, len(prefill_queue))):
            candidate = prefill_queue.popleft()
            other = candidate.request
            if (
                len(selected) < limit
                and not candidate.result.done()
                and other.current_input is not None
                and int(other.current_input.shape[1]) == chunk_len
                and other.sampling == request.sampling
                and (other.prompt_cursor + chunk_len >= len(other.prompt_token_ids))
                == finishes
            ):
                selected.append(candidate)
            else:
                deferred.append(candidate)
        prefill_queue.extendleft(reversed(deferred))
        return selected


class DecodeBatchSelector:
    def __init__(self, *, max_batch_size: int, selection_window: int):
        self.max_batch_size = max(1, int(max_batch_size))
        self.selection_window = max(self.max_batch_size, int(selection_window))

    def _sampling_key(self, request: RequestState):
        return (
            request.sampling.temperature,
            request.sampling.top_p,
            request.sampling.top_k,
            request.sampling.repetition_penalty,
        )

    def select(self, decode_queue: Deque[RequestState]) -> List[RequestState]:
        window: List[RequestState] = []
        while decode_queue and len(window) < self.selection_window:
            window.append(decode_queue.popleft())

        cohorts: dict[tuple, List[int]] = {}
        for idx, request in enumerate(window):
            cohort_key = self._sampling_key(request)
            cohorts.setdefault(cohort_key, []).append(idx)

        # Oldest request anchors the cohort, so a large busy cohort cannot
        # indefinitely starve different sampling parameters.
        best_indices = next(iter(cohorts.values()), [])
        selected_positions = set(best_indices[: self.max_batch_size])
        batch = [
            request for idx, request in enumerate(window) if idx in selected_positions
        ]
        remaining = [
            request
            for idx, request in enumerate(window)
            if idx not in selected_positions
        ]

        for request in reversed(remaining):
            decode_queue.appendleft(request)
        return batch
