from typing import Deque, List

from inference.types import RequestState


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
