from dataclasses import dataclass

from inference.types import RequestState


@dataclass(frozen=True)
class AdmissionDecision:
    admitted: bool
    reason: str = ""


class FifoAdmissionPolicy:
    def __init__(self, *, max_concurrent_requests: int):
        self.max_concurrent_requests = max(1, int(max_concurrent_requests))

    def decide(
        self, *, active_requests: int, request: RequestState
    ) -> AdmissionDecision:
        if active_requests >= self.max_concurrent_requests:
            return AdmissionDecision(admitted=False, reason="max_concurrent_requests")
        if request.status != "queued":
            return AdmissionDecision(admitted=False, reason="request_not_queued")
        return AdmissionDecision(admitted=True)
