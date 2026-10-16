import threading
from dataclasses import dataclass, field

from .types import Usage


@dataclass
class ServiceMetrics:
    model: str
    requests_total: int = 0
    requests_failed: int = 0
    tokens_prompt_total: int = 0
    tokens_completion_total: int = 0
    latency_total_s: float = 0.0
    last_latency_s: float = 0.0
    generation: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_generation(self, *, usage: Usage, result, stream: bool) -> None:
        with self._lock:
            self.requests_total += 1
            if getattr(result, "error_message", None) is not None:
                self.requests_failed += 1
            self.tokens_prompt_total += usage.prompt_tokens
            self.tokens_completion_total += usage.completion_tokens
            self.last_latency_s = float(getattr(result, "total_latency_s", 0.0))
            self.latency_total_s += self.last_latency_s
            self.generation = {
                "stream": stream,
                "last_queue_wait_s": float(getattr(result, "queue_wait_s", 0.0)),
                "last_prefill_s": float(getattr(result, "prefill_s", 0.0)),
                "last_decode_s": float(getattr(result, "decode_s", 0.0)),
                "last_total_latency_s": float(getattr(result, "total_latency_s", 0.0)),
                "last_model_tokens_per_s": float(
                    getattr(result, "model_tokens_per_s", 0.0)
                ),
                "last_prefill_steps": int(getattr(result, "prefill_steps", 0)),
                "last_decode_steps": int(getattr(result, "decode_steps", 0)),
                "last_ttft_s": getattr(result, "ttft_s", None),
                "last_tpot_s": getattr(result, "tpot_s", None),
                "last_stop_reason": getattr(result, "stop_reason", None),
            }

    def record_stream(self, *, usage: Usage, elapsed_s: float, failed: bool) -> None:
        with self._lock:
            self.requests_total += 1
            if failed:
                self.requests_failed += 1
            self.tokens_prompt_total += usage.prompt_tokens
            self.tokens_completion_total += usage.completion_tokens
            self.last_latency_s = float(elapsed_s)
            self.latency_total_s += self.last_latency_s
            self.generation = {
                "stream": True,
                "last_total_latency_s": float(elapsed_s),
            }

    def snapshot(self) -> dict:
        with self._lock:
            avg_latency = (
                self.latency_total_s / self.requests_total
                if self.requests_total
                else 0.0
            )
            return {
                "model": self.model,
                "requests_total": self.requests_total,
                "requests_failed": self.requests_failed,
                "tokens_prompt_total": self.tokens_prompt_total,
                "tokens_completion_total": self.tokens_completion_total,
                "latency_s": {
                    "last": self.last_latency_s,
                    "avg": avg_latency,
                },
                "generation": dict(self.generation),
            }
