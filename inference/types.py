from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set, Union

import torch

from .config import SamplingConfig


class RequestStatus(str, Enum):
    QUEUED = "queued"
    ACTIVE = "active"
    FINISHED = "finished"
    ERROR = "error"
    CANCELLED = "cancelled"


class StopReason(str, Enum):
    EOS = "eos"
    MAX_NEW_TOKENS = "max_new_tokens"
    CONTEXT_LIMIT = "context_limit"
    CAPACITY_EXCEEDED = "capacity_exceeded"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    QUEUE_FULL = "queue_full"
    ERROR = "error"


@dataclass(frozen=True)
class GenerateRequest:
    request_id: str
    prompt: str
    sampling: Optional[SamplingConfig] = None
    max_new_tokens: Optional[int] = None
    timeout_s: Optional[float] = None


@dataclass(frozen=True)
class EngineStats:
    scheduled_tokens: int = 0
    live_kv_tokens: int = 0
    active_requests: int = 0
    queued_requests: int = 0
    kv_blocks_total: int = 0
    kv_blocks_used: int = 0
    prefill_batch_size: int = 0
    decode_batch_size: int = 0
    completed_requests: int = 0
    cancelled_requests: int = 0
    failed_requests: int = 0
    queue_wait_ms: float = 0.0
    prefill_latency_ms: float = 0.0
    decode_step_latency_ms: float = 0.0
    prefix_cache_enabled: bool = False
    prefix_cache_entries: int = 0
    prefix_cache_hits: int = 0
    prefix_cache_misses: int = 0


@dataclass
class RequestState:
    request_id: Union[int, str]
    prompt_token_ids: List[int]
    sampling: SamplingConfig
    max_new_tokens: int
    eos_token_id: Optional[int]
    generated_ids: List[int] = field(default_factory=list)
    all_token_ids: List[int] = field(default_factory=list)
    seen_token_ids: Set[int] = field(default_factory=set)
    current_input: Optional[torch.Tensor] = None
    block_table: List[int] = field(default_factory=list)
    prompt_cursor: int = 0
    live_kv_tokens: int = 0
    status: str = RequestStatus.QUEUED.value
    stop_reason: Optional[str] = None
    error_message: Optional[str] = None
    sampling_generator: Optional[torch.Generator] = None
    created_at_s: float = 0.0
    deadline_s: Optional[float] = None
    first_scheduled_at_s: Optional[float] = None
    first_token_at_s: Optional[float] = None
    last_token_at_s: Optional[float] = None
    emitted_text: str = ""
    finished_at_s: Optional[float] = None
    prefill_time_s: float = 0.0
    prefill_steps: int = 0
    decode_time_s: float = 0.0
    decode_steps: int = 0

    @classmethod
    def from_prompt(
        cls,
        request_id: Union[int, str],
        prompt_token_ids: List[int],
        sampling: SamplingConfig,
        max_new_tokens: int,
        eos_token_id: Optional[int],
        created_at_s: float,
    ) -> "RequestState":
        return cls(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            sampling=sampling,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            created_at_s=created_at_s,
        )


@dataclass
class GenerateResult:
    request_id: Union[int, str]
    text: str
    token_ids: List[int]
    stop_reason: str
    error_message: Optional[str] = None
    queue_wait_s: float = 0.0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    total_latency_s: float = 0.0
    model_tokens_per_s: float = 0.0
    prefill_steps: int = 0
    decode_steps: int = 0
    ttft_s: Optional[float] = None
    tpot_s: Optional[float] = None


@dataclass(frozen=True)
class StreamEvent:
    kind: str
    text: str = ""
    token_id: Optional[int] = None
    generated_token_count: int = 0
    stop_reason: Optional[str] = None
    error_message: Optional[str] = None


__all__ = [
    "EngineStats",
    "GenerateRequest",
    "GenerateResult",
    "RequestState",
    "RequestStatus",
    "StopReason",
    "StreamEvent",
]
