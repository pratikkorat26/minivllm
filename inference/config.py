from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    seed: Optional[int] = None

    def __post_init__(self):
        if not all(
            math.isfinite(x)
            for x in (self.temperature, self.top_p, self.repetition_penalty)
        ):
            raise ValueError("sampling values must be finite")
        if (
            self.temperature < 0
            or not 0 < self.top_p <= 1
            or self.top_k < 0
            or self.repetition_penalty < 1
        ):
            raise ValueError("invalid sampling parameters")


@dataclass(frozen=True)
class EngineConfig:
    use_instruct_model: bool = True
    max_new_tokens: int = 180
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    decode_batch_size: int = 4
    decode_selection_window: int = 8
    max_queue_size: int = 128
    max_concurrent_requests: int = 16
    max_batch_tokens: int = 256
    request_timeout_s: Optional[float] = None
    max_kv_cache_tokens: int = 32_768
    kv_block_size: int = 16
    num_kv_blocks: Optional[int] = None
    prefill_chunk_size: Optional[int] = None
    enable_prefix_cache: bool = False
    max_prefix_cache_entries: int = 64

    def __post_init__(self):
        for name in (
            "decode_batch_size",
            "decode_selection_window",
            "max_queue_size",
            "max_concurrent_requests",
            "max_batch_tokens",
            "max_kv_cache_tokens",
            "kv_block_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("prefill_chunk_size", "num_kv_blocks", "request_timeout_s"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive and finite")
        if self.max_new_tokens < 0 or self.max_prefix_cache_entries < 0:
            raise ValueError("token and cache-entry limits must be nonnegative")
