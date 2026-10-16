import json
import logging
import time
import uuid
from contextlib import aclosing
from typing import AsyncIterator, List, Optional, Tuple

from config.settings import RuntimeSettings
from inference import EngineConfig, GenerateRequest, LLMEngine, SamplingConfig
from runtime import GemmaRuntime, resolve_device

from .errors import AppError
from .metrics import ServiceMetrics
from .prompting import messages_to_gemma_prompt
from .types import (
    ChatCompletionEvent,
    ChatCompletionRequest,
    ChatCompletionResult,
    Usage,
)

LOGGER = logging.getLogger("app")
SUPPORTED_MODEL = "gemma-3-270m-it"


def _now_unix() -> int:
    return int(time.time())


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _finish_reason(stop_reason: str) -> str:
    if stop_reason == "eos":
        return "stop"
    if stop_reason in ("max_new_tokens", "context_limit", "capacity_exceeded"):
        return "length"
    return "stop"


def _first_stop_index(text: str, stop_sequences: List[str]) -> Optional[int]:
    indexes = [text.find(sequence) for sequence in stop_sequences if sequence in text]
    return min(indexes) if indexes else None


def _trim_stop(text: str, stop_sequences: Optional[List[str]]) -> Tuple[str, bool]:
    if not stop_sequences:
        return text, False
    stop_index = _first_stop_index(text, stop_sequences)
    if stop_index is None:
        return text, False
    return text[:stop_index], True


def _config_value(config, name: str, default):
    return getattr(config, name, default) if config is not None else default


class ChatCompletionService:
    def __init__(self, config=None, runtime=None, engine=None) -> None:
        self.config = config or RuntimeSettings()
        self.model_name = SUPPORTED_MODEL
        self.default_max_tokens = int(_config_value(config, "default_max_tokens", 128))
        self.max_request_tokens = int(_config_value(config, "max_request_tokens", 4096))
        self.default_sampling = SamplingConfig(
            temperature=float(_config_value(config, "temperature", 0.8)),
            top_p=float(_config_value(config, "top_p", 0.9)),
            top_k=int(_config_value(config, "top_k", 50)),
            repetition_penalty=float(_config_value(config, "repetition_penalty", 1.1)),
        )
        self.runtime = runtime or GemmaRuntime(
            choose_model=str(_config_value(config, "model_size", "270m")),
            use_instruct_model=bool(_config_value(config, "use_instruct_model", True)),
            device=resolve_device(str(_config_value(config, "device", "auto"))),
        )
        self.engine = engine or LLMEngine(
            runtime=self.runtime,
            config=EngineConfig(
                choose_model=str(_config_value(config, "model_size", "270m")),
                use_instruct_model=False,
                max_new_tokens=self.default_max_tokens,
                sampling=self.default_sampling,
                decode_batch_size=int(_config_value(config, "decode_batch_size", 4)),
                decode_selection_window=int(
                    _config_value(config, "decode_selection_window", 8)
                ),
                max_queue_size=int(_config_value(config, "max_queue_size", 128)),
                max_concurrent_requests=int(
                    _config_value(config, "max_concurrent_requests", 16)
                ),
                max_batch_tokens=int(_config_value(config, "max_batch_tokens", 256)),
                request_timeout_s=_config_value(config, "request_timeout_s", None),
                max_kv_cache_tokens=int(
                    _config_value(config, "max_kv_cache_tokens", 32_768)
                ),
                kv_block_size=int(_config_value(config, "kv_block_size", 16)),
                num_kv_blocks=_config_value(config, "num_kv_blocks", None),
                prefill_chunk_size=_config_value(config, "prefill_chunk_size", None),
                enable_prefix_cache=bool(
                    _config_value(config, "enable_prefix_cache", False)
                ),
                max_prefix_cache_entries=int(
                    _config_value(config, "max_prefix_cache_entries", 64)
                ),
            ),
        )
        self.metrics = ServiceMetrics(model=self.model_name)

    def _validate_request(self, request: ChatCompletionRequest) -> None:
        if (
            request.max_tokens is not None
            and request.max_tokens > self.max_request_tokens
        ):
            raise AppError(
                f"max_tokens must be <= {self.max_request_tokens}",
                error_type="invalid_request_error",
                code="max_tokens_exceeded",
            )

    def _raise_for_result_error(self, result) -> None:
        if result.error_message is None:
            return
        if result.stop_reason in ("capacity_exceeded", "queue_full"):
            raise AppError(
                result.error_message,
                error_type="rate_limit_error",
                code=result.stop_reason,
            )
        if result.stop_reason == "timeout":
            raise AppError(
                "request timed out", error_type="server_error", code="timeout"
            )
        if result.stop_reason == "context_limit":
            raise AppError(
                "prompt exceeds model context length",
                error_type="invalid_request_error",
                code="context_length_exceeded",
            )
        raise AppError(
            "model execution failed", error_type="server_error", code="runtime_error"
        )

    def list_models(self) -> dict:
        return {
            "object": "list",
            "data": [
                {
                    "id": self.model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                }
            ],
        }

    def stats(self) -> dict:
        snapshot = self.metrics.snapshot()
        if hasattr(self.engine, "stats"):
            engine_stats = self.engine.stats()
            snapshot["scheduler"] = {
                "queue_size": engine_stats.queued_requests,
                "active_requests": engine_stats.active_requests,
                "scheduled_tokens": engine_stats.scheduled_tokens,
                "live_kv_tokens": engine_stats.live_kv_tokens,
                "kv_blocks_total": engine_stats.kv_blocks_total,
                "kv_blocks_used": engine_stats.kv_blocks_used,
                "prefill_batch_size": engine_stats.prefill_batch_size,
                "decode_batch_size": engine_stats.decode_batch_size,
                "queue_wait_ms": engine_stats.queue_wait_ms,
                "prefill_latency_ms": engine_stats.prefill_latency_ms,
                "decode_step_latency_ms": engine_stats.decode_step_latency_ms,
                "completed_requests": engine_stats.completed_requests,
                "cancelled_requests": engine_stats.cancelled_requests,
                "failed_requests": engine_stats.failed_requests,
                "prefix_cache_enabled": engine_stats.prefix_cache_enabled,
                "prefix_cache_entries": engine_stats.prefix_cache_entries,
                "prefix_cache_hits": engine_stats.prefix_cache_hits,
                "prefix_cache_misses": engine_stats.prefix_cache_misses,
            }
        return snapshot

    def metrics_text(self) -> str:
        snapshot = self.metrics.snapshot()
        model = snapshot["model"]
        lines = [
            "# HELP gemma_requests_total Total chat completion requests.",
            "# TYPE gemma_requests_total counter",
            f'gemma_requests_total{{model="{model}"}} {snapshot["requests_total"]}',
            f'gemma_requests_failed_total{{model="{model}"}} {snapshot["requests_failed"]}',
            f'gemma_prompt_tokens_total{{model="{model}"}} {snapshot["tokens_prompt_total"]}',
            f'gemma_completion_tokens_total{{model="{model}"}} {snapshot["tokens_completion_total"]}',
            f'gemma_generation_last_latency_seconds{{model="{model}"}} {snapshot["latency_s"]["last"]}',
        ]
        if hasattr(self.engine, "stats"):
            engine_stats = self.engine.stats()
            lines.extend(
                [
                    f'gemma_kv_blocks_total{{model="{model}"}} {engine_stats.kv_blocks_total}',
                    f'gemma_kv_blocks_used{{model="{model}"}} {engine_stats.kv_blocks_used}',
                    f'gemma_scheduler_queue_size{{model="{model}"}} {engine_stats.queued_requests}',
                    f'gemma_scheduler_active_requests{{model="{model}"}} {engine_stats.active_requests}',
                    f'gemma_scheduler_prefill_batch_size{{model="{model}"}} {engine_stats.prefill_batch_size}',
                    f'gemma_scheduler_decode_batch_size{{model="{model}"}} {engine_stats.decode_batch_size}',
                    f'gemma_scheduler_queue_wait_ms{{model="{model}"}} {engine_stats.queue_wait_ms}',
                    f'gemma_scheduler_prefill_latency_ms{{model="{model}"}} {engine_stats.prefill_latency_ms}',
                    f'gemma_scheduler_decode_step_latency_ms{{model="{model}"}} {engine_stats.decode_step_latency_ms}',
                    f'gemma_scheduler_completed_requests{{model="{model}"}} {engine_stats.completed_requests}',
                    f'gemma_scheduler_cancelled_requests{{model="{model}"}} {engine_stats.cancelled_requests}',
                    f'gemma_scheduler_failed_requests{{model="{model}"}} {engine_stats.failed_requests}',
                    f'gemma_prefix_cache_enabled{{model="{model}"}} {int(engine_stats.prefix_cache_enabled)}',
                    f'gemma_prefix_cache_entries{{model="{model}"}} {engine_stats.prefix_cache_entries}',
                    f'gemma_prefix_cache_hits{{model="{model}"}} {engine_stats.prefix_cache_hits}',
                    f'gemma_prefix_cache_misses{{model="{model}"}} {engine_stats.prefix_cache_misses}',
                ]
            )
        lines.append("")
        return "\n".join(lines)

    async def shutdown(self) -> None:
        await self.engine.shutdown()

    def _log_completion(
        self, *, usage: Usage, finish_reason: str, result, stream: bool
    ) -> None:
        LOGGER.info(
            json.dumps(
                {
                    "event": "chat_completion",
                    "model": self.model_name,
                    "stream": stream,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "finish_reason": finish_reason,
                    "queue_wait_s": getattr(result, "queue_wait_s", 0.0),
                    "prefill_s": getattr(result, "prefill_s", 0.0),
                    "decode_s": getattr(result, "decode_s", 0.0),
                    "total_latency_s": getattr(result, "total_latency_s", 0.0),
                    "model_tokens_per_s": getattr(result, "model_tokens_per_s", 0.0),
                }
            )
        )

    def _usage(self, prompt: str, completion_tokens: int) -> Usage:
        prompt_tokens = len(self.runtime.tokenizer.encode(prompt))
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    async def create(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        self._validate_request(request)
        prompt = messages_to_gemma_prompt(request.messages)
        result = await self.engine.generate(
            GenerateRequest(
                request_id=_completion_id(),
                prompt=prompt,
                sampling=request.sampling,
                max_new_tokens=request.max_tokens,
            )
        )
        self._raise_for_result_error(result)

        text, stopped_by_sequence = _trim_stop(result.text, request.stop)
        finish_reason = (
            "stop" if stopped_by_sequence else _finish_reason(result.stop_reason)
        )
        usage = self._usage(prompt, completion_tokens=len(result.token_ids))
        self.metrics.record_generation(usage=usage, result=result, stream=False)
        self._log_completion(
            usage=usage, finish_reason=finish_reason, result=result, stream=False
        )
        return ChatCompletionResult(
            request_id=str(result.request_id),
            model=self.model_name,
            content=text,
            finish_reason=finish_reason,
            usage=usage,
            created=_now_unix(),
        )

    async def stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionEvent]:
        self._validate_request(request)
        completion_id = _completion_id()
        created = _now_unix()
        started = time.perf_counter()
        prompt = messages_to_gemma_prompt(request.messages)

        completion_tokens = 0
        accumulated_text = ""
        emitted_chars = 0
        final_stop_reason = "eos"
        failed = False
        yield ChatCompletionEvent(
            request_id=completion_id,
            model=self.model_name,
            created=created,
            role="assistant",
        )
        try:
            async with aclosing(
                self.engine.stream(
                    GenerateRequest(
                        request_id=completion_id,
                        prompt=prompt,
                        sampling=request.sampling,
                        max_new_tokens=request.max_tokens,
                    )
                )
            ) as events:
                async for event in events:
                    if event.kind == "text":
                        if not event.text:
                            continue
                        completion_tokens = event.generated_token_count or (
                            completion_tokens + 1
                        )
                        accumulated_text += event.text
                        stop_index = _first_stop_index(
                            accumulated_text, request.stop or []
                        )
                        stopped_by_sequence = stop_index is not None
                        safe_end = (
                            len(accumulated_text) if stop_index is None else stop_index
                        )
                        if not stopped_by_sequence:
                            # Hold a suffix that could become a split stop string.
                            hold = max(
                                (
                                    length
                                    for stop in request.stop or []
                                    for length in range(1, len(stop))
                                    if accumulated_text.endswith(stop[:length])
                                ),
                                default=0,
                            )
                            safe_end -= hold
                        text = accumulated_text[emitted_chars:safe_end]
                        emitted_chars = safe_end
                        if stopped_by_sequence:
                            final_stop_reason = "stop"
                        if text:
                            yield ChatCompletionEvent(
                                request_id=completion_id,
                                model=self.model_name,
                                created=created,
                                content=text,
                            )
                        if stopped_by_sequence:
                            await self.engine.abort(completion_id)
                            break
                        continue

                    completion_tokens = event.generated_token_count
                    final_stop_reason = event.stop_reason or "eos"
                    if event.error_message is not None:
                        failed = True
                        yield ChatCompletionEvent(
                            request_id=completion_id,
                            model=self.model_name,
                            created=created,
                            error={
                                "message": "model execution failed",
                                "type": "server_error",
                                "code": event.stop_reason or "runtime_error",
                            },
                        )
                        break
        except Exception:
            failed = True
            LOGGER.exception("streaming chat completion failed")
            yield ChatCompletionEvent(
                request_id=completion_id,
                model=self.model_name,
                created=created,
                error={
                    "message": "model execution failed",
                    "type": "server_error",
                    "code": "runtime_error",
                },
            )

        if (
            not failed
            and final_stop_reason != "stop"
            and emitted_chars < len(accumulated_text)
        ):
            yield ChatCompletionEvent(
                request_id=completion_id,
                model=self.model_name,
                created=created,
                content=accumulated_text[emitted_chars:],
            )

        usage = self._usage(prompt, completion_tokens=completion_tokens)
        elapsed_s = time.perf_counter() - started
        self.metrics.record_stream(usage=usage, elapsed_s=elapsed_s, failed=failed)
        if request.include_usage:
            yield ChatCompletionEvent(
                request_id=completion_id,
                model=self.model_name,
                created=created,
                usage=usage,
            )

        yield ChatCompletionEvent(
            request_id=completion_id,
            model=self.model_name,
            created=created,
            finish_reason=_finish_reason(final_stop_reason),
        )
        yield ChatCompletionEvent(
            request_id=completion_id,
            model=self.model_name,
            created=created,
            done=True,
        )
