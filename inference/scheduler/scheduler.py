"""Async request lifecycle scheduler for one paged model executor."""

import asyncio
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import AsyncIterator, Deque, Dict, Optional

from inference.executor import PagedModelExecutor
from inference.types import (
    EngineStats,
    GenerateRequest,
    GenerateResult,
    RequestState,
    StopReason,
    StreamEvent,
)

from .batch import DecodeBatchSelector


@dataclass
class ScheduledRequest:
    request: RequestState
    stream: bool
    result: "asyncio.Future[GenerateResult]"
    events: "asyncio.Queue[StreamEvent]" = field(
        default_factory=lambda: asyncio.Queue(maxsize=64)
    )
    cancel_requested: bool = False

    @property
    def request_id(self) -> str:
        return str(self.request.request_id)


class AsyncScheduler:
    def __init__(self, executor: PagedModelExecutor, config) -> None:
        self.executor = executor
        self.config = config
        self.selector = DecodeBatchSelector(
            max_batch_size=min(config.decode_batch_size, config.max_batch_tokens),
            selection_window=config.decode_selection_window,
        )
        self.waiting: Deque[ScheduledRequest] = deque()
        self.prefill: Deque[ScheduledRequest] = deque()
        self.decode: Deque[ScheduledRequest] = deque()
        self.active: Dict[str, ScheduledRequest] = {}
        self.known: Dict[str, ScheduledRequest] = {}
        self._wake = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._closed = False
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
        self._prefill_turn = False
        self.last_scheduled_tokens = 0
        self.completed = self.cancelled = self.failed = 0
        self.last_prefill_batch_size = self.last_decode_batch_size = 0
        self.last_queue_wait_ms = self.last_prefill_latency_ms = (
            self.last_decode_step_latency_ms
        ) = 0.0

    async def submit(
        self, request: GenerateRequest, *, stream: bool
    ) -> ScheduledRequest:
        if self._closed:
            raise RuntimeError("inference scheduler is shut down")
        request_id = str(request.request_id)
        if request_id in self.known:
            raise ValueError(f"duplicate request_id: {request_id}")
        now = time.perf_counter()
        state = self.executor.build_request(
            request_id, request.prompt, request.sampling, request.max_new_tokens, now
        )
        timeout = (
            request.timeout_s
            if request.timeout_s is not None
            else self.config.request_timeout_s
        )
        if timeout is not None and (not 0 < float(timeout) < float("inf")):
            raise ValueError("timeout_s must be positive and finite")
        state.deadline_s = None if timeout is None else now + float(timeout)
        online = ScheduledRequest(
            request=state,
            stream=stream,
            result=asyncio.get_running_loop().create_future(),
        )
        if len(self.known) >= max(1, int(self.config.max_queue_size)):
            self._complete(
                online, StopReason.QUEUE_FULL.value, "scheduler queue is full"
            )
            return online
        self.known[request_id] = online
        self.waiting.append(online)
        self._ensure_task()
        self._wake.set()
        return online

    async def generate(self, request: GenerateRequest) -> GenerateResult:
        online = await self.submit(request, stream=False)
        try:
            return await asyncio.shield(online.result)
        finally:
            if not online.result.done():
                await self.abort(online.request_id)

    async def stream(self, request: GenerateRequest) -> AsyncIterator[StreamEvent]:
        online = await self.submit(request, stream=True)
        try:
            while True:
                event = await online.events.get()
                yield event
                if event.kind == "done":
                    return
        finally:
            if not online.result.done():
                await self.abort(online.request_id)

    async def abort(self, request_id: str) -> None:
        online = self.known.get(str(request_id))
        if online is not None and not online.result.done():
            online.cancel_requested = True
            self._wake.set()

    async def shutdown(self) -> None:
        self._closed = True
        for online in list(self.known.values()):
            online.cancel_requested = True
        self._wake.set()
        # The worker may still be writing KV. Join scheduler before reclamation.
        if self._task is not None:
            await self._task
        self.executor.capacity.clear_prefix_cache()
        self._worker.shutdown(wait=True)

    def stats(self) -> EngineStats:
        return EngineStats(
            active_requests=len(self.active),
            queued_requests=len(self.waiting),
            scheduled_tokens=self.last_scheduled_tokens,
            live_kv_tokens=sum(o.request.live_kv_tokens for o in self.active.values()),
            kv_blocks_total=self.executor.capacity.num_blocks,
            kv_blocks_used=self.executor.capacity.allocated_blocks,
            prefill_batch_size=self.last_prefill_batch_size,
            decode_batch_size=self.last_decode_batch_size,
            completed_requests=self.completed,
            cancelled_requests=self.cancelled,
            failed_requests=self.failed,
            queue_wait_ms=self.last_queue_wait_ms,
            prefill_latency_ms=self.last_prefill_latency_ms,
            decode_step_latency_ms=self.last_decode_step_latency_ms,
            prefix_cache_enabled=self.executor.capacity.prefix_cache_enabled,
            prefix_cache_entries=self.executor.capacity.prefix_cache_size,
            prefix_cache_hits=self.executor.prefix_cache_hits,
            prefix_cache_misses=self.executor.prefix_cache_misses,
        )

    def _ensure_task(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(
                self._run(), name="llm-inference-scheduler"
            )

    async def _run(self) -> None:
        while not self._closed or self._has_work():
            if not self._has_work():
                self._wake.clear()
                await self._wake.wait()
                continue
            self._expire_or_cancel()
            self.waiting = deque(o for o in self.waiting if not o.result.done())
            self._admit()
            budget = int(self.config.max_batch_tokens)
            # Decode first; when it consumes the whole budget, reserve the next
            # turn for prefill. Neither phase can starve the other.
            if self.decode and not (self.prefill and self._prefill_turn):
                budget -= await self._run_decode()
            self._prefill_turn = bool(self.prefill and budget <= 0)
            for _ in range(len(self.prefill)):
                if budget <= 0 or self._closed:
                    break
                budget -= await self._run_prefill(budget)
            self.last_scheduled_tokens = self.config.max_batch_tokens - budget
            await asyncio.sleep(0)

    def _has_work(self) -> bool:
        return bool(self.waiting or self.prefill or self.decode or self.active)

    def _admit(self) -> None:
        while self.waiting:
            online = self.waiting[0]
            request = online.request
            if len(self.active) >= self.config.max_concurrent_requests:
                break
            self.waiting.popleft()
            if request.max_new_tokens <= 0:
                self._complete(online, StopReason.MAX_NEW_TOKENS.value)
                continue
            if len(request.prompt_token_ids) >= self.executor.context_limit():
                self._complete(
                    online,
                    StopReason.CONTEXT_LIMIT.value,
                    "context length limit exceeded",
                )
                continue
            if not self.executor.can_admit(request):
                if self.active or self.prefill or self.decode:
                    self.waiting.appendleft(online)
                    break
                self._complete(
                    online,
                    StopReason.CAPACITY_EXCEEDED.value,
                    "KV cache capacity exceeded",
                )
                continue
            request.first_scheduled_at_s = time.perf_counter()
            self.last_queue_wait_ms = (
                request.first_scheduled_at_s - request.created_at_s
            ) * 1000
            try:
                self.executor.initialize(request)
            except Exception as exc:
                self._complete(online, StopReason.ERROR.value, str(exc))
                continue
            self.active[online.request_id] = online
            self.prefill.append(online)

    def _expire_or_cancel(self) -> None:
        for online in list(self.known.values()):
            if online.cancel_requested:
                self._complete(online, StopReason.CANCELLED.value)
            elif self._expired(online.request):
                self._complete(online, StopReason.TIMEOUT.value, "request timed out")

    async def _run_prefill(self, budget: int) -> int:
        online = self.prefill.popleft()
        if online.result.done():
            return 0
        before = len(online.request.generated_ids)
        started = time.perf_counter()
        try:
            count = min(budget, int(online.request.current_input.shape[1]))
            completed_prefill = await asyncio.get_running_loop().run_in_executor(
                self._worker, self.executor.prefill, online.request, budget
            )
        except Exception as exc:
            self._complete(online, StopReason.ERROR.value, str(exc))
            return 0
        self.last_prefill_batch_size, self.last_prefill_latency_ms = (
            1,
            (time.perf_counter() - started) * 1000,
        )
        self._expire_or_cancel()
        if online.result.done():
            return count
        self._emit_token(online, before)
        if online.request.status in ("finished", "error"):
            self._complete(online)
        elif completed_prefill:
            self.decode.append(online)
        else:
            self.prefill.append(online)
        return count

    async def _run_decode(self) -> int:
        by_id = {o.request_id: o for o in self.decode if not o.result.done()}
        candidates = deque(o.request for o in by_id.values())
        selected = [by_id[str(r.request_id)] for r in self.selector.select(candidates)]
        self.decode = deque(by_id[str(r.request_id)] for r in candidates)
        if not selected:
            return 0
        batch_tokens = len(selected)  # Every decode request contributes one token.
        before = {
            online.request_id: len(online.request.generated_ids) for online in selected
        }
        started = time.perf_counter()
        try:
            eligible = await asyncio.get_running_loop().run_in_executor(
                self._worker,
                self.executor.decode,
                [online.request for online in selected],
            )
        except Exception as exc:
            for online in selected:
                self._complete(online, StopReason.ERROR.value, str(exc))
            return batch_tokens
        self.last_decode_batch_size = len(eligible)
        self.last_decode_step_latency_ms = (time.perf_counter() - started) * 1000
        self._expire_or_cancel()
        for online in selected:
            if online.result.done():
                continue
            self._emit_token(online, before[online.request_id])
            if online.request.status in ("finished", "error"):
                self._complete(online)
            else:
                self.decode.append(online)
        return batch_tokens

    def _emit_token(self, online: ScheduledRequest, previous_count: int) -> None:
        request = online.request
        if online.stream and len(request.generated_ids) > previous_count:
            text = self.executor.token_text(request)
            if text:
                try:
                    # Keep a slot for the terminal event, including when the
                    # last generated token fills a slow consumer's queue.
                    if online.events.qsize() >= online.events.maxsize - 1:
                        raise asyncio.QueueFull
                    online.events.put_nowait(
                        StreamEvent(
                            kind="text",
                            text=text,
                            token_id=request.generated_ids[-1],
                            generated_token_count=len(request.generated_ids),
                        )
                    )
                except asyncio.QueueFull:
                    self._complete(
                        online, StopReason.ERROR.value, "stream consumer too slow"
                    )

    def _complete(
        self,
        online: ScheduledRequest,
        reason: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        if online.result.done():
            return
        request = online.request
        if reason is not None:
            self.executor.finish(request, reason, error)
        self.active.pop(online.request_id, None)
        self.known.pop(online.request_id, None)
        result = self.executor.result(request)
        if result.stop_reason == StopReason.CANCELLED.value:
            self.cancelled += 1
        elif request.status == "error" or result.error_message is not None:
            self.failed += 1
        else:
            self.completed += 1
        if online.events.full():
            online.events.get_nowait()
        online.events.put_nowait(
            StreamEvent(
                kind="done",
                generated_token_count=len(request.generated_ids),
                stop_reason=result.stop_reason,
                error_message=result.error_message,
            )
        )
        online.result.set_result(result)

    @staticmethod
    def _expired(request: RequestState) -> bool:
        return (
            request.deadline_s is not None and time.perf_counter() >= request.deadline_s
        )
