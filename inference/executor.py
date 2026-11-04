"""Synchronous, single-device paged-model execution primitives.

This module deliberately contains no queueing or asyncio code.  A scheduler is
the sole caller and therefore the sole owner of model and KV-cache mutation.
"""

import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

from lfm3.tokenizer import apply_chat_template

from .config import EngineConfig, SamplingConfig
from .kv import KVBlockManager
from .sampling import apply_repetition_penalty_, sample_next_token
from .types import GenerateResult, RequestState, StreamingTextState


def _next_power_of_two(value: int) -> int:
    """Round a positive tensor dimension up to a stable graph-size bucket."""
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


@dataclass
class DecodeMetadata:
    """One bounded cache entry, owned exclusively by the execution worker."""

    signature: tuple
    previous_lengths: tuple
    tables: torch.Tensor
    lengths: torch.Tensor
    slots: Optional[torch.Tensor]


class PagedModelExecutor:
    def __init__(self, runtime: object, config: EngineConfig) -> None:
        self.runtime = runtime
        self.config = config
        self.use_cuda_graph_buckets = bool(getattr(runtime, "compile_model", False))
        self.graph_batch_capacity = (
            _next_power_of_two(min(config.decode_batch_size, config.max_batch_tokens))
            if self.use_cuda_graph_buckets
            else 0
        )

        self.supports_prefix_cache = bool(
            getattr(runtime.model, "supports_prefix_cache", True)
        )
        if config.enable_prefix_cache and not self.supports_prefix_cache:
            raise ValueError(
                "prefix caching is not supported by this model because conv "
                "layers keep recurrent state outside the paged KV cache"
            )

        self.capacity = KVBlockManager(
            max_kv_cache_tokens=config.max_kv_cache_tokens,
            block_size=config.kv_block_size,
            num_blocks=config.num_kv_blocks,
            enable_prefix_cache=config.enable_prefix_cache,
            max_prefix_cache_entries=config.max_prefix_cache_entries,
        )
        self.paged_kv_caches = runtime.model.init_paged_kv_caches(
            num_blocks=self.capacity.num_blocks + self.graph_batch_capacity,
            block_size=self.capacity.block_size,
            device=runtime.device,
        )

        self.conv_state_caches = None
        self._conv_slots: dict[object, int] = {}
        self._free_conv_slots: list[int] = []
        if hasattr(runtime.model, "init_conv_state_caches"):
            self.conv_state_caches = runtime.model.init_conv_state_caches(
                max_slots=config.max_concurrent_requests + self.graph_batch_capacity,
                device=runtime.device,
                dtype=getattr(runtime, "dtype", None),
            )
            self._free_conv_slots = list(range(config.max_concurrent_requests))

        if getattr(runtime, "compile_model", False):
            self._mark_cache_addresses_static()

        self.prefix_cache_hits = 0
        self.prefix_cache_misses = 0
        self._decode_metadata = None
        self._token_copy_buffer = None
        self._token_copy_event = None
        self.last_decode_step_count = 1

    def _mark_cache_addresses_static(self) -> None:
        """Keep mutable cache buffers at stable addresses for CUDA graph replay."""
        mark_static_address = getattr(torch._dynamo, "mark_static_address", None)
        if mark_static_address is None:
            raise RuntimeError(
                "compiled serving requires torch._dynamo.mark_static_address"
            )

        for cache in self.paged_kv_caches:
            if cache is not None:
                mark_static_address(cache.k_blocks)
                mark_static_address(cache.v_blocks)
        if self.conv_state_caches is not None:
            for state in self.conv_state_caches:
                mark_static_address(state)

    def _acquire_conv_slot(self, request: RequestState) -> None:
        if self.conv_state_caches is None:
            return
        request_id = request.request_id
        if request_id in self._conv_slots:
            return
        if not self._free_conv_slots:
            raise RuntimeError("conv state slot pool exhausted")
        slot = self._free_conv_slots.pop()
        self._conv_slots[request_id] = slot
        for cache in self.conv_state_caches:
            cache[slot].zero_()

    def _release_conv_slot(self, request: RequestState) -> None:
        if self.conv_state_caches is None:
            return
        slot = self._conv_slots.pop(request.request_id, None)
        if slot is not None:
            self._free_conv_slots.append(slot)

    def build_request(
        self,
        request_id: str,
        prompt: str,
        sampling: Optional[SamplingConfig],
        max_new_tokens: Optional[int],
        created_at_s: float,
    ) -> RequestState:
        if self.config.use_instruct_model:
            prompt = apply_chat_template(prompt)
        token_ids = self.runtime.tokenizer.encode(prompt)
        if not token_ids:
            raise ValueError("prompt must encode to at least one token")
        if max_new_tokens is not None and max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        return RequestState.from_prompt(
            request_id=request_id,
            prompt_token_ids=token_ids,
            sampling=sampling or self.config.sampling,
            max_new_tokens=self.config.max_new_tokens
            if max_new_tokens is None
            else max_new_tokens,
            eos_token_id=self.runtime.tokenizer.eos_token_id,
            created_at_s=created_at_s,
        )

    def context_limit(self) -> int:
        return int(self.runtime.model.context_length)

    def can_admit(self, request: RequestState) -> bool:
        return self.capacity.can_allocate_for(
            request, self._reservation_tokens(request)
        )

    def _reservation_tokens(self, request: RequestState) -> int:
        return min(
            self.context_limit(),
            len(request.prompt_token_ids) + max(0, request.max_new_tokens - 1),
        )

    def initialize(self, request: RequestState) -> None:
        request.status = "active"
        request.generated_ids = []
        request.emitted_text = ""
        request.streaming_text = None
        request.all_token_ids = list(request.prompt_token_ids)
        request.seen_token_ids = set(request.prompt_token_ids)
        request.block_table = []
        cached_tokens = self.capacity.init_request_from_cache(request, scope="default")
        if self.capacity.prefix_cache_enabled:
            if cached_tokens:
                self.prefix_cache_hits += 1
            else:
                self.prefix_cache_misses += 1
        if not self.capacity.ensure_capacity(
            request, self._reservation_tokens(request)
        ):
            raise RuntimeError("KV reservation failed")
        request.prompt_cursor = request.live_kv_tokens = cached_tokens
        request.stop_reason = request.error_message = None
        request.sampling_generator = self._sampling_generator(
            getattr(request.sampling, "seed", None)
        )
        self._acquire_conv_slot(request)
        request.prefill_time_s = request.decode_time_s = 0.0
        request.prefill_steps = request.decode_steps = 0
        request.current_input = self._next_prefill_chunk(request)

    def finish(
        self, request: RequestState, reason: str, error_message: Optional[str] = None
    ) -> None:
        request.status = reason if reason in ("error", "cancelled") else "finished"
        request.stop_reason, request.error_message = reason, error_message
        request.finished_at_s = time.perf_counter()
        self.capacity.release(request)
        self._release_conv_slot(request)

    def result(self, request: RequestState) -> GenerateResult:
        first = request.first_scheduled_at_s or request.created_at_s
        finished = request.finished_at_s or time.perf_counter()
        text_ids = list(request.generated_ids)
        if (
            request.stop_reason == "eos"
            and text_ids
            and text_ids[-1] == request.eos_token_id
        ):
            text_ids.pop()
        model_s = request.prefill_time_s + request.decode_time_s
        return GenerateResult(
            request_id=request.request_id,
            text=self.runtime.tokenizer.decode(text_ids) if text_ids else "",
            token_ids=list(request.generated_ids),
            stop_reason=request.stop_reason or "error",
            error_message=request.error_message,
            queue_wait_s=max(0.0, first - request.created_at_s),
            prefill_s=request.prefill_time_s,
            decode_s=request.decode_time_s,
            total_latency_s=max(0.0, finished - request.created_at_s),
            model_tokens_per_s=0.0
            if model_s <= 0
            else len(request.generated_ids) / model_s,
            prefill_steps=request.prefill_steps,
            decode_steps=request.decode_steps,
            ttft_s=None
            if request.first_token_at_s is None
            else request.first_token_at_s - request.created_at_s,
            tpot_s=None
            if len(request.generated_ids) < 2
            else (request.last_token_at_s - request.first_token_at_s)
            / (len(request.generated_ids) - 1),
        )

    def prefill(
        self, request: RequestState, token_budget: Optional[int] = None
    ) -> bool:
        """Run one prefill chunk; return true once its first token is sampled."""
        return bool(self.prefill_batch([request], token_budget))

    def prefill_batch(
        self, requests: List[RequestState], token_budget: Optional[int] = None
    ) -> List[RequestState]:
        """Run equal-length chunks with one sampling cohort and completion phase.

        The scheduler forms compatible batches. No padding is needed, so every
        row writes only real tokens into its KV pages and convolution state.
        """
        if not requests:
            return []
        chunk_budget = None if token_budget is None else token_budget // len(requests)
        if chunk_budget is not None and chunk_budget <= 0:
            raise ValueError("prefill budget must cover at least one token per row")
        for request in requests:
            if request.current_input is None:
                request.current_input = self._next_prefill_chunk(request)
            if request.current_input is not None and chunk_budget is not None:
                request.current_input = request.current_input[:, :chunk_budget]
        started = time.perf_counter()
        eligible = self._eligible(requests)
        if not eligible:
            return []
        chunk_len = int(eligible[0].current_input.shape[1])
        finishes_prompt = eligible[0].prompt_cursor + chunk_len >= len(
            eligible[0].prompt_token_ids
        )
        logits = self._forward(
            eligible,
            compute_logits=finishes_prompt,
            use_compiled=self.use_cuda_graph_buckets,
        )
        elapsed = time.perf_counter() - started
        for request in eligible:
            request.prefill_time_s += elapsed
            request.prefill_steps += 1
            request.prompt_cursor += chunk_len
        if not finishes_prompt:
            for request in eligible:
                request.current_input = self._next_prefill_chunk(request)
            return []
        for request in eligible:
            self.capacity.cache_prefix(
                "default", request.prompt_token_ids, request.block_table
            )
        tokens = self._sample(logits, eligible)
        token_ids = self._copy_token_ids(tokens)
        for request, token, token_id in zip(eligible, tokens.split(1), token_ids):
            self._record_token(request, token, token_id)
        return eligible

    def decode(
        self,
        requests: List[RequestState],
        *,
        on_device_work_queued: Optional[Callable[[], None]] = None,
        max_steps: Optional[int] = None,
    ) -> List[RequestState]:
        self.last_decode_step_count = 1
        eligible = self._eligible(requests)
        if not eligible:
            return []
        self.last_decode_step_count = self._decode_chunk_size(eligible)
        if max_steps is not None:
            self.last_decode_step_count = min(self.last_decode_step_count, max_steps)
        if self.last_decode_step_count > 1:
            return self._decode_chunk(
                eligible, self.last_decode_step_count, on_device_work_queued
            )
        started = time.perf_counter()
        logits = self._forward(
            eligible, use_compiled=self.use_cuda_graph_buckets, pad_decode_batch=True
        )
        tokens = self._sample(logits, eligible)
        token_ids = self._copy_token_ids(tokens, on_device_work_queued)
        elapsed = time.perf_counter() - started
        for request, token, token_id in zip(eligible, tokens.split(1), token_ids):
            request.decode_time_s += elapsed
            request.decode_steps += 1
            self._record_token(request, token, token_id)
        return eligible

    def _decode_chunk_size(self, requests):
        """Bound exact greedy steps by fairness, reservation and context limits."""
        if (
            self.config.decode_steps_per_round == 1
            or not self.use_cuda_graph_buckets
            or any(
                r.sampling.temperature != 0 or r.sampling.repetition_penalty != 1
                for r in requests
            )
        ):
            return 1
        return max(
            1,
            min(
                self.config.decode_steps_per_round,
                self.config.max_batch_tokens // len(requests),
                min(r.max_new_tokens - len(r.generated_ids) for r in requests),
                min(self.context_limit() - r.live_kv_tokens for r in requests),
            ),
        )

    def _decode_chunk(self, requests, steps, on_device_work_queued):
        """Chain exact GPU argmax outputs without intermediate host barriers.

        Pages remain private and reserved for every queued position. An EOS row
        may do unused work until this bounded chunk ends; only tokens through its
        first EOS are recorded. No guessed tokens or draft model are involved.
        """
        started = time.perf_counter()
        outputs, inputs = [], None
        for _ in range(steps):
            logits = self._forward(
                requests, use_compiled=True, pad_decode_batch=True, input_batch=inputs
            )
            inputs = self._sample(logits, requests)
            outputs.append(inputs)
        tokens = torch.cat(outputs, dim=1)
        ids = self._copy_token_ids(tokens, on_device_work_queued)
        elapsed = time.perf_counter() - started
        for row, (request, final_input) in enumerate(
            zip(requests, outputs[-1].split(1))
        ):
            request.decode_time_s += elapsed
            request.decode_steps += self._record_token_ids(
                request, ids[row * steps : (row + 1) * steps]
            )
            if request.status == "active":
                # Intermediate GPU views are never consumed by another forward;
                # only the final sampled tensor becomes the next chunk's input.
                request.current_input = final_input
        return requests

    def _copy_token_ids(self, tokens, on_device_work_queued=None):
        """Queue one pinned D2H copy, then let independent host work proceed.

        The worker owns this buffer until the event completes. No consumer may
        read it, or release model state, before this method returns.
        """
        if tokens.is_cuda and self.config.enable_pinned_token_copy:
            count = tokens.numel()
            if (
                self._token_copy_buffer is None
                or self._token_copy_buffer.numel() < count
            ):
                self._token_copy_buffer = torch.empty(
                    _next_power_of_two(count), dtype=tokens.dtype, pin_memory=True
                )
                self._token_copy_event = torch.cuda.Event()
            host = self._token_copy_buffer[:count]
            host.copy_(tokens.flatten(), non_blocking=True)
            self._token_copy_event.record(torch.cuda.current_stream(tokens.device))
            if on_device_work_queued is not None:
                on_device_work_queued()
            self._token_copy_event.synchronize()
            return host.tolist()
        if on_device_work_queued is not None:
            on_device_work_queued()
        return tokens.flatten().tolist()

    def token_text(
        self, request: RequestState, *, token_count: Optional[int] = None, finished=None
    ) -> str:
        if not request.generated_ids:
            return ""
        count = len(request.generated_ids) if token_count is None else token_count
        if request.generated_ids[count - 1] == request.eos_token_id:
            count -= 1
        if finished is None:
            finished = request.status != "active"
        if (
            self.config.enable_incremental_detokenization
            and request.streaming_text is None
        ):
            factory = getattr(self.runtime.tokenizer, "create_stream_decoder", None)
            decoder = factory() if factory is not None else None
            if decoder is not None:
                request.streaming_text = StreamingTextState(decoder)
        state = request.streaming_text
        if state is not None and not finished:
            # Prefixes are immutable even if the worker appends a newer token.
            # Decode only the suffix; do not copy or decode the entire history.
            suffix = request.generated_ids[state.consumed_tokens : count]
            state.pending_text += state.decode(suffix) if suffix else ""
            state.consumed_tokens = count
            if state.pending_text.endswith("\ufffd"):
                return ""
            delta, state.pending_text = state.pending_text, ""
            state.emitted_chars += len(delta)
            return delta
        # Decode once at completion to flush an unfinished UTF-8 tail. Generic
        # tokenizers without incremental support retain their existing behavior.
        text = self.runtime.tokenizer.decode(request.generated_ids[:count])
        if text.endswith("\ufffd") and not finished:
            return ""
        emitted_chars = (
            state.emitted_chars if state is not None else len(request.emitted_text)
        )
        delta = text[emitted_chars:]
        if state is not None:
            state.emitted_chars = len(text)
        request.emitted_text = text
        return delta

    def _sampling_generator(self, seed: Optional[int]):
        if seed is None:
            return None
        generator = torch.Generator(device=self.runtime.device)
        generator.manual_seed(int(seed))
        return generator

    def _next_prefill_chunk(self, request: RequestState):
        if request.prompt_cursor >= len(request.prompt_token_ids):
            return None
        size = min(
            self.config.max_batch_tokens,
            self.config.prefill_chunk_size or self.config.max_batch_tokens,
        )
        ids = request.prompt_token_ids[
            request.prompt_cursor : request.prompt_cursor + max(1, int(size))
        ]
        return (
            torch.tensor(ids, device=self.runtime.device).unsqueeze(0) if ids else None
        )

    def _eligible(self, requests: List[RequestState]) -> List[RequestState]:
        eligible = []
        for request in requests:
            if request.current_input is None:
                self.finish(request, "error", "request tensors are not initialized")
            elif len(request.all_token_ids) >= self.context_limit():
                self.finish(request, "context_limit")
            elif self.capacity.ensure_capacity(
                request, request.live_kv_tokens + int(request.current_input.shape[1])
            ):
                eligible.append(request)
            else:
                self.finish(request, "capacity_exceeded", "KV cache capacity exceeded")
        return eligible

    def _forward(
        self,
        eligible: List[RequestState],
        *,
        compute_logits: bool = True,
        use_compiled: bool = False,
        pad_decode_batch: bool = False,
        input_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        inputs = (
            input_batch
            if input_batch is not None
            else torch.cat([request.current_input for request in eligible], dim=0)
        )
        active_batch_size = len(eligible)
        padded_batch_size = active_batch_size
        if pad_decode_batch and self.use_cuda_graph_buckets:
            padded_batch_size = min(
                self.graph_batch_capacity, _next_power_of_two(active_batch_size)
            )
            if padded_batch_size > active_batch_size:
                padding = inputs.new_zeros(
                    padded_batch_size - active_batch_size, inputs.shape[1]
                )
                inputs = torch.cat((inputs, padding), dim=0)

        max_seq_len = max(r.live_kv_tokens for r in eligible) + inputs.shape[1]
        width = max(len(request.block_table) for request in eligible)
        if use_compiled:
            # Keep attention's dense gather and block-table shapes stable across
            # decode positions. Masking still uses each request's true KV length.
            max_seq_len = _next_power_of_two(max_seq_len)
            blocks_for_attention = (
                max_seq_len + self.capacity.block_size - 1
            ) // self.capacity.block_size
            width = _next_power_of_two(max(width, blocks_for_attention))
        tables, lengths, slot_tensor = self._batch_metadata(
            eligible,
            padded_batch_size,
            width,
            cache_metadata=(
                pad_decode_batch
                and use_compiled
                and self.config.enable_decode_metadata_cache
            ),
        )
        conv_kwargs = (
            {}
            if slot_tensor is None
            else {
                "conv_state_caches": self.conv_state_caches,
                "conv_slots": slot_tensor,
            }
        )
        if use_compiled:
            # RoPE buffers outlive a forward. Grow them eagerly so they never
            # retain graph-pool outputs overwritten at the next step boundary.
            for name in ("rope_sliding", "rope_full"):
                rope = getattr(self.runtime.model, name, None)
                if rope is not None:
                    rope.ensure_capacity(
                        max_seq_len,
                        device=self.runtime.device,
                        restore_cache_dtype=True,
                    )
            torch.compiler.cudagraph_mark_step_begin()
        with torch.inference_mode():
            logits = self.runtime.model(
                inputs,
                block_tables=tables,
                kv_lens=lengths,
                paged_kv_caches=self.paged_kv_caches,
                compute_logits=compute_logits,
                logits_last_only=True,
                max_seq_len=max_seq_len,
                **conv_kwargs,
            )
        if padded_batch_size > active_batch_size:
            logits = logits[:active_batch_size]
        for request in eligible:
            request.live_kv_tokens += int(request.current_input.shape[1])
        return logits

    def _batch_metadata(self, eligible, padded_batch_size, width, *, cache_metadata):
        """Validate a stable decode cohort before reusing device metadata."""
        active_batch_size = len(eligible)
        slots = None
        if self.conv_state_caches is not None:
            try:
                slots = [self._conv_slots[r.request_id] for r in eligible]
            except KeyError as exc:
                raise RuntimeError(
                    "missing conv state slot for active request"
                ) from exc
            slots.extend(
                self.config.max_concurrent_requests + lane - active_batch_size
                for lane in range(active_batch_size, padded_batch_size)
            )

        signature = (
            (
                tuple((id(r), tuple(r.block_table)) for r in eligible),
                padded_batch_size,
                width,
                None if slots is None else tuple(slots),
            )
            if cache_metadata
            else None
        )
        current_lengths = tuple(r.live_kv_tokens for r in eligible)
        cached = self._decode_metadata if cache_metadata else None
        reuse = (
            cached is not None
            and cached.signature == signature
            and current_lengths == tuple(n + 1 for n in cached.previous_lengths)
        )
        if reuse:
            # All operations use the worker's current stream: the preceding
            # forward consumes these lengths before this increment can execute.
            cached.lengths[:active_batch_size].add_(1)
            cached.previous_lengths = current_lengths
            tables, lengths, slot_tensor = cached.tables, cached.lengths, cached.slots
        else:
            tables, lengths, slot_tensor = self._make_batch_metadata(
                eligible, padded_batch_size, width, slots
            )
            if cache_metadata:
                self._decode_metadata = DecodeMetadata(
                    signature, current_lengths, tables, lengths, slot_tensor
                )
        return tables, lengths, slot_tensor

    def _make_batch_metadata(self, eligible, padded_batch_size, width, slots):
        active_batch_size = len(eligible)
        table_rows = [
            r.block_table + [-1] * (width - len(r.block_table)) for r in eligible
        ]
        for lane in range(active_batch_size, padded_batch_size):
            scratch_block = self.capacity.num_blocks + lane - active_batch_size
            table_rows.append([scratch_block] + [-1] * (width - 1))
        tables = torch.tensor(table_rows, dtype=torch.long, device=self.runtime.device)
        lengths = torch.tensor(
            [request.live_kv_tokens for request in eligible],
            dtype=torch.long,
            device=self.runtime.device,
        )
        if padded_batch_size > active_batch_size:
            padding = torch.zeros(
                padded_batch_size - active_batch_size,
                dtype=lengths.dtype,
                device=lengths.device,
            )
            lengths = torch.cat((lengths, padding))
        slot_tensor = (
            None
            if slots is None
            else torch.tensor(slots, dtype=torch.long, device=self.runtime.device)
        )
        return tables, lengths, slot_tensor

    def _sample(
        self, logits: torch.Tensor, requests: List[RequestState]
    ) -> torch.Tensor:
        # Model already returns [B, 1, V] when logits_last_only=True.
        next_logits = logits[:, -1, :]
        if requests[0].sampling.repetition_penalty > 1.0:
            next_logits = next_logits.clone()
        next_logits = apply_repetition_penalty_(
            next_logits,
            [r.seen_token_ids for r in requests],
            penalty=requests[0].sampling.repetition_penalty,
        )
        if all(r.sampling_generator is None for r in requests):
            return sample_next_token(
                next_logits,
                temperature=requests[0].sampling.temperature,
                top_p=requests[0].sampling.top_p,
                top_k=requests[0].sampling.top_k,
            )
        return torch.cat(
            [
                sample_next_token(
                    next_logits[i : i + 1],
                    temperature=r.sampling.temperature,
                    top_p=r.sampling.top_p,
                    top_k=r.sampling.top_k,
                    generator=r.sampling_generator,
                )
                for i, r in enumerate(requests)
            ]
        )

    def _record_token(
        self, request: RequestState, token: torch.Tensor, next_id: Optional[int] = None
    ) -> None:
        if next_id is None:
            next_id = int(token.item())
        self._record_token_ids(request, [next_id])
        if request.status == "active":
            request.current_input = token

    def _record_token_ids(self, request: RequestState, next_ids: List[int]) -> int:
        """Commit a row in bulk, stopping at its first EOS before releasing KV."""
        eos = request.eos_token_id
        stopped = eos is not None and eos in next_ids
        if stopped:
            next_ids = next_ids[: next_ids.index(eos) + 1]
        now = time.perf_counter()
        if request.first_token_at_s is None:
            request.first_token_at_s = now
        request.last_token_at_s = now
        request.generated_ids.extend(next_ids)
        history = next_ids[:-1] if stopped else next_ids
        request.all_token_ids.extend(history)
        request.seen_token_ids.update(history)
        if stopped:
            self.finish(request, "eos")
        elif len(request.generated_ids) >= request.max_new_tokens:
            self.finish(request, "max_new_tokens")
        return len(next_ids)
