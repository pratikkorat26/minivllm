"""Public async inference-engine façade."""

from typing import AsyncIterator
from contextlib import aclosing

from .config import EngineConfig
from .executor import PagedModelExecutor
from .scheduler.scheduler import AsyncScheduler
from .types import EngineStats, GenerateRequest, GenerateResult, StreamEvent


class LLMEngine:
    """Single-device async request engine.

    Model execution and KV mutation are serialized by :class:`AsyncScheduler`.
    This type intentionally exposes only request-oriented operations.
    """

    def __init__(self, runtime: object, config: EngineConfig) -> None:
        self.runtime = runtime
        self.config = config
        self.executor = PagedModelExecutor(runtime, config)
        self.scheduler = AsyncScheduler(self.executor, config)

    async def generate(self, request: GenerateRequest) -> GenerateResult:
        return await self.scheduler.generate(request)

    async def stream(self, request: GenerateRequest) -> AsyncIterator[StreamEvent]:
        async with aclosing(self.scheduler.stream(request)) as events:
            async for event in events:
                yield event

    async def abort(self, request_id: str) -> None:
        await self.scheduler.abort(request_id)

    def stats(self) -> EngineStats:
        return self.scheduler.stats()

    async def shutdown(self) -> None:
        await self.scheduler.shutdown()
