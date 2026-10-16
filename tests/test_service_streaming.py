import asyncio
from types import SimpleNamespace

from app.service import ChatCompletionService
from app.types import ChatCompletionRequest, ChatMessage
from inference.types import StreamEvent


class ScriptedEngine:
    def __init__(self, chunks):
        self.chunks, self.aborted, self.closed = chunks, False, False

    async def stream(self, request):
        try:
            for index, text in enumerate(self.chunks):
                yield StreamEvent(
                    kind="text", text=text, generated_token_count=index + 1
                )
            yield StreamEvent(
                kind="done",
                generated_token_count=len(self.chunks) + 1,
                stop_reason="eos",
            )
        finally:
            self.closed = True

    async def abort(self, request_id):
        self.aborted = True


def collect(chunks, stops):
    engine = ScriptedEngine(chunks)
    service = ChatCompletionService(
        runtime=SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda text: [1])),
        engine=engine,
    )
    request = ChatCompletionRequest(
        messages=[ChatMessage("user", "hello")],
        model="gemma-3-270m-it",
        stop=stops,
        include_usage=True,
    )

    async def run():
        return [event async for event in service.stream(request)]

    return asyncio.run(run()), engine


def test_split_stop_never_leaks_and_iterator_closes():
    events, engine = collect(["hello <", "ST", "OP> rest"], ["<STOP>"])
    assert "".join(event.content or "" for event in events) == "hello "
    assert engine.aborted and engine.closed


def test_unfinished_stop_prefix_flushes_at_eos_and_usage_includes_eos():
    events, engine = collect(["hello <", "S"], ["<STOP>"])
    assert "".join(event.content or "" for event in events) == "hello <S"
    assert next(event.usage for event in events if event.usage).completion_tokens == 3
    assert not engine.aborted and engine.closed
