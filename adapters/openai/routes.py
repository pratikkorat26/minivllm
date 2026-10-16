from contextlib import asynccontextmanager, aclosing
import asyncio
import json
import logging
import time
from typing import Callable

from app import ChatCompletionService
from app.errors import AppError
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from .errors import APIError, app_error_to_api_error, error_response
from .mapper import chat_completion_to_openai, to_app_request
from .schemas import ChatCompletionRequest
from .sse import event_to_sse

ServiceFactory = Callable[[], ChatCompletionService]
LOGGER = logging.getLogger("adapters.openai")
DEFAULT_MAX_REQUEST_BYTES = 1_048_576


def _service_unavailable_message(fastapi_app: FastAPI) -> str:
    startup_error = getattr(fastapi_app.state, "startup_error", None)
    if startup_error:
        return "service unavailable: startup failed"
    return "service unavailable: startup incomplete"


def get_service(fastapi_app: FastAPI) -> ChatCompletionService:
    service = getattr(fastapi_app.state, "service", None)
    if service is None:
        raise HTTPException(
            status_code=503, detail=_service_unavailable_message(fastapi_app)
        )
    return service


def create_app(
    service_factory: ServiceFactory = ChatCompletionService,
    *,
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI):
        if not hasattr(fastapi_app.state, "service"):
            fastapi_app.state.service = None
        fastapi_app.state.startup_error = None
        if fastapi_app.state.service is None:
            try:
                fastapi_app.state.service = service_factory()
            except Exception as exc:
                fastapi_app.state.startup_error = str(exc) or exc.__class__.__name__
                LOGGER.exception("service startup failed")
        yield
        service = getattr(fastapi_app.state, "service", None)
        if service is not None and hasattr(service, "shutdown"):
            await service.shutdown()

    fastapi_app = FastAPI(
        title="Gemma OpenAI-like API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @fastapi_app.middleware("http")
    async def request_logging_and_size_limit(request: Request, call_next):
        content_length = request.headers.get("content-length")
        try:
            body_size = int(content_length) if content_length is not None else 0
            if body_size < 0:
                raise ValueError
        except ValueError:
            return JSONResponse(
                status_code=400,
                content=error_response(
                    APIError(
                        "invalid content-length",
                        status_code=400,
                        error_type="invalid_request_error",
                        code="invalid_content_length",
                    )
                ),
            )
        if body_size > max_request_bytes:
            return JSONResponse(
                status_code=413,
                content=error_response(
                    APIError(
                        "request body too large",
                        status_code=413,
                        error_type="invalid_request_error",
                        code="request_body_too_large",
                    )
                ),
            )
        started = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - started) * 1000
        LOGGER.info(
            json.dumps(
                {
                    "event": "http_request",
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration_ms, 3),
                }
            )
        )
        return response

    @fastapi_app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @fastapi_app.get("/readyz")
    async def readyz() -> dict:
        service = getattr(fastapi_app.state, "service", None)
        if service is None:
            raise HTTPException(
                status_code=503, detail=_service_unavailable_message(fastapi_app)
            )
        return {"status": "ready"}

    @fastapi_app.get("/v1/models")
    async def list_models(request: Request):
        service = getattr(request.app.state, "service", None)
        if service is not None and hasattr(service, "list_models"):
            return JSONResponse(service.list_models())
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": "gemma-3-270m-it",
                        "object": "model",
                        "created": 0,
                        "owned_by": "local",
                    }
                ],
            }
        )

    @fastapi_app.get("/stats")
    async def stats(request: Request):
        service = get_service(request.app)
        if hasattr(service, "stats"):
            return JSONResponse(service.stats())
        return JSONResponse({"model": "unknown", "requests_total": 0})

    @fastapi_app.get("/metrics")
    async def metrics(request: Request):
        service = get_service(request.app)
        if hasattr(service, "metrics_text"):
            return PlainTextResponse(
                service.metrics_text(), media_type="text/plain; version=0.0.4"
            )
        return PlainTextResponse("", media_type="text/plain")

    @fastapi_app.post("/v1/chat/completions")
    async def chat_completions(payload: ChatCompletionRequest, request: Request):
        service = get_service(request.app)
        try:
            app_request = to_app_request(payload)
            if payload.stream:

                async def sse_events():
                    async with aclosing(service.stream(app_request)) as events:
                        async for event in events:
                            yield event_to_sse(event)

                return StreamingResponse(
                    sse_events(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )
            generation = asyncio.create_task(service.create(app_request))
            try:
                while not generation.done():
                    done, _ = await asyncio.wait({generation}, timeout=0.05)
                    if done:
                        break
                    if await request.is_disconnected():
                        raise HTTPException(
                            status_code=499, detail="client disconnected"
                        )
                result = await generation
            finally:
                if not generation.done():
                    generation.cancel()
                    await asyncio.gather(generation, return_exceptions=True)
            return JSONResponse(chat_completion_to_openai(result))
        except AppError as exc:
            api_error = app_error_to_api_error(exc)
            return JSONResponse(
                status_code=api_error.status_code, content=error_response(api_error)
            )
        except RuntimeError:
            LOGGER.exception("chat completion failed")
            api_error = APIError("model execution failed")
            return JSONResponse(status_code=500, content=error_response(api_error))

    return fastapi_app


app = create_app()
