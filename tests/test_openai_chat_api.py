import logging
import unittest

try:
    from app.types import ChatCompletionEvent, ChatCompletionResult, Usage
    from fastapi.testclient import TestClient
    from adapters.openai.routes import create_app
    from adapters.openai.schemas import SUPPORTED_MODEL
    FASTAPI_AVAILABLE = True
except ModuleNotFoundError:
    FASTAPI_AVAILABLE = False


class FakeChatService:
    def __init__(self):
        self.requests_total = 0

    async def create(self, request):
        self.requests_total += 1
        return ChatCompletionResult(
            request_id="chatcmpl-test", model=SUPPORTED_MODEL, content="hello",
            finish_reason="stop", usage=Usage(10, 1, 11), created=0,
        )

    async def stream(self, request):
        self.requests_total += 1
        common = {"request_id": "chatcmpl-test", "model": SUPPORTED_MODEL, "created": 0}
        yield ChatCompletionEvent(**common, role="assistant")
        yield ChatCompletionEvent(**common, content="hello")
        if request.include_usage:
            yield ChatCompletionEvent(**common, usage=Usage(10, 1, 11))
        yield ChatCompletionEvent(**common, finish_reason="stop")
        yield ChatCompletionEvent(**common, done=True)

    def list_models(self):
        return {
            "object": "list",
            "data": [{"id": SUPPORTED_MODEL, "object": "model", "created": 0, "owned_by": "local"}],
        }

    def stats(self):
        return {
            "model": SUPPORTED_MODEL,
            "requests_total": self.requests_total,
            "requests_failed": 0,
            "tokens_prompt_total": 10 * self.requests_total,
            "tokens_completion_total": self.requests_total,
            "latency_s": {"last": 0.0, "avg": 0.0},
            "generation": {},
        }

    def metrics_text(self):
        return f'lfm3_requests_total{{model="{SUPPORTED_MODEL}"}} {self.requests_total}\n'


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed")
class OpenAIChatAPITests(unittest.TestCase):
    def setUp(self):
        self.app = create_app(service_factory=FakeChatService)
        self.client_cm = TestClient(self.app)
        self.client = self.client_cm.__enter__()

    def tearDown(self):
        self.client_cm.__exit__(None, None, None)

    def test_non_stream_response_shape(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": SUPPORTED_MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["choices"][0]["message"]["role"], "assistant")
        self.assertIn("usage", payload)

    def test_stream_response_has_done(self):
        with self.client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": SUPPORTED_MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        ) as response:
            self.assertEqual(response.status_code, 200)
            body = "".join(part for part in response.iter_text())
            self.assertIn("data: [DONE]", body)

    def test_rejects_unknown_model(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_readyz_reports_ready_after_startup(self):
        response = self.client.get("/readyz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ready"})

    def test_models_response_shape(self):
        response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["object"], "list")
        self.assertEqual(payload["data"][0]["id"], SUPPORTED_MODEL)

    def test_stats_response_shape(self):
        self.client.post(
            "/v1/chat/completions",
            json={
                "model": SUPPORTED_MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        response = self.client.get("/stats")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["model"], SUPPORTED_MODEL)
        self.assertEqual(payload["requests_total"], 1)

    def test_metrics_response_shape(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/plain", response.headers["content-type"])
        self.assertIn("lfm3_requests_total", response.text)

    def test_accepts_extended_sampling_fields(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": SUPPORTED_MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "top_k": 20,
                "repetition_penalty": 1.2,
                "seed": 123,
                "stop": ["\n"],
            },
        )
        self.assertEqual(response.status_code, 200)

    def test_rejects_too_many_messages(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": SUPPORTED_MODEL,
                "messages": [{"role": "user", "content": "Hi"}] * 65,
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_rejects_empty_stop_sequence(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": SUPPORTED_MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "stop": [""],
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_stream_response_can_include_usage(self):
        with self.client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": SUPPORTED_MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        ) as response:
            self.assertEqual(response.status_code, 200)
            body = "".join(part for part in response.iter_text())
            self.assertIn('"usage"', body)


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed")
class OpenAIChatAPIStartupFailureTests(unittest.TestCase):
    def setUp(self):
        def failing_service_factory():
            raise RuntimeError("model init failed")

        self.logger = logging.getLogger("adapters.openai")
        self.previous_logger_disabled = self.logger.disabled
        self.logger.disabled = True
        self.app = create_app(service_factory=failing_service_factory)
        self.client_cm = TestClient(self.app)
        self.client = self.client_cm.__enter__()

    def tearDown(self):
        self.client_cm.__exit__(None, None, None)
        self.logger.disabled = self.previous_logger_disabled

    def test_readyz_reports_startup_failure(self):
        response = self.client.get("/readyz")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "service unavailable: startup failed")

    def test_chat_completion_fails_fast_when_service_unavailable(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": SUPPORTED_MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "service unavailable: startup failed")


if __name__ == "__main__":
    unittest.main()
