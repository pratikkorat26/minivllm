import importlib.util
import unittest


HTTP_DEPENDENCIES_AVAILABLE = all(
    importlib.util.find_spec(name) is not None for name in ("fastapi", "pydantic")
)


class CanonicalImportTests(unittest.TestCase):
    def test_inference_and_runtime_imports(self):
        from inference import EngineConfig, LLMEngine, SamplingConfig
        from inference.kv import KVBlockManager
        from runtime import Lfm3Runtime

        self.assertIsNotNone(EngineConfig)
        self.assertIsNotNone(Lfm3Runtime)
        self.assertIsNotNone(KVBlockManager)
        self.assertIsNotNone(LLMEngine)
        self.assertIsNotNone(SamplingConfig)

    @unittest.skipUnless(HTTP_DEPENDENCIES_AVAILABLE, "HTTP dependencies are not installed")
    def test_openai_adapter_imports(self):
        from adapters.openai.routes import create_app
        from adapters.openai.schemas import ChatCompletionRequest, SUPPORTED_MODEL
        from app import ChatCompletionService

        self.assertEqual(SUPPORTED_MODEL, "LiquidAI/LFM3-1.7B")
        self.assertIsNotNone(ChatCompletionRequest)
        self.assertIsNotNone(ChatCompletionService)
        self.assertIsNotNone(create_app)


if __name__ == "__main__":
    unittest.main()
