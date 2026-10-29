import unittest

from adapters.openai.run import main
from config.settings import RuntimeSettings, ServerSettings, parse_settings


class RuntimeConfigTests(unittest.TestCase):
    def test_defaults_use_loopback_host(self):
        settings = parse_settings(argv=[], env={})

        self.assertEqual(settings.server, ServerSettings(host="127.0.0.1"))
        self.assertEqual(settings.runtime, RuntimeSettings())

    def test_env_overrides_defaults(self):
        settings = parse_settings(
            argv=[],
            env={
                "LFM3_API_HOST": "0.0.0.0",
                "LFM3_API_PORT": "9000",
                "LFM3_API_RELOAD": "yes",
                "LFM3_API_LOG_LEVEL": "debug",
                "LFM3_API_MODEL_ID": "LiquidAI/LFM3-1.7B",
                "LFM3_API_QUANTIZATION": "int8",
                "LFM3_API_COMPILE_MODEL": "true",
                "LFM3_API_USE_INSTRUCT_MODEL": "false",
                "LFM3_API_DEVICE": "cpu",
                "LFM3_API_DEFAULT_MAX_TOKENS": "64",
                "LFM3_API_MAX_REQUEST_TOKENS": "512",
                "LFM3_API_TEMPERATURE": "0.5",
                "LFM3_API_TOP_P": "0.75",
                "LFM3_API_TOP_K": "20",
                "LFM3_API_REPETITION_PENALTY": "1.2",
                "LFM3_API_DECODE_SELECTION_WINDOW": "6",
                "LFM3_API_MAX_QUEUE_SIZE": "7",
                "LFM3_API_MAX_CONCURRENT_REQUESTS": "3",
                "LFM3_API_MAX_BATCH_TOKENS": "96",
                "LFM3_API_DECODE_BATCH_SIZE": "2",
                "LFM3_API_REQUEST_TIMEOUT_S": "12.5",
                "LFM3_API_MAX_KV_CACHE_TOKENS": "1024",
                "LFM3_API_KV_BLOCK_SIZE": "32",
                "LFM3_API_NUM_KV_BLOCKS": "8",
                "LFM3_API_PREFILL_CHUNK_SIZE": "128",
                "LFM3_API_ENABLE_PREFIX_CACHE": "true",
                "LFM3_API_MAX_PREFIX_CACHE_ENTRIES": "32",
            },
        )

        self.assertEqual(settings.server.host, "0.0.0.0")
        self.assertEqual(settings.server.port, 9000)
        self.assertTrue(settings.server.reload)
        self.assertEqual(settings.server.log_level, "debug")
        self.assertEqual(settings.runtime.model_id, "LiquidAI/LFM3-1.7B")
        self.assertEqual(settings.runtime.quantization, "int8")
        self.assertTrue(settings.runtime.compile_model)
        self.assertFalse(settings.runtime.use_instruct_model)
        self.assertEqual(settings.runtime.device, "cpu")
        self.assertEqual(settings.runtime.default_max_tokens, 64)
        self.assertEqual(settings.runtime.max_request_tokens, 512)
        self.assertEqual(settings.runtime.temperature, 0.5)
        self.assertEqual(settings.runtime.top_p, 0.75)
        self.assertEqual(settings.runtime.top_k, 20)
        self.assertEqual(settings.runtime.repetition_penalty, 1.2)
        self.assertEqual(settings.runtime.decode_selection_window, 6)
        self.assertEqual(settings.runtime.max_queue_size, 7)
        self.assertEqual(settings.runtime.max_concurrent_requests, 3)
        self.assertEqual(settings.runtime.max_batch_tokens, 96)
        self.assertEqual(settings.runtime.decode_batch_size, 2)
        self.assertEqual(settings.runtime.request_timeout_s, 12.5)
        self.assertEqual(settings.runtime.max_kv_cache_tokens, 1024)
        self.assertEqual(settings.runtime.kv_block_size, 32)
        self.assertEqual(settings.runtime.num_kv_blocks, 8)
        self.assertEqual(settings.runtime.prefill_chunk_size, 128)
        self.assertTrue(settings.runtime.enable_prefix_cache)
        self.assertEqual(settings.runtime.max_prefix_cache_entries, 32)

    def test_cli_overrides_env(self):
        settings = parse_settings(
            argv=[
                "--host",
                "localhost",
                "--port",
                "7000",
                "--no-reload",
                "--log-level",
                "warning",
                "--model-id",
                "LiquidAI/LFM3-1.7B",
                "--quantization",
                "int4",
                "--compile-model",
                "--no-use-instruct-model",
                "--default-max-tokens",
                "33",
                "--temperature",
                "0.2",
                "--decode-batch-size",
                "5",
                "--request-timeout-s",
                "1.5",
                "--enable-prefix-cache",
                "--max-prefix-cache-entries",
                "16",
                "--num-kv-blocks",
                "none",
            ],
            env={
                "LFM3_API_HOST": "0.0.0.0",
                "LFM3_API_PORT": "9000",
                "LFM3_API_RELOAD": "true",
                "LFM3_API_LOG_LEVEL": "debug",
                "LFM3_API_USE_INSTRUCT_MODEL": "true",
                "LFM3_API_COMPILE_MODEL": "false",
                "LFM3_API_DEFAULT_MAX_TOKENS": "64",
                "LFM3_API_TEMPERATURE": "0.5",
                "LFM3_API_DECODE_BATCH_SIZE": "8",
                "LFM3_API_REQUEST_TIMEOUT_S": "9.0",
                "LFM3_API_NUM_KV_BLOCKS": "8",
            },
        )

        self.assertEqual(settings.server.host, "localhost")
        self.assertEqual(settings.server.port, 7000)
        self.assertFalse(settings.server.reload)
        self.assertEqual(settings.server.log_level, "warning")
        self.assertEqual(settings.runtime.model_id, "LiquidAI/LFM3-1.7B")
        self.assertEqual(settings.runtime.quantization, "int4")
        self.assertTrue(settings.runtime.compile_model)
        self.assertFalse(settings.runtime.use_instruct_model)
        self.assertEqual(settings.runtime.default_max_tokens, 33)
        self.assertEqual(settings.runtime.temperature, 0.2)
        self.assertEqual(settings.runtime.decode_batch_size, 5)
        self.assertEqual(settings.runtime.request_timeout_s, 1.5)
        self.assertTrue(settings.runtime.enable_prefix_cache)
        self.assertEqual(settings.runtime.max_prefix_cache_entries, 16)
        self.assertIsNone(settings.runtime.num_kv_blocks)

    def test_invalid_env_bool_reports_variable_name(self):
        with self.assertRaisesRegex(ValueError, "LFM3_API_RELOAD"):
            parse_settings(argv=[], env={"LFM3_API_RELOAD": "sometimes"})

    def test_run_main_passes_server_settings_to_runner(self):
        calls = []

        def fake_runner(*args, **kwargs):
            calls.append((args, kwargs))

        def fake_app_factory(service_factory):
            service = service_factory()
            return {"service_config": service.config}

        class FakeService:
            def __init__(self, config):
                self.config = config

        main(
            argv=["--host", "localhost", "--port", "7001", "--reload", "--log-level", "debug"],
            env={},
            runner=fake_runner,
            app_factory=fake_app_factory,
            service_cls=FakeService,
        )

        self.assertEqual(len(calls), 1)
        args, kwargs = calls[0]
        self.assertEqual(len(args), 1)
        self.assertEqual(args[0]["service_config"].default_max_tokens, RuntimeSettings().default_max_tokens)
        self.assertEqual(args[0]["service_config"].max_kv_cache_tokens, RuntimeSettings().max_kv_cache_tokens)
        self.assertEqual(kwargs, {"host": "localhost", "port": 7001, "reload": True, "log_level": "debug"})


if __name__ == "__main__":
    unittest.main()
