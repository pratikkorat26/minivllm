import argparse
import os
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence, TypeVar


ENV_PREFIX = "LFM3_API_"


@dataclass(frozen=True)
class ServerSettings:
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = False
    log_level: str = "info"


@dataclass(frozen=True)
class RuntimeSettings:
    model_id: str = "LiquidAI/LFM3-1.7B"
    quantization: str = "none"
    compile_model: bool = False
    use_instruct_model: bool = True
    device: str = "auto"
    default_max_tokens: int = 128
    max_request_tokens: int = 4096
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
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


@dataclass(frozen=True)
class Settings:
    server: ServerSettings = field(default_factory=ServerSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)


T = TypeVar("T")


def _env_name(field_name: str) -> str:
    return f"{ENV_PREFIX}{field_name.upper()}"


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"expected a boolean value, got {value!r}")


def _parse_optional_int(value: str) -> Optional[int]:
    normalized = value.strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    return int(value)


def _parse_optional_float(value: str) -> Optional[float]:
    normalized = value.strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    return float(value)


def _from_env(
    env: Mapping[str, str],
    field_name: str,
    parser: Callable[[str], T],
    default: T,
) -> T:
    raw = env.get(_env_name(field_name))
    if raw is None:
        return default
    try:
        return parser(raw)
    except ValueError as exc:
        raise ValueError(f"{_env_name(field_name)}: {exc}") from exc


def _add_bool_override(parser: argparse.ArgumentParser, name: str) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        f"--{name.replace('_', '-')}",
        dest=name,
        action="store_true",
        default=argparse.SUPPRESS,
    )
    group.add_argument(
        f"--no-{name.replace('_', '-')}",
        dest=name,
        action="store_false",
        default=argparse.SUPPRESS,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LFM3 OpenAI-like API server.")
    parser.add_argument("--host", default=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=argparse.SUPPRESS)
    _add_bool_override(parser, "reload")
    parser.add_argument("--log-level", default=argparse.SUPPRESS)

    parser.add_argument("--model-id", default=argparse.SUPPRESS)
    parser.add_argument(
        "--quantization",
        choices=("none", "int8", "int4"),
        default=argparse.SUPPRESS,
        help="Weight-only quantization; int8 is recommended, int4 is experimental",
    )
    _add_bool_override(parser, "compile_model")
    _add_bool_override(parser, "use_instruct_model")
    parser.add_argument("--device", default=argparse.SUPPRESS)
    parser.add_argument("--default-max-tokens", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max-request-tokens", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--temperature", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--top-p", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--top-k", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--repetition-penalty", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--decode-batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--decode-selection-window", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max-queue-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max-concurrent-requests", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max-batch-tokens", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--request-timeout-s", type=_parse_optional_float, default=argparse.SUPPRESS)
    parser.add_argument("--max-kv-cache-tokens", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--kv-block-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num-kv-blocks", type=_parse_optional_int, default=argparse.SUPPRESS)
    parser.add_argument("--prefill-chunk-size", type=_parse_optional_int, default=argparse.SUPPRESS)
    _add_bool_override(parser, "enable_prefix_cache")
    parser.add_argument("--max-prefix-cache-entries", type=int, default=argparse.SUPPRESS)
    return parser


def _override(args: argparse.Namespace, field_name: str, current: T) -> T:
    return getattr(args, field_name) if hasattr(args, field_name) else current


def load_settings_from_env(env: Mapping[str, str] = os.environ) -> Settings:
    server_defaults = ServerSettings()
    runtime_defaults = RuntimeSettings()

    return Settings(
        server=ServerSettings(
            host=_from_env(env, "host", str, server_defaults.host),
            port=_from_env(env, "port", int, server_defaults.port),
            reload=_from_env(env, "reload", _parse_bool, server_defaults.reload),
            log_level=_from_env(env, "log_level", str, server_defaults.log_level),
        ),
        runtime=RuntimeSettings(
            model_id=_from_env(env, "model_id", str, runtime_defaults.model_id),
            quantization=_from_env(
                env, "quantization", str, runtime_defaults.quantization
            ),
            compile_model=_from_env(
                env, "compile_model", _parse_bool, runtime_defaults.compile_model
            ),
            use_instruct_model=_from_env(
                env,
                "use_instruct_model",
                _parse_bool,
                runtime_defaults.use_instruct_model,
            ),
            device=_from_env(env, "device", str, runtime_defaults.device),
            default_max_tokens=_from_env(
                env,
                "default_max_tokens",
                int,
                runtime_defaults.default_max_tokens,
            ),
            max_request_tokens=_from_env(
                env,
                "max_request_tokens",
                int,
                runtime_defaults.max_request_tokens,
            ),
            temperature=_from_env(env, "temperature", float, runtime_defaults.temperature),
            top_p=_from_env(env, "top_p", float, runtime_defaults.top_p),
            top_k=_from_env(env, "top_k", int, runtime_defaults.top_k),
            repetition_penalty=_from_env(
                env,
                "repetition_penalty",
                float,
                runtime_defaults.repetition_penalty,
            ),
            decode_batch_size=_from_env(
                env,
                "decode_batch_size",
                int,
                runtime_defaults.decode_batch_size,
            ),
            decode_selection_window=_from_env(
                env,
                "decode_selection_window",
                int,
                runtime_defaults.decode_selection_window,
            ),
            max_queue_size=_from_env(
                env,
                "max_queue_size",
                int,
                runtime_defaults.max_queue_size,
            ),
            max_concurrent_requests=_from_env(
                env,
                "max_concurrent_requests",
                int,
                runtime_defaults.max_concurrent_requests,
            ),
            max_batch_tokens=_from_env(
                env,
                "max_batch_tokens",
                int,
                runtime_defaults.max_batch_tokens,
            ),
            request_timeout_s=_from_env(
                env,
                "request_timeout_s",
                _parse_optional_float,
                runtime_defaults.request_timeout_s,
            ),
            max_kv_cache_tokens=_from_env(
                env,
                "max_kv_cache_tokens",
                int,
                runtime_defaults.max_kv_cache_tokens,
            ),
            kv_block_size=_from_env(env, "kv_block_size", int, runtime_defaults.kv_block_size),
            num_kv_blocks=_from_env(
                env,
                "num_kv_blocks",
                _parse_optional_int,
                runtime_defaults.num_kv_blocks,
            ),
            prefill_chunk_size=_from_env(
                env,
                "prefill_chunk_size",
                _parse_optional_int,
                runtime_defaults.prefill_chunk_size,
            ),
            enable_prefix_cache=_from_env(
                env,
                "enable_prefix_cache",
                _parse_bool,
                runtime_defaults.enable_prefix_cache,
            ),
            max_prefix_cache_entries=_from_env(
                env,
                "max_prefix_cache_entries",
                int,
                runtime_defaults.max_prefix_cache_entries,
            ),
        ),
    )


def parse_settings(
    argv: Optional[Sequence[str]] = None,
    env: Mapping[str, str] = os.environ,
) -> Settings:
    settings = load_settings_from_env(env)
    args = build_parser().parse_args(argv)

    server = ServerSettings(
        host=_override(args, "host", settings.server.host),
        port=_override(args, "port", settings.server.port),
        reload=_override(args, "reload", settings.server.reload),
        log_level=_override(args, "log_level", settings.server.log_level),
    )
    runtime = RuntimeSettings(
        model_id=_override(args, "model_id", settings.runtime.model_id),
        quantization=_override(args, "quantization", settings.runtime.quantization),
        compile_model=_override(
            args, "compile_model", settings.runtime.compile_model
        ),
        use_instruct_model=_override(
            args,
            "use_instruct_model",
            settings.runtime.use_instruct_model,
        ),
        device=_override(args, "device", settings.runtime.device),
        default_max_tokens=_override(
            args,
            "default_max_tokens",
            settings.runtime.default_max_tokens,
        ),
        max_request_tokens=_override(
            args,
            "max_request_tokens",
            settings.runtime.max_request_tokens,
        ),
        temperature=_override(args, "temperature", settings.runtime.temperature),
        top_p=_override(args, "top_p", settings.runtime.top_p),
        top_k=_override(args, "top_k", settings.runtime.top_k),
        repetition_penalty=_override(
            args,
            "repetition_penalty",
            settings.runtime.repetition_penalty,
        ),
        decode_batch_size=_override(
            args,
            "decode_batch_size",
            settings.runtime.decode_batch_size,
        ),
        decode_selection_window=_override(
            args,
            "decode_selection_window",
            settings.runtime.decode_selection_window,
        ),
        max_queue_size=_override(args, "max_queue_size", settings.runtime.max_queue_size),
        max_concurrent_requests=_override(
            args,
            "max_concurrent_requests",
            settings.runtime.max_concurrent_requests,
        ),
        max_batch_tokens=_override(args, "max_batch_tokens", settings.runtime.max_batch_tokens),
        request_timeout_s=_override(
            args,
            "request_timeout_s",
            settings.runtime.request_timeout_s,
        ),
        max_kv_cache_tokens=_override(
            args,
            "max_kv_cache_tokens",
            settings.runtime.max_kv_cache_tokens,
        ),
        kv_block_size=_override(args, "kv_block_size", settings.runtime.kv_block_size),
        num_kv_blocks=_override(args, "num_kv_blocks", settings.runtime.num_kv_blocks),
        prefill_chunk_size=_override(
            args,
            "prefill_chunk_size",
            settings.runtime.prefill_chunk_size,
        ),
        enable_prefix_cache=_override(
            args,
            "enable_prefix_cache",
            settings.runtime.enable_prefix_cache,
        ),
        max_prefix_cache_entries=_override(
            args,
            "max_prefix_cache_entries",
            settings.runtime.max_prefix_cache_entries,
        ),
    )
    return Settings(server=server, runtime=runtime)
