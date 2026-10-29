import os
from pathlib import Path

import torch

from lfm3.model import build_lfm3_1_7b
from lfm3.quantization import QUANTIZATION_CHOICES, quantize_model
from lfm3.utilities import load_weights_into_lfm3

DEFAULT_MODEL_ID = "LiquidAI/LFM3-1.7B"


def _get_hf_downloaders():
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "huggingface_hub is required when local model/tokenizer files are missing."
        ) from exc
    return hf_hub_download


def _load_safetensor_file(path: str) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "safetensors is required to load LFM3 model weights."
        ) from exc
    return load_file(path)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_device(configured: str = "auto") -> torch.device:
    configured = str(configured).lower()
    if configured == "auto":
        return get_device()
    return torch.device(configured)


def preferred_dtype(device: torch.device) -> torch.dtype:
    """Pick a compute dtype that is fast and stable on the target device."""
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def configure_cpu_threads() -> None:
    raw = os.environ.get("LFM3_NUM_THREADS")
    if not raw:
        return
    torch.set_num_threads(max(1, int(raw)))


def local_model_dir(model_id: str = DEFAULT_MODEL_ID) -> Path:
    return Path(model_id).name


def download_weights(model_id: str, local_dir: str) -> dict[str, torch.Tensor]:
    local_weights = Path(local_dir) / "model.safetensors"
    if local_weights.exists():
        return _load_safetensor_file(str(local_weights))

    hf_hub_download = _get_hf_downloaders()
    weights_file = hf_hub_download(
        repo_id=model_id,
        filename="model.safetensors",
        local_dir=local_dir,
    )
    return _load_safetensor_file(weights_file)


def resolve_tokenizer_path(model_id: str, local_dir: str) -> str:
    tokenizer_path = Path(local_dir) / "tokenizer.json"
    if tokenizer_path.exists():
        return str(tokenizer_path)
    fallback = Path("tokenizer.json")
    if fallback.exists():
        return str(fallback)
    hf_hub_download = _get_hf_downloaders()
    return hf_hub_download(
        repo_id=model_id, filename="tokenizer.json", local_dir=local_dir
    )


class Lfm3Tokenizer:
    def __init__(self, tokenizer_file_path: str):
        try:
            from tokenizers import Tokenizer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "tokenizers is required to load tokenizer.json."
            ) from exc
        self._tok = Tokenizer.from_file(str(tokenizer_file_path))
        self.bos_token_id = self._tok.token_to_id("<|startoftext|>")
        self.eos_token_id = self._tok.token_to_id("<|im_end|>")
        self.pad_token_id = self._tok.token_to_id("<|pad|>")

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False)


class Lfm3Runtime:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        use_instruct_model: bool = True,
        device: torch.device | None = None,
        local_dir: str | None = None,
        quantization: str = "none",
        compile_model: bool | None = None,
    ):
        if quantization not in QUANTIZATION_CHOICES:
            raise ValueError(
                f"unknown quantization {quantization!r}; "
                f"choose from {QUANTIZATION_CHOICES}"
            )
        self.device = device or get_device()
        if self.device.type == "cpu":
            configure_cpu_threads()
        self.dtype = preferred_dtype(self.device)
        self.model_id = model_id
        self.use_instruct_model = use_instruct_model
        self.local_dir = local_dir or str(local_model_dir(model_id))
        self.quantization = quantization
        if compile_model is None:
            compile_model = os.environ.get("LFM3_ENABLE_COMPILE", "0") == "1"
        self.compile_model = bool(
            self.device.type == "cuda"
            and compile_model
            and os.environ.get("LFM3_DISABLE_COMPILE", "0") != "1"
        )

        self.model = build_lfm3_1_7b()

        weights_dict = download_weights(
            model_id=self.model_id,
            local_dir=self.local_dir,
        )
        load_weights_into_lfm3(self.model, weights_dict)
        del weights_dict

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        if self.quantization != "none":
            self.model = quantize_model(
                self.model, self.quantization, compute_dtype=self.dtype
            )
        self.model.dtype = self.dtype

        tokenizer_path = resolve_tokenizer_path(
            model_id=self.model_id, local_dir=self.local_dir
        )
        self.tokenizer = Lfm3Tokenizer(tokenizer_path)

        # Compilation is optional: it improves measured throughput after a long
        # warmup, but CUDA graph capture is skipped for the mutating KV cache.
        if self.compile_model:
            self.model = torch.compile(self.model, mode="reduce-overhead")
        self.model.eval()
