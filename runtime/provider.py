import json
import os
from pathlib import Path
from typing import Dict, Optional

import torch

from gemma3.model import GEMMA3_CONFIG_270M, build_gemma3_270m
from gemma3.utilities import load_weights_into_gemma


def _get_hf_downloaders():
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "huggingface_hub is required when local model/tokenizer files are missing."
        ) from exc
    return hf_hub_download, snapshot_download


def _load_safetensor_file(path: str) -> Dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "safetensors is required to load Gemma model weights."
        ) from exc
    return load_file(path)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Paged KV gather/scatter is currently much faster on CPU than MPS for this model.
    # Opt into MPS explicitly with --device mps / GEMMA_API_DEVICE=mps when desired.
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
    # MPS float16 has been unstable with this paged-attention path; prefer fp32.
    return torch.float32


def configure_cpu_threads() -> None:
    raw = os.environ.get("GEMMA_NUM_THREADS")
    if not raw:
        return
    torch.set_num_threads(max(1, int(raw)))


def build_repo_id(choose_model: str, use_instruct_model: bool) -> str:
    suffix = "-it" if use_instruct_model else ""
    return f"google/gemma-3-{choose_model}{suffix}"


def download_weights(repo_id: str, choose_model: str, local_dir: str) -> Dict[str, torch.Tensor]:
    if choose_model == "270m":
        local_weights = Path(local_dir) / "model.safetensors"
        if local_weights.exists():
            return _load_safetensor_file(str(local_weights))

        hf_hub_download, _ = _get_hf_downloaders()
        weights_file = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            local_dir=local_dir,
        )
        return _load_safetensor_file(weights_file)

    _, snapshot_download = _get_hf_downloaders()
    repo_dir = snapshot_download(repo_id=repo_id, local_dir=local_dir)
    index_path = Path(repo_dir) / "model.safetensors.index.json"
    with open(index_path, "r", encoding="utf-8") as file_handle:
        index = json.load(file_handle)

    weights_dict: Dict[str, torch.Tensor] = {}
    for filename in set(index["weight_map"].values()):
        shard_path = Path(repo_dir) / filename
        shard = _load_safetensor_file(str(shard_path))
        weights_dict.update(shard)
    return weights_dict


def resolve_tokenizer_path(repo_id: str, local_dir: str) -> str:
    tokenizer_path = Path(local_dir) / "tokenizer.json"
    if tokenizer_path.exists():
        return str(tokenizer_path)
    fallback = Path("tokenizer.json")
    if fallback.exists():
        return str(fallback)
    hf_hub_download, _ = _get_hf_downloaders()
    return hf_hub_download(repo_id=repo_id, filename="tokenizer.json", local_dir=local_dir)


class GemmaTokenizer:
    def __init__(self, tokenizer_file_path: str, *, use_instruct_model: bool = True):
        try:
            from tokenizers import Tokenizer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "tokenizers is required to load tokenizer.json."
            ) from exc
        self._tok = Tokenizer.from_file(str(Path(tokenizer_file_path)))
        self.end_of_turn_id = self._tok.encode("<end_of_turn>").ids[-1]
        self.pad_token_id = self.end_of_turn_id
        self.eos_token_id = self.end_of_turn_id if use_instruct_model else self._tok.token_to_id("<eos>")

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False)


class GemmaRuntime:
    def __init__(
        self,
        choose_model: str = "270m",
        use_instruct_model: bool = True,
        device: Optional[torch.device] = None,
    ):
        if choose_model != "270m":
            raise ValueError("This runtime currently supports only choose_model='270m'")

        self.device = device or get_device()
        if self.device.type == "cpu":
            configure_cpu_threads()
        self.dtype = preferred_dtype(self.device)
        self.choose_model = choose_model
        self.use_instruct_model = use_instruct_model
        self.repo_id = build_repo_id(choose_model=choose_model, use_instruct_model=use_instruct_model)
        self.local_dir = Path(self.repo_id).name

        self.model = build_gemma3_270m()

        weights_dict = download_weights(
            repo_id=self.repo_id,
            choose_model=self.choose_model,
            local_dir=self.local_dir,
        )
        load_weights_into_gemma(self.model, GEMMA3_CONFIG_270M, weights_dict)
        del weights_dict

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.dtype = self.dtype

        tokenizer_path = resolve_tokenizer_path(repo_id=self.repo_id, local_dir=self.local_dir)
        self.tokenizer = GemmaTokenizer(tokenizer_path, use_instruct_model=use_instruct_model)

        # reduce-overhead / CUDA graphs help on CUDA; on CPU/MPS they often slow cold starts.
        if self.device.type == "cuda" and os.environ.get("GEMMA_ENABLE_COMPILE", "0") == "1" and os.environ.get("GEMMA_DISABLE_COMPILE", "0") != "1":
            self.model = torch.compile(self.model, mode="reduce-overhead")
        self.model.eval()
