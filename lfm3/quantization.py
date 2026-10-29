"""Weight-only quantization for LFM3 linear layers.

Two opt-in formats are supported:

* ``int8``: per-output-channel symmetric int8 weights. On CUDA this uses
  PyTorch's native ``_weight_int8pack_mm`` kernel; elsewhere it falls back to
  on-the-fly dequantization. It was token-identical on the tested real prompt;
  validate it against representative prompts for a production workload.
* ``int4``: NF4 weights via bitsandbytes (CUDA only), with double
  quantization. This is the smallest option, but it is experimental on
  LFM3-1.7B: in testing it changed the model's stop behavior on a short
  arithmetic prompt. Prefer ``int8`` unless the extra ~0.65 GiB is required.

Embeddings and the tied output head stay in the compute dtype. The output
head shares its storage with the token embedding, so quantizing it separately
would both break the tie and use more memory.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

QUANTIZATION_CHOICES = ("none", "int8", "int4")


class Int8Linear(nn.Module):
    """Linear layer with per-output-channel symmetric int8 weights."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "qweight",
            torch.empty(out_features, in_features, dtype=torch.int8, device=device),
        )
        self.register_buffer(
            "scales",
            torch.empty(out_features, dtype=torch.float32, device=device),
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear: nn.Linear) -> "Int8Linear":
        quantized = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        weight = linear.weight.float()
        scales = weight.abs().amax(dim=1).clamp_min(1e-8) / 127.0
        qweight = torch.round(weight / scales[:, None]).clamp(-127, 127)
        quantized.qweight.copy_(qweight.to(torch.int8))
        quantized.scales.copy_(scales)
        if linear.bias is not None:
            quantized.bias.copy_(linear.bias)
        return quantized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, self.in_features)
        if not flat.is_contiguous():
            flat = flat.contiguous()

        if flat.device.type == "cuda" and hasattr(torch, "_weight_int8pack_mm"):
            out = torch._weight_int8pack_mm(flat, self.qweight, self.scales)
            out = out.to(flat.dtype)
        else:
            weight = self.qweight.to(flat.dtype) * self.scales.to(flat.dtype)[:, None]
            out = F.linear(flat, weight)

        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*original_shape[:-1], self.out_features)


def _replace_linear_modules(
    module: nn.Module,
    replace: Callable[[nn.Linear], nn.Module],
    *,
    prefix: str = "",
    skip_names: frozenset[str] = frozenset(),
    skip_suffixes: tuple[str, ...] = (),
) -> None:
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            if full_name in skip_names or full_name.endswith(skip_suffixes):
                continue
            setattr(module, name, replace(child))
        else:
            _replace_linear_modules(
                child,
                replace,
                prefix=full_name,
                skip_names=skip_names,
                skip_suffixes=skip_suffixes,
            )


def _tied_output_head_name(model: nn.Module) -> frozenset[str]:
    head = getattr(model, "out_head", None)
    tok_emb = getattr(model, "tok_emb", None)
    if (
        isinstance(head, nn.Linear)
        and isinstance(tok_emb, nn.Embedding)
        and head.weight is tok_emb.weight
    ):
        return frozenset({"out_head"})
    return frozenset()


def quantize_model_int8(model: nn.Module) -> nn.Module:
    """Replace every non-tied Linear with an :class:`Int8Linear` in place."""
    skip_names = _tied_output_head_name(model)
    _replace_linear_modules(
        model,
        lambda linear: Int8Linear.from_linear(linear),
        skip_names=skip_names,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


def quantize_model_int4(
    model: nn.Module,
    *,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Replace every non-tied Linear with a bitsandbytes NF4 layer in place."""
    try:
        import bitsandbytes as bnb
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "int4 quantization requires bitsandbytes; install the 'quant' extra"
        ) from exc

    skip_names = _tied_output_head_name(model)

    def replace(linear: nn.Linear):
        quantized = bnb.nn.Linear4bit(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            compute_dtype=compute_dtype,
            quant_type="nf4",
            compress_statistics=True,
        )
        quantized.weight = bnb.nn.Params4bit(
            linear.weight.data,
            requires_grad=False,
            quant_type="nf4",
            compress_statistics=True,
        )
        if linear.bias is not None:
            quantized.bias = nn.Parameter(linear.bias.data.clone())
        return quantized.to(linear.weight.device)

    _replace_linear_modules(
        model, replace, skip_names=skip_names, skip_suffixes=("gate_proj",)
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


def quantize_model(
    model: nn.Module,
    quantization: str,
    *,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    if quantization == "none":
        return model
    if quantization == "int8":
        return quantize_model_int8(model)
    if quantization == "int4":
        if not torch.cuda.is_available():
            raise ValueError("int4 quantization requires CUDA")
        return quantize_model_int4(model, compute_dtype=compute_dtype)
    raise ValueError(
        f"unknown quantization {quantization!r}; choose from {QUANTIZATION_CHOICES}"
    )
