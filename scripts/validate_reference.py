"""Compare native LFM3 execution with the Transformers reference implementation.

The LFM3 checkpoint is loaded from ``--model-dir``. The reference architecture
and config come from ``LiquidAI/LFM3-1.7B`` remote code, which is cached by
Transformers. This script compares logits and greedy tokens for both a chat
prompt and a raw prompt, including the per-request short-convolution state.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from safetensors.torch import load_file

from lfm3.model import LFM3_CONFIG_1_7B, Lfm3Model
from lfm3.tokenizer import apply_chat_template
from lfm3.utilities import load_weights_into_lfm3
from runtime.provider import Lfm3Tokenizer

MODEL_ID = "LiquidAI/LFM3-1.7B"


def _load_reference(model_id, dtype, device):
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config_cls = type(
        AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    )
    model_cls = get_class_from_dynamic_module(
        "modeling_lfm3.Lfm3ForCausalLM", model_id
    )
    hf_config = config_cls.from_pretrained(model_id, trust_remote_code=True)
    model = model_cls(hf_config).to(dtype=dtype)
    model.to(device).eval()
    return model


def _greedy_step(reference, full_ids, native, tables, cache, conv_state, conv_slots):
    """Return native/reference logits for the next token after ``full_ids``."""
    reference_logits = reference(full_ids, use_cache=False).logits[:, -1].float()
    cursor = full_ids.shape[1] - 1
    native_logits = native(
        full_ids[:, -1:],
        block_tables=tables,
        kv_lens=torch.tensor([cursor], device=full_ids.device),
        paged_kv_caches=cache,
        conv_state_caches=conv_state,
        conv_slots=conv_slots,
        logits_last_only=True,
    )[:, -1].float()
    return native_logits, reference_logits


def validate_prompt(
    *,
    reference,
    native,
    tokenizer,
    prompt,
    max_new_tokens,
    block_size,
):
    prompt_ids = tokenizer.encode(prompt)
    total_tokens = len(prompt_ids) + max_new_tokens
    num_blocks = (total_tokens + block_size - 1) // block_size + 1
    cache = native.init_paged_kv_caches(
        num_blocks=num_blocks, block_size=block_size
    )
    conv_state = native.init_conv_state_caches(
        max_slots=1,
        device=native.tok_emb.weight.device,
        dtype=native.dtype,
    )
    conv_slots = torch.tensor(
        [0], device=native.tok_emb.weight.device, dtype=torch.long
    )
    tables = torch.arange(num_blocks, device=native.tok_emb.weight.device).unsqueeze(
        0
    )

    full_ids = torch.tensor(
        [prompt_ids], device=native.tok_emb.weight.device, dtype=torch.long
    )
    native_ids = []
    reference_ids = []
    step_rows = []

    with torch.inference_mode():
        # Prefill once, then teacher-force both implementations with the
        # reference token at each position.
        native_logits = native(
            full_ids,
            block_tables=tables,
            kv_lens=torch.tensor([0], device=full_ids.device),
            paged_kv_caches=cache,
            conv_state_caches=conv_state,
            conv_slots=conv_slots,
            logits_last_only=True,
            max_seq_len=len(prompt_ids),
        )[:, -1].float()
        reference_logits = reference(full_ids, use_cache=False).logits[:, -1].float()

        for step in range(max_new_tokens):
            error = (native_logits - reference_logits).abs().max().item()
            native_token = int(native_logits.argmax(-1).item())
            reference_token = int(reference_logits.argmax(-1).item())
            native_top2 = torch.topk(native_logits, 2, dim=-1).values[0]
            reference_top2 = torch.topk(reference_logits, 2, dim=-1).values[0]
            step_rows.append(
                {
                    "step": step,
                    "max_abs_logit_error": error,
                    "native_token_id": native_token,
                    "reference_token_id": reference_token,
                    "native_top2_gap": (native_top2[0] - native_top2[1]).item(),
                    "reference_top2_gap": (
                        reference_top2[0] - reference_top2[1]
                    ).item(),
                }
            )
            native_ids.append(native_token)
            reference_ids.append(reference_token)
            if step + 1 >= max_new_tokens:
                break

            next_token = torch.tensor(
                [[reference_token]], device=full_ids.device, dtype=torch.long
            )
            full_ids = torch.cat([full_ids, next_token], dim=1)
            native_logits, reference_logits = _greedy_step(
                reference,
                full_ids,
                native,
                tables,
                cache,
                conv_state,
                conv_slots,
            )

    return {
        "prompt_tokens": len(prompt_ids),
        "max_abs_logit_error": max(row["max_abs_logit_error"] for row in step_rows),
        "native_token_ids": native_ids,
        "reference_token_ids": reference_ids,
        "greedy_match": native_ids == reference_ids,
        "steps": step_rows,
        "native_text": tokenizer.decode(native_ids),
        "reference_text": tokenizer.decode(reference_ids),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("LFM3-1.7B"))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/lfm3_reference.json")
    )
    parser.add_argument(
        "--dtype", choices=["float32", "bfloat16"], default="bfloat16"
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if any tested greedy token differs",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    if device.type == "cuda" and dtype == torch.float32:
        raise SystemExit(
            "float32 validation needs more than 8 GB when both models are resident; "
            "use --dtype bfloat16 or run on CPU"
        )
    if not (args.model_dir / "model.safetensors").exists():
        raise SystemExit(f"missing {args.model_dir / 'model.safetensors'}")

    state = load_file(str(args.model_dir / "model.safetensors"))
    reference = _load_reference(args.model_id, dtype, device)
    missing, unexpected = reference.load_state_dict(state, strict=False)
    if unexpected or any(not key.endswith("lm_head.weight") for key in missing):
        raise SystemExit(
            f"reference checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )

    native = Lfm3Model(LFM3_CONFIG_1_7B)
    load_weights_into_lfm3(native, state)
    del state
    native.to(device=device, dtype=dtype).eval()
    native.dtype = dtype

    tokenizer = Lfm3Tokenizer(str(args.model_dir / "tokenizer.json"))
    prompts = [
        (
            "chat",
            apply_chat_template("What is 2+2? Answer in one short sentence."),
        ),
        ("raw", "The purpose of a KV cache is"),
    ]

    rows = []
    for label, prompt in prompts:
        row = validate_prompt(
            reference=reference,
            native=native,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            block_size=args.block_size,
        )
        row["label"] = label
        row["prompt"] = prompt
        rows.append(row)
        print(
            f"{label}: greedy_match={row['greedy_match']} "
            f"max_abs_logit_error={row['max_abs_logit_error']:.6g} "
            f"native={row['native_token_ids']} "
            f"reference={row['reference_token_ids']}"
        )

    result = {
        "kind": "reference_validation",
        "model_dir": str(args.model_dir),
        "model_id": args.model_id,
        "torch": torch.__version__,
        "dtype": args.dtype,
        "device": str(device),
        "block_size": args.block_size,
        "max_new_tokens": args.max_new_tokens,
        "reference": "Transformers remote-code LFM3, full context recomputed each step",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    mismatches = [row["label"] for row in rows if not row["greedy_match"]]
    if mismatches:
        print(
            "Greedy mismatch in "
            + ", ".join(mismatches)
            + "; inspect the saved per-step logits and top-2 gaps."
        )
        if args.strict:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
