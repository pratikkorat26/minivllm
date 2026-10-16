"""Compare local Gemma weights with Transformers without any network requests."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from gemma3.model import GEMMA3_CONFIG_270M, Gemma3Model
from gemma3.utilities import load_weights_into_gemma
from runtime.provider import GemmaTokenizer
from safetensors.torch import load_file


def main():
    from transformers import Gemma3ForCausalLM

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("gemma-3-270m"))
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/reference.json")
    )
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--attention", choices=["eager", "sdpa"], default="eager")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    config = dict(GEMMA3_CONFIG_270M, dtype=torch.float32)
    model = Gemma3Model(config)
    load_weights_into_gemma(
        model, config, load_file(str(args.model_dir / "model.safetensors"))
    )
    model.to(device=device, dtype=dtype).eval()
    model.dtype = dtype
    reference = (
        Gemma3ForCausalLM.from_pretrained(
            str(args.model_dir),
            torch_dtype=dtype,
            local_files_only=True,
            attn_implementation=args.attention,
        )
        .to(device)
        .eval()
    )
    tokenizer = GemmaTokenizer(
        str(args.model_dir / "tokenizer.json"), use_instruct_model=False
    )
    rows = []
    with torch.inference_mode():
        for prompt in [
            "The purpose of a KV cache is",
            "Continuous batching improves GPU utilization because",
            "A useful engineering tradeoff in serving is",
        ]:
            ids = torch.tensor([tokenizer.encode(prompt)], device=device)
            cache = model.init_paged_kv_caches(
                num_blocks=16, block_size=16, device=device
            )
            tables = torch.arange(16, device=device).unsqueeze(0)
            full_ids = ids.clone()
            native_ids, reference_ids = [], []
            mismatches = []
            max_error = 0.0
            for step in range(8):
                cursor = 0 if step == 0 else full_ids.shape[1] - 1
                actual = model(
                    full_ids if step == 0 else full_ids[:, -1:],
                    block_tables=tables,
                    kv_lens=torch.tensor([cursor], device=device),
                    paged_kv_caches=cache,
                    max_seq_len=full_ids.shape[1],
                )[:, -1].float()
                expected = reference(full_ids, use_cache=False).logits[:, -1].float()
                max_error = max(max_error, (actual - expected).abs().max().item())
                native_ids.append(actual.argmax(-1).item())
                reference_ids.append(expected.argmax(-1).item())
                if native_ids[-1] != reference_ids[-1]:
                    native_id, reference_id = native_ids[-1], reference_ids[-1]
                    mismatches.append(
                        dict(
                            step=step,
                            reference_logit_gap=(
                                expected[0, reference_id] - expected[0, native_id]
                            ).item(),
                            native_logit_gap=(
                                actual[0, native_id] - actual[0, reference_id]
                            ).item(),
                        )
                    )
                full_ids = torch.cat(
                    [full_ids, expected.argmax(-1, keepdim=True)], dim=1
                )
            rows.append(
                dict(
                    prompt=prompt,
                    max_abs_logit_error=max_error,
                    native_token_ids=native_ids,
                    reference_token_ids=reference_ids,
                    greedy_match=native_ids == reference_ids,
                    mismatches=mismatches,
                )
            )
    result = dict(
        kind="reference_validation",
        model_dir=str(args.model_dir),
        torch=torch.__version__,
        dtype=args.dtype,
        device=str(device),
        reference=f"Transformers {args.attention} attention, full context recomputed at each step",
        rows=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not all(row["greedy_match"] for row in rows):
        raise SystemExit("Greedy token mismatch; inspect saved logits error and dtype.")


if __name__ == "__main__":
    main()
