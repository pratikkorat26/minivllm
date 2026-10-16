import torch


def load_weights_into_gemma(model, param_config, params):
    def _to_tensor(value, like: torch.Tensor) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(dtype=like.dtype, device=like.device)
        return torch.as_tensor(value, dtype=like.dtype, device=like.device)

    def copy_param_(left: torch.Tensor, key: str, *, required: bool = True) -> None:
        if key not in params:
            if required:
                raise KeyError(f"Missing required tensor: {key}")
            return

        right = _to_tensor(params[key], like=left)
        if right.shape != left.shape:
            raise ValueError(
                f"Shape mismatch in tensor '{key}'. Left: {left.shape}, Right: {right.shape}"
            )

        with torch.no_grad():
            left.copy_(right)

    # Embedding weights
    copy_param_(model.tok_emb.weight, "model.embed_tokens.weight")

    # Iterate over transformer layers
    for layer_idx in range(param_config["n_layers"]):
        block = model.blocks[layer_idx]
        att = block.att

        # Attention projections
        copy_param_(
            att.q_proj.weight, f"model.layers.{layer_idx}.self_attn.q_proj.weight"
        )
        copy_param_(
            att.k_proj.weight, f"model.layers.{layer_idx}.self_attn.k_proj.weight"
        )
        copy_param_(
            att.v_proj.weight, f"model.layers.{layer_idx}.self_attn.v_proj.weight"
        )
        copy_param_(
            att.out_proj.weight, f"model.layers.{layer_idx}.self_attn.o_proj.weight"
        )

        # QK normalization weights (optional depending on checkpoint/config)
        if att.q_norm is not None:
            copy_param_(
                att.q_norm.scale,
                f"model.layers.{layer_idx}.self_attn.q_norm.weight",
                required=False,
            )
        if att.k_norm is not None:
            copy_param_(
                att.k_norm.scale,
                f"model.layers.{layer_idx}.self_attn.k_norm.weight",
                required=False,
            )

        # Feed forward weights
        copy_param_(
            block.ff.gate.weight, f"model.layers.{layer_idx}.mlp.gate_proj.weight"
        )
        copy_param_(block.ff.up.weight, f"model.layers.{layer_idx}.mlp.up_proj.weight")
        copy_param_(
            block.ff.down.weight, f"model.layers.{layer_idx}.mlp.down_proj.weight"
        )

        # LayerNorm weights
        copy_param_(
            block.input_layernorm.scale,
            f"model.layers.{layer_idx}.input_layernorm.weight",
        )
        copy_param_(
            block.post_attention_layernorm.scale,
            f"model.layers.{layer_idx}.post_attention_layernorm.weight",
        )

        # Pre- and post-feedforward norms (optional in some exports)
        pre_key = f"model.layers.{layer_idx}.pre_feedforward_layernorm.weight"
        post_key = f"model.layers.{layer_idx}.post_feedforward_layernorm.weight"
        copy_param_(block.pre_feedforward_layernorm.scale, pre_key, required=False)
        copy_param_(block.post_feedforward_layernorm.scale, post_key, required=False)

    # Final LayerNorm
    copy_param_(model.final_norm.scale, "model.norm.weight")

    # Output head
    if "lm_head.weight" in params:
        copy_param_(model.out_head.weight, "lm_head.weight")
    else:
        model.out_head.weight = model.tok_emb.weight
        print("Model uses weight tying.")
