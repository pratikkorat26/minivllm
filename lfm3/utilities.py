import torch


def load_weights_into_lfm3(model, params):
    """Copy a Hugging Face LFM3 checkpoint into :class:`Lfm3Model`."""

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
                f"Shape mismatch in tensor '{key}'. "
                f"Left: {left.shape}, Right: {right.shape}"
            )
        with torch.no_grad():
            left.copy_(right)

    copy_param_(model.tok_emb.weight, "model.embed_tokens.weight")

    for layer_idx, block in enumerate(model.blocks):
        prefix = f"model.layers.{layer_idx}"
        copy_param_(block.operator_norm.weight, f"{prefix}.operator_norm.weight")
        copy_param_(
            block.post_operator_norm.weight, f"{prefix}.post_operator_norm.weight"
        )
        copy_param_(block.ffn_norm.weight, f"{prefix}.ffn_norm.weight")
        copy_param_(block.post_ffn_norm.weight, f"{prefix}.post_ffn_norm.weight")

        # Copy into views: no temporary concatenation of checkpoint tensors.
        gate_weight, up_weight = block.feed_forward.gate_up_proj.weight.chunk(2, dim=0)
        copy_param_(gate_weight, f"{prefix}.feed_forward.w1.weight")
        copy_param_(
            block.feed_forward.w2.weight, f"{prefix}.feed_forward.w2.weight"
        )
        copy_param_(up_weight, f"{prefix}.feed_forward.w3.weight")

        if block.is_attention:
            att = block.self_attn
            q_weight, k_weight, v_weight = att.qkv_proj.weight.split(
                att.qkv_sizes, dim=0
            )
            copy_param_(q_weight, f"{prefix}.self_attn.q_proj.weight")
            copy_param_(k_weight, f"{prefix}.self_attn.k_proj.weight")
            copy_param_(v_weight, f"{prefix}.self_attn.v_proj.weight")
            copy_param_(att.o_proj.weight, f"{prefix}.self_attn.o_proj.weight")
            copy_param_(
                att.gate_proj.weight, f"{prefix}.self_attn.gate_proj.weight"
            )
            copy_param_(att.q_norm.weight, f"{prefix}.self_attn.q_norm.weight")
            copy_param_(att.k_norm.weight, f"{prefix}.self_attn.k_norm.weight")
        else:
            conv = block.conv
            copy_param_(conv.conv.weight, f"{prefix}.conv.conv.weight")
            copy_param_(conv.in_proj.weight, f"{prefix}.conv.in_proj.weight")
            copy_param_(conv.out_proj.weight, f"{prefix}.conv.out_proj.weight")
            if conv.bias:
                copy_param_(conv.conv.bias, f"{prefix}.conv.conv.bias")
                copy_param_(conv.in_proj.bias, f"{prefix}.conv.in_proj.bias")
                copy_param_(conv.out_proj.bias, f"{prefix}.conv.out_proj.bias")

    copy_param_(model.embedding_norm.weight, "model.embedding_norm.weight")

    if "lm_head.weight" in params:
        copy_param_(model.out_head.weight, "lm_head.weight", required=False)
    else:
        model.out_head.weight = model.tok_emb.weight
