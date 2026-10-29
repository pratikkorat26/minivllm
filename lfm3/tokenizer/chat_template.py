def apply_chat_template(user_text: str, *, enable_thinking: bool = False) -> str:
    """Format one user turn for the LFM3 post-trained chat template."""
    opening = "<think>" if enable_thinking else "<think></think>"
    return (
        "<|startoftext|><|im_start|>user\n"
        f"{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n{opening}"
    )
