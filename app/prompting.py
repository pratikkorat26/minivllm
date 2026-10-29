import re
from typing import List

from .types import ChatMessage

DANGEROUS_TOKENS = re.compile(
    r"<\|im_start\|>|<\|im_end\|>|<\|startoftext\|>|<\|endoftext\|>|<\|pad\|>",
    re.IGNORECASE,
)
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

MAX_CONTENT_LENGTH = 32_768


def sanitize_user_content(content: str) -> str:
    content = CONTROL_CHARS.sub("", content)
    content = DANGEROUS_TOKENS.sub("", content)
    return content


def messages_to_lfm3_prompt(
    messages: List[ChatMessage], *, enable_thinking: bool = False
) -> str:
    parts = ["<|startoftext|>"]
    for message in messages:
        content = sanitize_user_content(message.content)
        if len(content) > MAX_CONTENT_LENGTH:
            content = content[:MAX_CONTENT_LENGTH]
        if message.role == "assistant" and "<think>" not in content:
            content = f"<think></think>{content}"
        parts.append(
            f"<|im_start|>{message.role}\n{content}<|im_end|>\n"
        )
    opening = "<think>" if enable_thinking else "<think></think>"
    parts.append(f"<|im_start|>assistant\n{opening}")
    return "".join(parts)
