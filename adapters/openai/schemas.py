from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


SUPPORTED_MODEL = "gemma-3-270m-it"
MAX_MESSAGES = 64
MAX_TOTAL_CONTENT_CHARS = 65_536
MAX_STOP_SEQUENCES = 4


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=32_768)


class ChatCompletionRequest(BaseModel):
    model: str = Field(default=SUPPORTED_MODEL)
    messages: List[ChatMessage] = Field(min_length=1, max_length=MAX_MESSAGES)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=4096)
    temperature: Optional[float] = Field(default=0.8, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, gt=0.0, le=1.0)
    top_k: Optional[int] = Field(default=50, ge=0, le=4096)
    repetition_penalty: Optional[float] = Field(default=1.1, ge=1.0, le=5.0)
    seed: Optional[int] = Field(default=None, ge=0)
    stop: Optional[Union[str, List[str]]] = None
    stream_options: Optional["StreamOptions"] = None
    stream: bool = False

    @field_validator("model")
    def validate_model(cls, value: str) -> str:
        if value != SUPPORTED_MODEL:
            raise ValueError(f"Only '{SUPPORTED_MODEL}' is supported")
        return value

    @field_validator("stop")
    def validate_stop(cls, value):
        if value is None:
            return value
        values = [value] if isinstance(value, str) else list(value)
        if not values:
            raise ValueError("stop must contain at least one sequence")
        if len(values) > MAX_STOP_SEQUENCES:
            raise ValueError(f"stop supports at most {MAX_STOP_SEQUENCES} sequences")
        if any(sequence == "" for sequence in values):
            raise ValueError("stop sequences must be non-empty")
        return values

    @field_validator("messages")
    def validate_total_content_length(cls, messages):
        total_chars = sum(len(message.content) for message in messages)
        if total_chars > MAX_TOTAL_CONTENT_CHARS:
            raise ValueError(
                f"messages contain more than {MAX_TOTAL_CONTENT_CHARS} characters"
            )
        return messages


class StreamOptions(BaseModel):
    include_usage: bool = False


ChatCompletionRequest.model_rebuild()
