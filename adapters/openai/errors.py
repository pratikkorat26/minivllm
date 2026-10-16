from typing import Union

from app.errors import AppError


class APIError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 500,
        error_type: str = "server_error",
        code: str = "runtime_error",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.code = code


def error_response(error: Union[APIError, AppError]) -> dict:
    return {
        "error": {
            "message": error.message,
            "type": error.error_type,
            "code": error.code,
        }
    }


def app_error_to_api_error(error: AppError) -> APIError:
    status_code = 500
    if error.code == "max_tokens_exceeded" or error.code == "context_length_exceeded":
        status_code = 400
    elif error.code in ("capacity_exceeded", "queue_full"):
        status_code = 429
    elif error.code == "timeout":
        status_code = 504
    return APIError(
        error.message,
        status_code=status_code,
        error_type=error.error_type,
        code=error.code,
    )
