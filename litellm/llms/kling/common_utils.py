from litellm.constants import KLING_DEFAULT_API_BASE
from litellm.secret_managers.main import get_secret_str

KLING_TASK_STATUS_MAP = {
    "submitted": "queued",
    "processing": "in_progress",
    "succeed": "completed",
    "failed": "failed",
}


def resolve_kling_api_base(api_base: str | None) -> str:
    base = api_base or get_secret_str("KLING_API_BASE") or KLING_DEFAULT_API_BASE
    return base.rstrip("/")


def strip_kling_prefix(model: str) -> str:
    stripped = model.removeprefix("kling/").strip("/")
    if not stripped:
        raise ValueError("Kling model id is empty after stripping the provider prefix")
    return stripped
