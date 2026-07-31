from collections.abc import Mapping
from types import MappingProxyType
from typing import Any  # noqa: TID251  # base transformation contracts type these payloads as Any

import litellm
from litellm.constants import MINIMAX_MEDIA_DEFAULT_API_BASE
from litellm.secret_managers.main import get_secret_str

EMPTY_MAP: Mapping[str, Any] = MappingProxyType({})  # mutable-ok: frozen shared empty mapping


def resolve_minimax_media_api_base(api_base: str | None) -> str:
    return (api_base or MINIMAX_MEDIA_DEFAULT_API_BASE).rstrip("/")


def strip_minimax_prefix(model: str) -> str:
    return model.removeprefix("minimax/")


def drop_none_values(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def minimax_bearer_headers(
    headers: Mapping[str, Any],
    api_key: str | None,
) -> dict:  # mutable-ok: validate_environment contracts return dict
    final_api_key = api_key or get_secret_str("MINIMAX_API_KEY") or litellm.api_key
    if not final_api_key:
        raise ValueError("MINIMAX_API_KEY is not set")
    return {**headers, "Authorization": f"Bearer {final_api_key}", "Content-Type": "application/json"}
