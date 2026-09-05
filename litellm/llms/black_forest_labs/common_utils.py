"""
Black Forest Labs Common Utilities

Common utilities, constants, and error handling for Black Forest Labs API.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import (
    Any,  # noqa: TID251  # base transformation contracts type these payloads as Any
    Final,
)
from urllib.parse import urlparse

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str

EMPTY_MAP: Mapping[str, Any] = MappingProxyType({})  # mutable-ok: frozen shared empty mapping


class BlackForestLabsError(BaseLLMException):
    """Exception class for Black Forest Labs API errors."""


# API Constants
DEFAULT_API_BASE: Final = "https://api.bfl.ai"


def resolve_bfl_api_base(api_base: str | None) -> str:
    """Resolve the BFL API base, honoring an explicit override then BFL_API_BASE then the default."""
    base_url = api_base or get_secret_str("BFL_API_BASE") or DEFAULT_API_BASE
    return base_url.rstrip("/")


def bfl_auth_headers(api_key: str | None) -> Mapping[str, str]:
    """Build the shared BFL request headers, resolving the x-key from arg or environment."""
    resolved_key = api_key or get_secret_str("BFL_API_KEY") or get_secret_str("BLACK_FOREST_LABS_API_KEY")

    if not resolved_key:
        raise BlackForestLabsError(
            status_code=401,
            message="BFL_API_KEY is not set. Please set it via environment variable or pass api_key parameter.",
        )

    return MappingProxyType(
        {  # mutable-ok: frozen header view, merged into the request headers by validate_environment
            "x-key": resolved_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
    )


# BFL uses regional subdomains (e.g. gateway.bfl.ai) for polling URLs that
# differ from the submission host (api.bfl.ai). We validate against the
# registered domain rather than doing a strict same-origin check.
_BFL_REGISTERED_DOMAIN: Final = "bfl.ai"


def assert_bfl_polling_url(polling_url: str) -> None:
    """Validate that a polling URL points to a BFL-controlled host.

    BFL returns polling URLs on subdomains like ``gateway.bfl.ai`` that differ
    from the submission host ``api.bfl.ai``. A strict same-origin check would
    reject these legitimate URLs. Instead we verify the host is ``bfl.ai`` or
    any subdomain of it, which keeps the SSRF guarantee (credentials only go
    to BFL-controlled infrastructure) without false-positives on regional hosts.

    Raises:
        BlackForestLabsError: If the polling URL scheme or host is not trusted.
    """
    parsed: Final = urlparse(polling_url)
    host: Final = (parsed.hostname or "").lower()

    if parsed.scheme != "https":
        raise BlackForestLabsError(
            status_code=502,
            message="Rejected polling URL: scheme must be https",
        )

    if host != _BFL_REGISTERED_DOMAIN and not host.endswith("." + _BFL_REGISTERED_DOMAIN):
        raise BlackForestLabsError(
            status_code=502,
            message="Rejected polling URL: host is not within the bfl.ai domain",
        )


# Polling configuration
DEFAULT_POLLING_INTERVAL: Final = 1.5  # seconds
DEFAULT_MAX_POLLING_TIME: Final = 300  # 5 minutes

# Model to endpoint mapping for image edit
IMAGE_EDIT_MODELS: Final[dict[str, str]] = {
    "flux-kontext-pro": "/v1/flux-kontext-pro",
    "flux-kontext-max": "/v1/flux-kontext-max",
    "flux-pro-1.0-fill": "/v1/flux-pro-1.0-fill",
    "flux-pro-1.0-expand": "/v1/flux-pro-1.0-expand",
    "flux-2-pro": "/v1/flux-2-pro",
    "flux-2-max": "/v1/flux-2-max",
    "flux-2-flex": "/v1/flux-2-flex",
    "flux-2-klein-9b": "/v1/flux-2-klein-9b",
    "flux-2-klein-4b": "/v1/flux-2-klein-4b",
}

# Model to endpoint mapping for video generation
FLUX_3_VIDEO_ENDPOINT = "/v1/flux-3-video"
VIDEO_GENERATION_MODELS: Mapping[str, str] = MappingProxyType(
    {  # mutable-ok: frozen constant lookup table
        "flux-3-video": FLUX_3_VIDEO_ENDPOINT,
    }
)

# Model to endpoint mapping for image generation
IMAGE_GENERATION_MODELS: Final[dict[str, str]] = {
    "flux-pro-1.1": "/v1/flux-pro-1.1",
    "flux-pro-1.1-ultra": "/v1/flux-pro-1.1-ultra",
    "flux-dev": "/v1/flux-dev",
    "flux-pro": "/v1/flux-pro",
    # Kontext models support both text-to-image and image editing
    "flux-kontext-pro": "/v1/flux-kontext-pro",
    "flux-kontext-max": "/v1/flux-kontext-max",
    "flux-2-pro": "/v1/flux-2-pro",
    "flux-2-max": "/v1/flux-2-max",
    "flux-2-flex": "/v1/flux-2-flex",
    "flux-2-klein-9b": "/v1/flux-2-klein-9b",
    "flux-2-klein-4b": "/v1/flux-2-klein-4b",
}
