"""
Deployed-capability reporting for the video endpoints.

Capability advertisement (what a customer-facing catalog says a model can do) and
capability execution (what the proxy image can actually carry out) live in different
repositories and deploy independently. Nothing forces them to agree, so a catalog
that ships ahead of the proxy advertises inputs the proxy cannot honor.

This module lets the running proxy answer for itself: for every configured video
deployment it reports the capability params that image actually executes, taken from
the same BaseVideoConfig the request path uses. A catalog can then intersect its
static registry against the live answer and publish only what is currently
executable, instead of trusting a build-time assumption.

The report is derived, never hand-maintained: it resolves each deployment through
get_llm_provider and ProviderConfigManager exactly as video_generation does, so it
cannot drift from the code that serves the request.
"""

from dataclasses import dataclass
from typing import (
    Any,  # noqa: TID251  # router deployment dicts are untyped at this boundary
    Iterable,
    Mapping,
    TypedDict,
)

import litellm
from litellm.utils import ProviderConfigManager
from litellm.videos.capabilities import (
    CAPABILITY_PARAMS,
    DeclaredCapabilityParams,
    UndeclaredCapabilityParams,
)


class VideoModelCapabilitiesJSON(TypedDict):
    model: str
    custom_llm_provider: str
    declared: bool
    capability_params: list[str]  # mutable-ok: JSON array; orjson serializes list, not tuple
    supported_openai_params: list[str]  # mutable-ok: JSON array; orjson serializes list, not tuple


class VideoCapabilityReportJSON(TypedDict):
    object: str
    capability_params_vocabulary: list[str]  # mutable-ok: JSON array; orjson serializes list, not tuple
    data: list[VideoModelCapabilitiesJSON]  # mutable-ok: JSON array; orjson serializes list, not tuple


@dataclass(frozen=True, slots=True)
class VideoModelCapabilities:
    model: str
    custom_llm_provider: str
    declared: bool
    capability_params: tuple[str, ...]
    supported_openai_params: tuple[str, ...]

    def to_json(self) -> VideoModelCapabilitiesJSON:
        return VideoModelCapabilitiesJSON(
            model=self.model,
            custom_llm_provider=self.custom_llm_provider,
            declared=self.declared,
            capability_params=list(self.capability_params),  # mutable-ok: JSON array; orjson needs list
            supported_openai_params=list(self.supported_openai_params),  # mutable-ok: JSON array; orjson needs list
        )


def _deployment_model(deployment: Mapping[str, Any]) -> str | None:
    litellm_params = deployment.get("litellm_params")
    if not isinstance(litellm_params, Mapping):
        return None
    model = litellm_params.get("model")
    return model if isinstance(model, str) and model else None


def _capabilities_for(model_name: str, litellm_model: str) -> VideoModelCapabilities | None:
    try:
        resolved_model, custom_llm_provider, _, _ = litellm.get_llm_provider(model=litellm_model)
        provider_config = ProviderConfigManager.get_provider_video_config(
            model=resolved_model,
            provider=litellm.LlmProviders(custom_llm_provider),
        )
    except Exception:  # noqa: BLE001  # an unroutable deployment must not fail the whole report
        return None

    if provider_config is None:
        return None

    match provider_config.get_capability_param_support(resolved_model):
        case UndeclaredCapabilityParams():
            declared, capability_params = False, ()
        case DeclaredCapabilityParams(supported=supported):
            declared, capability_params = True, tuple(sorted(supported & CAPABILITY_PARAMS))

    try:
        supported_openai_params = tuple(provider_config.get_supported_openai_params(resolved_model) or ())
    except Exception:  # noqa: BLE001  # a provider that cannot list its params still reports capability params
        supported_openai_params = ()

    return VideoModelCapabilities(
        model=model_name,
        custom_llm_provider=custom_llm_provider,
        declared=declared,
        capability_params=capability_params,
        supported_openai_params=supported_openai_params,
    )


def build_video_capability_report(
    deployments: Iterable[Mapping[str, Any]],
) -> VideoCapabilityReportJSON:
    """
    Report the capability params every configured video deployment can execute.

    Entries are keyed by proxy model_name and answer "if a video create were routed
    to this name, what would this image execute". Video configs resolve per provider,
    not per model, so a provider that serves video at all yields an entry for each of
    its deployments including chat and image ones; a consumer looks up only the names
    it routes video to, so that surplus is inert.

    Two negatives are NOT the same and must not be collapsed. A missing entry means
    the provider has no video config at all. An entry with declared=false means the
    provider has a video config that has not been audited, so its capability surface
    is unknown; reading that as a denial would withhold capabilities that work.
    """
    named = tuple(
        (model_name, litellm_model)
        for model_name, litellm_model in (
            (deployment.get("model_name"), _deployment_model(deployment)) for deployment in deployments
        )
        if isinstance(model_name, str) and model_name and litellm_model is not None
    )
    # dict() keeps the LAST pair for a duplicate key; reversing keeps the first
    # deployment configured under a model_name, matching the router's own precedence.
    unique = dict(reversed(named))  # mutable-ok: one-shot dedupe, read only as a Mapping below
    resolved = {  # mutable-ok: one-shot comprehension, never mutated after construction
        model_name: capabilities
        for model_name, capabilities in (
            (model_name, _capabilities_for(model_name, litellm_model)) for model_name, litellm_model in unique.items()
        )
        if capabilities is not None
    }

    return VideoCapabilityReportJSON(
        object="list",
        capability_params_vocabulary=sorted(CAPABILITY_PARAMS),  # mutable-ok: JSON array; orjson needs list
        data=[resolved[model_name].to_json() for model_name in sorted(resolved)],  # mutable-ok: JSON array
    )
