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
    Collection,
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
    custom_llm_providers: list[str]  # mutable-ok: JSON array; orjson serializes list, not tuple
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
    custom_llm_providers: tuple[str, ...]
    declared: bool
    capability_params: tuple[str, ...]
    supported_openai_params: tuple[str, ...]

    def to_json(self) -> VideoModelCapabilitiesJSON:
        return VideoModelCapabilitiesJSON(
            model=self.model,
            custom_llm_providers=list(self.custom_llm_providers),  # mutable-ok: JSON array; orjson needs list
            declared=self.declared,
            capability_params=list(self.capability_params),  # mutable-ok: JSON array; orjson needs list
            supported_openai_params=list(self.supported_openai_params),  # mutable-ok: JSON array; orjson needs list
        )


@dataclass(frozen=True, slots=True)
class _RouteCapabilities:
    """What one deployment behind a model_name executes."""

    custom_llm_provider: str
    declared: bool
    capability_params: frozenset[str]
    supported_openai_params: frozenset[str]


def _deployment_route(deployment: Mapping[str, Any]) -> tuple[str, str | None] | None:
    """The (model, explicit provider) pair the request path would route with."""
    litellm_params = deployment.get("litellm_params")
    if not isinstance(litellm_params, Mapping):
        return None
    model = litellm_params.get("model")
    if not isinstance(model, str) or not model:
        return None
    provider = litellm_params.get("custom_llm_provider")
    return model, provider if isinstance(provider, str) and provider else None


def _capabilities_for(litellm_model: str, custom_llm_provider: str | None) -> _RouteCapabilities | None:
    try:
        # The deployment's own custom_llm_provider is passed through: a config with an
        # unprefixed or custom model identifier routes on that field, so inferring a
        # provider from the model string alone would report a surface the route never
        # serves (or drop the deployment entirely when inference raises).
        resolved_model, resolved_provider, _, _ = litellm.get_llm_provider(
            model=litellm_model,
            custom_llm_provider=custom_llm_provider,
        )
        provider_config = ProviderConfigManager.get_provider_video_config(
            model=resolved_model,
            provider=litellm.LlmProviders(resolved_provider),
        )
    except Exception:  # noqa: BLE001  # an unroutable deployment must not fail the whole report
        return None

    if provider_config is None:
        return None

    match provider_config.get_capability_param_support(resolved_model):
        case UndeclaredCapabilityParams():
            declared, capability_params = False, frozenset()
        case DeclaredCapabilityParams(supported=supported):
            declared, capability_params = True, supported & CAPABILITY_PARAMS

    try:
        supported_openai_params = frozenset(provider_config.get_supported_openai_params(resolved_model) or ())
    except Exception:  # noqa: BLE001  # a provider that cannot list its params still reports capability params
        supported_openai_params = frozenset()

    return _RouteCapabilities(
        custom_llm_provider=resolved_provider,
        declared=declared,
        capability_params=capability_params,
        supported_openai_params=supported_openai_params,
    )


def _intersect(param_sets: Iterable[frozenset[str]]) -> frozenset[str]:
    merged: frozenset[str] | None = None
    for params in param_sets:
        merged = params if merged is None else merged & params
    return merged if merged is not None else frozenset()


def _merge_routes(model_name: str, routes: tuple[_RouteCapabilities, ...]) -> VideoModelCapabilities:
    """
    Reduce every deployment behind a model_name to what all of them can execute.

    The router may pick any healthy deployment under a name, so a capability only one
    of them handles would 400 intermittently once routing lands elsewhere. Reporting
    the intersection is therefore the only answer a catalog can rely on, and a single
    undeclared route makes the whole name undeclared: its surface is unknown, and
    claiming the others' audited set would promise what that route may not serve.
    """
    declared = all(route.declared for route in routes)
    capability_params = _intersect(route.capability_params for route in routes) if declared else frozenset()
    return VideoModelCapabilities(
        model=model_name,
        # mutable-ok: one-shot dedupe of the providers behind the name
        custom_llm_providers=tuple(sorted({route.custom_llm_provider for route in routes})),
        declared=declared,
        capability_params=tuple(sorted(capability_params)),
        supported_openai_params=tuple(sorted(_intersect(route.supported_openai_params for route in routes))),
    )


def build_video_capability_report(
    deployments: Iterable[Mapping[str, Any]],
    visible_models: Collection[str],
) -> VideoCapabilityReportJSON:
    """
    Report the capability params every configured video deployment can execute.

    ``visible_models`` is the set of model names the calling key may route to, resolved
    the same way /v1/models resolves it. A key or team scoped to a subset of models must
    not learn the deployment metadata of the rest, and a scoped catalog must not be
    handed capabilities for models it cannot call.

    Entries are keyed by proxy model_name and answer "if a video create were routed
    to this name, what would this image execute". Video configs resolve per provider,
    not per model, so a provider that serves video at all yields an entry for each of
    its deployments including chat and image ones; a consumer looks up only the names
    it routes video to, so that surplus is inert.

    A name backed by several deployments reports every provider behind it and only the
    capabilities common to all of them, because the router is free to route to any of
    them; see _merge_routes.

    Two negatives are NOT the same and must not be collapsed. A missing entry means
    the provider has no video config at all. An entry with declared=false means the
    provider has a video config that has not been audited, so its capability surface
    is unknown; reading that as a denial would withhold capabilities that work.
    """
    by_model_name: dict[str, list[_RouteCapabilities]] = {}  # mutable-ok: one-shot accumulator
    for deployment in deployments:
        model_name = deployment.get("model_name")
        route = _deployment_route(deployment)
        if not isinstance(model_name, str) or not model_name or route is None:
            continue
        if model_name not in visible_models:
            continue
        capabilities = _capabilities_for(*route)
        if capabilities is not None:
            by_model_name.setdefault(model_name, []).append(capabilities)  # mutable-ok: accumulator

    return VideoCapabilityReportJSON(
        object="list",
        capability_params_vocabulary=sorted(CAPABILITY_PARAMS),  # mutable-ok: JSON array; orjson needs list
        data=[  # mutable-ok: JSON array; orjson needs list
            _merge_routes(model_name, tuple(by_model_name[model_name])).to_json()
            for model_name in sorted(by_model_name)
        ],
    )
