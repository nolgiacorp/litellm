"""
Capability-parameter contract for the video generation path.

A *capability parameter* is a video-create parameter that materially changes what
the customer receives: the frames the model is conditioned on, the reference media
it must honor, whether a soundtrack is rendered. Dropping one does not degrade the
output, it silently substitutes a different one, and the caller is billed for a
result that ignored what they asked for.

Every other layer of LiteLLM treats an unsupported parameter as droppable, which is
the right default for tuning knobs (``temperature`` on a model that has none) and the
wrong one here. This module scopes strictness to a closed vocabulary, so the general
``drop_params`` affordance keeps working everywhere else:

1. Only the video create path consults it. Chat, embeddings and image generation are
   untouched, as is ``litellm.drop_params`` / ``additional_drop_params`` semantics for
   every parameter outside :data:`CAPABILITY_PARAMS`.
2. Only names in :data:`CAPABILITY_PARAMS` are gated. A provider that forwards novel
   parameters verbatim keeps doing so; nothing new starts 4xx-ing because a provider
   gained a parameter this vocabulary has never heard of.
3. Only providers that opt in are enforced. A config that does not override
   :meth:`BaseVideoConfig.get_capability_param_support` reports
   :class:`UndeclaredCapabilityParams` and behaves exactly as it does today. Silence
   means "this provider has not been audited", never "this provider supports
   everything".
4. ``drop_params`` does not suppress the gate. Making a capability parameter
   droppable is precisely the failure being fixed, so the two settings are
   deliberately independent: ``drop_params`` governs tuning knobs, this governs
   what the customer actually receives.
"""

from dataclasses import dataclass
from typing import Any, Mapping, NoReturn  # noqa: TID251  # OpenAI-shaped video params are untyped at this boundary

import litellm

CAPABILITY_PARAMS: frozenset[str] = frozenset(
    (
        "input_reference",
        "image_url",
        "end_image_url",
        "image_urls",
        "video_urls",
        "audio_urls",
        "reference_audios",
        "base_video_url",
        "bitrate_mode",
        "generate_audio",
        "negative_prompt",
    )
)
"""
The closed vocabulary this module enforces.

``input_reference`` and ``image_url`` are the same start-frame slot under two names;
callers routinely send both, so a provider that honors one must declare both.

``negative_prompt`` is here for the same reason the frame and reference slots are:
it constrains what the model may render, so discarding it returns a different video
than the caller asked for and bills them for it. It is the one member with no
``GET /models`` capability flag behind it, because it is published unconditionally on
the video request rather than advertised per model.
"""


@dataclass(frozen=True, slots=True)
class DeclaredCapabilityParams:
    """The provider has been audited: ``supported`` is exhaustive for this model."""

    supported: frozenset[str]


@dataclass(frozen=True, slots=True)
class UndeclaredCapabilityParams:
    """The provider has not been audited; the gate does not run for it."""


CapabilityParamSupport = DeclaredCapabilityParams | UndeclaredCapabilityParams


@dataclass(frozen=True, slots=True)
class UnsupportedCapabilityParams:
    """A refusal, modeled as a value so exactly one function raises."""

    model: str
    custom_llm_provider: str
    requested: tuple[str, ...]
    supported: tuple[str, ...]


def _is_requested(value: Any) -> bool:  # noqa: TID251  # video params are OpenAI-shaped and untyped at this boundary
    if value is None:
        return False
    if isinstance(value, (str, bytes, list, tuple, dict, set, frozenset)):
        return len(value) > 0
    return True


def check_capability_params(
    model: str,
    custom_llm_provider: str,
    support: CapabilityParamSupport,
    requested_params: Mapping[str, Any],  # noqa: TID251  # OpenAI-shaped request params
) -> UnsupportedCapabilityParams | None:
    """
    Return a refusal when the request carries capability parameters this model cannot
    execute, or ``None`` when there is nothing to refuse.
    """
    match support:
        case UndeclaredCapabilityParams():
            return None
        case DeclaredCapabilityParams(supported=supported):
            unsupported = tuple(
                sorted(
                    name
                    for name, value in requested_params.items()
                    if name in CAPABILITY_PARAMS and name not in supported and _is_requested(value)
                )
            )
            if not unsupported:
                return None
            return UnsupportedCapabilityParams(
                model=model,
                custom_llm_provider=custom_llm_provider,
                requested=unsupported,
                supported=tuple(sorted(supported & CAPABILITY_PARAMS)),
            )


def raise_public(failure: UnsupportedCapabilityParams) -> NoReturn:
    """Map the refusal onto LiteLLM's existing public 400 contract."""
    named = ", ".join(failure.requested)
    supported = ", ".join(failure.supported) or "none"
    raise litellm.BadRequestError(
        message=(
            f"Model '{failure.model}' does not support video parameter(s): {named}. "
            f"The deployed provider integration has no handling for them, so honoring the request would return a "
            f"video that silently ignored them; refusing instead. "
            f"Capability parameters this model does support: {supported}."
        ),
        model=failure.model,
        llm_provider=failure.custom_llm_provider,
    )
