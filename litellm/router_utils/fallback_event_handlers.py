import hashlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

import litellm
from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import get_metadata_variable_name_from_kwargs
from litellm.litellm_core_utils.redact_messages import should_redact_message_logging
from litellm.litellm_core_utils.secret_redaction import redact_string
from litellm.litellm_core_utils.sensitive_data_masker import mask_sensitive_structure
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.router_utils.add_retry_fallback_headers import (
    add_fallback_headers_to_response,
    get_fallback_error_info,
)
from litellm.router_utils.batch_utils import _get_router_metadata_variable_name
from litellm.router_utils.cooldown_handlers import (
    _first_present,  # pyright: ignore[reportPrivateUsage] - shared internal helper, used across router_utils
    _set_cooldown_deployments,  # pyright: ignore[reportPrivateUsage] - shared helper, used across router_utils
    cast_exception_status_to_int,
    is_advisor_orchestration_failure,
)
from litellm.router_utils.router_callbacks.track_deployment_metrics import (
    increment_deployment_failures_for_current_minute,
)
from litellm.types.router import LiteLLMParamsTypedDict
from litellm.types.utils import all_litellm_params

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any

# Status codes a generic API call's caller-supplied resource id can trigger on its own
# (e.g. a nonexistent file/batch/thread id), independent of the selected deployment's health.
_REQUEST_SCOPED_STATUS_CODES: Final = frozenset((404,))


def _trigger_cooldown_for_failed_deployment(
    litellm_router: LitellmRouter,
    kwargs: Mapping[str, Any],
    exception: Exception,
) -> None:
    """
    Trigger cooldown for a failed fallback deployment.

    In the fallback path the normal failure-callback cooldown is skipped because the
    Logging object sets has_logged_async_failure=True after the first failure and
    blocks all subsequent failure callbacks. This helper ensures every failed
    fallback deployment is evaluated for cooldown regardless.
    """
    try:
        if is_advisor_orchestration_failure(exception):
            verbose_router_logger.debug(
                "Not triggering cooldown for fallback deployment: failure originated "
                "from advisor orchestration, not the selected deployment."
            )
            return

        exception_status: Final[str | int] = getattr(exception, "status_code", "")

        # Generic API calls (files, batches, threads, rerank, ...) take a caller-supplied
        # resource id, so a 404 there usually means "that id doesn't exist" rather than
        # "this deployment is unhealthy". Left unguarded, one bad id would 404 every
        # deployment in the fallback chain and cool all of them down from a single request.
        if (
            kwargs.get("original_generic_function") is not None
            and cast_exception_status_to_int(exception_status) in _REQUEST_SCOPED_STATUS_CODES
        ):
            verbose_router_logger.debug(
                "Not triggering cooldown for fallback deployment: status %s on a generic API "
                "call is caller-attributable, not a deployment health signal.",
                exception_status,
            )
            return

        # The proxy's `x-litellm-timeout` header lets a caller set an arbitrarily short
        # timeout, which litellm.Timeout reports as status 408 regardless of the deployment's
        # actual health. Left unguarded, a caller could force a 408 on every deployment in
        # the fallback chain from a single request with a near-zero timeout.
        if kwargs.get("client_side_timeout") and cast_exception_status_to_int(exception_status) == 408:
            verbose_router_logger.debug(
                "Not triggering cooldown for fallback deployment: a caller-supplied "
                "x-litellm-timeout caused this 408, not deployment health."
            )
            return

        # Only Router._set_failed_deployment_id_on_exception()'s server-stamped id is
        # trusted here: a metadata-bucket lookup (e.g. "metadata"/"litellm_metadata")
        # can't reliably tell a caller-supplied bucket from a router-authored one
        # without knowing this call's function_name, so a client with permission to
        # set metadata could otherwise get an arbitrary deployment cooled down.
        deployment_id: Final[str | None] = getattr(exception, "failed_deployment_id", None)

        if deployment_id is None:
            verbose_router_logger.debug("Cannot trigger cooldown for fallback: no failed_deployment_id on exception")
            return

        # Priority: deployment config > response header > router default, matching
        # Router.deployment_callback_on_failure's precedence for the primary path.
        deployment_dict: Final = litellm_router.get_model_info(id=deployment_id)
        deployment_cooldown: Final = (
            _first_present(
                deployment_dict.get("model_info"), deployment_dict.get("litellm_params"), key="cooldown_time"
            )
            if deployment_dict is not None
            else None
        )
        exception_headers: Final = litellm.litellm_core_utils.exception_mapping_utils._get_response_headers(
            original_exception=exception
        )
        _get_retry_after: Final = (
            litellm.utils._get_retry_after_from_exception_header  # pyright: ignore[reportPrivateUsage] - as router.py
        )
        header_cooldown: Final = (
            _get_retry_after(response_headers=exception_headers) if exception_headers is not None else None
        )
        time_to_cooldown: Final = (
            deployment_cooldown
            if deployment_cooldown is not None and deployment_cooldown >= 0
            else (
                header_cooldown
                if header_cooldown is not None and header_cooldown >= 0
                else litellm_router.cooldown_time
            )
        )

        increment_deployment_failures_for_current_minute(
            litellm_router_instance=litellm_router,
            deployment_id=deployment_id,
        )
        _set_cooldown_deployments(
            litellm_router_instance=litellm_router,
            exception_status=exception_status,
            original_exception=exception,
            deployment=deployment_id,
            time_to_cooldown=time_to_cooldown,
        )

        verbose_router_logger.debug("Triggered cooldown for fallback deployment %s", deployment_id)
    except Exception as e:  # noqa: BLE001 - best-effort cooldown trigger must never break the fallback response itself
        verbose_router_logger.debug("Error triggering cooldown for fallback deployment: %s", e)


def fallback_attempt_key(fallback_target: object) -> str | None:
    """
    Identity of one fallback attempt, so the same attempt is never made twice per request.

    A bare model group name and a `{"model": name}` entry describe the same attempt. An
    entry carrying anything else describes a different one and keeps its own identity: a
    client-side fallback list overrides request params such as `messages`, and the router
    re-targets the group that just failed by attaching `_target_order` or
    `_excluded_deployment_ids` to select a different set of deployments inside it. The
    payload is hashed rather than kept, so a large `messages` override does not make the
    request hold a second copy of itself.

    Returns None for a shape with no usable identity, which is never skipped.
    """
    if isinstance(fallback_target, str):
        return fallback_target
    if not isinstance(fallback_target, dict):
        return None
    model: Final = fallback_target.get("model")
    if tuple(fallback_target) == ("model",) and isinstance(model, str):
        return model
    serialized: Final = json.dumps(fallback_target, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


@dataclass(slots=True)
class AttemptedFallbackTargets:
    """
    The fallback attempts a single request has already made.

    One instance is created on the first fallback hop and shared by reference for the rest
    of the walk, so an attempt made in one branch is not repeated in a sibling branch.
    Without it the walk enumerates paths rather than attempts: a fallback graph containing
    a cycle retries one deterministic failure once per path through the cycle, and a
    client-side fallback list is re-walked at every level of the recursion.
    """

    keys: frozenset[str] = frozenset()

    def __contains__(self, key: str) -> bool:
        return key in self.keys

    def record(self, key: str) -> None:
        self.keys = self.keys | frozenset((key,))


# Keys a fallback entry can carry without changing the request the provider
# rejected: the routing keys the router adds itself when it re-points the same
# request, and the LiteLLM-level settings (api_key, api_base, timeout, metadata,
# num_retries, ...) that configure the call but never reach the provider payload.
# An entry made only of these resends the rejected payload verbatim.
_ROUTER_FALLBACK_ENTRY_KEYS = frozenset(("model", "_target_order", "_excluded_deployment_ids"))
_NON_PAYLOAD_FALLBACK_KEYS = _ROUTER_FALLBACK_ENTRY_KEYS | frozenset(all_litellm_params) | frozenset(("timeout",))


def get_fallback_model_name(fallback: object) -> str | None:
    """Return only the routing identifier from a fallback entry."""
    if isinstance(fallback, str):
        return fallback
    if isinstance(fallback, dict) and isinstance(fallback.get("model"), str):
        return fallback["model"]
    return None


def _fallback_names_for_log(fallbacks: Sequence[object]) -> str:
    names = (get_fallback_model_name(fallback) for fallback in fallbacks)
    return f"[{', '.join(repr(name) for name in names)}]"


def _fallback_error_for_log(
    error: Exception,
    kwargs: dict,  # mutable-ok: existing router kwargs are updated throughout fallback handling
    limit: int = 600,
) -> str:
    if should_redact_message_logging(
        {  # mutable-ok: callback redaction API requires this request metadata mapping
            "litellm_params": kwargs,
            "standard_callback_dynamic_params": kwargs.get("standard_callback_dynamic_params")
            or {  # mutable-ok: callback redaction API requires this dynamic parameter mapping
                "turn_off_message_logging": kwargs.get("turn_off_message_logging")
            },
        }
    ):
        return "redacted-by-litellm"
    flat = " ".join(redact_string(str(error)).split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


@runtime_checkable
class _ObjectAsyncIterator(Protocol):
    def __aiter__(self) -> AsyncIterator[object]: ...

    async def __anext__(self) -> object: ...


class _FallbackSuccessAsyncIterator:
    def __init__(
        self,
        inner: _ObjectAsyncIterator,
        on_success: Callable[[], Awaitable[None]],
    ) -> None:
        self._inner = inner
        self._on_success = on_success
        self._success_logged = False

    def __aiter__(self) -> "_FallbackSuccessAsyncIterator":
        return self

    async def __anext__(self) -> object:
        try:
            return await self._inner.__anext__()
        except StopAsyncIteration:
            if not self._success_logged:
                self._success_logged = True
                await self._on_success()
            raise

    async def aclose(self) -> None:
        aclose = getattr(self._inner, "aclose", None)
        if callable(aclose):
            await aclose()


def _attach_custom_stream_success_log(
    response: CustomStreamWrapper,
    on_success: Callable[[], Awaitable[None]],
) -> CustomStreamWrapper:
    stream_class = type(response)
    stream_aiter = stream_class.__aiter__
    stream_anext = stream_class.__anext__

    def __aiter__(self: CustomStreamWrapper) -> AsyncIterator[object]:
        iterator = stream_aiter(self)
        if iterator is self or not isinstance(iterator, _ObjectAsyncIterator):
            return self
        return _FallbackSuccessAsyncIterator(iterator, on_success)

    async def __anext__(self: CustomStreamWrapper) -> object:
        try:
            return await stream_anext(self)
        except StopAsyncIteration:
            if not getattr(self, "_fallback_success_logged", False):
                self._fallback_success_logged = True
                await on_success()
            raise

    completion_class = type(
        f"_FallbackSuccess{stream_class.__name__}",
        (stream_class,),
        {  # mutable-ok: type() requires a mutable namespace mapping
            "__aiter__": __aiter__,
            "__anext__": __anext__,
        },
    )
    response.__class__ = completion_class
    return response


async def _finalize_fallback_success(
    response: object,
    fallback_model_name: str | None,
    original_model_group: str,
    kwargs: dict,  # mutable-ok: existing router kwargs feed fallback callbacks
    original_exception: Exception,
) -> object:
    success_logged = False

    async def log_success() -> None:
        nonlocal success_logged
        if success_logged:
            return
        success_logged = True
        verbose_router_logger.warning("router_fallback_succeeded model_group=%s", fallback_model_name)
        await log_success_fallback_event(
            original_model_group=original_model_group,
            kwargs=kwargs,
            original_exception=original_exception,
        )

    if isinstance(response, CustomStreamWrapper):
        try:
            return _attach_custom_stream_success_log(response, log_success)
        except TypeError:
            await log_success()
            return response
    if isinstance(response, _ObjectAsyncIterator):
        return _FallbackSuccessAsyncIterator(response, log_success)

    await log_success()
    return response


def is_router_availability_error(error: Exception) -> bool:
    """
    True when the router, not a provider, produced the error.

    The router reuses litellm.BadRequestError for its own configuration and
    availability failures: no healthy deployment in a model group, a model name
    matching deployments from several teams, no deployment configured for
    pass-through. Those say nothing about the request, so an explicitly configured
    fallback to a healthy group is the right answer and has to keep working.

    Router-generated errors carry no llm_provider, because no provider was reached
    to produce them. That signal is necessary but not sufficient: some provider
    param validators raise litellm.UnsupportedParamsError without naming the
    provider (litellm/images/utils.py, the provider transformations), and those
    are refusals of the request. Their type says so regardless of llm_provider, so
    they are classified first; anything else without a provider falls back to
    being treated as a router failure, which keeps failover working.
    """
    if isinstance(error, litellm.UnsupportedParamsError):
        return False
    return not getattr(error, "llm_provider", None)


def fallback_transforms_request(fallback: Any) -> bool:
    """
    True when a fallback entry rewrites the request the provider rejected.

    A client-side fallback such as {"model": "backup", "messages": [...]} is a
    deliberate request repair: it sends a different payload, so a rejection of the
    original does not predict its outcome. Entries that only re-point or
    reconfigure the call ("backup", {"model": ..., "_target_order": ...},
    {"model": "backup", "api_key": ...}) resend the rejected payload unchanged and
    therefore cannot repair a rejection.
    """
    if not isinstance(fallback, dict):
        return False
    return any(key not in _NON_PAYLOAD_FALLBACK_KEYS for key in fallback)


def get_request_transforming_fallbacks(fallbacks: Sequence[Any] | None) -> tuple[Any, ...]:
    """
    The client-side fallbacks that repair the request itself, if any.

    Only the non-standard (client-side) fallback formats can carry request
    overrides; the standard {model_group: [...]} mapping never does.
    """
    if not _check_non_standard_fallback_format(fallbacks=fallbacks):
        return ()
    return tuple(fallback for fallback in fallbacks or () if fallback_transforms_request(fallback))


def is_request_rejection(error: Exception) -> bool:
    """
    True when the error means the REQUEST is invalid, rather than the deployment
    being unhealthy.

    Fallbacks exist to route around a sick deployment: a 5xx, a timeout, a rate
    limit, exhausted capacity. Retrying those elsewhere is the whole point. A
    deliberate rejection of the request as written is the opposite case. Every
    deployment that implements the same contract will reject it identically, so
    failing over cannot fix it; it can only find a deployment whose contract
    happens to differ, silently substituting a provider the caller never asked
    for and billing them for it.

    That is not hypothetical. A video request carrying a capability param the
    primary provider cannot execute is refused by
    litellm/videos/capabilities.py, and before this check the router treated
    that refusal as a failed submit and re-ran the request on a fallback twin
    that did accept the param, so the caller silently got a different provider's
    render instead of the 400 the gate raised.

    Scoped to 400 and 422, the two statuses that mean "the request is wrong".
    Auth, permission and not-found (401/403/404) keep falling over, since those
    describe a broken deployment rather than a bad request, and 408/409/429/5xx
    are untouched.

    Context-window and content-policy errors are excluded even though they are
    400s: they have their own dedicated fallback lists, and honoring those is a
    deliberate feature rather than a silent substitution. Router-generated 400s
    (see is_router_availability_error) are excluded too, since they describe the
    router's own configuration rather than the request.
    """
    if isinstance(error, (litellm.ContextWindowExceededError, litellm.ContentPolicyViolationError)):
        return False
    if not isinstance(error, (litellm.BadRequestError, litellm.UnprocessableEntityError)):
        return False
    return not is_router_availability_error(error)


def _check_stripped_model_group(model_group: str, fallback_key: str) -> bool:
    """
    Handles wildcard routing scenario

    where fallbacks set like:
    [{"gpt-3.5-turbo": ["claude-3-haiku"]}]

    but model_group is like:
    "openai/gpt-3.5-turbo"

    Returns:
    - True if the stripped model group == fallback_key
    """
    for provider in litellm.provider_list:
        if isinstance(provider, Enum):
            _provider = provider.value
        else:
            _provider = provider
        if model_group.startswith(f"{_provider}/"):
            stripped_model_group = model_group.replace(f"{_provider}/", "")
            if stripped_model_group == fallback_key:
                return True
    return False


PRE_ROUTING_SELECTED_MODEL_KEY: Final = "pre_routing_selected_model"
_ROUTER_METADATA_BUCKETS: Final = ("metadata", "litellm_metadata")


def record_pre_routing_selection(request_kwargs: Mapping[str, Any] | None, selected_model: str) -> None:
    """
    Remember which model a pre-routing hook picked, so fallback lookup can key off it.

    Fallback resolution runs on an outer kwargs dict that ``**kwargs`` already copied, so
    writing the model there is invisible by the time routing picks a tier. The metadata
    buckets are nested dicts shared by reference across those copies, which is how the
    router already carries values back up.

    The write goes through the proxy-internal bucket resolver, never into both buckets:
    on /v1/messages the top-level ``metadata`` dict is the provider's own request field,
    so a blanket write would forward the tier stamp upstream.
    """
    if request_kwargs is None:
        return
    bucket: Final = request_kwargs.get(get_metadata_variable_name_from_kwargs(request_kwargs))
    if isinstance(bucket, dict):
        bucket[PRE_ROUTING_SELECTED_MODEL_KEY] = selected_model


def clear_pre_routing_selection(request_kwargs: Mapping[str, object] | None) -> None:
    """
    Drop any selection the router did not make itself on this hop.

    The buckets carry whatever the caller sent, so an inbound value is the caller
    choosing a fallback chain rather than the router choosing a tier. A fallback hop
    also inherits the previous hop's selection, which would key its own failure off
    the tier that already failed. Clearing at the start of every hop leaves only a
    value the pre-routing hook wrote while routing that hop.
    """
    if request_kwargs is None:
        return
    for bucket in (request_kwargs.get(name) for name in _ROUTER_METADATA_BUCKETS):
        if isinstance(bucket, dict) and PRE_ROUTING_SELECTED_MODEL_KEY in bucket:
            del bucket[PRE_ROUTING_SELECTED_MODEL_KEY]


def get_pre_routing_selection(kwargs: Mapping[str, Any]) -> str | None:
    """The model a pre-routing hook selected for this request, if one did."""
    buckets: Final = (kwargs.get(name) for name in _ROUTER_METADATA_BUCKETS)
    selections: Final = (bucket.get(PRE_ROUTING_SELECTED_MODEL_KEY) for bucket in buckets if isinstance(bucket, dict))
    return next((selected for selected in selections if isinstance(selected, str) and selected), None)


DISABLE_FALLBACKS_METADATA_KEY: Final = "_disable_fallbacks"


def record_disable_fallbacks(request_kwargs: Mapping[str, Any] | None, disabled: bool) -> None:
    """
    Write-or-clear the request's disable_fallbacks verdict into the router-internal metadata
    bucket. The wrapper pops the raw kwarg before any downstream frame runs, so the refusal
    gate (which decides whether to convert a refusal into a recoverable error) needs this
    carrier to know recovery is impossible.
    """
    from litellm.litellm_core_utils.core_helpers import get_metadata_variable_name_from_kwargs

    if request_kwargs is None:
        return
    bucket: Final = request_kwargs.get(get_metadata_variable_name_from_kwargs(request_kwargs))
    if not isinstance(bucket, dict):
        return
    if disabled:
        bucket[DISABLE_FALLBACKS_METADATA_KEY] = True
    else:
        bucket.pop(DISABLE_FALLBACKS_METADATA_KEY, None)


def fallbacks_disabled_for_request(kwargs: Mapping[str, Any]) -> bool:
    """True when this request opted out of fallbacks, read from the raw kwarg (pre-pop
    snapshots keep it) or the router-internal bucket the wrapper stamps after popping it."""
    if kwargs.get("disable_fallbacks") is True:
        return True
    buckets: Final = (kwargs.get(name) for name in _ROUTER_METADATA_BUCKETS)
    return any(isinstance(bucket, dict) and bucket.get(DISABLE_FALLBACKS_METADATA_KEY) is True for bucket in buckets)


def fallback_lookup_groups(kwargs: Mapping[str, Any], model_group: str | None) -> tuple[str, ...]:
    """
    Ordered keys for resolving a fallback chain: the tier a pre-routing hook selected wins,
    then the routed group, then the requested group. The routed group differs when Claude Code
    session affinity remaps a subagent's concrete model to its bound router.
    """
    metadata: Final = kwargs.get(get_metadata_variable_name_from_kwargs(kwargs))
    routed_group_value: Final = metadata.get("model_group") if isinstance(metadata, Mapping) else None
    routed_group: Final = routed_group_value if isinstance(routed_group_value, str) else None
    ordered: Final = (get_pre_routing_selection(kwargs), routed_group, model_group)
    return tuple(dict.fromkeys(group for group in ordered if group))


def _resolved_a_specific_chain(
    fallbacks: list[Any],  # mutable-ok: mirrors get_fallback_model_group's contract
    result: tuple[list[str] | None, int | None],  # mutable-ok: mirrors get_fallback_model_group's contract
) -> bool:
    resolved, generic_idx = result
    if resolved is None:
        return False
    return generic_idx is None or resolved is not fallbacks[generic_idx]["*"]


def get_fallback_model_group_for_lookup_groups(
    fallbacks: list[Any],  # mutable-ok: mirrors get_fallback_model_group's contract
    lookup_groups: tuple[str, ...],
) -> tuple[list[str] | None, int | None]:  # mutable-ok: mirrors get_fallback_model_group's contract
    """
    First lookup group with a specifically-keyed chain wins; the generic "*" chain applies
    only after every group missed, so a catch-all cannot shadow a later group's own chain.
    """
    results: Final = tuple(get_fallback_model_group(fallbacks=fallbacks, model_group=group) for group in lookup_groups)
    specific: Final = next((result for result in results if _resolved_a_specific_chain(fallbacks, result)), None)
    if specific is not None:
        return specific
    return next((result for result in results if result[0] is not None), (None, None))


def get_fallback_model_group(fallbacks: list[Any], model_group: str) -> tuple[list[str] | None, int | None]:
    """
    Returns:
    - fallback_model_group: List[str] of fallback model groups. example: ["gpt-4", "gpt-3.5-turbo"]
    - generic_fallback_idx: int of the index of the generic fallback in the fallbacks list.

    Checks:
    - exact match
    - stripped model group match
    - generic fallback
    """
    generic_fallback_idx: int | None = None
    stripped_model_fallback: list[str] | None = None
    fallback_model_group: list[str] | None = None
    ## check for specific model group-specific fallbacks
    for idx, item in enumerate(fallbacks):
        if isinstance(item, dict):
            if list(item.keys())[0] == model_group:  # check exact match
                fallback_model_group = item[model_group]
                break
            elif _check_stripped_model_group(
                model_group=model_group, fallback_key=list(item.keys())[0]
            ):  # check generic fallback
                stripped_model_fallback = item[list(item.keys())[0]]
            elif list(item.keys())[0] == "*":  # check generic fallback
                generic_fallback_idx = idx
        elif isinstance(item, str):
            fallback_model_group = [item]
    ## if none, check for generic fallback
    if fallback_model_group is None:
        if stripped_model_fallback is not None:
            fallback_model_group = stripped_model_fallback
        elif generic_fallback_idx is not None:
            fallback_model_group = fallbacks[generic_fallback_idx]["*"]

    return fallback_model_group, generic_fallback_idx


PROVIDER_SCOPED_RESOURCE_KEYS: Final = ("input_file_id", "training_file", "batch_id", "file_id", "fine_tuning_job_id")
PROVIDER_SCOPED_RESOURCE_FUNCTION_NAMES: Final = frozenset(
    {
        "_acreate_batch",
        "_acancel_batch",
        "acreate_fine_tuning_job",
        "acancel_fine_tuning_job",
        "aretrieve_fine_tuning_job",
        "afile_content",
        "afile_delete",
    }
)
PROVIDER_SCOPED_CREATION_FUNCTION_NAMES: Final = frozenset({"_acreate_file"})


def _get_fallback_target_model_group(fallback_entry: str | Mapping[str, object]) -> str | None:
    if isinstance(fallback_entry, str):
        return fallback_entry
    target: Final = fallback_entry.get("model")
    return target if isinstance(target, str) else None


async def _is_fallback_target_authorized(
    litellm_router: LitellmRouter,
    fallback_entry: str | Mapping[str, object],
    original_model_group: str,
    kwargs: Mapping[str, object],
) -> bool:
    access_check: Final = litellm_router.fallback_access_check
    target: Final = _get_fallback_target_model_group(fallback_entry)
    if access_check is None or target is None or target == original_model_group:
        return True
    if await access_check(model=target, request_kwargs=kwargs, llm_router=litellm_router):
        return True
    verbose_router_logger.info(
        "Skipping fallback to model_group = %s: caller is not authorized to call it",
        mask_sensitive_structure(fallback_entry),
    )
    return False


def references_provider_scoped_resource(kwargs: Mapping[str, object]) -> bool:
    """
    True when a file, batch, or fine-tuning job operation names an id that only exists
    under one provider's credentials.

    Each of those ids lives in the account of the deployment that issued it. Handing it to
    a different model group asks a provider about an id it never issued, which costs an
    extra round trip that can only answer not-found. Generic calls dispatched through
    `Router._ageneric_api_call_with_fallbacks` carry the real handler in
    `original_generic_function`, so both slots are checked. Gating on the handler name
    keeps completion-style requests eligible for cross-group fallback even when a caller
    passes a stray extra body field that happens to share one of these key names.
    """
    handler_names: Final = tuple(
        getattr(kwargs.get(function_key), "__name__", None)
        for function_key in ("original_function", "original_generic_function")
    )
    if all(name not in PROVIDER_SCOPED_RESOURCE_FUNCTION_NAMES for name in handler_names):
        return False
    return any(kwargs.get(key) for key in PROVIDER_SCOPED_RESOURCE_KEYS)


def creates_provider_scoped_resource(kwargs: Mapping[str, object]) -> bool:
    """
    True when the request creates a resource that will live under one provider's credentials.

    A file uploaded for batches or fine-tuning is stored in the account of the deployment
    that handled it, and its id is only usable against the model group the caller named.
    Letting the upload fall back to a different model group silently stores the file with
    the wrong provider, and every later use of the returned id fails.
    """
    return getattr(kwargs.get("original_function"), "__name__", None) in PROVIDER_SCOPED_CREATION_FUNCTION_NAMES


async def run_async_fallback(
    *args: tuple[Any],
    litellm_router: LitellmRouter,
    fallback_model_group: list[str],
    original_model_group: str,
    original_exception: Exception,
    max_fallbacks: int,
    fallback_depth: int,
    include_fallback_errors: bool = False,
    **kwargs,
) -> Any:
    """
    Loops through all the fallback model groups and calls kwargs["original_function"] with the arguments and keyword arguments provided.

    If the call is successful, it logs the success and returns the response.
    If the call fails, it logs the failure and continues to the next fallback model group.
    If all fallback model groups fail, it raises the most recent exception.

    Args:
        litellm_router: The litellm router instance.
        *args: Positional arguments.
        fallback_model_group: List[str] of fallback model groups. example: ["gpt-4", "gpt-3.5-turbo"]
        original_model_group: The original model group. example: "gpt-3.5-turbo"
        original_exception: The original exception.
        **kwargs: Keyword arguments. `attempted_targets` carries the fallback attempts
            already made for this request, created on the first hop and shared by reference
            for the rest of the walk. A target already in it is skipped, so neither a
            fallback graph that loops back on itself nor a client-side fallback list
            re-walked at each level can repeat an attempt that has already failed. Identity
            comes from `fallback_attempt_key`, so an entry that overrides request params or
            re-targets the failed group with a different deployment selection stays distinct
            from a bare name.

    Returns:
        The response from the successful fallback model group.
    Raises:
        The most recent exception if all fallback model groups fail.
    """

    ### BASE CASE ### MAX FALLBACK DEPTH REACHED
    if fallback_depth >= max_fallbacks:
        raise original_exception

    error_from_fallbacks = original_exception
    fallback_errors = (get_fallback_error_info(original_exception),)
    metadata_variable_name: Final = _get_router_metadata_variable_name(
        function_name=getattr(kwargs.get("original_function"), "__name__", None)
    )
    same_model_group_only: Final = references_provider_scoped_resource(kwargs) or creates_provider_scoped_resource(
        kwargs
    )
    # Read out of kwargs and narrowed here rather than declared as a parameter: every caller
    # reaches this function by spreading a loosely-typed kwargs dict, so a declared parameter
    # would carry an annotation that no call site can actually be checked against.
    carried_targets: Final = kwargs.get("attempted_targets")
    attempted: Final = (
        carried_targets if isinstance(carried_targets, AttemptedFallbackTargets) else AttemptedFallbackTargets()
    )
    failed_model_group: Final = get_pre_routing_selection(kwargs) or original_model_group
    attempted.record(failed_model_group)
    rejected_request: Exception | None = None
    trigger_logged = False

    for mg in fallback_model_group:
        fallback_model_name = get_fallback_model_name(mg)
        if mg == failed_model_group:
            continue
        # Once the request itself has been rejected, only entries that rewrite it
        # can still help; the rest resend the rejected payload.
        if rejected_request is not None and not fallback_transforms_request(mg):
            continue
        if same_model_group_only and _get_fallback_target_model_group(mg) != original_model_group:
            verbose_router_logger.info(
                "Skipping fallback to model_group = %s: request names a resource owned by model_group = %s",
                mask_sensitive_structure(mg),
                original_model_group,
            )
            continue
        if not await _is_fallback_target_authorized(litellm_router, mg, original_model_group, kwargs):
            continue
        attempt_key = fallback_attempt_key(mg)
        if attempt_key is not None:
            if attempt_key in attempted:
                verbose_router_logger.info(
                    "Skipping fallback to model_group = %s, already attempted for this request",
                    mask_sensitive_structure(mg),
                )
                continue
            attempted.record(attempt_key)
        try:
            # LOGGING
            kwargs = litellm_router.log_retry(kwargs=kwargs, e=original_exception)
            if not trigger_logged:
                trigger_logged = True
                verbose_router_logger.warning(
                    "router_fallback_triggered model_group=%s error_type=%s status_code=%s fallbacks=%s error=%s",
                    original_model_group,
                    type(original_exception).__name__,
                    getattr(original_exception, "status_code", None),
                    _fallback_names_for_log(fallback_model_group),
                    _fallback_error_for_log(original_exception, kwargs),
                )
            # WARNING, not INFO: the proxy's default log level hides INFO, and a
            # hop being taken is exactly what operators need to see (which
            # deployment actually served a request the caller billed to
            # another). Pairs with router_fallback_triggered in Router.
            verbose_router_logger.warning(
                "router_fallback_attempt model_group=%s from=%s",
                fallback_model_name,
                kwargs.get("model"),
            )
            kwargs.pop("_target_order", None)  # rebind-ok: next hop must not inherit the previous order target
            if isinstance(mg, str):
                kwargs["model"] = mg
            elif isinstance(mg, dict):
                kwargs.update(mg)
            fallback_depth = fallback_depth + 1
            _hop_metadata = dict(kwargs.get(metadata_variable_name) or {})
            _original_model_group_stamp = _hop_metadata.pop("original_model_group", original_model_group)
            _hop_metadata.pop("model_group", None)
            _hop_metadata.pop("attempted_fallbacks", None)
            _hop_metadata["original_model_group"] = _original_model_group_stamp
            _hop_metadata["model_group"] = kwargs.get("model", None)
            _hop_metadata["attempted_fallbacks"] = fallback_depth
            kwargs[metadata_variable_name] = _hop_metadata
            kwargs["fallback_depth"] = fallback_depth
            kwargs["max_fallbacks"] = max_fallbacks
            kwargs["attempted_targets"] = attempted
            if include_fallback_errors:
                kwargs["include_fallback_errors"] = include_fallback_errors
            response = await litellm_router.async_function_with_fallbacks(*args, **kwargs)
            response = add_fallback_headers_to_response(
                response=response,
                attempted_fallbacks=fallback_depth,
                fallback_errors=(list(fallback_errors) if include_fallback_errors else None),
            )

            return await _finalize_fallback_success(
                response=response,
                fallback_model_name=fallback_model_name,
                original_model_group=original_model_group,
                kwargs=kwargs,
                original_exception=original_exception,
            )
        except Exception as e:
            error_from_fallbacks = e
            fallback_errors = fallback_errors + (get_fallback_error_info(e),)
            await log_failure_fallback_event(
                original_model_group=original_model_group,
                kwargs=kwargs,
                original_exception=original_exception,
            )
            logging_obj = kwargs.get("litellm_logging_obj")
            if logging_obj is not None and logging_obj.model_call_details.get("has_logged_async_failure", False):
                _trigger_cooldown_for_failed_deployment(
                    litellm_router=litellm_router,
                    kwargs=kwargs,
                    exception=e,
                )
            # A request rejection is terminal for every entry that resends the same
            # request: continuing only finds one whose contract happens to accept it
            # and bills the caller for a substitution they never asked for. Entries
            # that rewrite the request are the exception, so the loop keeps going for
            # them and raises the rejection only if none of them recovers.
            if is_request_rejection(e) and not fallback_transforms_request(mg):
                rejected_request = e
    raise error_from_fallbacks


async def log_success_fallback_event(original_model_group: str, kwargs: dict, original_exception: Exception):
    """
    Log a successful fallback event to all registered callbacks.

    Uses LoggingCallbackManager.get_custom_loggers_for_type() to get deduplicated
    CustomLogger instances from all callback lists.

    Args:
        original_model_group (str): The original model group before fallback.
        kwargs (dict): kwargs for the request

    Note:
        Errors during logging are caught and reported but do not interrupt the process.
    """
    # Get deduplicated CustomLogger instances from all callback lists
    custom_loggers: Final = litellm.logging_callback_manager.get_custom_loggers_for_type(CustomLogger)

    for _callback_custom_logger in custom_loggers:
        try:
            await _callback_custom_logger.log_success_fallback_event(
                original_model_group=original_model_group,
                kwargs=kwargs,
                original_exception=original_exception,
            )
        except Exception as e:
            verbose_router_logger.error("Error in log_success_fallback_event: %s", e)


async def log_failure_fallback_event(original_model_group: str, kwargs: dict, original_exception: Exception):
    """
    Log a failed fallback event to all registered callbacks.

    Uses LoggingCallbackManager.get_custom_loggers_for_type() to get deduplicated
    CustomLogger instances from all callback lists.

    Args:
        original_model_group (str): The original model group before fallback.
        kwargs (dict): kwargs for the request

    Note:
        Errors during logging are caught and reported but do not interrupt the process.
    """
    # Get deduplicated CustomLogger instances from all callback lists
    custom_loggers: Final = litellm.logging_callback_manager.get_custom_loggers_for_type(CustomLogger)

    for _callback_custom_logger in custom_loggers:
        try:
            await _callback_custom_logger.log_failure_fallback_event(
                original_model_group=original_model_group,
                kwargs=kwargs,
                original_exception=original_exception,
            )
        except Exception as e:
            verbose_router_logger.error("Error in log_failure_fallback_event: %s", e)


def _check_non_standard_fallback_format(fallbacks: Sequence[Any] | None) -> bool:
    """
    Checks if the fallbacks list is a list of strings or a list of dictionaries.

    If
    - List[str]: e.g. ["claude-3-haiku", "openai/o-1"]
    - List[Dict[<LiteLLMParamsTypedDict>, Any]]: e.g. [{"model": "claude-3-haiku", "messages": [{"role": "user", "content": "Hey, how's it going?"}]}]

    If [{"gpt-3.5-turbo": ["claude-3-haiku"]}] then standard format.
    """
    if fallbacks is None or not isinstance(fallbacks, list) or len(fallbacks) == 0:
        return False
    if all(isinstance(item, str) for item in fallbacks):
        return True
    elif all(isinstance(item, dict) for item in fallbacks):
        for item in fallbacks:
            for key in LiteLLMParamsTypedDict.__annotations__:
                if key in item:
                    # If the value is a list, it's likely a standard fallback model group mapping
                    # (e.g. {"model": ["backup"]}) rather than a parameter override.
                    if not isinstance(item[key], list):
                        return True

    return False


def run_non_standard_fallback_format(fallbacks: list[str] | list[dict[str, Any]], model_group: str):
    pass
