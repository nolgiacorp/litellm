from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import litellm
from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.sensitive_data_masker import mask_sensitive_structure
from litellm.router_utils.add_retry_fallback_headers import (
    add_fallback_headers_to_response,
    get_fallback_error_info,
)
from litellm.types.router import LiteLLMParamsTypedDict
from litellm.types.utils import all_litellm_params

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any


# Keys a fallback entry can carry without changing the request the provider
# rejected: the routing keys the router adds itself when it re-points the same
# request, and the LiteLLM-level settings (api_key, api_base, timeout, metadata,
# num_retries, ...) that configure the call but never reach the provider payload.
# An entry made only of these resends the rejected payload verbatim.
_ROUTER_FALLBACK_ENTRY_KEYS = frozenset(("model", "_target_order", "_excluded_deployment_ids"))
_NON_PAYLOAD_FALLBACK_KEYS = _ROUTER_FALLBACK_ENTRY_KEYS | frozenset(all_litellm_params) | frozenset(("timeout",))


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


def get_fallback_model_group(fallbacks: List[Any], model_group: str) -> Tuple[Optional[List[str]], Optional[int]]:
    """
    Returns:
    - fallback_model_group: List[str] of fallback model groups. example: ["gpt-4", "gpt-3.5-turbo"]
    - generic_fallback_idx: int of the index of the generic fallback in the fallbacks list.

    Checks:
    - exact match
    - stripped model group match
    - generic fallback
    """
    generic_fallback_idx: Optional[int] = None
    stripped_model_fallback: Optional[List[str]] = None
    fallback_model_group: Optional[List[str]] = None
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


async def run_async_fallback(
    *args: Tuple[Any],
    litellm_router: LitellmRouter,
    fallback_model_group: List[str],
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
        **kwargs: Keyword arguments.

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
    rejected_request: Exception | None = None

    for mg in fallback_model_group:
        if mg == original_model_group:
            continue
        # Once the request itself has been rejected, only entries that rewrite it
        # can still help; the rest resend the rejected payload.
        if rejected_request is not None and not fallback_transforms_request(mg):
            continue
        try:
            # LOGGING
            kwargs = litellm_router.log_retry(kwargs=kwargs, e=original_exception)
            verbose_router_logger.info(f"Falling back to model_group = {mask_sensitive_structure(mg)}")
            if isinstance(mg, str):
                kwargs["model"] = mg
            elif isinstance(mg, dict):
                kwargs.update(mg)
            kwargs.setdefault("metadata", {}).update(
                {"model_group": kwargs.get("model", None)}
            )  # update model_group used, if fallbacks are done
            fallback_depth = fallback_depth + 1
            kwargs["fallback_depth"] = fallback_depth
            kwargs["max_fallbacks"] = max_fallbacks
            if include_fallback_errors:
                kwargs["include_fallback_errors"] = include_fallback_errors
            response = await litellm_router.async_function_with_fallbacks(*args, **kwargs)
            verbose_router_logger.info("Successful fallback b/w models.")
            response = add_fallback_headers_to_response(
                response=response,
                attempted_fallbacks=fallback_depth,
                fallback_errors=(list(fallback_errors) if include_fallback_errors else None),
            )
            # callback for successfull_fallback_event():
            await log_success_fallback_event(
                original_model_group=original_model_group,
                kwargs=kwargs,
                original_exception=original_exception,
            )
            return response
        except Exception as e:
            error_from_fallbacks = e
            fallback_errors = fallback_errors + (get_fallback_error_info(e),)
            await log_failure_fallback_event(
                original_model_group=original_model_group,
                kwargs=kwargs,
                original_exception=original_exception,
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
    custom_loggers = litellm.logging_callback_manager.get_custom_loggers_for_type(CustomLogger)

    for _callback_custom_logger in custom_loggers:
        try:
            await _callback_custom_logger.log_success_fallback_event(
                original_model_group=original_model_group,
                kwargs=kwargs,
                original_exception=original_exception,
            )
        except Exception as e:
            verbose_router_logger.error(f"Error in log_success_fallback_event: {str(e)}")


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
    custom_loggers = litellm.logging_callback_manager.get_custom_loggers_for_type(CustomLogger)

    for _callback_custom_logger in custom_loggers:
        try:
            await _callback_custom_logger.log_failure_fallback_event(
                original_model_group=original_model_group,
                kwargs=kwargs,
                original_exception=original_exception,
            )
        except Exception as e:
            verbose_router_logger.error(f"Error in log_failure_fallback_event: {str(e)}")


def _check_non_standard_fallback_format(fallbacks: Optional[Sequence[Any]]) -> bool:
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
            for key in LiteLLMParamsTypedDict.__annotations__.keys():
                if key in item:
                    # If the value is a list, it's likely a standard fallback model group mapping
                    # (e.g. {"model": ["backup"]}) rather than a parameter override.
                    if not isinstance(item[key], list):
                        return True

    return False


def run_non_standard_fallback_format(fallbacks: Union[List[str], List[Dict[str, Any]]], model_group: str):
    pass
