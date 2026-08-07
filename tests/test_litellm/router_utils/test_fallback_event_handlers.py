import json

import httpx
import pytest

import litellm
from litellm import Router
from litellm.router_utils.fallback_event_handlers import (
    get_fallback_model_group,
    is_request_rejection,
    run_async_fallback,
)


class StreamingWrapper:
    def __init__(self):
        self._hidden_params = {"additional_headers": {}}


class FakeRouter:
    def log_retry(self, kwargs, e):
        return kwargs

    async def async_function_with_fallbacks(self, *args, **kwargs):
        return StreamingWrapper()


class AlwaysFailRouter:
    def log_retry(self, kwargs, e):
        return kwargs

    async def async_function_with_fallbacks(self, *args, **kwargs):
        raise RuntimeError("fallback model also failed")


@pytest.mark.asyncio
async def test_run_async_fallback_adds_errors_when_opted_in():
    response = await run_async_fallback(
        litellm_router=FakeRouter(),
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("upstream limited request"),
        max_fallbacks=3,
        fallback_depth=0,
        include_fallback_errors=True,
    )

    additional_headers = response._hidden_params["additional_headers"]
    assert additional_headers["x-litellm-attempted-fallbacks"] == 1
    assert json.loads(additional_headers["x-litellm-fallback-errors"]) == [
        {
            "message": "upstream limited request",
            "type": "RuntimeError",
            "param": None,
            "code": None,
        }
    ]


@pytest.mark.asyncio
async def test_run_async_fallback_omits_errors_without_opt_in():
    response = await run_async_fallback(
        litellm_router=FakeRouter(),
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("upstream limited request"),
        max_fallbacks=3,
        fallback_depth=0,
    )

    additional_headers = response._hidden_params["additional_headers"]
    assert additional_headers["x-litellm-attempted-fallbacks"] == 1
    assert "x-litellm-fallback-errors" not in additional_headers


@pytest.mark.asyncio
async def test_run_async_fallback_raises_when_all_fallbacks_fail():
    with pytest.raises(RuntimeError, match="fallback model also failed"):
        await run_async_fallback(
            litellm_router=AlwaysFailRouter(),
            fallback_model_group=["fallback-model"],
            original_model_group="primary-model",
            original_exception=RuntimeError("original request failed"),
            max_fallbacks=3,
            fallback_depth=0,
            include_fallback_errors=True,
        )


class RecordingRouter:
    def __init__(self):
        self.received_kwargs = None

    def log_retry(self, kwargs, e):
        return kwargs

    async def async_function_with_fallbacks(self, *args, **kwargs):
        self.received_kwargs = kwargs
        return StreamingWrapper()


@pytest.mark.asyncio
async def test_run_async_fallback_forwards_include_fallback_errors_to_nested_call():
    """A nested fallback (multi-hop) must keep collecting errors, so the opt-in
    flag has to reach the nested async_function_with_fallbacks call."""
    router = RecordingRouter()
    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("upstream limited request"),
        max_fallbacks=3,
        fallback_depth=0,
        include_fallback_errors=True,
    )

    assert router.received_kwargs.get("include_fallback_errors") is True


@pytest.mark.asyncio
async def test_run_async_fallback_does_not_forward_flag_without_opt_in():
    router = RecordingRouter()
    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("upstream limited request"),
        max_fallbacks=3,
        fallback_depth=0,
    )

    assert "include_fallback_errors" not in router.received_kwargs


@pytest.mark.asyncio
async def test_run_async_fallback_skips_original_model_group():
    response = await run_async_fallback(
        litellm_router=FakeRouter(),
        fallback_model_group=["primary-model", "fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("original failed"),
        max_fallbacks=3,
        fallback_depth=0,
    )

    assert response._hidden_params["additional_headers"]["x-litellm-attempted-fallbacks"] == 1


def test_get_fallback_model_group_does_not_mutate_fallbacks():
    """A string fallback must be resolved without mutating the caller's
    fallbacks list, which is the live router config shared across requests."""
    fallbacks = [{"gpt-3.5-turbo": ["claude-3-haiku"]}, "gpt-4o-mini"]

    fallback_model_group, _ = get_fallback_model_group(
        fallbacks=fallbacks, model_group="unmatched-model"
    )

    assert fallback_model_group == ["gpt-4o-mini"]
    assert fallbacks == [{"gpt-3.5-turbo": ["claude-3-haiku"]}, "gpt-4o-mini"]


# --- request rejections must not fall over --------------------------------


@pytest.mark.parametrize(
    "error,terminal",
    (
        (litellm.BadRequestError(message="bad", model="m", llm_provider="p"), True),
        (litellm.UnsupportedParamsError(message="unsupported", model="m", llm_provider="p"), True),
        (
            litellm.UnprocessableEntityError(
                message="unprocessable",
                model="m",
                llm_provider="p",
                response=httpx.Response(status_code=422, request=httpx.Request("POST", "https://litellm.ai")),
            ),
            True,
        ),
        # Dedicated fallback lists exist for these two, so they stay failover-able.
        (litellm.ContextWindowExceededError(message="ctx", model="m", llm_provider="p"), False),
        (litellm.ContentPolicyViolationError(message="policy", model="m", llm_provider="p"), False),
        # Deployment health, not request validity.
        (litellm.RateLimitError(message="429", model="m", llm_provider="p"), False),
        (litellm.AuthenticationError(message="401", model="m", llm_provider="p"), False),
        (litellm.InternalServerError(message="500", model="m", llm_provider="p"), False),
        (litellm.Timeout(message="timeout", model="m", llm_provider="p"), False),
        (litellm.ServiceUnavailableError(message="503", model="m", llm_provider="p"), False),
    ),
    ids=lambda v: type(v).__name__ if isinstance(v, Exception) else str(v),
)
def test_is_request_rejection_classifies_by_whether_the_request_or_the_deployment_is_at_fault(error, terminal):
    assert is_request_rejection(error) is terminal


def _two_group_router() -> Router:
    """primary with a fallback twin, mirroring a direct provider plus its reseller."""
    return Router(
        model_list=[
            {"model_name": "primary", "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-fake"}},
            {"model_name": "twin", "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-fake"}},
        ],
        fallbacks=[{"primary": ["twin"]}],
        num_retries=0,
    )


async def _call(router: Router, failure: Exception):
    """Run the primary through the real fallback path, failing it with `failure`."""
    attempted: list = []  # mutable-ok: test recorder for which model groups were called

    async def original_function(**kwargs):
        model = kwargs.get("model")
        attempted.append(model)
        if model == "primary":
            raise failure
        return {"served_by": model}

    response = await router.async_function_with_fallbacks(
        model="primary",
        original_function=original_function,
        messages=[{"role": "user", "content": "hi"}],
        metadata={},
    )
    return response, attempted


@pytest.mark.asyncio
async def test_validation_rejection_does_not_fall_over_to_the_twin():
    """
    The capability gate refuses a param the primary cannot execute. Falling over
    would run the request on a provider whose contract happens to accept it, so the
    caller silently receives a different provider's output instead of the refusal.
    """
    router = _two_group_router()
    rejection = litellm.BadRequestError(
        message="Model 'primary' does not support video parameter(s): negative_prompt.",
        model="primary",
        llm_provider="openai",
    )

    with pytest.raises(litellm.BadRequestError, match="negative_prompt"):
        await _call(router, rejection)


@pytest.mark.asyncio
async def test_validation_rejection_never_reaches_the_twin_at_all():
    """The twin must not be invoked; a billed generation on it is the actual harm."""
    router = _two_group_router()
    rejection = litellm.BadRequestError(message="bad param", model="primary", llm_provider="openai")

    attempted: list = []  # mutable-ok: test recorder

    async def original_function(**kwargs):
        attempted.append(kwargs.get("model"))
        if kwargs.get("model") == "primary":
            raise rejection
        return {"served_by": kwargs.get("model")}

    with pytest.raises(litellm.BadRequestError):
        await router.async_function_with_fallbacks(
            model="primary",
            original_function=original_function,
            messages=[{"role": "user", "content": "hi"}],
            metadata={},
        )

    assert attempted == ["primary"], f"the twin was called despite a request rejection: {attempted}"


@pytest.mark.asyncio
async def test_upstream_server_error_still_falls_over_to_the_twin():
    """
    The other half. Making rejections terminal must not cost us real failover: a sick
    deployment is exactly what the fallback group exists for.
    """
    router = _two_group_router()
    outage = litellm.InternalServerError(message="upstream exploded", model="primary", llm_provider="openai")

    response, attempted = await _call(router, outage)

    assert response["served_by"] == "twin"
    assert attempted == ["primary", "twin"]


@pytest.mark.asyncio
async def test_rate_limit_still_falls_over_to_the_twin():
    """Capacity errors are the other canonical failover case."""
    router = _two_group_router()
    throttled = litellm.RateLimitError(message="slow down", model="primary", llm_provider="openai")

    response, attempted = await _call(router, throttled)

    assert response["served_by"] == "twin"
    assert attempted == ["primary", "twin"]
