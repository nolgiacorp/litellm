import json
import logging

import httpx
import pytest

import litellm
from litellm import Router
from litellm._logging import verbose_router_logger
from litellm.router_utils.fallback_event_handlers import (
    _fallback_error_for_log,
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


class AsyncFallbackStream:
    def __init__(self, error=None):
        self._items = iter(["chunk"])
        self._error = error

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = next(self._items, None)
        if item is not None:
            return item
        if self._error is not None:
            raise self._error
        raise StopAsyncIteration


class StreamingRouter(FakeRouter):
    def __init__(self, error=None):
        self.error = error

    async def async_function_with_fallbacks(self, *args, **kwargs):
        return AsyncFallbackStream(self.error)


def test_fallback_error_text_is_bounded_and_single_line():
    assert _fallback_error_for_log(RuntimeError("a  b\n\tc"), {}) == "a b c"
    rendered = _fallback_error_for_log(RuntimeError("x" * 5000), {})
    assert len(rendered) == 601 and rendered.endswith("…")


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


@pytest.mark.asyncio
async def test_stream_fallback_logs_success_only_after_completion(caplog):
    caplog.set_level(logging.WARNING, logger=verbose_router_logger.name)
    response = await run_async_fallback(
        litellm_router=StreamingRouter(),
        fallback_model_group=[{"model": "fallback-model", "_target_order": 2}],
        original_model_group="primary-model",
        original_exception=RuntimeError("original request failed"),
        max_fallbacks=3,
        fallback_depth=0,
        model="primary-model",
    )

    warnings = [record.getMessage() for record in caplog.records]
    assert "router_fallback_attempt model_group=fallback-model from=primary-model" in warnings
    assert not any(message.startswith("router_fallback_succeeded ") for message in warnings)

    assert [item async for item in response] == ["chunk"]
    assert "router_fallback_succeeded model_group=fallback-model" in [
        record.getMessage() for record in caplog.records
    ]


@pytest.mark.asyncio
async def test_stream_fallback_does_not_log_success_when_iteration_fails(caplog):
    caplog.set_level(logging.WARNING, logger=verbose_router_logger.name)
    response = await run_async_fallback(
        litellm_router=StreamingRouter(RuntimeError("stream failed")),
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("original request failed"),
        max_fallbacks=3,
        fallback_depth=0,
        model="primary-model",
    )

    with pytest.raises(RuntimeError, match="stream failed"):
        [item async for item in response]
    assert not any(
        record.getMessage().startswith("router_fallback_succeeded ")
        for record in caplog.records
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

    fallback_model_group, _ = get_fallback_model_group(fallbacks=fallbacks, model_group="unmatched-model")

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
        # Router-generated 400s: the router's own availability/config failures carry
        # no provider, and a healthy backup group is exactly what they need.
        (
            litellm.BadRequestError(
                message="You passed in model=primary. There are no healthy deployments for this model",
                model="primary",
                llm_provider="",
            ),
            False,
        ),
        # A provider param validator that does not name its provider: the type says
        # the request is at fault, so the missing llm_provider must not read as a
        # router failure.
        (litellm.UnsupportedParamsError(message="size is not supported for model m", model="m"), True),
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


@pytest.mark.asyncio
async def test_router_availability_error_still_falls_over_to_the_twin():
    """
    The router reuses BadRequestError for its own outages ("no healthy deployments").
    The request is fine there, so a configured fallback to a healthy group must run.
    """
    router = _two_group_router()
    unavailable = litellm.BadRequestError(
        message="You passed in model=primary. There are no healthy deployments for this model",
        model="primary",
        llm_provider="",
    )

    response, attempted = await _call(router, unavailable)

    assert response["served_by"] == "twin"
    assert attempted == ["primary", "twin"]


@pytest.mark.asyncio
async def test_rejection_stops_the_outer_fallback_loop_before_the_second_twin():
    """
    Primary fails retryably, so failover is correct; the first twin then rejects the
    request. That rejection is terminal: continuing to the second twin would find a
    contract that happens to accept the request and bill for it.
    """
    router = Router(
        model_list=[
            {"model_name": "primary", "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-fake"}},
            {"model_name": "twin_a", "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-fake"}},
            {"model_name": "twin_b", "litellm_params": {"model": "openai/gpt-4.1-mini", "api_key": "sk-fake"}},
        ],
        fallbacks=[{"primary": ["twin_a", "twin_b"]}],
        num_retries=0,
    )

    attempted: list = []  # mutable-ok: test recorder

    async def original_function(**kwargs):
        model = kwargs.get("model")
        attempted.append(model)
        if model == "primary":
            raise litellm.InternalServerError(message="upstream exploded", model=model, llm_provider="openai")
        if model == "twin_a":
            raise litellm.BadRequestError(message="bad param", model=model, llm_provider="openai")
        return {"served_by": model}

    with pytest.raises(litellm.BadRequestError, match="bad param"):
        await router.async_function_with_fallbacks(
            model="primary",
            original_function=original_function,
            messages=[{"role": "user", "content": "hi"}],
            metadata={},
        )

    assert attempted == ["primary", "twin_a"], f"the second twin ran after a request rejection: {attempted}"


@pytest.mark.asyncio
async def test_request_transforming_client_fallback_still_repairs_a_rejection():
    """
    A client-side fallback that rewrites the request is a deliberate repair, not a
    silent provider substitution, so a 400 on the original request must not skip it.
    """
    router = _two_group_router()

    attempted: list = []  # mutable-ok: test recorder

    async def original_function(**kwargs):
        attempted.append((kwargs.get("model"), kwargs["messages"][0]["content"]))
        if kwargs["messages"][0]["content"] == "rejected":
            raise litellm.BadRequestError(message="bad param", model=kwargs.get("model"), llm_provider="openai")
        return {"served_by": kwargs.get("model")}

    response = await router.async_function_with_fallbacks(
        model="primary",
        original_function=original_function,
        messages=[{"role": "user", "content": "rejected"}],
        metadata={},
        fallbacks=[{"model": "twin", "messages": [{"role": "user", "content": "repaired"}]}],
    )

    assert response["served_by"] == "twin"
    assert attempted == [("primary", "rejected"), ("twin", "repaired")]


@pytest.mark.asyncio
async def test_plain_client_side_fallback_list_does_not_run_after_a_rejection():
    """
    A bare client-side list re-points the same request, so it is the substitution
    this change prevents, not a repair.
    """
    router = _two_group_router()

    attempted: list = []  # mutable-ok: test recorder

    async def original_function(**kwargs):
        attempted.append(kwargs.get("model"))
        if kwargs.get("model") == "primary":
            raise litellm.BadRequestError(message="bad param", model="primary", llm_provider="openai")
        return {"served_by": kwargs.get("model")}

    with pytest.raises(litellm.BadRequestError, match="bad param"):
        await router.async_function_with_fallbacks(
            model="primary",
            original_function=original_function,
            messages=[{"role": "user", "content": "hi"}],
            metadata={},
            fallbacks=["twin"],
        )

    assert attempted == ["primary"]


@pytest.mark.asyncio
async def test_connection_only_client_fallback_does_not_run_after_a_rejection():
    """
    api_key/api_base/timeout reconfigure the call but leave the payload alone, so an
    entry carrying only those resends the rejected request: not a repair.
    """
    router = _two_group_router()

    attempted: list = []  # mutable-ok: test recorder

    async def original_function(**kwargs):
        attempted.append(kwargs.get("model"))
        if kwargs.get("model") == "primary":
            raise litellm.BadRequestError(message="bad param", model="primary", llm_provider="openai")
        return {"served_by": kwargs.get("model")}

    with pytest.raises(litellm.BadRequestError, match="bad param"):
        await router.async_function_with_fallbacks(
            model="primary",
            original_function=original_function,
            messages=[{"role": "user", "content": "hi"}],
            metadata={},
            fallbacks=[{"model": "twin", "api_key": "sk-other", "api_base": "https://other", "timeout": 5}],
        )

    assert attempted == ["primary"]


@pytest.mark.asyncio
async def test_run_async_fallback_skips_plain_entries_but_keeps_a_later_repair_entry():
    """
    twin_a rejects the unchanged request, which rules out every later entry that
    resends it (twin_c), but not the repair entry that follows: that one sends a
    different payload, so the rejection says nothing about it.
    """
    attempted: list = []  # mutable-ok: test recorder

    class RejectingRouter:
        def log_retry(self, kwargs, e):
            return kwargs

        async def async_function_with_fallbacks(self, *args, **kwargs):
            content = kwargs["messages"][0]["content"]
            attempted.append((kwargs["model"], content))
            if content == "rejected":
                raise litellm.BadRequestError(message="bad param", model=kwargs["model"], llm_provider="openai")
            return {"served_by": kwargs["model"]}

    response = await run_async_fallback(
        litellm_router=RejectingRouter(),
        fallback_model_group=[
            {"model": "twin_a"},
            {"model": "twin_c"},
            {"model": "twin_b", "messages": [{"role": "user", "content": "repaired"}]},
        ],
        original_model_group="primary",
        original_exception=litellm.InternalServerError(message="500", model="primary", llm_provider="openai"),
        max_fallbacks=5,
        fallback_depth=0,
        model="primary",
        messages=[{"role": "user", "content": "rejected"}],
    )

    assert response["served_by"] == "twin_b"
    assert attempted == [("twin_a", "rejected"), ("twin_b", "repaired")]
