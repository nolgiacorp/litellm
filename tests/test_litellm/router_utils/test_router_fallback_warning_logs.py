"""The router's fallback path logs at WARNING, with the PRIMARY failure attached.

The proxy's default log level hides INFO, and the fallback chain only ever
surfaces its LAST hop's error — so before these lines existed a primary that
failed on every call, or a fallback hop that was itself broken, was invisible
until a customer hit it (2026-08-30: every deepseek-v4-flash mid-stream failure
fell back to a Gemini hop that 400'd on thought signatures, and nothing logged
the deepseek failure at all). Operators alert on `router_fallback_triggered`.
"""

import logging

import pytest

import litellm
from litellm import Router
from litellm._logging import verbose_router_logger


@pytest.fixture
def router_with_fallback():
    return Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-primary", "mock_response": "unused"},
            },
            {
                "model_name": "backup",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-backup",
                    "mock_response": "from backup",
                },
            },
        ],
        fallbacks=[{"primary": ["backup"]}],
        set_verbose=False,
    )


@pytest.mark.asyncio
async def test_fallback_warns_with_primary_error(router_with_fallback, caplog):
    caplog.set_level(logging.WARNING, logger=verbose_router_logger.name)
    primary_error = litellm.ServiceUnavailableError(
        message="upstream stream died: connection reset by peer",
        model="openai/gpt-4o-mini",
        llm_provider="openai",
    )

    response = await router_with_fallback.async_function_with_fallbacks_common_utils(
        e=primary_error,
        disable_fallbacks=False,
        fallbacks=[{"primary": ["backup"]}],
        context_window_fallbacks=None,
        content_policy_fallbacks=None,
        model_group="primary",
        args=(),
        kwargs={
            "model": "primary",
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {},
            "original_function": router_with_fallback._acompletion,
        },
    )
    assert response.choices[0].message.content == "from backup"

    warnings = [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]
    triggered = [message for message in warnings if message.startswith("router_fallback_triggered ")]
    assert len(triggered) == 1, warnings
    assert "model_group=primary" in triggered[0]
    assert "error_type=ServiceUnavailableError" in triggered[0]
    assert "status_code=503" in triggered[0]
    assert "connection reset by peer" in triggered[0], "the PRIMARY failure rides the warning"
    assert "fallbacks=" in triggered[0]

    assert any(message.startswith("router_fallback_attempt model_group=backup") for message in warnings), warnings
    assert any(message.startswith("router_fallback_succeeded model_group=backup") for message in warnings), warnings


@pytest.mark.asyncio
async def test_no_trigger_warning_without_a_matching_fallback(caplog):
    router = Router(model_list=[], fallbacks=[], set_verbose=False)
    caplog.set_level(logging.WARNING, logger=verbose_router_logger.name)
    primary_error = litellm.ServiceUnavailableError(
        message="upstream unavailable",
        model="primary",
        llm_provider="openai",
    )

    with pytest.raises(litellm.ServiceUnavailableError):
        await router.async_function_with_fallbacks_common_utils(
            e=primary_error,
            disable_fallbacks=False,
            fallbacks=[],
            context_window_fallbacks=None,
            content_policy_fallbacks=None,
            model_group="primary",
            args=(),
            kwargs={"model": "primary", "metadata": {}},
        )

    assert not any(
        record.getMessage().startswith("router_fallback_triggered ")
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_fallback_warning_redacts_message_content(router_with_fallback, caplog):
    caplog.set_level(logging.WARNING, logger=verbose_router_logger.name)
    secret = "customer prompt must not be logged"
    primary_error = litellm.ServiceUnavailableError(
        message=secret,
        model="openai/gpt-4o-mini",
        llm_provider="openai",
    )

    await router_with_fallback.async_function_with_fallbacks_common_utils(
        e=primary_error,
        disable_fallbacks=False,
        fallbacks=[{"model": "backup", "messages": [{"role": "user", "content": secret}]}],
        context_window_fallbacks=None,
        content_policy_fallbacks=None,
        model_group="primary",
        args=(),
        kwargs={
            "model": "primary",
            "messages": [{"role": "user", "content": secret}],
            "metadata": {},
            "original_function": router_with_fallback._acompletion,
            "standard_callback_dynamic_params": {"turn_off_message_logging": True},
        },
    )

    warnings = [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]
    assert not any(secret in warning for warning in warnings)
    assert any("fallbacks=['backup'] error=redacted-by-litellm" in warning for warning in warnings)
    assert any("router_fallback_attempt model_group=backup" in warning for warning in warnings)
