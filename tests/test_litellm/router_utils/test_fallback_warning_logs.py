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
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-backup", "mock_response": "from backup"},
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


def test_fallback_error_text_is_bounded_and_single_line():
    from litellm.router import _truncate_for_log

    assert _truncate_for_log("a  b\n\tc") == "a b c"
    long = "x" * 5000
    rendered = _truncate_for_log(long)
    assert len(rendered) == 601 and rendered.endswith("…")
