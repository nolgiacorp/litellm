"""Cached-prompt-token billing for the fleet brain (NOL-376).

`openrouter/deepseek/deepseek-v4-flash-0731` is priced in both maps so spend does
not rest solely on the litellm-config `model_info` pin. The map entry originally
spelled its cache-read rate `input_cost_per_token_cache_hit`, which ModelInfoBase
declares but no cost calculator reads: `_calculate_input_cost` bills cached prompt
tokens at `cache_read_input_token_cost`, defaulting to 0.0 when the key is absent.
Cache hits therefore billed at $0 instead of $0.018/M.

These tests bill through the local map (LITELLM_LOCAL_MODEL_COST_MAP=True, what prod
runs) so a regression back to the unread key shows up as the money it costs.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.types.utils import PromptTokensDetailsWrapper, Usage

MODEL = "openrouter/deepseek/deepseek-v4-flash-0731"
INPUT_RATE = 9e-08
CACHE_READ_RATE = 1.8e-08
OUTPUT_RATE = 1.8e-07


@pytest.fixture
def local_cost_map(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    original = litellm.model_cost
    litellm.model_cost = litellm.get_model_cost_map(url="")
    yield
    litellm.model_cost = original


def test_cached_prompt_tokens_bill_at_the_cache_read_rate(local_cost_map):
    """The 800 cached tokens must cost $0.018/M, not $0."""
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=100,
        total_tokens=1100,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=800, text_tokens=200),
    )

    prompt_cost, completion_cost = generic_cost_per_token(
        model=MODEL,
        usage=usage,
        custom_llm_provider="openrouter",
    )

    assert prompt_cost == pytest.approx(200 * INPUT_RATE + 800 * CACHE_READ_RATE)
    assert completion_cost == pytest.approx(100 * OUTPUT_RATE)
    # The bug this pins: a zero cache-read rate would have charged only the
    # uncached remainder, silently dropping 80% of the prompt off the invoice.
    assert prompt_cost > 200 * INPUT_RATE


def test_uncached_request_is_unaffected(local_cost_map):
    usage = Usage(prompt_tokens=1000, completion_tokens=100, total_tokens=1100)

    prompt_cost, completion_cost = generic_cost_per_token(
        model=MODEL,
        usage=usage,
        custom_llm_provider="openrouter",
    )

    assert prompt_cost == pytest.approx(1000 * INPUT_RATE)
    assert completion_cost == pytest.approx(100 * OUTPUT_RATE)


def test_model_info_exposes_the_consumed_cache_read_key(local_cost_map):
    info = litellm.get_model_info(model=MODEL, custom_llm_provider="openrouter")
    assert info["cache_read_input_token_cost"] == CACHE_READ_RATE
