"""
Test suite for XAI cost calculation functionality.
"""

import math
import os
import pytest
import sys

import litellm
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    PromptTokensDetailsWrapper,
    Usage,
)

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.llms.xai.cost_calculator import cost_per_token, cost_per_web_search_request


class TestXAICostCalculator:
    """Test suite for XAI cost calculation functionality."""

    def setup_method(self):
        """Set up test environment."""
        # Load the main model cost map directly to ensure we have the latest pricing
        import json

        try:
            with open("model_prices_and_context_window.json", "r") as f:
                model_cost_map = json.load(f)
            litellm.model_cost = model_cost_map
        except FileNotFoundError:
            # Fallback to default behavior
            os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
            litellm.model_cost = litellm.get_model_cost_map(url="")

    def test_basic_cost_calculation(self):
        """Test basic cost calculation without reasoning tokens."""
        usage = Usage(prompt_tokens=12, completion_tokens=125, total_tokens=137)

        prompt_cost, completion_cost = cost_per_token(model="grok-3-mini", usage=usage)

        # Expected costs for grok-3-mini:
        # Input: 12 tokens * $3e-7 = $0.0000036
        # Output: 125 tokens * $5e-7 = $0.0000625
        expected_prompt_cost = 12 * 3e-7
        expected_completion_cost = 125 * 5e-7

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_reasoning_tokens_cost_calculation(self):
        """Test cost calculation with reasoning tokens from completion_tokens_details."""
        usage = Usage(
            prompt_tokens=12,
            completion_tokens=125,
            total_tokens=1086,
            completion_tokens_details=CompletionTokensDetailsWrapper(
                accepted_prediction_tokens=0,
                audio_tokens=0,
                reasoning_tokens=949,
                rejected_prediction_tokens=0,
                text_tokens=None,  # Not set, but doesn't matter for XAI billing
            ),
        )

        prompt_cost, completion_cost = cost_per_token(model="grok-3-mini", usage=usage)

        # Expected costs for grok-3-mini:
        # Input: 12 tokens * $3e-7 = $0.0000036
        # Completion: (125 + 949) tokens * $5e-7 = $0.000537
        expected_prompt_cost = 12 * 3e-7
        expected_completion_cost = (125 + 949) * 5e-7

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_reasoning_and_text_tokens_cost_calculation(self):
        """Test cost calculation with both reasoning and text tokens."""
        usage = Usage(
            prompt_tokens=12,
            completion_tokens=125,
            total_tokens=1086,
            completion_tokens_details=CompletionTokensDetailsWrapper(
                accepted_prediction_tokens=0,
                audio_tokens=0,
                reasoning_tokens=949,
                rejected_prediction_tokens=0,
                text_tokens=76,  # Explicitly set (but ignored in XAI billing)
            ),
        )

        prompt_cost, completion_cost = cost_per_token(model="grok-3-mini", usage=usage)

        # Expected costs for grok-3-mini:
        # Input: 12 tokens * $3e-7 = $0.0000036
        # Completion: (125 + 949) tokens * $5e-7 = $0.000537
        # Note: text_tokens field is ignored, only completion_tokens + reasoning_tokens matters
        expected_prompt_cost = 12 * 3e-7
        expected_completion_cost = (125 + 949) * 5e-7

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_grok_4_cost_calculation(self):
        """Test cost calculation for grok-4 model."""
        usage = Usage(
            prompt_tokens=10,
            completion_tokens=200,
            total_tokens=360,
            completion_tokens_details=CompletionTokensDetailsWrapper(
                accepted_prediction_tokens=0,
                audio_tokens=0,
                reasoning_tokens=150,
                rejected_prediction_tokens=0,
                text_tokens=50,  # Ignored in XAI billing
            ),
        )

        prompt_cost, completion_cost = cost_per_token(model="grok-4", usage=usage)

        # Expected costs for grok-4:
        # Input: 10 tokens * $3e-6 = $0.00003
        # Completion: (200 + 150) tokens * $1.5e-5 = $0.00525
        expected_prompt_cost = 10 * 3e-6
        expected_completion_cost = (200 + 150) * 1.5e-5

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_grok_3_fast_beta_cost_calculation(self):
        """Test cost calculation for grok-3-fast-beta model."""
        usage = Usage(
            prompt_tokens=20,
            completion_tokens=300,
            total_tokens=520,
            completion_tokens_details=CompletionTokensDetailsWrapper(
                accepted_prediction_tokens=0,
                audio_tokens=0,
                reasoning_tokens=200,
                rejected_prediction_tokens=0,
                text_tokens=100,  # Ignored in XAI billing
            ),
        )

        prompt_cost, completion_cost = cost_per_token(
            model="grok-3-fast-beta", usage=usage
        )

        # Expected costs for grok-3-fast-beta:
        # Input: 20 tokens * $5e-6 = $0.0001
        # Completion: (300 + 200) tokens * $2.5e-5 = $0.0125
        expected_prompt_cost = 20 * 5e-6
        expected_completion_cost = (300 + 200) * 2.5e-5

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_edge_case_no_completion_tokens_details(self):
        """Test cost calculation when completion_tokens_details is not present."""
        usage = Usage(prompt_tokens=12, completion_tokens=125, total_tokens=137)

        prompt_cost, completion_cost = cost_per_token(model="grok-3-mini", usage=usage)

        # Should fall back to basic calculation
        expected_prompt_cost = 12 * 3e-7
        expected_completion_cost = 125 * 5e-7

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_edge_case_large_reasoning_tokens(self):
        """Test cost calculation when reasoning_tokens is larger than completion_tokens."""
        usage = Usage(
            prompt_tokens=12,
            completion_tokens=50,  # Less than reasoning_tokens
            total_tokens=162,
            completion_tokens_details=CompletionTokensDetailsWrapper(
                accepted_prediction_tokens=0,
                audio_tokens=0,
                reasoning_tokens=100,  # More than completion_tokens
                rejected_prediction_tokens=0,
                text_tokens=None,
            ),
        )

        prompt_cost, completion_cost = cost_per_token(model="grok-3-mini", usage=usage)

        # Expected costs:
        # Input: 12 tokens * $3e-7 = $0.0000036
        # Completion: (50 + 100) tokens * $5e-7 = $0.000075
        expected_prompt_cost = 12 * 3e-7
        expected_completion_cost = (50 + 100) * 5e-7

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_tiered_pricing_above_128k_tokens(self):
        """Test tiered pricing for tokens above 128k."""
        # Test with grok-4-fast-reasoning which has tiered pricing
        usage = Usage(
            prompt_tokens=150000,  # Above 128k threshold
            completion_tokens=100000,  # Above 128k threshold
            total_tokens=300000,
            completion_tokens_details=CompletionTokensDetailsWrapper(
                accepted_prediction_tokens=0,
                audio_tokens=0,
                reasoning_tokens=50000,  # Total completion tokens = 100000 + 50000 = 150000 > 128k
                rejected_prediction_tokens=0,
                text_tokens=None,
            ),
        )

        prompt_cost, completion_cost = cost_per_token(
            model="xai/grok-4-fast-reasoning", usage=usage
        )

        # Expected costs for grok-4-fast-reasoning with tiered pricing:
        # Input: 150000 tokens * $0.4e-6 (ALL tokens at tiered rate since input > 128k) = $0.06
        # Completion: (100000 + 50000) tokens * $1e-6 (tiered rate since input > 128k) = $0.15
        expected_prompt_cost = 150000 * 0.4e-6
        expected_completion_cost = (100000 + 50000) * 1e-6

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_tiered_pricing_below_128k_tokens(self):
        """Test that regular pricing is used for tokens below 128k threshold."""
        # Test with grok-4-fast-reasoning which has tiered pricing
        usage = Usage(
            prompt_tokens=100000,  # Below 128k threshold
            completion_tokens=50000,
            total_tokens=160000,
            completion_tokens_details=CompletionTokensDetailsWrapper(
                accepted_prediction_tokens=0,
                audio_tokens=0,
                reasoning_tokens=10000,
                rejected_prediction_tokens=0,
                text_tokens=None,
            ),
        )

        prompt_cost, completion_cost = cost_per_token(
            model="xai/grok-4-fast-reasoning", usage=usage
        )

        # Expected costs for grok-4-fast-reasoning with regular pricing:
        # Input: 100000 tokens * $0.2e-6 (regular rate) = $0.02
        # Completion: (50000 + 10000) tokens * $0.5e-6 (regular rate) = $0.03
        expected_prompt_cost = 100000 * 0.2e-6
        expected_completion_cost = (50000 + 10000) * 0.5e-6

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_tiered_pricing_grok_4_latest(self):
        """Test tiered pricing for grok-4-latest model."""
        usage = Usage(
            prompt_tokens=200000,  # Above 128k threshold
            completion_tokens=100000,
            total_tokens=350000,
            completion_tokens_details=CompletionTokensDetailsWrapper(
                accepted_prediction_tokens=0,
                audio_tokens=0,
                reasoning_tokens=50000,
                rejected_prediction_tokens=0,
                text_tokens=None,
            ),
        )

        prompt_cost, completion_cost = cost_per_token(
            model="xai/grok-4-latest", usage=usage
        )

        # Expected costs for grok-4-latest with tiered pricing:
        # Input: 200000 tokens * $6e-6 (ALL tokens at tiered rate since input > 128k) = $1.2
        # Completion: (100000 + 50000) tokens * $30e-6 (tiered rate since input > 128k) = $4.5
        expected_prompt_cost = 200000 * 6e-6
        expected_completion_cost = (100000 + 50000) * 30e-6

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_tiered_pricing_output_tokens_below_128k(self):
        """Test that output tokens get tiered rate when input tokens > 128k, even if output tokens < 128k."""
        usage = Usage(
            prompt_tokens=150000,  # Above 128k threshold
            completion_tokens=50000,  # Below 128k threshold
            total_tokens=210000,
            completion_tokens_details=CompletionTokensDetailsWrapper(
                accepted_prediction_tokens=0,
                audio_tokens=0,
                reasoning_tokens=10000,  # Total completion tokens = 50000 + 10000 = 60000 < 128k
                rejected_prediction_tokens=0,
                text_tokens=None,
            ),
        )

        prompt_cost, completion_cost = cost_per_token(
            model="xai/grok-4-fast-reasoning", usage=usage
        )

        # Expected costs for grok-4-fast-reasoning:
        # Input: 150000 tokens * $0.4e-6 (ALL tokens at tiered rate since input > 128k) = $0.06
        # Completion: (50000 + 10000) tokens * $1e-6 (tiered rate since input > 128k) = $0.06
        expected_prompt_cost = 150000 * 0.4e-6
        expected_completion_cost = (50000 + 10000) * 1e-6

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_tiered_pricing_model_without_tiered_pricing(self):
        """Test that models without tiered pricing use regular pricing even above 128k."""
        usage = Usage(
            prompt_tokens=150000,  # Above 128k threshold
            completion_tokens=50000,
            total_tokens=200000,
        )

        prompt_cost, completion_cost = cost_per_token(model="grok-3-mini", usage=usage)

        # grok-3-mini doesn't have tiered pricing, so should use regular rates:
        # Input: 150000 tokens * $3e-7 (regular rate) = $0.045
        # Completion: 50000 tokens * $5e-7 (regular rate) = $0.025
        expected_prompt_cost = 150000 * 3e-7
        expected_completion_cost = 50000 * 5e-7

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_already_normalised_usage_does_not_double_count_reasoning(self):
        """Cost calc must not double-bill when Usage is already OpenAI-normalised."""
        usage = Usage(
            prompt_tokens=12,
            completion_tokens=200,
            total_tokens=212,
            completion_tokens_details=CompletionTokensDetailsWrapper(
                accepted_prediction_tokens=0,
                audio_tokens=0,
                reasoning_tokens=100,
                rejected_prediction_tokens=0,
                text_tokens=None,
            ),
        )

        prompt_cost, completion_cost = cost_per_token(model="grok-3-mini", usage=usage)

        expected_prompt_cost = 12 * 3e-7
        expected_completion_cost = 200 * 5e-7

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_web_search_cost_calculation(self):
        """Test web search cost calculation for X.AI models."""
        # Test with web_search_requests in prompt_tokens_details (primary path)
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                text_tokens=100,
                web_search_requests=3,  # 3 sources used
            ),
        )

        web_search_cost = cost_per_web_search_request(usage=usage, model_info={})

        # Expected cost: 3 sources * $0.025 per source = $0.075
        expected_cost = 3 * (25.0 / 1000.0)  # 3 * $0.025

        assert math.isclose(web_search_cost, expected_cost, rel_tol=1e-10)
        assert math.isclose(web_search_cost, 0.075, rel_tol=1e-10)

    def test_web_search_cost_fallback_calculation(self):
        """Test web search cost calculation using fallback num_sources_used."""
        # Test fallback: num_sources_used on usage object
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        # Manually set num_sources_used (as done by transformation layer)
        setattr(usage, "num_sources_used", 5)

        web_search_cost = cost_per_web_search_request(usage=usage, model_info={})

        # Expected cost: 5 sources * $0.025 per source = $0.125
        expected_cost = 5 * (25.0 / 1000.0)  # 5 * $0.025

        assert math.isclose(web_search_cost, expected_cost, rel_tol=1e-10)
        assert math.isclose(web_search_cost, 0.125, rel_tol=1e-10)

    def test_web_search_no_sources_used(self):
        """Test web search cost calculation when no sources are used."""
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                text_tokens=100,
                web_search_requests=0,  # No web search
            ),
        )

        web_search_cost = cost_per_web_search_request(usage=usage, model_info={})

        # Expected cost: 0 sources * $0.025 per source = $0.0
        assert web_search_cost == 0.0

    def test_web_search_cost_without_prompt_tokens_details(self):
        """Test web search cost calculation when prompt_tokens_details is None."""
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )

        web_search_cost = cost_per_web_search_request(usage=usage, model_info={})

        # Expected cost: No web search data = $0.0
        assert web_search_cost == 0.0

    def test_grok_4_20_beta_reasoning_cost_calculation(self):
        """Test cost calculation for grok-4.20-beta-0309-reasoning model."""
        usage = Usage(prompt_tokens=100, completion_tokens=200, total_tokens=300)

        prompt_cost, completion_cost = cost_per_token(
            model="grok-4.20-beta-0309-reasoning", usage=usage
        )

        # Input: 100 tokens * $2e-6 = $0.0002
        # Output: 200 tokens * $6e-6 = $0.0012
        expected_prompt_cost = 100 * 2e-6
        expected_completion_cost = 200 * 6e-6

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_grok_4_20_beta_non_reasoning_cost_calculation(self):
        """Test cost calculation for grok-4.20-beta-0309-non-reasoning model."""
        usage = Usage(prompt_tokens=50, completion_tokens=100, total_tokens=150)

        prompt_cost, completion_cost = cost_per_token(
            model="grok-4.20-beta-0309-non-reasoning", usage=usage
        )

        # Input: 50 tokens * $2e-6 = $0.0001
        # Output: 100 tokens * $6e-6 = $0.0006
        expected_prompt_cost = 50 * 2e-6
        expected_completion_cost = 100 * 6e-6

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_grok_4_20_multi_agent_cost_calculation(self):
        """Test cost calculation for grok-4.20-multi-agent-beta-0309 model."""
        usage = Usage(prompt_tokens=200, completion_tokens=300, total_tokens=500)

        prompt_cost, completion_cost = cost_per_token(
            model="grok-4.20-multi-agent-beta-0309", usage=usage
        )

        # Input: 200 tokens * $2e-6 = $0.0004
        # Output: 300 tokens * $6e-6 = $0.0018
        expected_prompt_cost = 200 * 2e-6
        expected_completion_cost = 300 * 6e-6

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)


class TestXAIImageCostCalculator:
    @pytest.fixture(autouse=True)
    def _local_model_cost_map(self, monkeypatch):
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        litellm.model_cost = litellm.get_model_cost_map(url="")

    def test_image_cost_uses_output_cost_per_image(self):
        from litellm.llms.xai.cost_calculator import image_cost_calculator
        from litellm.types.utils import ImageObject, ImageResponse

        response = ImageResponse()
        response.data = [ImageObject(url="https://img.x.ai/1.png"), ImageObject(url="https://img.x.ai/2.png")]
        cost = image_cost_calculator(model="grok-imagine-image", image_response=response)
        assert math.isclose(cost, 2 * 0.02, rel_tol=1e-10)

    def test_image_quality_cost(self):
        from litellm.llms.xai.cost_calculator import image_cost_calculator
        from litellm.types.utils import ImageObject, ImageResponse

        response = ImageResponse()
        response.data = [ImageObject(url="https://img.x.ai/1.png")]
        cost = image_cost_calculator(model="grok-imagine-image-quality", image_response=response)
        assert math.isclose(cost, 0.05, rel_tol=1e-10)

    def test_image_cost_rejects_non_image_response(self):
        from litellm.llms.xai.cost_calculator import image_cost_calculator

        with pytest.raises(ValueError, match="ImageResponse"):
            image_cost_calculator(model="grok-imagine-image", image_response={"data": []})

    def test_video_models_have_per_second_pricing(self):
        import litellm

        for model, expected in (("grok-imagine-video", 0.05), ("grok-imagine-video-1.5", 0.08)):
            info = litellm.get_model_info(model=model, custom_llm_provider="xai")
            assert math.isclose(info.get("output_cost_per_video_per_second", 0), expected, rel_tol=1e-10)


class TestXAIGrokImagineCanonicalMapCost:
    """Regression for NOL-107: the canonical price map shipped zero grok-imagine
    entries while the backup carried all four, breaking the canonical==backup
    invariant NOL-90 established. The deployed proxy loads the backup
    (``LITELLM_LOCAL_MODEL_COST_MAP=True``), so live COGS was correct, but any
    consumer of the canonical file resolves grok-imagine pricing to $0 and the
    drift leaves the working backup entries one regeneration away from loss.

    These load the canonical map into ``litellm.model_cost`` and drive the real
    ``completion_cost`` / image cost paths so the entries have to exist in the
    canonical file for the assertions to hold; they fail (get_model_info raises
    "not mapped") on the pre-fix canonical map.
    """

    @pytest.fixture(autouse=True)
    def _canonical_model_cost_map(self, monkeypatch):
        import json

        with open("model_prices_and_context_window.json", "r") as f:
            canonical = json.load(f)
        monkeypatch.setattr(litellm, "model_cost", canonical)

    @staticmethod
    def _video_response(duration_seconds):
        from unittest.mock import MagicMock

        response = MagicMock()
        response.usage = MagicMock()
        response.usage.duration_seconds = duration_seconds
        response.usage.video_resolution = None
        type(response)._hidden_params = {}
        return response

    @pytest.mark.parametrize(
        "model, duration_seconds, expected",
        [
            ("xai/grok-imagine-video", 5.0, 0.25),
            ("xai/grok-imagine-video-1.5", 5.0, 0.40),
        ],
    )
    def test_video_per_second_cost_from_canonical_map(self, model, duration_seconds, expected):
        from litellm.cost_calculator import completion_cost
        from litellm.types.utils import CallTypes

        cost = completion_cost(
            completion_response=self._video_response(duration_seconds),
            model=model,
            custom_llm_provider="xai",
            call_type=CallTypes.acreate_video.value,
        )
        assert math.isclose(cost, expected, rel_tol=1e-10)

    def test_video_cost_is_computed_once_not_doubled(self):
        """NOL-107 double-count guard: a single 45s grok-imagine-video-1.5 job
        must resolve to exactly 45 * $0.08 = $3.60, matching the live spend row,
        and never stack to $7.20."""
        from litellm.cost_calculator import completion_cost
        from litellm.types.utils import CallTypes

        cost = completion_cost(
            completion_response=self._video_response(45.0),
            model="xai/grok-imagine-video-1.5",
            custom_llm_provider="xai",
            call_type=CallTypes.acreate_video.value,
        )
        assert math.isclose(cost, 3.60, rel_tol=1e-10)

    @pytest.mark.parametrize(
        "model, num_images, expected",
        [
            ("grok-imagine-image", 2, 0.04),
            ("grok-imagine-image-quality", 1, 0.05),
        ],
    )
    def test_image_cost_from_canonical_map(self, model, num_images, expected):
        from litellm.llms.xai.cost_calculator import image_cost_calculator
        from litellm.types.utils import ImageObject, ImageResponse

        response = ImageResponse()
        response.data = [ImageObject(url="https://img.x.ai/%d.png" % i) for i in range(num_images)]
        cost = image_cost_calculator(model=model, image_response=response)
        assert math.isclose(cost, expected, rel_tol=1e-10)
