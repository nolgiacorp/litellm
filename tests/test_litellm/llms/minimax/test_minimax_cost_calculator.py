import math

import pytest

import litellm
from litellm.llms.minimax.cost_calculator import image_cost_calculator
from litellm.types.utils import ImageObject, ImageResponse


class TestMinimaxCostCalculator:
    @pytest.fixture(autouse=True)
    def _local_model_cost_map(self, monkeypatch):
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        litellm.model_cost = litellm.get_model_cost_map(url="")

    def test_image_cost_uses_output_cost_per_image(self):
        response = ImageResponse()
        response.data = [ImageObject(url="https://cdn.example.com/1.png"), ImageObject(url="https://cdn.example.com/2.png")]
        cost = image_cost_calculator(model="image-01", image_response=response)
        assert math.isclose(cost, 2 * 0.0035, rel_tol=1e-10)

    def test_image_cost_rejects_non_image_response(self):
        with pytest.raises(ValueError, match="ImageResponse"):
            image_cost_calculator(model="image-01", image_response={"data": []})

    @pytest.mark.parametrize(
        "model,expected",
        [
            ("MiniMax-H3", 0.13),
            ("MiniMax-Hailuo-2.3", 0.056),
            ("MiniMax-Hailuo-2.3-Fast", 0.032),
        ],
    )
    def test_video_models_have_per_second_pricing(self, model, expected):
        info = litellm.get_model_info(model=model, custom_llm_provider="minimax")
        assert math.isclose(info.get("output_cost_per_video_per_second", 0), expected, rel_tol=1e-10)

    def test_image_cost_routing_dispatches_to_minimax(self):
        from litellm.litellm_core_utils.llm_cost_calc.utils import CostCalculatorUtils

        response = ImageResponse()
        response.data = [ImageObject(url="https://cdn.example.com/1.png")]
        cost = CostCalculatorUtils.route_image_generation_cost_calculator(
            model="image-01",
            completion_response=response,
            custom_llm_provider="minimax",
        )
        assert math.isclose(cost, 0.0035, rel_tol=1e-10)
