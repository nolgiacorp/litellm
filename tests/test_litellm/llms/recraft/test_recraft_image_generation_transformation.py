import pytest


class TestRecraftV4Pricing:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("recraftv4_1", 0.035),
            ("recraftv4_1_pro", 0.21),
            ("recraftv4_1_utility", 0.035),
            ("recraftv4_1_utility_pro", 0.21),
            ("recraftv4", 0.04),
            ("recraftv4_pro", 0.25),
        ],
    )
    def test_v4_models_have_published_pricing(self, model, expected, monkeypatch):
        import math

        import litellm

        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        litellm.model_cost = litellm.get_model_cost_map(url="")
        info = litellm.get_model_info(model=model, custom_llm_provider="recraft")
        assert math.isclose(info.get("output_cost_per_image", 0), expected, rel_tol=1e-10)
