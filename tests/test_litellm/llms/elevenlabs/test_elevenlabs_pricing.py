import math

import pytest


class TestElevenLabsTTSPricing:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("eleven_v3", 0.0001),
            ("eleven_multilingual_v2", 0.0001),
            ("eleven_turbo_v2_5", 5e-05),
            ("eleven_flash_v2_5", 5e-05),
        ],
    )
    def test_tts_models_have_current_api_rates(self, model, expected, monkeypatch):
        import litellm

        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        litellm.model_cost = litellm.get_model_cost_map(url="")
        info = litellm.get_model_info(model=model, custom_llm_provider="elevenlabs")
        assert math.isclose(info.get("input_cost_per_character", 0), expected, rel_tol=1e-10)
