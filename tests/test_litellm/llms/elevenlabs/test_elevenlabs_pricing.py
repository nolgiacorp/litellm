import json
import math
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
ROOT_COST_MAP = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_COST_MAP = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"


class TestElevenLabsTTSPricing:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("eleven_v3", 0.0001),
            ("eleven_multilingual_v2", 0.0001),
            ("eleven_turbo_v2_5", 5e-05),
            ("eleven_flash_v2_5", 5e-05),
            ("eleven_v3_conversational", 5e-05),
        ],
    )
    def test_tts_models_have_current_api_rates(self, model, expected, monkeypatch):
        import litellm

        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        litellm.model_cost = litellm.get_model_cost_map(url="")
        info = litellm.get_model_info(model=model, custom_llm_provider="elevenlabs")
        assert math.isclose(info.get("input_cost_per_character", 0), expected, rel_tol=1e-10)


@pytest.mark.parametrize("cost_map", [ROOT_COST_MAP, BACKUP_COST_MAP], ids=["root", "backup"])
def test_eleven_v3_conversational_is_priced_in_both_cost_maps(cost_map):
    """A proxy on its defaults fetches the root map while one running with
    LITELLM_LOCAL_MODEL_COST_MAP reads the packaged backup, so a TTS model
    present in only one of them bills $0 for half the fleet. eleven_v3_conversational
    was in neither and logged nothing at all."""
    entry = json.loads(cost_map.read_text())["elevenlabs/eleven_v3_conversational"]
    assert math.isclose(entry["input_cost_per_character"], 5e-05, rel_tol=1e-10)
    assert entry["mode"] == "audio_speech"
    assert entry["litellm_provider"] == "elevenlabs"
