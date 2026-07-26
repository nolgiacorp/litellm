from unittest.mock import Mock

import httpx
import pytest

from litellm.llms.xai.image_generation.transformation import XAIImageGenerationConfig
from litellm.types.utils import ImageResponse

MODEL = "xai/grok-imagine-image"


def _response(payload, status_code=200):
    request = httpx.Request("POST", "https://api.x.ai/v1/images/generations")
    return httpx.Response(status_code, json=payload, request=request)


class TestXAIImageGenerationTransformation:
    def setup_method(self):
        self.config = XAIImageGenerationConfig()
        self.logging_obj = Mock()

    def test_get_complete_url_default(self, monkeypatch):
        monkeypatch.delenv("XAI_API_BASE", raising=False)
        url = self.config.get_complete_url(
            api_base=None, api_key=None, model=MODEL, optional_params={}, litellm_params={}
        )
        assert url == "https://api.x.ai/v1/images/generations"

    def test_get_complete_url_handles_v1_base(self):
        url = self.config.get_complete_url(
            api_base="https://api.x.ai/v1", api_key=None, model=MODEL, optional_params={}, litellm_params={}
        )
        assert url == "https://api.x.ai/v1/images/generations"

    def test_validate_environment_sets_bearer(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        headers = self.config.validate_environment(
            headers={}, model=MODEL, messages=[], optional_params={}, litellm_params={}, api_key="xai-secret"
        )
        assert headers["Authorization"] == "Bearer xai-secret"

    def test_validate_environment_requires_key(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        monkeypatch.setattr("litellm.xai_key", None, raising=False)
        with pytest.raises(ValueError, match="XAI_API_KEY"):
            self.config.validate_environment(
                headers={}, model=MODEL, messages=[], optional_params={}, litellm_params={}
            )

    def test_map_size_to_aspect_ratio(self):
        mapped = self.config.map_openai_params(
            non_default_params={"size": "1920x1080"}, optional_params={}, model=MODEL, drop_params=False
        )
        assert mapped["aspect_ratio"] == "16:9"
        assert "size" not in mapped

    def test_map_passthrough_resolution_and_aspect_ratio(self):
        mapped = self.config.map_openai_params(
            non_default_params={"resolution": "2k", "aspect_ratio": "3:2", "n": 2},
            optional_params={},
            model=MODEL,
            drop_params=False,
        )
        assert mapped == {"resolution": "2k", "aspect_ratio": "3:2", "n": 2}

    def test_map_rejects_unsupported_without_drop_params(self):
        with pytest.raises(ValueError, match="not supported"):
            self.config.map_openai_params(
                non_default_params={"quality": "hd"}, optional_params={}, model=MODEL, drop_params=False
            )

    def test_map_drops_unsupported_with_drop_params(self):
        mapped = self.config.map_openai_params(
            non_default_params={"quality": "hd", "n": 1}, optional_params={}, model=MODEL, drop_params=True
        )
        assert mapped == {"n": 1}

    def test_request_body_strips_provider_prefix(self):
        body = self.config.transform_image_generation_request(
            model=MODEL,
            prompt="a red fox",
            optional_params={"aspect_ratio": "1:1", "resolution": "1k", "n": 2},
            litellm_params={},
            headers={},
        )
        assert body == {
            "model": "grok-imagine-image",
            "prompt": "a red fox",
            "aspect_ratio": "1:1",
            "resolution": "1k",
            "n": 2,
        }

    def test_request_body_merges_extra_body(self):
        body = self.config.transform_image_generation_request(
            model=MODEL,
            prompt="a red fox",
            optional_params={"extra_body": {"resolution": "2k"}},
            litellm_params={},
            headers={},
        )
        assert body["resolution"] == "2k"

    def test_response_parses_urls_and_b64(self):
        model_response = ImageResponse()
        result = self.config.transform_image_generation_response(
            model=MODEL,
            raw_response=_response({"data": [{"url": "https://img.x.ai/1.png"}, {"b64_json": "Zm9v"}]}),
            model_response=model_response,
            logging_obj=self.logging_obj,
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )
        assert len(result.data) == 2
        assert result.data[0].url == "https://img.x.ai/1.png"
        assert result.data[1].b64_json == "Zm9v"

    def test_provider_image_generation_config_registry(self):
        import litellm
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_image_generation_config(
            model=MODEL, provider=litellm.LlmProviders.XAI
        )
        assert isinstance(config, XAIImageGenerationConfig)
