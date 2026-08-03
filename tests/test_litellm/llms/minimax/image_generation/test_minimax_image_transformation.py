from unittest.mock import Mock

import httpx
import pytest

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.minimax.image_generation.transformation import (
    MinimaxImageGenerationConfig,
)
from litellm.types.utils import ImageResponse

MODEL = "minimax/image-01"
API_BASE = "https://api.minimax.io"


def _response(payload, status_code=200):
    request = httpx.Request("POST", f"{API_BASE}/v1/image_generation")
    return httpx.Response(status_code, json=payload, request=request)


class TestMinimaxImageTransformation:
    def setup_method(self):
        self.config = MinimaxImageGenerationConfig()
        self.logging_obj = Mock()

    def test_validate_environment_sets_bearer(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        headers = self.config.validate_environment(
            headers={},
            model=MODEL,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="mm-secret",
        )
        assert headers["Authorization"] == "Bearer mm-secret"

    def test_validate_environment_requires_key(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        monkeypatch.setattr("litellm.api_key", None, raising=False)
        with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
            self.config.validate_environment(
                headers={},
                model=MODEL,
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=None,
            )

    def test_validate_environment_prefers_minimax_key_over_global_key(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "mm-env-secret")
        monkeypatch.setattr("litellm.api_key", "other-provider-key", raising=False)
        headers = self.config.validate_environment(
            headers={},
            model=MODEL,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
        )
        assert headers["Authorization"] == "Bearer mm-env-secret"

    def test_validate_environment_falls_back_to_global_key(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        monkeypatch.setattr("litellm.api_key", "global-key", raising=False)
        headers = self.config.validate_environment(
            headers={},
            model=MODEL,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
        )
        assert headers["Authorization"] == "Bearer global-key"

    def test_get_complete_url_targets_image_generation(self):
        url = self.config.get_complete_url(
            api_base=None,
            api_key=None,
            model=MODEL,
            optional_params={},
            litellm_params={},
        )
        assert url == f"{API_BASE}/v1/image_generation"

    def test_map_openai_params_size_to_aspect_ratio(self):
        mapped = self.config.map_openai_params(
            non_default_params={"size": "1280x720", "n": 3, "response_format": "url"},
            optional_params={},
            model=MODEL,
            drop_params=False,
        )
        assert mapped == {"aspect_ratio": "16:9", "n": 3, "response_format": "url"}

    @pytest.mark.parametrize(
        "size, expected",
        [
            ("256x256", "1:1"),
            ("512x512", "1:1"),
            ("1024x1024", "1:1"),
            ("1536x1024", "3:2"),
            ("1024x768", "4:3"),
        ],
    )
    def test_map_openai_params_reduces_size_to_supported_ratio(self, size, expected):
        mapped = self.config.map_openai_params(
            non_default_params={"size": size},
            optional_params={},
            model=MODEL,
            drop_params=False,
        )
        assert mapped["aspect_ratio"] == expected

    def test_map_openai_params_passthrough_native_fields(self):
        mapped = self.config.map_openai_params(
            non_default_params={
                "aspect_ratio": "21:9",
                "seed": 7,
                "prompt_optimizer": True,
                "subject_reference": [{"type": "character", "image_file": "https://img.example.com/face.png"}],
            },
            optional_params={},
            model=MODEL,
            drop_params=False,
        )
        assert mapped["aspect_ratio"] == "21:9"
        assert mapped["seed"] == 7
        assert mapped["prompt_optimizer"] is True
        assert mapped["subject_reference"] == [
            {"type": "character", "image_file": "https://img.example.com/face.png"}
        ]

    def test_map_openai_params_raises_on_unsupported(self):
        with pytest.raises(ValueError, match="not supported"):
            self.config.map_openai_params(
                non_default_params={"totally_unknown": 1},
                optional_params={},
                model=MODEL,
                drop_params=False,
            )

    def test_map_openai_params_drops_unsupported_when_allowed(self):
        mapped = self.config.map_openai_params(
            non_default_params={"totally_unknown": 1, "n": 2},
            optional_params={},
            model=MODEL,
            drop_params=True,
        )
        assert mapped == {"n": 2}

    def test_transform_request_strips_prefix_and_forwards_params(self):
        body = self.config.transform_image_generation_request(
            model=MODEL,
            prompt="a red fox",
            optional_params={"aspect_ratio": "16:9", "n": 2, "response_format": "url", "user": "u1", "size": None},
            litellm_params={},
            headers={},
        )
        assert body == {
            "model": "image-01",
            "prompt": "a red fox",
            "aspect_ratio": "16:9",
            "n": 2,
            "response_format": "url",
        }

    def test_transform_request_merges_extra_body(self):
        body = self.config.transform_image_generation_request(
            model=MODEL,
            prompt="a red fox",
            optional_params={"extra_body": {"prompt_optimizer": True, "width": 1024, "height": 1024}},
            litellm_params={},
            headers={},
        )
        assert body["prompt_optimizer"] is True
        assert body["width"] == 1024
        assert body["height"] == 1024

    def test_transform_request_maps_image_url_to_subject_reference(self):
        body = self.config.transform_image_generation_request(
            model=MODEL,
            prompt="same character on a beach",
            optional_params={"image_url": "https://img.example.com/character.png"},
            litellm_params={},
            headers={},
        )
        assert body["subject_reference"] == (
            {"type": "character", "image_file": "https://img.example.com/character.png"},
        )
        assert "image_url" not in body

    def test_transform_request_explicit_subject_reference_wins_over_image_url(self):
        explicit = [{"type": "character", "image_file": "https://img.example.com/explicit.png"}]
        body = self.config.transform_image_generation_request(
            model=MODEL,
            prompt="same character on a beach",
            optional_params={"image_url": "https://img.example.com/character.png", "subject_reference": explicit},
            litellm_params={},
            headers={},
        )
        assert body["subject_reference"] == explicit
        assert "image_url" not in body

    def test_transform_response_url_images(self):
        response = self.config.transform_image_generation_response(
            model=MODEL,
            raw_response=_response(
                {
                    "id": "trace-1",
                    "data": {"image_urls": ["https://cdn.example.com/a.png", "https://cdn.example.com/b.png"]},
                    "metadata": {"success_count": 2, "failed_count": 0},
                    "base_resp": {"status_code": 0, "status_msg": "success"},
                }
            ),
            model_response=ImageResponse(),
            logging_obj=self.logging_obj,
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )
        assert [image.url for image in response.data] == [
            "https://cdn.example.com/a.png",
            "https://cdn.example.com/b.png",
        ]

    def test_transform_response_base64_images(self):
        response = self.config.transform_image_generation_response(
            model=MODEL,
            raw_response=_response(
                {
                    "data": {"image_base64": ["aGVsbG8="]},
                    "base_resp": {"status_code": 0, "status_msg": "success"},
                }
            ),
            model_response=ImageResponse(),
            logging_obj=self.logging_obj,
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )
        assert response.data[0].b64_json == "aGVsbG8="

    def test_transform_response_base_resp_error_raises(self):
        with pytest.raises(BaseLLMException) as excinfo:
            self.config.transform_image_generation_response(
                model=MODEL,
                raw_response=_response({"base_resp": {"status_code": 1008, "status_msg": "insufficient balance"}}),
                model_response=ImageResponse(),
                logging_obj=self.logging_obj,
                request_data={},
                optional_params={},
                litellm_params={},
                encoding=None,
            )
        assert excinfo.value.status_code == 402
        assert "insufficient balance" in excinfo.value.message

    def test_transform_response_http_error_preserves_status(self):
        raw_response = _response(
            {"error": {"type": "rate_limit_error", "message": "rate limit, please retry later (1002)"}},
            status_code=429,
        )
        with pytest.raises(BaseLLMException) as excinfo:
            self.config.transform_image_generation_response(
                model=MODEL,
                raw_response=raw_response,
                model_response=ImageResponse(),
                logging_obj=self.logging_obj,
                request_data={},
                optional_params={},
                litellm_params={},
                encoding=None,
            )
        assert excinfo.value.status_code == 429
        assert "rate limit" in excinfo.value.message

    def test_transform_response_http_error_without_json_body_preserves_status(self):
        request = httpx.Request("POST", f"{API_BASE}/v1/image_generation")
        raw_response = httpx.Response(502, content=b"<html>bad gateway</html>", request=request)
        with pytest.raises(BaseLLMException) as excinfo:
            self.config.transform_image_generation_response(
                model=MODEL,
                raw_response=raw_response,
                model_response=ImageResponse(),
                logging_obj=self.logging_obj,
                request_data={},
                optional_params={},
                litellm_params={},
                encoding=None,
            )
        assert excinfo.value.status_code == 502
        assert "bad gateway" in excinfo.value.message

    def test_transform_response_no_images_raises(self):
        with pytest.raises(ValueError, match="no images"):
            self.config.transform_image_generation_response(
                model=MODEL,
                raw_response=_response(
                    {"data": {"image_urls": []}, "base_resp": {"status_code": 0, "status_msg": "success"}}
                ),
                model_response=ImageResponse(),
                logging_obj=self.logging_obj,
                request_data={},
                optional_params={},
                litellm_params={},
                encoding=None,
            )

    def test_provider_image_generation_config_registry(self):
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_image_generation_config(
            model=MODEL, provider=litellm.LlmProviders.MINIMAX
        )
        assert isinstance(config, MinimaxImageGenerationConfig)
