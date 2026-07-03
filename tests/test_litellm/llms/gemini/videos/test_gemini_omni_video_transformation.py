"""
Tests for Gemini Omni (Interactions API) video generation transformation.
"""

import base64
import json
import os
from unittest.mock import Mock

import httpx
import pytest

from litellm.llms.gemini.videos.omni_transformation import (
    INTERACTIONS_API_REVISION,
    GeminiOmniVideoConfig,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.utils import encode_video_id_with_provider
from litellm.utils import ProviderConfigManager
from litellm.types.utils import LlmProviders

MODEL = "gemini-omni-flash-preview"
API_BASE = "https://generativelanguage.googleapis.com"


def _response(payload: dict, status_code: int = 200, request_headers: dict = None) -> httpx.Response:
    request = httpx.Request("GET", f"{API_BASE}/v1beta/interactions/v1_abc", headers=request_headers or {})
    return httpx.Response(status_code=status_code, json=payload, request=request)


class TestGeminiOmniVideoConfig:
    def setup_method(self):
        self.config = GeminiOmniVideoConfig()
        self.mock_logging_obj = Mock()

    def test_provider_config_dispatch(self):
        omni = ProviderConfigManager.get_provider_video_config(
            model=MODEL, provider=LlmProviders.GEMINI
        )
        assert isinstance(omni, GeminiOmniVideoConfig)

        from litellm.llms.gemini.videos.transformation import GeminiVideoConfig

        veo = ProviderConfigManager.get_provider_video_config(
            model="veo-3.1-generate-preview", provider=LlmProviders.GEMINI
        )
        assert isinstance(veo, GeminiVideoConfig)
        assert not isinstance(veo, GeminiOmniVideoConfig)

    def test_get_supported_openai_params(self):
        params = self.config.get_supported_openai_params(MODEL)
        assert params == ["model", "prompt", "seconds", "size"]

    def test_validate_environment_sets_headers(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        headers = self.config.validate_environment(headers={}, model=MODEL)
        assert headers["x-goog-api-key"] == "test-key"
        assert headers["Content-Type"] == "application/json"
        assert headers["Api-Revision"] == INTERACTIONS_API_REVISION

    def test_validate_environment_requires_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        import litellm

        monkeypatch.setattr(litellm, "api_key", None)
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            self.config.validate_environment(headers={}, model=MODEL)

    def test_get_complete_url(self):
        url = self.config.get_complete_url(model=MODEL, api_base=None, litellm_params={})
        assert url == f"{API_BASE}/v1beta/interactions"

    def test_get_complete_url_without_model_returns_base(self):
        url = self.config.get_complete_url(model="", api_base=None, litellm_params={})
        assert url == API_BASE

    def test_map_openai_params_size_to_aspect_ratio(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={"size": "720x1280"},
            model=MODEL,
            drop_params=False,
        )
        assert mapped["aspect_ratio"] == "9:16"
        assert "size" not in mapped

    def test_map_openai_params_passes_through_extra_params(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "aspect_ratio": "16:9",
                "duration_seconds": 6,
                "negative_prompt": "text overlays",
            },
            model=MODEL,
            drop_params=False,
        )
        assert mapped["aspect_ratio"] == "16:9"
        assert mapped["duration_seconds"] == 6
        assert mapped["negative_prompt"] == "text overlays"

    def test_transform_video_create_request(self):
        request_data, files, api_base = self.config.transform_video_create_request(
            model="gemini/gemini-omni-flash-preview",
            prompt="A marble rolling on a track.",
            api_base=f"{API_BASE}/v1beta/interactions",
            video_create_optional_request_params={"aspect_ratio": "9:16"},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert request_data["model"] == MODEL
        assert request_data["input"] == "A marble rolling on a track."
        assert request_data["response_format"] == {"type": "video", "aspect_ratio": "9:16"}
        assert request_data["background"] is True
        assert files == []

    def test_transform_video_create_request_folds_duration_and_negative_prompt(self):
        request_data, _, _ = self.config.transform_video_create_request(
            model=MODEL,
            prompt="A cat playing with yarn.",
            api_base=f"{API_BASE}/v1beta/interactions",
            video_create_optional_request_params={
                "duration_seconds": 6,
                "negative_prompt": "captions",
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert request_data["input"] == (
            "A cat playing with yarn. The video must be exactly 6 seconds long. Do not include: captions."
        )

    def test_transform_video_create_request_with_image_url(self, monkeypatch):
        image_bytes = b"png-bytes"
        download_response = Mock()
        download_response.content = image_bytes
        download_response.headers = {"content-type": "image/png"}
        download_response.raise_for_status = Mock()
        mock_client = Mock()
        mock_client.get.return_value = download_response

        import litellm

        monkeypatch.setattr(litellm, "module_level_client", mock_client)

        request_data, _, _ = self.config.transform_video_create_request(
            model=MODEL,
            prompt="Animate this drawing.",
            api_base=f"{API_BASE}/v1beta/interactions",
            video_create_optional_request_params={"image_url": "https://storage.example/signed.png"},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert request_data["input"] == [
            {"type": "image", "data": base64.b64encode(image_bytes).decode(), "mime_type": "image/png"},
            {"type": "text", "text": "Animate this drawing."},
        ]
        assert request_data["generation_config"] == {"video_config": {"task": "image_to_video"}}
        mock_client.get.assert_called_once_with(url="https://storage.example/signed.png")

    def test_transform_video_create_request_ignores_unsupported_aspect_ratio(self):
        request_data, _, _ = self.config.transform_video_create_request(
            model=MODEL,
            prompt="prompt",
            api_base=f"{API_BASE}/v1beta/interactions",
            video_create_optional_request_params={"aspect_ratio": "1:1"},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert request_data["response_format"] == {"type": "video"}

    def test_transform_video_create_response(self):
        raw = _response({"id": "v1_abc123", "status": "in_progress", "object": "interaction", "model": MODEL})
        video = self.config.transform_video_create_response(
            model=MODEL,
            raw_response=raw,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="gemini",
        )
        assert video.status == "processing"
        assert video.id == encode_video_id_with_provider("v1_abc123", "gemini", MODEL)
        assert video.usage["video_resolution"] == "720p"
        assert video.usage["duration_seconds"] > 0

    def test_transform_video_create_response_without_id_raises(self):
        raw = _response({"status": "in_progress"})
        with pytest.raises(ValueError, match="No interaction id"):
            self.config.transform_video_create_response(
                model=MODEL,
                raw_response=raw,
                logging_obj=self.mock_logging_obj,
            )

    def test_status_retrieve_request_url(self):
        video_id = encode_video_id_with_provider("v1_abc123", "gemini", MODEL)
        url, params = self.config.transform_video_status_retrieve_request(
            video_id=video_id,
            api_base=API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert url == f"{API_BASE}/v1beta/interactions/v1_abc123"
        assert params == {}

    @pytest.mark.parametrize(
        "interaction_status,expected",
        [
            ("completed", "completed"),
            ("in_progress", "processing"),
            ("failed", "failed"),
            ("cancelled", "failed"),
            ("budget_exceeded", "failed"),
        ],
    )
    def test_status_retrieve_response_mapping(self, interaction_status, expected):
        raw = _response({"id": "v1_abc123", "status": interaction_status})
        video = self.config.transform_video_status_retrieve_response(
            raw_response=raw,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="gemini",
        )
        assert video.status == expected

    def test_status_retrieve_response_failed_carries_error(self):
        raw = _response({"id": "v1_abc123", "status": "failed"})
        video = self.config.transform_video_status_retrieve_response(
            raw_response=raw,
            logging_obj=self.mock_logging_obj,
        )
        assert video.status == "failed"
        assert video.error is not None
        assert "failed" in video.error["message"]

    def test_content_response_decodes_inline_base64(self):
        video_bytes = b"fake-mp4-bytes"
        raw = _response(
            {
                "id": "v1_abc123",
                "status": "completed",
                "steps": [
                    {"type": "user_input", "content": [{"type": "text", "text": "prompt"}]},
                    {"type": "thought", "content": [{"type": "thought", "text": "..."}]},
                    {
                        "type": "model_output",
                        "content": [
                            {
                                "type": "video",
                                "mime_type": "video/mp4",
                                "data": base64.b64encode(video_bytes).decode(),
                            }
                        ],
                    },
                ],
            }
        )
        assert (
            self.config.transform_video_content_response(raw_response=raw, logging_obj=self.mock_logging_obj)
            == video_bytes
        )

    def test_content_response_incomplete_raises(self):
        raw = _response({"id": "v1_abc123", "status": "in_progress", "steps": []})
        with pytest.raises(ValueError, match="not complete"):
            self.config.transform_video_content_response(raw_response=raw, logging_obj=self.mock_logging_obj)

    def test_content_response_without_video_part_raises(self):
        raw = _response(
            {
                "id": "v1_abc123",
                "status": "completed",
                "steps": [{"type": "model_output", "content": [{"type": "text", "text": "no video"}]}],
            }
        )
        with pytest.raises(ValueError, match="No video output"):
            self.config.transform_video_content_response(raw_response=raw, logging_obj=self.mock_logging_obj)

    def test_content_response_downloads_uri_fallback(self, monkeypatch):
        video_bytes = b"uri-mp4-bytes"
        download_response = Mock()
        download_response.content = video_bytes
        download_response.raise_for_status = Mock()
        mock_client = Mock()
        mock_client.get.return_value = download_response

        import litellm

        monkeypatch.setattr(litellm, "module_level_client", mock_client)

        raw = _response(
            {
                "id": "v1_abc123",
                "status": "completed",
                "steps": [
                    {
                        "type": "model_output",
                        "content": [
                            {
                                "type": "video",
                                "mime_type": "video/mp4",
                                "uri": f"{API_BASE}/v1beta/files/xyz:download?alt=media",
                            }
                        ],
                    }
                ],
            },
            request_headers={"x-goog-api-key": "test-key"},
        )
        result = self.config.transform_video_content_response(raw_response=raw, logging_obj=self.mock_logging_obj)
        assert result == video_bytes
        mock_client.get.assert_called_once()
        _, kwargs = mock_client.get.call_args
        assert kwargs["headers"]["x-goog-api-key"] == "test-key"

    def test_video_remix_not_supported(self):
        with pytest.raises(NotImplementedError):
            self.config.transform_video_remix_request(
                video_id="v1_abc",
                prompt="edit it",
                api_base=API_BASE,
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

    def test_model_registered_for_video_generation(self):
        import litellm
        from litellm import get_model_info

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")
        info = get_model_info("gemini/gemini-omni-flash-preview")
        assert info["mode"] == "video_generation"
        assert info["output_cost_per_second"] == 0.10
