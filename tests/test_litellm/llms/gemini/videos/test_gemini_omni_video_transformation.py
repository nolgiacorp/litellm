"""
Tests for Gemini Omni (Interactions API) video generation transformation.
"""

import base64
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

    @pytest.mark.parametrize("model", [MODEL, "gemini-omni-1.1-flash"])
    def test_provider_config_dispatch(self, model):
        omni = ProviderConfigManager.get_provider_video_config(model=model, provider=LlmProviders.GEMINI)
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
        assert request_data["store"] is True
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
        monkeypatch.setattr(litellm, "user_url_validation", False)

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
        mock_client.get.assert_called_once_with("https://storage.example/signed.png", follow_redirects=True)

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

    def test_video_status_and_content_dispatch_to_omni_config_from_encoded_id(self):
        from unittest.mock import MagicMock, patch

        from litellm.videos import main as videos_main

        handler = MagicMock()
        video_id = encode_video_id_with_provider("v1_abc123", "gemini", MODEL)
        with patch.object(videos_main, "base_llm_http_handler", handler):
            videos_main.video_status(video_id=video_id)
            videos_main.video_content(video_id=video_id)

        status_config = handler.video_status_handler.call_args.kwargs["video_status_provider_config"]
        assert isinstance(status_config, GeminiOmniVideoConfig)
        content_config = handler.video_content_handler.call_args.kwargs["video_content_provider_config"]
        assert isinstance(content_config, GeminiOmniVideoConfig)

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
        expected_rates = {
            "gemini/gemini-omni-flash-preview": 0.10,
            "gemini/gemini-omni-1.1-flash": 0.10136,
        }
        for model, expected_rate in expected_rates.items():
            info = get_model_info(model)
            assert info["mode"] == "video_generation"
            assert info["output_cost_per_second"] == expected_rate


class TestGeminiOmniEditMode:
    """EDIT mode (https://ai.google.dev/gemini-api/docs/omni, "Edit your own videos"):
    the customer's clip arrives on the fal-shaped video_urls slot and is sent as the
    interaction's video input part with task=edit; the prompt names only the change."""

    def setup_method(self):
        self.config = GeminiOmniVideoConfig()

    def _client_serving(self, content: bytes, content_type: str = "video/mp4"):
        download = Mock()
        download.content = content
        download.headers = {"content-type": content_type}
        download.raise_for_status = Mock()
        client = Mock()
        client.get.return_value = download
        return client

    def test_capability_params_declare_video_urls(self):
        support = self.config.get_capability_param_support(MODEL)
        assert "video_urls" in support.supported
        assert "image_urls" not in support.supported, "Omni edit takes one source clip, not element images"

    def test_edit_request_inlines_a_small_source_clip(self, monkeypatch):
        import litellm

        client = self._client_serving(b"mp4-bytes")
        monkeypatch.setattr(litellm, "module_level_client", client)
        monkeypatch.setattr(litellm, "user_url_validation", False)

        request_data, _, _ = self.config.transform_video_create_request(
            model=MODEL,
            prompt="Add drifting fog along the floor. Keep everything else the same.",
            api_base=f"{API_BASE}/v1beta/interactions",
            video_create_optional_request_params={
                "video_urls": ["https://storage.example/source.mp4"],
                "seconds": 5,
                "aspect_ratio": "16:9",
                "negative_prompt": "text",
            },
            litellm_params=GenericLiteLLMParams(),
            headers={"x-goog-api-key": "k"},
        )
        assert request_data["input"] == [
            {"type": "video", "mime_type": "video/mp4", "data": base64.b64encode(b"mp4-bytes").decode()},
            {
                "type": "text",
                "text": "Add drifting fog along the floor. Keep everything else the same. Do not include: text.",
            },
        ]
        assert request_data["generation_config"] == {"video_config": {"task": "edit"}}
        # The edit follows the source clip: no duration clause, no aspect ratio.
        assert "seconds long" not in request_data["input"][1]["text"]
        assert request_data["response_format"] == {"type": "video"}
        client.get.assert_called_once_with("https://storage.example/source.mp4", follow_redirects=True)

    def test_edit_request_maps_quicktime_onto_the_interactions_enum(self, monkeypatch):
        import litellm

        monkeypatch.setattr(litellm, "module_level_client", self._client_serving(b"mov", "video/quicktime"))
        monkeypatch.setattr(litellm, "user_url_validation", False)
        request_data, _, _ = self.config.transform_video_create_request(
            model=MODEL,
            prompt="Make it rain.",
            api_base=f"{API_BASE}/v1beta/interactions",
            video_create_optional_request_params={"video_urls": ["https://storage.example/source.mov"]},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert request_data["input"][0]["mime_type"] == "video/mov"

    def test_edit_request_refuses_a_start_frame_beside_the_source_clip(self):
        with pytest.raises(ValueError, match="source clip only"):
            self.config.transform_video_create_request(
                model=MODEL,
                prompt="Add fog.",
                api_base=f"{API_BASE}/v1beta/interactions",
                video_create_optional_request_params={
                    "video_urls": ["https://storage.example/source.mp4"],
                    "image_url": "https://storage.example/frame.png",
                },
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

    def test_edit_request_refuses_more_than_one_source_clip(self):
        with pytest.raises(ValueError, match="exactly one source video"):
            self.config.transform_video_create_request(
                model=MODEL,
                prompt="Add fog.",
                api_base=f"{API_BASE}/v1beta/interactions",
                video_create_optional_request_params={
                    "video_urls": ["https://storage.example/a.mp4", "https://storage.example/b.mp4"],
                },
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

    def test_edit_request_refuses_a_malformed_video_urls_value(self):
        with pytest.raises(ValueError, match="list of https URL strings"):
            self.config.transform_video_create_request(
                model=MODEL,
                prompt="Add fog.",
                api_base=f"{API_BASE}/v1beta/interactions",
                video_create_optional_request_params={"video_urls": [{"url": "https://storage.example/a.mp4"}]},
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

    def test_edit_request_uploads_a_large_source_clip_through_the_files_api(self, monkeypatch):
        import litellm
        from litellm.llms.gemini.videos import omni_transformation

        monkeypatch.setattr(omni_transformation, "_INLINE_VIDEO_MAX_BYTES", 4)
        client = self._client_serving(b"mp4-bytes-larger-than-budget")
        start = Mock()
        start.raise_for_status = Mock()
        start.headers = {
            "x-goog-upload-url": "https://generativelanguage.googleapis.com/upload/v1beta/files?upload_id=u1"
        }
        finalize = Mock()
        finalize.raise_for_status = Mock()
        finalize.json.return_value = {
            "file": {
                "name": "files/abc",
                "uri": "https://generativelanguage.googleapis.com/v1beta/files/abc",
                "state": "PROCESSING",
                "mimeType": "video/mp4",
            }
        }
        poll = Mock()
        poll.raise_for_status = Mock()
        poll.json.return_value = {"name": "files/abc", "state": "ACTIVE"}
        client.post.side_effect = [start, finalize]
        # The first GET downloads the source, the second polls the file state.
        client.get.side_effect = [client.get.return_value, poll]
        monkeypatch.setattr(litellm, "module_level_client", client)
        monkeypatch.setattr(litellm, "user_url_validation", False)
        monkeypatch.setattr(omni_transformation.time, "sleep", lambda _s: None)

        request_data, _, _ = self.config.transform_video_create_request(
            model=MODEL,
            prompt="Add fog.",
            api_base=f"{API_BASE}/v1beta/interactions",
            video_create_optional_request_params={"video_urls": ["https://storage.example/source.mp4"]},
            litellm_params=GenericLiteLLMParams(),
            headers={"x-goog-api-key": "secret"},
        )
        assert request_data["input"][0] == {
            "type": "video",
            "mime_type": "video/mp4",
            "uri": "https://generativelanguage.googleapis.com/v1beta/files/abc",
        }
        start_call, finalize_call = client.post.call_args_list
        assert start_call.args[0] == f"{API_BASE}/upload/v1beta/files"
        assert start_call.kwargs["headers"]["X-Goog-Upload-Command"] == "start"
        assert start_call.kwargs["headers"]["x-goog-api-key"] == "secret"
        assert finalize_call.args[0].startswith(f"{API_BASE}/upload/v1beta/files?upload_id=")
        assert finalize_call.kwargs["headers"]["X-Goog-Upload-Command"] == "upload, finalize"
        assert finalize_call.kwargs["data"] == b"mp4-bytes-larger-than-budget"
        poll_call = client.get.call_args_list[1]
        assert poll_call.args[0] == f"{API_BASE}/v1beta/files/abc"

    def test_edit_request_fails_when_the_files_api_rejects_the_clip(self, monkeypatch):
        import litellm
        from litellm.llms.gemini.videos import omni_transformation

        monkeypatch.setattr(omni_transformation, "_INLINE_VIDEO_MAX_BYTES", 1)
        client = self._client_serving(b"mp4")
        start = Mock()
        start.raise_for_status = Mock()
        start.headers = {"x-goog-upload-url": "https://g/upload?u=1"}
        finalize = Mock()
        finalize.raise_for_status = Mock()
        finalize.json.return_value = {"file": {"name": "files/x", "uri": "https://g/v1beta/files/x", "state": "FAILED"}}
        client.post.side_effect = [start, finalize]
        monkeypatch.setattr(litellm, "module_level_client", client)
        monkeypatch.setattr(litellm, "user_url_validation", False)
        with pytest.raises(ValueError, match="state 'FAILED'"):
            self.config.transform_video_create_request(
                model=MODEL,
                prompt="Add fog.",
                api_base=f"{API_BASE}/v1beta/interactions",
                video_create_optional_request_params={"video_urls": ["https://storage.example/source.mp4"]},
                litellm_params=GenericLiteLLMParams(),
                headers={"x-goog-api-key": "secret"},
            )

    def test_generation_without_a_source_clip_is_unchanged(self):
        request_data, _, _ = self.config.transform_video_create_request(
            model=MODEL,
            prompt="A cat.",
            api_base=f"{API_BASE}/v1beta/interactions",
            video_create_optional_request_params={"seconds": 6, "aspect_ratio": "9:16", "video_urls": []},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert request_data["input"] == "A cat. The video must be exactly 6 seconds long."
        assert request_data["response_format"] == {"type": "video", "aspect_ratio": "9:16"}
        assert "generation_config" not in request_data
