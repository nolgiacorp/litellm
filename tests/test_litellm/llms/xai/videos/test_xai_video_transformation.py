from unittest.mock import Mock

import httpx
import pytest

from litellm.llms.xai.videos.transformation import XAIVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
)

MODEL = "xai/grok-imagine-video"
API_BASE = "https://api.x.ai"


def _response(payload, status_code=200, request_id="req-1"):
    request = httpx.Request("GET", f"{API_BASE}/v1/videos/{request_id}")
    return httpx.Response(status_code, json=payload, request=request)


class TestXAIVideoTransformation:
    def setup_method(self):
        self.config = XAIVideoConfig()
        self.logging_obj = Mock()

    def test_validate_environment_sets_bearer(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        headers = self.config.validate_environment(headers={}, model=MODEL, api_key="xai-secret")
        assert headers["Authorization"] == "Bearer xai-secret"

    def test_validate_environment_requires_key(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        monkeypatch.setattr("litellm.xai_key", None, raising=False)
        with pytest.raises(ValueError, match="XAI_API_KEY"):
            self.config.validate_environment(headers={}, model=MODEL, api_key=None)

    def test_get_complete_url_default(self, monkeypatch):
        monkeypatch.delenv("XAI_API_BASE", raising=False)
        assert self.config.get_complete_url(model=MODEL, api_base=None, litellm_params={}) == API_BASE

    def test_get_complete_url_strips_v1_suffix(self):
        url = self.config.get_complete_url(model=MODEL, api_base="https://custom.example.com/v1/", litellm_params={})
        assert url == "https://custom.example.com"

    def test_map_openai_params_full_mapping(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "seconds": "10",
                "size": "720x1280",
                "aspect_ratio": "16:9",
                "input_reference": "https://img.example.com/start.png",
                "extra_body": {"resolution": "720p", "reference_images": ["https://img.example.com/a.png"]},
            },
            model=MODEL,
            drop_params=False,
        )
        assert mapped["duration"] == 10
        assert mapped["aspect_ratio"] == "9:16"
        assert mapped["resolution"] == "720p"
        assert mapped["reference_images"] == [{"url": "https://img.example.com/a.png"}]
        assert "image" not in mapped

    def test_map_openai_params_prefers_input_reference_over_fal_image_url(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "input_reference": "https://img.example.com/canonical-start.png",
                "image_url": "https://img.example.com/fal-start.png",
            },
            model=MODEL,
            drop_params=False,
        )
        assert mapped == {"image": {"url": "https://img.example.com/canonical-start.png"}}

    def test_map_openai_params_maps_fal_image_url(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={"image_url": "https://img.example.com/start.png"},
            model=MODEL,
            drop_params=False,
        )
        assert mapped == {"image": {"url": "https://img.example.com/start.png"}}

    def test_map_openai_params_maps_fal_image_urls(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "image_urls": [
                    "https://img.example.com/character-a.png",
                    "",
                    "https://img.example.com/character-b.png",
                ]
            },
            model=MODEL,
            drop_params=False,
        )
        assert mapped["reference_images"] == [
            {"url": "https://img.example.com/character-a.png"},
            {"url": "https://img.example.com/character-b.png"},
        ]
        assert "image" not in mapped
        assert "image_urls" not in mapped

    def test_map_openai_params_prefers_reference_images_over_fal_image_url(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "image_url": "https://img.example.com/start.png",
                "image_urls": ["https://img.example.com/character.png"],
            },
            model=MODEL,
            drop_params=False,
        )
        assert mapped == {"reference_images": [{"url": "https://img.example.com/character.png"}]}

    def test_map_openai_params_forwards_reference_audios_voice_objects(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "extra_body": {"reference_audios": [{"voice_id": "eve"}, {"voice_id": "ara"}]},
            },
            model="xai/grok-imagine-video-1.5",
            drop_params=False,
        )
        assert mapped["reference_audios"] == [{"voice_id": "eve"}, {"voice_id": "ara"}]

    def test_map_openai_params_normalizes_reference_audios_from_voice_id_strings(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "extra_body": {"reference_audios": ["eve", "", "rex"]},
            },
            model="xai/grok-imagine-video-1.5",
            drop_params=False,
        )
        assert mapped["reference_audios"] == [{"voice_id": "eve"}, {"voice_id": "rex"}]

    def test_reference_audios_are_mutually_exclusive_with_start_frame_image(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "input_reference": "https://img.example.com/start.png",
                "extra_body": {"reference_audios": [{"voice_id": "eve"}]},
            },
            model="xai/grok-imagine-video-1.5",
            drop_params=False,
        )
        assert "image" not in mapped
        assert mapped["reference_audios"] == [{"voice_id": "eve"}]

    def test_transform_video_create_request_sends_reference_audios_to_xai(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={"extra_body": {"reference_audios": [{"voice_id": "eve"}]}},
            model="xai/grok-imagine-video-1.5",
            drop_params=False,
        )
        body, _files, url = self.config.transform_video_create_request(
            model="xai/grok-imagine-video-1.5",
            prompt="<AUDIO_0> narrates the scene",
            api_base=API_BASE,
            video_create_optional_request_params=mapped,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert url == f"{API_BASE}/v1/videos/generations"
        assert body["reference_audios"] == [{"voice_id": "eve"}]

    def test_map_openai_params_drops_fal_only_params(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "seconds": 8,
                "duration": "12.9",
                "duration_seconds": 5,
                "seed": 1,
                "negative_prompt": "x",
                "generate_audio": True,
                "end_image_url": "https://img.example.com/end.png",
                "video_urls": ["https://video.example.com/reference.mp4"],
                "audio_urls": ["https://audio.example.com/reference.mp3"],
                "bitrate_mode": "high",
            },
            model=MODEL,
            drop_params=False,
        )
        assert mapped == {"duration": 8}

    def test_map_openai_params_normalizes_explicit_xai_fields(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "duration": "8.9",
                "aspect_ratio": "16:9",
                "resolution": "1080p",
                "reference_images": "https://img.example.com/character.png",
            },
            model=MODEL,
            drop_params=False,
        )
        assert mapped == {
            "duration": 8,
            "aspect_ratio": "16:9",
            "resolution": "1080p",
            "reference_images": [{"url": "https://img.example.com/character.png"}],
        }

    def test_map_openai_params_wraps_start_frame_image_as_image_url_struct(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={"image_url": "https://img.example.com/start.png"},
            model=MODEL,
            drop_params=False,
        )
        assert mapped["image"] == {"url": "https://img.example.com/start.png"}
        assert not isinstance(mapped["image"], str)

    def test_map_openai_params_wraps_every_reference_image_as_image_url_struct(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "image_urls": ["https://img.example.com/a.png", "https://img.example.com/b.png"]
            },
            model=MODEL,
            drop_params=False,
        )
        assert mapped["reference_images"] == [
            {"url": "https://img.example.com/a.png"},
            {"url": "https://img.example.com/b.png"},
        ]
        assert all(isinstance(item, dict) and set(item) == {"url"} for item in mapped["reference_images"])

    def test_transform_video_create_request_sends_image_url_structs_to_xai(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={"image_urls": ["https://img.example.com/character.png"]},
            model=MODEL,
            drop_params=False,
        )
        body, _files, url = self.config.transform_video_create_request(
            model=MODEL,
            prompt="@Image1 waves at the camera",
            api_base=API_BASE,
            video_create_optional_request_params=mapped,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert url == f"{API_BASE}/v1/videos/generations"
        assert body["reference_images"] == [{"url": "https://img.example.com/character.png"}]

    def test_map_openai_params_rejects_bad_duration(self):
        with pytest.raises(ValueError, match="duration"):
            self.config.map_openai_params(
                video_create_optional_params={"seconds": "long"},
                model=MODEL,
                drop_params=False,
            )

    def test_create_request_body_and_url(self):
        body, files, url = self.config.transform_video_create_request(
            model=MODEL,
            prompt="a rocket launch",
            api_base=API_BASE,
            video_create_optional_request_params={"duration": 8, "aspect_ratio": "16:9", "resolution": "1080p"},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert url == f"{API_BASE}/v1/videos/generations"
        assert files == []
        assert body == {
            "model": "grok-imagine-video",
            "prompt": "a rocket launch",
            "duration": 8,
            "aspect_ratio": "16:9",
            "resolution": "1080p",
        }

    def test_create_response_encodes_provider_and_usage(self):
        video = self.config.transform_video_create_response(
            model=MODEL,
            raw_response=_response({"request_id": "req-42"}),
            logging_obj=self.logging_obj,
            custom_llm_provider="xai",
            request_data={"duration": 8},
        )
        assert video.status == "queued"
        assert video.seconds == "8"
        assert video.usage["duration_seconds"] == 8.0
        assert video.id != "req-42"
        decoded = decode_video_id_with_provider(video.id)
        assert decoded.get("video_id") == "req-42"
        assert decoded.get("custom_llm_provider") == "xai"

    def test_create_response_missing_request_id_raises(self):
        with pytest.raises(ValueError, match="request_id"):
            self.config.transform_video_create_response(
                model=MODEL,
                raw_response=_response({"unexpected": True}),
                logging_obj=self.logging_obj,
            )

    @pytest.mark.parametrize(
        "raw_status,expected",
        [("pending", "in_progress"), ("done", "completed"), ("failed", "failed"), ("expired", "failed")],
    )
    def test_status_response_mapping(self, raw_status, expected):
        video = self.config.transform_video_status_retrieve_response(
            raw_response=_response({"request_id": "req-1", "status": raw_status}),
            logging_obj=self.logging_obj,
        )
        assert video.status == expected

    def test_status_done_carries_actual_duration(self):
        video = self.config.transform_video_status_retrieve_response(
            raw_response=_response(
                {
                    "request_id": "req-1",
                    "status": "done",
                    "video": {"url": "https://vidgen.x.ai/v.mp4", "duration": 9.5},
                }
            ),
            logging_obj=self.logging_obj,
        )
        assert video.status == "completed"
        assert video.usage["duration_seconds"] == 9.5

    def test_status_failed_carries_error(self):
        video = self.config.transform_video_status_retrieve_response(
            raw_response=_response({"request_id": "req-1", "status": "failed", "error": "moderated"}),
            logging_obj=self.logging_obj,
        )
        assert video.status == "failed"
        assert video.error == {"code": "failed", "message": "moderated"}

    def test_task_url_from_encoded_id(self):
        encoded = encode_video_id_with_provider("req-7", "xai", "grok-imagine-video")
        url, params = self.config.transform_video_status_retrieve_request(
            video_id=encoded,
            api_base=API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert url == f"{API_BASE}/v1/videos/req-7"
        assert params == {}

    def test_extract_video_url(self):
        assert (
            XAIVideoConfig._extract_video_url({"status": "done", "video": {"url": "https://vidgen.x.ai/v.mp4"}})
            == "https://vidgen.x.ai/v.mp4"
        )

    @pytest.mark.parametrize("raw_status", ["failed", "expired"])
    def test_extract_video_url_raises_on_terminal_failure(self, raw_status):
        with pytest.raises(ValueError, match="failed"):
            XAIVideoConfig._extract_video_url({"status": raw_status, "error": "boom"})

    def test_extract_video_url_raises_when_pending(self):
        with pytest.raises(ValueError, match="still be processing"):
            XAIVideoConfig._extract_video_url({"status": "pending"})

    def test_provider_video_config_registry(self):
        import litellm
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_video_config(model=MODEL, provider=litellm.LlmProviders.XAI)
        assert isinstance(config, XAIVideoConfig)
