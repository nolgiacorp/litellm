from unittest.mock import Mock

import httpx
import pytest

from litellm.llms.kling.videos.transformation import KlingVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
)

MODEL = "kling/kling-v3"
API_BASE = "https://api-singapore.klingai.com/v1"


def _status_response(payload, task_kind="text2video", task_id="t-1", status_code=200):
    request = httpx.Request("GET", f"{API_BASE}/videos/{task_kind}/{task_id}")
    return httpx.Response(status_code, json=payload, request=request)


class TestKlingVideoTransformation:
    def setup_method(self):
        self.config = KlingVideoConfig()
        self.logging_obj = Mock()

    def test_validate_environment_sets_bearer_jwt(self):
        headers = self.config.validate_environment(headers={}, model=MODEL, api_key="A" * 32 + ":" + "S" * 32)
        assert headers["Authorization"].startswith("Bearer ")
        assert headers["Content-Type"] == "application/json"

    def test_get_complete_url_default_includes_v1(self, monkeypatch):
        monkeypatch.delenv("KLING_API_BASE", raising=False)
        url = self.config.get_complete_url(model=MODEL, api_base=None, litellm_params={})
        assert url == API_BASE

    def test_get_complete_url_strips_trailing_slash(self):
        url = self.config.get_complete_url(model=MODEL, api_base="https://custom.example.com/v1/", litellm_params={})
        assert url == "https://custom.example.com/v1"

    @pytest.mark.parametrize(
        "resolution,expected_mode",
        [("720p", "std"), ("1080p", "pro"), ("4k", "4k"), ("4K", "4k")],
    )
    def test_map_resolution_to_mode_all_tiers(self, resolution, expected_mode):
        mapped = self.config.map_openai_params(
            video_create_optional_params={"resolution": resolution},
            model=MODEL,
            drop_params=False,
        )
        assert mapped["mode"] == expected_mode
        assert "resolution" not in mapped

    def test_map_rejects_unknown_resolution(self):
        with pytest.raises(ValueError, match="Unsupported Kling video resolution"):
            self.config.map_openai_params(
                video_create_optional_params={"resolution": "8k"},
                model=MODEL,
                drop_params=False,
            )

    def test_map_seconds_and_size(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={"seconds": 5, "size": "1920x1080"},
            model=MODEL,
            drop_params=False,
        )
        assert mapped["duration"] == "5"
        assert mapped["aspect_ratio"] == "16:9"

    def test_map_forwards_extra_body_and_input_reference(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "input_reference": "https://img/x.png",
                "extra_body": {"negative_prompt": "blurry", "audio": True},
            },
            model=MODEL,
            drop_params=False,
        )
        assert mapped["image"] == "https://img/x.png"
        assert mapped["negative_prompt"] == "blurry"
        assert mapped["audio"] is True
        assert "extra_body" not in mapped

    @pytest.mark.parametrize(
        "resolution,expected_mode",
        [("720p", "std"), ("1080p", "pro"), ("4k", "4k")],
    )
    def test_create_request_resolution_reaches_body_as_mode(self, resolution, expected_mode):
        mapped = self.config.map_openai_params(
            video_create_optional_params={"resolution": resolution, "seconds": 5},
            model=MODEL,
            drop_params=False,
        )
        data, files, url = self.config.transform_video_create_request(
            model=MODEL,
            prompt="a cat playing piano",
            api_base=API_BASE,
            video_create_optional_request_params=mapped,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert url == f"{API_BASE}/videos/text2video"
        assert data["model_name"] == "kling-v3"
        assert data["mode"] == expected_mode
        assert data["prompt"] == "a cat playing piano"
        assert data["duration"] == "5"
        assert files == []

    def test_create_request_defaults_mode_to_pro(self):
        data, _, _ = self.config.transform_video_create_request(
            model=MODEL,
            prompt="x",
            api_base=API_BASE,
            video_create_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert data["mode"] == "pro"

    def test_create_request_image_triggers_image2video(self):
        data, _, url = self.config.transform_video_create_request(
            model=MODEL,
            prompt="animate",
            api_base=API_BASE,
            video_create_optional_request_params={"image": "https://img/x.png"},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert url == f"{API_BASE}/videos/image2video"
        assert data["image"] == "https://img/x.png"

    def test_create_response_encodes_kind_and_task_id(self):
        response = Mock(spec=httpx.Response)
        response.json.return_value = {
            "code": 0,
            "data": {"task_id": "task-123", "task_status": "submitted"},
        }
        video_obj = self.config.transform_video_create_response(
            model=MODEL,
            raw_response=response,
            logging_obj=self.logging_obj,
            custom_llm_provider="kling",
            request_data={"image": "https://img/x.png", "duration": "5"},
        )
        assert isinstance(video_obj, VideoObject)
        assert video_obj.status == "queued"
        assert video_obj.id.startswith("video_")
        decoded = decode_video_id_with_provider(video_obj.id)
        assert decoded["video_id"] == "task-123"
        assert decoded["custom_llm_provider"] == "kling"
        assert decoded["model_id"] == "image2video"
        assert video_obj.seconds == "5"

    def test_create_response_raises_on_error_code(self):
        response = Mock(spec=httpx.Response)
        response.json.return_value = {"code": 1002, "message": "AK/SK not supported"}
        with pytest.raises(Exception, match="AK/SK not supported"):
            self.config.transform_video_create_response(
                model=MODEL,
                raw_response=response,
                logging_obj=self.logging_obj,
                custom_llm_provider="kling",
                request_data={},
            )

    def test_create_response_raises_when_task_id_missing(self):
        response = Mock(spec=httpx.Response)
        response.json.return_value = {"code": 0, "data": {}}
        with pytest.raises(ValueError, match="missing data.task_id"):
            self.config.transform_video_create_response(
                model=MODEL,
                raw_response=response,
                logging_obj=self.logging_obj,
                custom_llm_provider="kling",
                request_data={},
            )

    @pytest.mark.parametrize("kind", ["text2video", "image2video"])
    def test_status_and_content_urls_reconstruct_from_kind(self, kind):
        encoded = encode_video_id_with_provider("task-7", "kling", kind)
        status_url, params = self.config.transform_video_status_retrieve_request(
            video_id=encoded,
            api_base=API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        content_url, _ = self.config.transform_video_content_request(
            video_id=encoded,
            api_base=API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert status_url == f"{API_BASE}/videos/{kind}/task-7"
        assert content_url == f"{API_BASE}/videos/{kind}/task-7"
        assert params == {}

    def test_status_url_ignores_untrusted_api_base_host_but_reuses_provided(self):
        encoded = encode_video_id_with_provider("task-7", "kling", "text2video")
        status_url, _ = self.config.transform_video_status_retrieve_request(
            video_id=encoded,
            api_base="https://attacker.example.com/v1",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert status_url == "https://attacker.example.com/v1/videos/text2video/task-7"

    def test_status_url_encodes_path_segment(self):
        encoded = encode_video_id_with_provider("../../../etc/passwd", "kling", "text2video")
        status_url, _ = self.config.transform_video_status_retrieve_request(
            video_id=encoded,
            api_base=API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert "..%2F..%2F..%2Fetc%2Fpasswd" in status_url

    def test_status_request_requires_kind_in_video_id(self):
        encoded = encode_video_id_with_provider("task-7", "kling", None)
        with pytest.raises(ValueError, match="kind encoded in the video_id"):
            self.config.transform_video_status_retrieve_request(
                video_id=encoded,
                api_base=API_BASE,
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

    def test_status_request_rejects_unknown_kind(self):
        encoded = encode_video_id_with_provider("task-7", "kling", "etc/passwd")
        with pytest.raises(ValueError, match="kind encoded in the video_id"):
            self.config.transform_video_content_request(
                video_id=encoded,
                api_base=API_BASE,
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

    @pytest.mark.parametrize(
        "task_status,expected",
        [
            ("submitted", "queued"),
            ("processing", "in_progress"),
            ("succeed", "completed"),
        ],
    )
    def test_status_response_maps_task_status(self, task_status, expected):
        response = _status_response({"code": 0, "data": {"task_id": "t-1", "task_status": task_status}})
        status_obj = self.config.transform_video_status_retrieve_response(
            raw_response=response,
            logging_obj=self.logging_obj,
            custom_llm_provider="kling",
        )
        assert status_obj.status == expected

    def test_status_response_failed_sets_error(self):
        response = _status_response(
            {
                "code": 0,
                "data": {
                    "task_id": "t-1",
                    "task_status": "failed",
                    "task_status_msg": "content moderation",
                },
            }
        )
        status_obj = self.config.transform_video_status_retrieve_response(
            raw_response=response,
            logging_obj=self.logging_obj,
            custom_llm_provider="kling",
        )
        assert status_obj.status == "failed"
        assert status_obj.error is not None
        assert status_obj.error["message"] == "content moderation"

    def test_status_response_tolerates_non_json(self):
        response = Mock(spec=httpx.Response)
        response.is_success = True
        response.json.side_effect = ValueError("no json")
        status_obj = self.config.transform_video_status_retrieve_response(
            raw_response=response,
            logging_obj=self.logging_obj,
            custom_llm_provider="kling",
        )
        assert status_obj.status == "in_progress"

    def test_extract_video_url_from_task_result(self):
        url = self.config._extract_video_url(
            {
                "code": 0,
                "data": {
                    "task_status": "succeed",
                    "task_result": {"videos": [{"url": "https://cdn/v.mp4"}]},
                },
            }
        )
        assert url == "https://cdn/v.mp4"

    def test_extract_video_url_raises_on_failed(self):
        with pytest.raises(ValueError, match="Kling video generation failed"):
            self.config._extract_video_url({"data": {"task_status": "failed", "task_status_msg": "nsfw"}})

    def test_extract_video_url_raises_when_missing(self):
        with pytest.raises(ValueError, match="Video URL not found"):
            self.config._extract_video_url({"data": {"task_status": "processing", "task_result": {}}})

    def test_remix_list_delete_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self.config.transform_video_remix_request(
                video_id="x",
                prompt="p",
                api_base=API_BASE,
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )
        with pytest.raises(NotImplementedError):
            self.config.transform_video_list_request(
                api_base=API_BASE, litellm_params=GenericLiteLLMParams(), headers={}
            )
        with pytest.raises(NotImplementedError):
            self.config.transform_video_delete_request(
                video_id="x",
                api_base=API_BASE,
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )


def test_provider_config_manager_returns_kling_video_config():
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_video_config(model="kling-v3", provider=LlmProviders.KLING)
    assert isinstance(config, KlingVideoConfig)


def test_get_llm_provider_routes_kling():
    from litellm import get_llm_provider

    model, provider, _, _ = get_llm_provider("kling/kling-v3")
    assert model == "kling-v3"
    assert provider == "kling"
