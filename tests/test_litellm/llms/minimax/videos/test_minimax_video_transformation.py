from unittest.mock import Mock

import httpx
import pytest

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.minimax.videos.transformation import MinimaxVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
)

V2_MODEL = "minimax/MiniMax-H3"
V1_MODEL = "minimax/MiniMax-Hailuo-2.3-Fast"
API_BASE = "https://api.minimax.io"


def _response(payload, status_code=200, url=f"{API_BASE}/v2/query/video_generation/424", headers=None):
    request = httpx.Request("GET", url, headers=headers or {})
    return httpx.Response(status_code, json=payload, request=request)


class _FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get(self, url, headers=None):
        self.calls.append({"url": url, "headers": headers or {}})
        payload = self._responses.pop(0)
        request = httpx.Request("GET", url)
        if isinstance(payload, bytes):
            return httpx.Response(200, content=payload, request=request)
        return httpx.Response(200, json=payload, request=request)


class TestMinimaxVideoTransformation:
    def setup_method(self):
        self.config = MinimaxVideoConfig()
        self.logging_obj = Mock()

    def test_validate_environment_sets_bearer(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        headers = self.config.validate_environment(headers={}, model=V2_MODEL, api_key="mm-secret")
        assert headers["Authorization"] == "Bearer mm-secret"
        assert headers["Content-Type"] == "application/json"

    def test_validate_environment_requires_key(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        monkeypatch.setattr("litellm.api_key", None, raising=False)
        with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
            self.config.validate_environment(headers={}, model=V2_MODEL, api_key=None)

    def test_get_complete_url_default(self):
        assert self.config.get_complete_url(model=V2_MODEL, api_base=None, litellm_params={}) == API_BASE

    def test_get_complete_url_strips_trailing_slash(self):
        url = self.config.get_complete_url(model=V2_MODEL, api_base="https://custom.example.com/", litellm_params={})
        assert url == "https://custom.example.com"

    def test_map_v2_text_to_video_defaults_ratio_and_resolution(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={"seconds": "5"},
            model=V2_MODEL,
            drop_params=False,
        )
        assert mapped == {"duration": 5, "resolution": "2K", "ratio": "16:9"}

    def test_map_v2_text_to_video_honors_explicit_ratio(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={"seconds": 4, "aspect_ratio": "9:16", "resolution": "2k"},
            model=V2_MODEL,
            drop_params=False,
        )
        assert mapped == {"duration": 4, "resolution": "2K", "ratio": "9:16"}

    def test_map_v2_text_to_video_replaces_adaptive_ratio(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={"seconds": 4, "aspect_ratio": "adaptive"},
            model=V2_MODEL,
            drop_params=False,
        )
        assert mapped["ratio"] == "16:9"

    def test_map_v2_size_converts_to_ratio(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={"seconds": 4, "size": "720x1280"},
            model=V2_MODEL,
            drop_params=False,
        )
        assert mapped["ratio"] == "9:16"

    def test_map_v2_drops_unsupported_params(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "seconds": 4,
                "negative_prompt": "blurry",
                "seed": 42,
                "generate_audio": True,
                "bitrate_mode": "high",
                "duration_seconds": 9,
            },
            model=V2_MODEL,
            drop_params=False,
        )
        assert mapped == {"duration": 4, "resolution": "2K", "ratio": "16:9"}

    def test_map_v2_defaults_duration_when_absent(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={},
            model=V2_MODEL,
            drop_params=False,
        )
        assert mapped["duration"] == 6

    def test_map_v2_rejects_bad_duration(self):
        with pytest.raises(ValueError, match="duration"):
            self.config.map_openai_params(
                video_create_optional_params={"seconds": "long"},
                model=V2_MODEL,
                drop_params=False,
            )

    def test_map_v2_first_frame_omits_ratio(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "seconds": 5,
                "input_reference": "https://img.example.com/start.png",
                "aspect_ratio": "16:9",
            },
            model=V2_MODEL,
            drop_params=False,
        )
        assert mapped["first_frame"] == "https://img.example.com/start.png"
        assert "ratio" not in mapped

    def test_map_v2_prefers_input_reference_over_image_url(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "input_reference": "https://img.example.com/canonical.png",
                "image_url": "https://img.example.com/mirror.png",
            },
            model=V2_MODEL,
            drop_params=False,
        )
        assert mapped["first_frame"] == "https://img.example.com/canonical.png"

    def test_map_v2_first_and_last_frame(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "seconds": 5,
                "input_reference": "https://img.example.com/start.png",
                "end_image_url": "https://img.example.com/end.png",
            },
            model=V2_MODEL,
            drop_params=False,
        )
        assert mapped["first_frame"] == "https://img.example.com/start.png"
        assert mapped["last_frame"] == "https://img.example.com/end.png"

    def test_map_v2_last_frame_without_first_raises(self):
        with pytest.raises(litellm.BadRequestError, match="first frame"):
            self.config.map_openai_params(
                video_create_optional_params={"end_image_url": "https://img.example.com/end.png"},
                model=V2_MODEL,
                drop_params=False,
            )

    def test_map_v2_frames_and_references_are_mutually_exclusive(self):
        with pytest.raises(litellm.BadRequestError, match="mutually exclusive"):
            self.config.map_openai_params(
                video_create_optional_params={
                    "input_reference": "https://img.example.com/start.png",
                    "image_urls": ["https://img.example.com/character.png"],
                },
                model=V2_MODEL,
                drop_params=False,
            )

    def test_map_v2_reference_media_lists(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "seconds": 5,
                "image_urls": ["https://img.example.com/a.png", "", "https://img.example.com/b.png"],
                "audio_urls": ["https://audio.example.com/ref.mp3"],
            },
            model=V2_MODEL,
            drop_params=False,
        )
        assert mapped["reference_images"] == ("https://img.example.com/a.png", "https://img.example.com/b.png")
        assert mapped["reference_audios"] == ("https://audio.example.com/ref.mp3",)
        assert "reference_videos" not in mapped
        assert "ratio" not in mapped

    def test_map_v2_reference_videos_rejected_because_input_seconds_are_billed(self):
        with pytest.raises(litellm.BadRequestError, match="usage.input_seconds"):
            self.config.map_openai_params(
                video_create_optional_params={
                    "seconds": 5,
                    "video_urls": ["https://video.example.com/ref.mp4"],
                },
                model=V2_MODEL,
                drop_params=False,
            )

    def test_map_v2_reference_media_keeps_explicit_ratio(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "image_urls": ["https://img.example.com/a.png"],
                "aspect_ratio": "1:1",
            },
            model=V2_MODEL,
            drop_params=False,
        )
        assert mapped["ratio"] == "1:1"

    def test_map_v2_input_reference_file_becomes_data_uri(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={"input_reference": ("frame.png", b"png-bytes", "image/png")},
            model=V2_MODEL,
            drop_params=False,
        )
        assert mapped["first_frame"].startswith("data:image/png;base64,")

    def test_map_v2_reads_resolution_from_extra_body(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={"seconds": 4, "extra_body": {"resolution": "768P"}},
            model=V2_MODEL,
            drop_params=False,
        )
        assert mapped["resolution"] == "768P"

    def test_map_legacy_params(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "seconds": 6,
                "resolution": "1080p",
                "input_reference": "https://img.example.com/start.png",
                "end_image_url": "https://img.example.com/end.png",
                "image_urls": ["https://img.example.com/character.png"],
                "negative_prompt": "blurry",
                "generate_audio": True,
                "aspect_ratio": "16:9",
            },
            model=V1_MODEL,
            drop_params=False,
        )
        assert mapped == {
            "duration": 6,
            "resolution": "1080P",
            "first_frame_image": "https://img.example.com/start.png",
        }

    def test_map_legacy_prompt_optimizer_passthrough(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={"seconds": 6, "prompt_optimizer": False},
            model=V1_MODEL,
            drop_params=False,
        )
        assert mapped["prompt_optimizer"] is False

    def test_create_request_v2_text_to_video(self):
        body, files, url = self.config.transform_video_create_request(
            model=V2_MODEL,
            prompt="a rocket launch",
            api_base=API_BASE,
            video_create_optional_request_params={"duration": 4, "resolution": "2K", "ratio": "16:9"},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert url == f"{API_BASE}/v2/video_generation"
        assert files == ()
        assert body["model"] == "MiniMax-H3"
        assert list(body["content"]) == [{"type": "text", "text": "a rocket launch"}]
        assert body["duration"] == 4
        assert body["resolution"] == "2K"
        assert body["ratio"] == "16:9"

    def test_create_request_v2_frames_become_content_items(self):
        body, _files, _url = self.config.transform_video_create_request(
            model=V2_MODEL,
            prompt="a girl grows up",
            api_base=API_BASE,
            video_create_optional_request_params={
                "duration": 5,
                "resolution": "2K",
                "first_frame": "https://img.example.com/start.png",
                "last_frame": "https://img.example.com/end.png",
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert list(body["content"]) == [
            {"type": "text", "text": "a girl grows up"},
            {"type": "image_url", "image_url": {"url": "https://img.example.com/start.png"}, "role": "first_frame"},
            {"type": "image_url", "image_url": {"url": "https://img.example.com/end.png"}, "role": "last_frame"},
        ]
        assert "first_frame" not in body
        assert "last_frame" not in body

    def test_create_request_v2_reference_content_items(self):
        body, _files, _url = self.config.transform_video_create_request(
            model=V2_MODEL,
            prompt="the model walks",
            api_base=API_BASE,
            video_create_optional_request_params={
                "duration": 5,
                "resolution": "2K",
                "reference_images": ["https://img.example.com/subject.png"],
                "reference_videos": ["https://video.example.com/motion.mp4"],
                "reference_audios": ["https://audio.example.com/voice.mp3"],
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert list(body["content"]) == [
            {"type": "text", "text": "the model walks"},
            {"type": "image_url", "image_url": {"url": "https://img.example.com/subject.png"}, "role": "reference_image"},
            {"type": "video_url", "video_url": {"url": "https://video.example.com/motion.mp4"}, "role": "reference_video"},
            {"type": "audio_url", "audio_url": {"url": "https://audio.example.com/voice.mp3"}, "role": "reference_audio"},
        ]

    def test_create_request_v1_body_and_url(self):
        body, files, url = self.config.transform_video_create_request(
            model=V1_MODEL,
            prompt="a mouse runs toward the camera",
            api_base=API_BASE,
            video_create_optional_request_params={
                "duration": 6,
                "resolution": "768P",
                "first_frame_image": "https://img.example.com/start.png",
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert url == f"{API_BASE}/v1/video_generation"
        assert files == ()
        assert body == {
            "model": "MiniMax-Hailuo-2.3-Fast",
            "prompt": "a mouse runs toward the camera",
            "duration": 6,
            "resolution": "768P",
            "first_frame_image": "https://img.example.com/start.png",
        }

    def test_create_response_v2_encodes_provider_and_model(self):
        video = self.config.transform_video_create_response(
            model=V2_MODEL,
            raw_response=_response({"task_id": "424010985738629"}, url=f"{API_BASE}/v2/video_generation"),
            logging_obj=self.logging_obj,
            custom_llm_provider="minimax",
            request_data={"duration": 4},
        )
        assert video.status == "queued"
        assert video.seconds == "4"
        assert video.usage["duration_seconds"] == 4.0
        decoded = decode_video_id_with_provider(video.id)
        assert decoded.get("video_id") == "424010985738629"
        assert decoded.get("custom_llm_provider") == "minimax"
        assert decoded.get("model_id") == "MiniMax-H3"

    def test_create_response_v1_encodes_hailuo_model(self):
        video = self.config.transform_video_create_response(
            model=V1_MODEL,
            raw_response=_response(
                {"task_id": "106916112212032", "base_resp": {"status_code": 0, "status_msg": "success"}},
                url=f"{API_BASE}/v1/video_generation",
            ),
            logging_obj=self.logging_obj,
            custom_llm_provider="minimax",
            request_data={"duration": 6},
        )
        assert video.status == "queued"
        decoded = decode_video_id_with_provider(video.id)
        assert decoded.get("video_id") == "106916112212032"
        assert decoded.get("model_id") == "MiniMax-Hailuo-2.3-Fast"

    def test_create_response_v1_base_resp_error_raises(self):
        with pytest.raises(BaseLLMException) as excinfo:
            self.config.transform_video_create_response(
                model=V1_MODEL,
                raw_response=_response(
                    {"base_resp": {"status_code": 1008, "status_msg": "insufficient balance"}},
                    url=f"{API_BASE}/v1/video_generation",
                ),
                logging_obj=self.logging_obj,
            )
        assert excinfo.value.status_code == 402
        assert "insufficient balance" in excinfo.value.message

    def test_create_response_missing_task_id_raises(self):
        with pytest.raises(ValueError, match="task_id"):
            self.config.transform_video_create_response(
                model=V2_MODEL,
                raw_response=_response({"unexpected": True}, url=f"{API_BASE}/v2/video_generation"),
                logging_obj=self.logging_obj,
            )

    def test_create_response_http_error_extracts_openai_message(self):
        with pytest.raises(BaseLLMException) as excinfo:
            self.config.transform_video_create_response(
                model=V2_MODEL,
                raw_response=_response(
                    {
                        "type": "error",
                        "error": {
                            "type": "insufficient_balance_error",
                            "message": "insufficient balance (1008)",
                            "http_code": "402",
                        },
                    },
                    status_code=402,
                    url=f"{API_BASE}/v2/video_generation",
                ),
                logging_obj=self.logging_obj,
            )
        assert excinfo.value.status_code == 402
        assert excinfo.value.message == "insufficient balance (1008)"

    def test_status_request_url_v2(self):
        encoded = encode_video_id_with_provider("424010985738629", "minimax", "MiniMax-H3")
        url, params = self.config.transform_video_status_retrieve_request(
            video_id=encoded,
            api_base=API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert url == f"{API_BASE}/v2/query/video_generation/424010985738629"
        assert params == {}

    def test_status_request_url_v1(self):
        encoded = encode_video_id_with_provider("106916112212032", "minimax", "MiniMax-Hailuo-2.3-Fast")
        url, params = self.config.transform_video_status_retrieve_request(
            video_id=encoded,
            api_base=API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert url == f"{API_BASE}/v1/query/video_generation?task_id=106916112212032"
        assert params == {}

    @pytest.mark.parametrize(
        "raw_status,expected",
        [
            ("queued", "queued"),
            ("running", "in_progress"),
            ("succeeded", "completed"),
            ("failed", "failed"),
            ("cancelled", "failed"),
            ("expired", "failed"),
            ("something_new", "in_progress"),
        ],
    )
    def test_status_response_v2_mapping(self, raw_status, expected):
        video = self.config.transform_video_status_retrieve_response(
            raw_response=_response({"task": {"id": "424", "status": raw_status}}),
            logging_obj=self.logging_obj,
        )
        assert video.status == expected

    def test_status_v2_success_uses_billed_total_seconds(self):
        video = self.config.transform_video_status_retrieve_response(
            raw_response=_response(
                {
                    "task": {
                        "id": "424",
                        "status": "succeeded",
                        "duration": 5,
                        "content": {"url": "https://cdn.example.com/v.mp4"},
                        "usage": {"total_seconds": 7, "input_seconds": 2, "output_seconds": 5, "image_count": 0},
                    }
                }
            ),
            logging_obj=self.logging_obj,
        )
        assert video.status == "completed"
        assert video.usage["duration_seconds"] == 7.0

    def test_status_v2_falls_back_to_task_duration(self):
        video = self.config.transform_video_status_retrieve_response(
            raw_response=_response({"task": {"id": "424", "status": "succeeded", "duration": 5}}),
            logging_obj=self.logging_obj,
        )
        assert video.usage["duration_seconds"] == 5.0

    def test_status_v2_failed_carries_error(self):
        video = self.config.transform_video_status_retrieve_response(
            raw_response=_response(
                {
                    "task": {
                        "id": "424",
                        "status": "failed",
                        "error": {"code": "1026", "message": "video description contains sensitive content"},
                    }
                }
            ),
            logging_obj=self.logging_obj,
        )
        assert video.status == "failed"
        assert video.error == {"code": "1026", "message": "video description contains sensitive content"}

    @pytest.mark.parametrize(
        "raw_status,expected",
        [
            ("Preparing", "queued"),
            ("Queueing", "queued"),
            ("Processing", "in_progress"),
            ("Success", "completed"),
            ("Fail", "failed"),
        ],
    )
    def test_status_response_v1_mapping(self, raw_status, expected):
        video = self.config.transform_video_status_retrieve_response(
            raw_response=_response(
                {"task_id": "106", "status": raw_status, "base_resp": {"status_code": 0, "status_msg": "success"}},
                url=f"{API_BASE}/v1/query/video_generation?task_id=106",
            ),
            logging_obj=self.logging_obj,
        )
        assert video.status == expected

    def test_status_v1_reencodes_legacy_marker(self):
        video = self.config.transform_video_status_retrieve_response(
            raw_response=_response(
                {"task_id": "106", "status": "Processing", "base_resp": {"status_code": 0, "status_msg": "success"}},
                url=f"{API_BASE}/v1/query/video_generation?task_id=106",
            ),
            logging_obj=self.logging_obj,
            custom_llm_provider="minimax",
        )
        decoded = decode_video_id_with_provider(video.id)
        assert decoded.get("video_id") == "106"
        assert "hailuo" in (decoded.get("model_id") or "").lower()

    def test_extract_v2_video_url(self):
        url = MinimaxVideoConfig._extract_v2_video_url(
            {"task": {"status": "succeeded", "content": {"url": "https://cdn.example.com/v.mp4"}}}
        )
        assert url == "https://cdn.example.com/v.mp4"

    @pytest.mark.parametrize("raw_status", ["failed", "cancelled", "expired"])
    def test_extract_v2_video_url_raises_on_terminal_failure(self, raw_status):
        with pytest.raises(ValueError, match=raw_status):
            MinimaxVideoConfig._extract_v2_video_url(
                {"task": {"status": raw_status, "error": {"code": "1026", "message": "boom"}}}
            )

    def test_extract_v2_video_url_raises_when_pending(self):
        with pytest.raises(ValueError, match="still be processing"):
            MinimaxVideoConfig._extract_v2_video_url({"task": {"status": "running"}})

    def test_content_v1_two_hop_file_retrieve(self, monkeypatch):
        fake_client = _FakeClient(
            [
                {
                    "file": {"file_id": 176844028768320, "download_url": "https://cdn.example.com/output.mp4"},
                    "base_resp": {"status_code": 0, "status_msg": "success"},
                },
                b"mp4-bytes",
            ]
        )
        monkeypatch.setattr(
            "litellm.llms.minimax.videos.transformation._get_httpx_client",
            lambda: fake_client,
        )
        raw_response = _response(
            {
                "task_id": "106",
                "status": "Success",
                "file_id": "176844028768320",
                "base_resp": {"status_code": 0, "status_msg": "success"},
            },
            url=f"{API_BASE}/v1/query/video_generation?task_id=106",
            headers={"Authorization": "Bearer mm-secret"},
        )
        content = self.config.transform_video_content_response(raw_response=raw_response, logging_obj=self.logging_obj)
        assert content == b"mp4-bytes"
        assert fake_client.calls[0]["url"] == f"{API_BASE}/v1/files/retrieve?file_id=176844028768320"
        assert fake_client.calls[0]["headers"]["Authorization"] == "Bearer mm-secret"
        assert fake_client.calls[1]["url"] == "https://cdn.example.com/output.mp4"

    def test_content_v1_fail_status_raises(self):
        raw_response = _response(
            {"task_id": "106", "status": "Fail", "base_resp": {"status_code": 0, "status_msg": "generation failed"}},
            url=f"{API_BASE}/v1/query/video_generation?task_id=106",
        )
        with pytest.raises(ValueError, match="generation failed"):
            self.config.transform_video_content_response(raw_response=raw_response, logging_obj=self.logging_obj)

    def test_content_v1_missing_file_id_raises(self):
        raw_response = _response(
            {"task_id": "106", "status": "Processing", "base_resp": {"status_code": 0, "status_msg": "success"}},
            url=f"{API_BASE}/v1/query/video_generation?task_id=106",
        )
        with pytest.raises(ValueError, match="still be processing"):
            self.config.transform_video_content_response(raw_response=raw_response, logging_obj=self.logging_obj)

    def test_content_v2_downloads_bytes(self, monkeypatch):
        fake_client = _FakeClient([b"mp4-bytes"])
        monkeypatch.setattr(
            "litellm.llms.minimax.videos.transformation._get_httpx_client",
            lambda: fake_client,
        )
        raw_response = _response(
            {"task": {"id": "424", "status": "succeeded", "content": {"url": "https://cdn.example.com/v.mp4"}}}
        )
        content = self.config.transform_video_content_response(raw_response=raw_response, logging_obj=self.logging_obj)
        assert content == b"mp4-bytes"
        assert fake_client.calls[0]["url"] == "https://cdn.example.com/v.mp4"

    def test_provider_video_config_registry(self):
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_video_config(
            model=V2_MODEL, provider=litellm.LlmProviders.MINIMAX
        )
        assert isinstance(config, MinimaxVideoConfig)
