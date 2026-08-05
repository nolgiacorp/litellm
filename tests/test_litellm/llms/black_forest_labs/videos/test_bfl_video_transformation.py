import base64
import io
from unittest.mock import Mock

import httpx
import pytest

from litellm.llms.black_forest_labs.common_utils import BlackForestLabsError
from litellm.llms.black_forest_labs.videos.transformation import BflVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
)

MODEL = "black_forest_labs/flux-3-video"
API_BASE = "https://api.bfl.ai"
POLLING_URL = "https://gateway.bfl.ai/v1/get_result?id=req-123"
PROVIDER = "black_forest_labs"


def _result_response(payload, request_url=POLLING_URL, status_code=200):
    request = httpx.Request("GET", request_url)
    return httpx.Response(status_code, json=payload, request=request)


class _RecordingClient:
    """Injected httpx handler stand-in that records the follow-up download call."""

    def __init__(self, response):
        self._response = response
        self.calls = []

    def get(self, url, headers=None, **kwargs):
        self.calls.append((url, headers))
        return self._response


class _RecordingAsyncClient(_RecordingClient):
    async def get(self, url, headers=None, **kwargs):
        self.calls.append((url, headers))
        return self._response


class TestBflVideoMapAndCreate:
    def setup_method(self):
        self.config = BflVideoConfig()
        self.logging_obj = Mock()

    def test_map_seconds_resolution_aspect_ratio(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={"seconds": 8, "resolution": "FHD", "aspect_ratio": "16:9"},
            model=MODEL,
            drop_params=False,
        )
        assert mapped["duration"] == 8
        assert mapped["resolution"] == "fhd"
        assert mapped["aspect_ratio"] == "16:9"

    def test_map_size_falls_back_to_aspect_ratio(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={"size": "1920x1080"},
            model=MODEL,
            drop_params=False,
        )
        assert mapped["aspect_ratio"] == "16:9"

    def test_map_duration_auto_passthrough(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={"seconds": "auto"},
            model=MODEL,
            drop_params=False,
        )
        assert mapped["duration"] == "auto"

    def test_map_platform_duration_seconds_alias_becomes_duration(self):
        # Regression (NOL-431): the platform sends duration_seconds; BFL's t2v schema forbids it
        # (extra_forbidden 422) and wants duration, so the alias must map to duration, never leak.
        mapped = self.config.map_openai_params(
            video_create_optional_params={"duration_seconds": 5},
            model=MODEL,
            drop_params=False,
        )
        assert mapped["duration"] == 5
        assert "duration_seconds" not in mapped

    def test_map_seconds_wins_and_duration_seconds_never_leaks(self):
        # Regression (NOL-431): nolgia-api mirrors the clip length into BOTH seconds and
        # duration_seconds; BFL must receive a single duration field and no duration_seconds.
        mapped = self.config.map_openai_params(
            video_create_optional_params={"seconds": 8, "duration_seconds": 8},
            model=MODEL,
            drop_params=False,
        )
        assert mapped["duration"] == 8
        assert "duration_seconds" not in mapped

    def test_create_request_body_carries_duration_not_duration_seconds(self):
        # Regression (NOL-431): the exact payload that 422'd in prod must produce a BFL body
        # with duration and without the extra_forbidden duration_seconds field.
        mapped = self.config.map_openai_params(
            video_create_optional_params={"seconds": 5, "duration_seconds": 5, "generate_audio": True},
            model=MODEL,
            drop_params=False,
        )
        data, _, _ = self.config.transform_video_create_request(
            model=MODEL,
            prompt="a calico cat stretching",
            api_base=API_BASE,
            video_create_optional_request_params=mapped,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert data["duration"] == 5
        assert "duration_seconds" not in data

    @pytest.mark.parametrize("value", [True, False])
    def test_generate_audio_top_level_is_forwarded(self, value):
        # Regression: generate_audio arriving as a top-level param must reach the BFL body.
        mapped = self.config.map_openai_params(
            video_create_optional_params={"generate_audio": value},
            model=MODEL,
            drop_params=False,
        )
        assert mapped["generate_audio"] is value

    @pytest.mark.parametrize("value", [True, False])
    def test_generate_audio_via_extra_body_is_forwarded(self, value):
        mapped = self.config.map_openai_params(
            video_create_optional_params={"extra_body": {"generate_audio": value}},
            model=MODEL,
            drop_params=False,
        )
        assert mapped["generate_audio"] is value

    def test_generate_audio_string_coerced_to_bool(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={"generate_audio": "false"},
            model=MODEL,
            drop_params=False,
        )
        assert mapped["generate_audio"] is False

    def test_map_forwards_unknown_native_params(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={"seed": 7, "start_video": "https://v/in.mp4", "mode": "v2v"},
            model=MODEL,
            drop_params=False,
        )
        assert mapped["seed"] == 7
        assert mapped["start_video"] == "https://v/in.mp4"
        assert mapped["mode"] == "v2v"

    def test_map_image_url_sets_i2v_and_keyframes(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={"image_url": "https://img/x.png"},
            model=MODEL,
            drop_params=False,
        )
        assert mapped["mode"] == "i2v"
        assert mapped["keyframes"] == ("https://img/x.png",)
        assert "image_url" not in mapped

    def test_map_input_reference_sets_i2v_and_keyframes(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={"input_reference": "https://img/x.png"},
            model=MODEL,
            drop_params=False,
        )
        assert mapped["mode"] == "i2v"
        assert mapped["keyframes"] == ("https://img/x.png",)

    def test_map_filelike_image_base64_encoded_into_keyframes(self):
        raw = b"\x89PNG\r\n\x1a\nfake-start-frame"
        mapped = self.config.map_openai_params(
            video_create_optional_params={"image": io.BytesIO(raw)},
            model=MODEL,
            drop_params=False,
        )
        assert mapped["mode"] == "i2v"
        assert mapped["keyframes"] == (base64.b64encode(raw).decode("utf-8"),)

    def test_create_request_carries_all_fields_and_defaults_t2v(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "seconds": 10,
                "resolution": "hd",
                "aspect_ratio": "9:16",
                "generate_audio": True,
            },
            model=MODEL,
            drop_params=False,
        )
        data, files, url = self.config.transform_video_create_request(
            model=MODEL,
            prompt="a neon city at night",
            api_base=API_BASE,
            video_create_optional_request_params=mapped,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert url == f"{API_BASE}/v1/flux-3-video"
        assert files == ()
        assert data["prompt"] == "a neon city at night"
        assert data["mode"] == "t2v"
        assert data["duration"] == 10
        assert data["resolution"] == "hd"
        assert data["aspect_ratio"] == "9:16"
        assert data["generate_audio"] is True

    @pytest.mark.parametrize("value", [True, False])
    def test_create_request_pins_generate_audio_end_to_end(self, value):
        mapped = self.config.map_openai_params(
            video_create_optional_params={"generate_audio": value},
            model=MODEL,
            drop_params=False,
        )
        data, _, _ = self.config.transform_video_create_request(
            model=MODEL,
            prompt="x",
            api_base=API_BASE,
            video_create_optional_request_params=mapped,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert data["generate_audio"] is value

    def test_create_request_i2v_mode_is_not_overwritten(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={"image_url": "https://img/x.png"},
            model=MODEL,
            drop_params=False,
        )
        data, _, _ = self.config.transform_video_create_request(
            model=MODEL,
            prompt="animate this",
            api_base=API_BASE,
            video_create_optional_request_params=mapped,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert data["mode"] == "i2v"
        assert data["keyframes"] == ("https://img/x.png",)

    def test_create_request_drops_none_values(self):
        data, _, _ = self.config.transform_video_create_request(
            model=MODEL,
            prompt="x",
            api_base=API_BASE,
            video_create_optional_request_params={"duration": None, "generate_audio": True},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert "duration" not in data
        assert data["generate_audio"] is True

    def test_map_end_frame_pins_start_and_end_keyframes(self):
        # NOL-442: start + end frame pinning maps to BFL's two-element keyframes array (i2v).
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "image_url": "https://img/start.png",
                "end_image_url": "https://img/end.png",
            },
            model=MODEL,
            drop_params=False,
        )
        assert mapped["mode"] == "i2v"
        assert mapped["keyframes"] == ("https://img/start.png", "https://img/end.png")

    def test_map_element_refs_become_timestamped_keyframes(self):
        # NOL-442: three or more storyboard frames pin to evenly spaced timestamps across the duration.
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "image_urls": ["https://img/a.png", "https://img/b.png", "https://img/c.png"],
                "seconds": 6,
            },
            model=MODEL,
            drop_params=False,
        )
        assert mapped["mode"] == "i2v"
        assert mapped["keyframes"] == (
            (0.0, "https://img/a.png"),
            (3.0, "https://img/b.png"),
            (6.0, "https://img/c.png"),
        )

    def test_map_start_plus_elements_plus_end_ordered_and_timestamped(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "image_url": "https://img/start.png",
                "image_urls": ["https://img/mid.png"],
                "end_image_url": "https://img/end.png",
                "duration_seconds": 10,
            },
            model=MODEL,
            drop_params=False,
        )
        assert mapped["mode"] == "i2v"
        assert mapped["keyframes"] == (
            (0.0, "https://img/start.png"),
            (5.0, "https://img/mid.png"),
            (10.0, "https://img/end.png"),
        )

    def test_map_video_refs_become_v2v_start_video(self):
        # NOL-442: a reference video maps to start_video and forces v2v continuation, never keyframes.
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "video_urls": ["https://v/clip.mp4"],
                "image_url": "https://img/ignored.png",
            },
            model=MODEL,
            drop_params=False,
        )
        assert mapped["mode"] == "v2v"
        assert mapped["start_video"] == "https://v/clip.mp4"
        assert "keyframes" not in mapped

    def test_nolgia_reference_aliases_never_leak_into_strict_body(self):
        # NOL-442 regression: BFL's flux-3-video body is strict (422 extra_forbidden), so the
        # platform's end_image_url / image_urls / video_urls aliases must be consumed, never forwarded.
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "image_url": "https://img/start.png",
                "end_image_url": "https://img/end.png",
                "image_urls": ["https://img/mid.png"],
                "seconds": 8,
            },
            model=MODEL,
            drop_params=False,
        )
        data, _, _ = self.config.transform_video_create_request(
            model=MODEL,
            prompt="a storyboard",
            api_base=API_BASE,
            video_create_optional_request_params=mapped,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert "end_image_url" not in data
        assert "image_urls" not in data
        assert "video_urls" not in data
        assert "image_url" not in data
        assert data["mode"] == "i2v"

    def test_map_explicit_keyframes_take_precedence(self):
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "keyframes": [[0, "https://img/a.png"], [2.5, "https://img/b.png"]],
                "image_url": "https://img/ignored.png",
            },
            model=MODEL,
            drop_params=False,
        )
        assert mapped["mode"] == "i2v"
        assert mapped["keyframes"] == ([0, "https://img/a.png"], [2.5, "https://img/b.png"])


class TestBflVideoEnvironmentAndUrls:
    def setup_method(self):
        self.config = BflVideoConfig()

    def test_validate_environment_sets_x_key(self):
        headers = self.config.validate_environment(headers={}, model=MODEL, api_key="secret-key")
        assert headers["x-key"] == "secret-key"
        assert headers["Content-Type"] == "application/json"

    def test_validate_environment_prefers_litellm_params_key(self):
        headers = self.config.validate_environment(
            headers={},
            model=MODEL,
            api_key=None,
            litellm_params=GenericLiteLLMParams(api_key="from-params"),
        )
        assert headers["x-key"] == "from-params"

    def test_validate_environment_raises_when_missing(self, monkeypatch):
        monkeypatch.delenv("BFL_API_KEY", raising=False)
        monkeypatch.delenv("BLACK_FOREST_LABS_API_KEY", raising=False)
        with pytest.raises(BlackForestLabsError, match="BFL_API_KEY is not set"):
            self.config.validate_environment(headers={}, model=MODEL, api_key=None)

    def test_get_complete_url_default(self, monkeypatch):
        monkeypatch.delenv("BFL_API_BASE", raising=False)
        url = self.config.get_complete_url(model=MODEL, api_base=None, litellm_params={})
        assert url == API_BASE

    def test_get_complete_url_strips_trailing_slash(self):
        url = self.config.get_complete_url(model=MODEL, api_base="https://custom.bfl.ai/", litellm_params={})
        assert url == "https://custom.bfl.ai"


class TestBflVideoCreateResponse:
    def setup_method(self):
        self.config = BflVideoConfig()
        self.logging_obj = Mock()

    def test_create_response_encodes_polling_url_and_queues(self):
        response = Mock(spec=httpx.Response)
        response.json.return_value = {"id": "req-123", "polling_url": POLLING_URL}
        video_obj = self.config.transform_video_create_response(
            model=MODEL,
            raw_response=response,
            logging_obj=self.logging_obj,
            custom_llm_provider=PROVIDER,
            request_data={"duration": 8, "aspect_ratio": "16:9"},
        )
        assert isinstance(video_obj, VideoObject)
        assert video_obj.status == "queued"
        assert video_obj.id.startswith("video_")
        decoded = decode_video_id_with_provider(video_obj.id)
        assert decoded["video_id"] == POLLING_URL
        assert decoded["custom_llm_provider"] == PROVIDER
        assert video_obj.seconds == "8"
        assert video_obj.size == "16x9"

    def test_create_response_raises_when_polling_url_missing(self):
        response = Mock(spec=httpx.Response)
        response.json.return_value = {"id": "req-123"}
        with pytest.raises(ValueError, match="missing id/polling_url"):
            self.config.transform_video_create_response(
                model=MODEL,
                raw_response=response,
                logging_obj=self.logging_obj,
                custom_llm_provider=PROVIDER,
                request_data={},
            )

    def test_create_response_rejects_non_bfl_polling_url(self):
        response = Mock(spec=httpx.Response)
        response.json.return_value = {"id": "req-123", "polling_url": "https://evil.example.com/get_result?id=x"}
        with pytest.raises(BlackForestLabsError, match="not within the bfl.ai domain"):
            self.config.transform_video_create_response(
                model=MODEL,
                raw_response=response,
                logging_obj=self.logging_obj,
                custom_llm_provider=PROVIDER,
                request_data={},
            )


class TestBflVideoStatusAndContentRequests:
    def setup_method(self):
        self.config = BflVideoConfig()

    def test_status_and_content_requests_decode_polling_url(self):
        encoded = encode_video_id_with_provider(POLLING_URL, PROVIDER, "flux-3-video")
        status_url, status_params = self.config.transform_video_status_retrieve_request(
            video_id=encoded,
            api_base=API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        content_url, content_params = self.config.transform_video_content_request(
            video_id=encoded,
            api_base=API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert status_url == POLLING_URL
        assert content_url == POLLING_URL
        assert status_params == {}
        assert content_params == {}

    def test_status_request_rejects_forged_non_bfl_polling_url(self):
        forged = encode_video_id_with_provider("https://evil.example.com/get_result?id=x", PROVIDER, "flux-3-video")
        with pytest.raises(BlackForestLabsError, match="not within the bfl.ai domain"):
            self.config.transform_video_status_retrieve_request(
                video_id=forged,
                api_base=API_BASE,
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

    def test_content_request_rejects_http_scheme(self):
        forged = encode_video_id_with_provider("http://gateway.bfl.ai/get_result?id=x", PROVIDER, "flux-3-video")
        with pytest.raises(BlackForestLabsError, match="scheme must be https"):
            self.config.transform_video_content_request(
                video_id=forged,
                api_base=API_BASE,
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )


class TestBflVideoStatusResponse:
    def setup_method(self):
        self.config = BflVideoConfig()
        self.logging_obj = Mock()

    @pytest.mark.parametrize(
        "bfl_status,expected",
        [("Ready", "completed"), ("Pending", "in_progress"), ("Task Queued", "queued")],
    )
    def test_status_maps_bfl_status(self, bfl_status, expected):
        response = _result_response({"id": "req-123", "status": bfl_status})
        status_obj = self.config.transform_video_status_retrieve_response(
            raw_response=response,
            logging_obj=self.logging_obj,
            custom_llm_provider=PROVIDER,
        )
        assert status_obj.status == expected

    def test_status_reencodes_polling_url_from_request(self):
        response = _result_response({"id": "req-123", "status": "Pending"})
        status_obj = self.config.transform_video_status_retrieve_response(
            raw_response=response,
            logging_obj=self.logging_obj,
            custom_llm_provider=PROVIDER,
        )
        decoded = decode_video_id_with_provider(status_obj.id)
        assert decoded["video_id"] == POLLING_URL

    def test_status_error_sets_error_payload(self):
        response = _result_response({"id": "req-123", "status": "Content Moderated"})
        status_obj = self.config.transform_video_status_retrieve_response(
            raw_response=response,
            logging_obj=self.logging_obj,
            custom_llm_provider=PROVIDER,
        )
        assert status_obj.status == "failed"
        assert status_obj.error is not None
        assert status_obj.error["message"] == "Content Moderated"

    def test_status_tolerates_non_json(self):
        response = Mock(spec=httpx.Response)
        response.is_success = True
        response.json.side_effect = ValueError("no json")
        status_obj = self.config.transform_video_status_retrieve_response(
            raw_response=response,
            logging_obj=self.logging_obj,
            custom_llm_provider=PROVIDER,
        )
        assert status_obj.status == "in_progress"


class TestBflVideoContentResponse:
    def setup_method(self):
        self.logging_obj = Mock()

    def test_extract_video_url_reads_result_sample(self):
        url = BflVideoConfig._extract_video_url({"status": "Ready", "result": {"sample": "https://cdn/v.mp4"}})
        assert url == "https://cdn/v.mp4"

    def test_extract_video_url_raises_on_failed_status(self):
        with pytest.raises(ValueError, match="flux-3-video generation failed"):
            BflVideoConfig._extract_video_url({"status": "Error", "result": {}})

    def test_extract_video_url_raises_when_missing(self):
        with pytest.raises(ValueError, match="Video URL not found"):
            BflVideoConfig._extract_video_url({"status": "Pending", "result": {}})

    def test_content_response_downloads_native_audio_bytes(self):
        clip_bytes = b"\x00\x00\x00\x18ftypmp42AUDIO-AAC-TRACK-\xde\xad\xbe\xef"
        sample_url = "https://delivery.bfl.ai/video/abc.mp4"
        download_client = _RecordingClient(
            httpx.Response(200, content=clip_bytes, request=httpx.Request("GET", sample_url))
        )
        config = BflVideoConfig(sync_client=download_client)

        out = config.transform_video_content_response(
            raw_response=_result_response({"status": "Ready", "result": {"sample": sample_url}}),
            logging_obj=self.logging_obj,
        )
        assert out == clip_bytes
        assert download_client.calls == [(sample_url, None)]

    async def test_async_content_response_downloads_native_audio_bytes(self):
        clip_bytes = b"\x00\x00\x00\x18ftypmp42AUDIO-AAC-TRACK-\xde\xad\xbe\xef"
        sample_url = "https://delivery.bfl.ai/video/abc.mp4"
        download_client = _RecordingAsyncClient(
            httpx.Response(200, content=clip_bytes, request=httpx.Request("GET", sample_url))
        )
        config = BflVideoConfig(async_client=download_client)

        out = await config.async_transform_video_content_response(
            raw_response=_result_response({"status": "Ready", "result": {"sample": sample_url}}),
            logging_obj=self.logging_obj,
        )
        assert out == clip_bytes
        assert download_client.calls == [(sample_url, None)]


def test_provider_config_manager_returns_bfl_video_config():
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_video_config(
        model="flux-3-video", provider=LlmProviders.BLACK_FOREST_LABS
    )
    assert isinstance(config, BflVideoConfig)


def test_get_llm_provider_routes_black_forest_labs():
    from litellm import get_llm_provider

    model, provider, _, _ = get_llm_provider("black_forest_labs/flux-3-video")
    assert model == "flux-3-video"
    assert provider == "black_forest_labs"
