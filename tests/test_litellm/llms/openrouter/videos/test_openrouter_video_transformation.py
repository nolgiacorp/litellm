import httpx
import pytest

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openrouter.videos.transformation import OpenRouterVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
)
from litellm.utils import ProviderConfigManager

_API_BASE = "https://openrouter.ai/api/v1"


def _config() -> OpenRouterVideoConfig:
    return OpenRouterVideoConfig()


def _response(
    status_code: int, *, json_body=None, text: str = "", method: str = "GET", url: str = _API_BASE
) -> httpx.Response:
    request = httpx.Request(method, url)
    if json_body is not None:
        return httpx.Response(status_code, json=json_body, request=request)
    return httpx.Response(status_code, text=text, request=request)


def test_provider_dispatch_returns_openrouter_video_config():
    cfg = ProviderConfigManager.get_provider_video_config("bytedance/seedance-2.0", litellm.LlmProviders.OPENROUTER)
    assert isinstance(cfg, OpenRouterVideoConfig)


def test_map_params_text_to_video_flattens_extra_body():
    mapped = _config().map_openai_params(
        {"seconds": "5", "size": "1280x720", "extra_body": {"resolution": "720p", "seed": 42, "generate_audio": False}},
        "bytedance/seedance-2.0",
        True,
    )
    assert mapped["duration"] == 5
    assert isinstance(mapped["duration"], int)
    assert mapped["size"] == "1280x720"
    assert mapped["resolution"] == "720p"
    assert mapped["seed"] == 42
    assert mapped["generate_audio"] is False
    # extra_body itself must not survive into the request body
    assert "extra_body" not in mapped


@pytest.mark.parametrize(
    "seconds,expected",
    [("5", 5), (5, 5), (5.0, 5), (6.9, 6), (None, None), ("abc", None)],
)
def test_duration_is_coerced_to_int_or_dropped(seconds, expected):
    mapped = _config().map_openai_params({"seconds": seconds}, "bytedance/seedance-2.0", True)
    assert mapped.get("duration") == expected


def test_input_reference_becomes_first_frame():
    mapped = _config().map_openai_params({"input_reference": "https://cdn/start.png"}, "bytedance/seedance-2.0", True)
    assert mapped["frame_images"] == [
        {"type": "image_url", "image_url": {"url": "https://cdn/start.png"}, "frame_type": "first_frame"}
    ]


def test_input_references_drive_character_consistency():
    # The core reason for this provider: reference-to-video for recurring characters.
    # Accept both a raw URL string and a {"url": ...} dict, normalize to typed objects.
    mapped = _config().map_openai_params(
        {"extra_body": {"input_references": ["https://cdn/lou.png", {"url": "https://cdn/remy.png"}]}},
        "bytedance/seedance-2.0",
        True,
    )
    assert mapped["input_references"] == [
        {"type": "image_url", "image_url": {"url": "https://cdn/lou.png"}},
        {"type": "image_url", "image_url": {"url": "https://cdn/remy.png"}},
    ]


def test_frame_images_honor_explicit_frame_type():
    mapped = _config().map_openai_params(
        {"extra_body": {"frame_images": [{"url": "https://cdn/end.png", "frame_type": "last_frame"}]}},
        "bytedance/seedance-2.0",
        True,
    )
    assert mapped["frame_images"] == [
        {"type": "image_url", "image_url": {"url": "https://cdn/end.png"}, "frame_type": "last_frame"}
    ]


def test_already_typed_reference_is_passed_through_unchanged():
    typed = {"type": "image_url", "image_url": {"url": "https://cdn/x.png"}}
    mapped = _config().map_openai_params({"extra_body": {"input_references": [typed]}}, "bytedance/seedance-2.0", True)
    assert mapped["input_references"] == [typed]


def test_empty_reference_lists_are_omitted():
    mapped = _config().map_openai_params({"seconds": 5}, "bytedance/seedance-2.0", True)
    assert "frame_images" not in mapped
    assert "input_references" not in mapped


def test_transform_create_request_strips_prefix_and_targets_videos():
    body, files, url = _config().transform_video_create_request(
        "openrouter/bytedance/seedance-2.0",
        "a cat",
        _API_BASE,
        {"duration": 5, "size": None},
        GenericLiteLLMParams(),
        {},
    )
    assert url == f"{_API_BASE}/videos"
    assert body["model"] == "bytedance/seedance-2.0"
    assert body["prompt"] == "a cat"
    assert body["duration"] == 5
    # None-valued params are dropped, not sent
    assert "size" not in body
    assert files == []


def test_validate_environment_sets_bearer_header():
    headers = _config().validate_environment({}, "bytedance/seedance-2.0", api_key="sk-or-abc")
    assert headers["Authorization"] == "Bearer sk-or-abc"
    assert headers["Content-Type"] == "application/json"


def test_validate_environment_without_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "api_key", None, raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        _config().validate_environment({}, "bytedance/seedance-2.0")


def test_create_response_encodes_id_and_records_usage():
    resp = _response(200, json_body={"id": "job_xyz", "status": "pending"}, method="POST", url=f"{_API_BASE}/videos")
    video = _config().transform_video_create_response(
        "openrouter/bytedance/seedance-2.0",
        resp,
        None,
        "openrouter",
        {"duration": 5, "size": "1280x720"},
    )
    assert video.status == "queued"
    assert video.usage == {"duration_seconds": 5.0}
    assert video.seconds == "5"

    decoded = decode_video_id_with_provider(video.id)
    assert decoded["custom_llm_provider"] == "openrouter"
    assert decoded["video_id"] == "job_xyz"


def test_create_response_missing_id_raises():
    resp = _response(200, json_body={"status": "pending"}, method="POST")
    with pytest.raises(ValueError, match="missing 'id'"):
        _config().transform_video_create_response("bytedance/seedance-2.0", resp, None, "openrouter", {})


def test_create_response_http_error_raises_llm_exception():
    resp = _response(401, text="unauthorized", method="POST")
    with pytest.raises(BaseLLMException):
        _config().transform_video_create_response("bytedance/seedance-2.0", resp, None, "openrouter", {})


def test_status_retrieve_request_rebuilds_url_from_encoded_id():
    # Security: the poll URL must be rebuilt from api_base + the decoded job id,
    # never from any URL an attacker could embed in a forged video_id.
    video_id = encode_video_id_with_provider("job_abc", "openrouter", "bytedance/seedance-2.0")
    url, params = _config().transform_video_status_retrieve_request(video_id, _API_BASE, GenericLiteLLMParams(), {})
    assert url == f"{_API_BASE}/videos/job_abc"
    assert params == {}


def test_status_retrieve_response_completed_surfaces_usage_and_reencodes_id():
    resp = _response(
        200,
        json_body={
            "id": "job_abc",
            "status": "completed",
            "usage": {"cost": 0.76},
            "unsigned_urls": ["https://cdn/out.mp4"],
        },
    )
    video = _config().transform_video_status_retrieve_response(resp, None, "openrouter")
    assert video.status == "completed"
    assert video.usage == {"cost": 0.76}
    assert video.error is None
    assert decode_video_id_with_provider(video.id)["video_id"] == "job_abc"


def test_status_retrieve_response_failed_sets_error():
    resp = _response(200, json_body={"id": "job_abc", "status": "failed", "error": "content policy"})
    video = _config().transform_video_status_retrieve_response(resp, None, "openrouter")
    assert video.status == "failed"
    assert video.error == {"code": "failed", "message": "content policy"}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("pending", "queued"),
        ("in_progress", "in_progress"),
        ("processing", "in_progress"),
        ("completed", "completed"),
        ("failed", "failed"),
        ("cancelled", "failed"),
        ("something-new", "queued"),
    ],
)
def test_status_map(raw, expected):
    resp = _response(200, json_body={"id": "job", "status": raw})
    assert _config().transform_video_status_retrieve_response(resp, None, "openrouter").status == expected


def test_content_request_targets_authenticated_content_endpoint():
    # Download must go through the bearer-authenticated /content route rebuilt from
    # the decoded job id, not the unsigned poll URLs (which 401 without a session).
    video_id = encode_video_id_with_provider("job_abc", "openrouter", "bytedance/seedance-2.0")
    url, params = _config().transform_video_content_request(video_id, _API_BASE, GenericLiteLLMParams(), {})
    assert url == f"{_API_BASE}/videos/job_abc/content?index=0"
    assert params == {}


def test_content_response_returns_raw_bytes():
    resp = httpx.Response(200, content=b"\x00\x01MP4BYTES", request=httpx.Request("GET", _API_BASE))
    assert _config().transform_video_content_response(resp, None) == b"\x00\x01MP4BYTES"


def test_content_response_http_error_raises_llm_exception():
    resp = _response(401, text="No cookie auth credentials found")
    with pytest.raises(BaseLLMException):
        _config().transform_video_content_response(resp, None)


def test_get_complete_url_defaults_to_openrouter(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_BASE", raising=False)
    assert _config().get_complete_url("bytedance/seedance-2.0", None, {}) == _API_BASE
    assert _config().get_complete_url("bytedance/seedance-2.0", f"{_API_BASE}/", {}) == _API_BASE


def test_unsupported_operations_raise_not_implemented():
    cfg = _config()
    with pytest.raises(NotImplementedError):
        cfg.transform_video_remix_request("v", "p", _API_BASE, GenericLiteLLMParams(), {})
    with pytest.raises(NotImplementedError):
        cfg.transform_video_list_request(_API_BASE, GenericLiteLLMParams(), {})
    with pytest.raises(NotImplementedError):
        cfg.transform_video_delete_request("v", _API_BASE, GenericLiteLLMParams(), {})
