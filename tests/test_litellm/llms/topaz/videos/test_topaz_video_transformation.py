import json
from unittest.mock import Mock

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.url_utils import SSRFError
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.topaz.common_utils import TopazModelInfo
from litellm.llms.topaz.videos import transformation as topaz_videos
from litellm.llms.topaz.videos.transformation import TopazVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.utils import encode_video_id_with_provider

MODEL = "topaz/prob-4"
UPLOAD_URL = "https://videocloud.s3.amazonaws.com/abc/source.mp4?X-Amz-Signature=deadbeef"
REQUEST_ID = "019fd8c7-8da6-7168-abfc-8684fe340cb9"


def _response(payload: object, status_code: int = 200, url: str = "https://api.topazlabs.com/x") -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        content=json.dumps(payload).encode(),
        request=httpx.Request("GET", url),
    )


@pytest.fixture(autouse=True)
def _trust_test_source_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    # The source relay fetches caller URLs through safe_get; the SSRF guard resolves DNS,
    # which no unit test should depend on. The guard itself is asserted explicitly below.
    monkeypatch.setattr(litellm, "user_url_validation", False, raising=False)


class _FakeSyncClient:
    def __init__(self, get_bytes: bytes = b"source-bytes") -> None:
        self._get_bytes = get_bytes
        self.puts: list[tuple[str, bytes, dict]] = []  # mutable-ok: test spy needs an append log
        self.gets: list[str] = []  # mutable-ok: test spy needs an append log

    def get(self, url: str, **kwargs: object) -> httpx.Response:
        self.gets.append(url)
        return httpx.Response(status_code=200, content=self._get_bytes, request=httpx.Request("GET", url))

    def put(self, url: str, content: bytes | None = None, headers: dict | None = None, **kwargs: object):
        self.puts.append((url, content or b"", headers or {}))
        return httpx.Response(status_code=200, request=httpx.Request("PUT", url))


def _mapped(**overrides: object) -> dict:
    config = TopazVideoConfig()
    params = {"input_reference": "https://cdn.example/clip.mp4", "resolution": "1080p", "seconds": 2, **overrides}
    return config.map_openai_params(video_create_optional_params=params, model=MODEL, drop_params=False)


def test_map_openai_params_resolves_resolution_alias_and_carries_source():
    mapped = _mapped()
    assert mapped["resolution"] == "1920x1080"
    assert mapped["input_reference"] == "https://cdn.example/clip.mp4"
    assert mapped["seconds"] == 2
    assert mapped["container"] == "mp4"


@pytest.mark.parametrize(
    ("alias", "expected"),
    [("720p", "1280x720"), ("1440p", "2560x1440"), ("2160p", "3840x2160"), ("4320p", "7680x4320")],
)
def test_map_openai_params_supports_every_published_resolution_tier(alias: str, expected: str):
    assert _mapped(resolution=alias)["resolution"] == expected


def test_map_openai_params_accepts_explicit_dimensions():
    assert _mapped(resolution="1920x816")["resolution"] == "1920x816"


@pytest.mark.parametrize("param", ["seed", "aspect_ratio", "generate_audio", "negative_prompt", "nonsense_knob"])
def test_map_openai_params_raises_on_unsupported_params_instead_of_dropping(param: str):
    with pytest.raises(litellm.BadRequestError) as excinfo:
        _mapped(**{param: "x"})
    assert param in str(excinfo.value)


def test_map_openai_params_still_raises_when_drop_params_is_enabled():
    config = TopazVideoConfig()
    with pytest.raises(litellm.BadRequestError):
        config.map_openai_params(
            video_create_optional_params={"input_reference": "u", "resolution": "1080p", "seed": 1},
            model=MODEL,
            drop_params=True,
        )


def test_map_openai_params_forwards_topaz_filter_and_output_knobs():
    mapped = _mapped(details=0.2, videoEncoder="H265")
    assert mapped["details"] == 0.2
    assert mapped["videoEncoder"] == "H265"


def test_map_openai_params_requires_a_resolution():
    config = TopazVideoConfig()
    with pytest.raises(litellm.BadRequestError, match="requires a target output resolution"):
        config.map_openai_params(video_create_optional_params={"input_reference": "u"}, model=MODEL, drop_params=False)


def test_map_openai_params_rejects_an_unusable_resolution():
    with pytest.raises(litellm.BadRequestError, match="unusable resolution"):
        _mapped(resolution="enormous")


def test_create_request_builds_the_topaz_express_body():
    config = TopazVideoConfig()
    body, files, url = config.transform_video_create_request(
        model=MODEL,
        prompt="",
        api_base=None,
        video_create_optional_request_params=_mapped(details=0.2, videoEncoder="H265"),
        litellm_params=None,
        headers={},
    )
    assert url == "https://api.topazlabs.com/video/express"
    assert files == ()
    assert body["source"] == {"container": "mp4"}
    assert body["filters"] == [{"model": "prob-4", "details": 0.2}]
    assert body["output"]["resolution"] == {"width": 1920, "height": 1080}
    assert body["output"]["videoEncoder"] == "H265"


def test_create_request_rejects_an_unknown_topaz_model_code():
    config = TopazVideoConfig()
    with pytest.raises(litellm.BadRequestError, match="Unknown Topaz enhancement model"):
        config.transform_video_create_request(
            model="topaz/not-a-real-model",
            prompt="",
            api_base=None,
            video_create_optional_request_params=_mapped(),
            litellm_params=None,
            headers={},
        )


def test_create_request_requires_source_footage():
    config = TopazVideoConfig()
    with pytest.raises(litellm.BadRequestError, match="requires source footage"):
        config.transform_video_create_request(
            model=MODEL,
            prompt="",
            api_base=None,
            video_create_optional_request_params={"resolution": "1920x1080"},
            litellm_params=None,
            headers={},
        )


def _created(config: TopazVideoConfig, payload: object) -> object:
    config.transform_video_create_request(
        model=MODEL,
        prompt="",
        api_base=None,
        video_create_optional_request_params=_mapped(),
        litellm_params=None,
        headers={},
    )
    return config.transform_video_create_response(
        model=MODEL,
        raw_response=_response(payload),
        logging_obj=None,
        custom_llm_provider="topaz",
        request_data=None,
    )


def test_create_response_relays_the_source_bytes_to_the_presigned_upload_url():
    client = _FakeSyncClient(get_bytes=b"the-real-clip")
    config = TopazVideoConfig(sync_client=client)
    video = _created(config, {"requestId": REQUEST_ID, "uploadId": "u", "uploadUrls": [UPLOAD_URL]})

    assert client.gets == ["https://cdn.example/clip.mp4"]
    assert len(client.puts) == 1
    url, content, headers = client.puts[0]
    assert url == UPLOAD_URL
    assert content == b"the-real-clip"
    assert headers["Content-Type"] == "video/mp4"
    assert video.status == "queued"
    assert video.usage == {"duration_seconds": 2.0}
    assert REQUEST_ID not in video.id


def test_create_response_refuses_a_multipart_upload_rather_than_truncating():
    client = _FakeSyncClient()
    config = TopazVideoConfig(sync_client=client)
    with pytest.raises(Exception, match="exactly one upload URL"):
        _created(config, {"requestId": REQUEST_ID, "uploadUrls": [UPLOAD_URL, UPLOAD_URL]})
    assert client.puts == []


def test_create_response_rejects_a_response_without_a_request_id():
    config = TopazVideoConfig(sync_client=_FakeSyncClient())
    with pytest.raises(Exception, match="no requestId"):
        _created(config, {"uploadUrls": [UPLOAD_URL]})


@pytest.mark.parametrize(
    ("topaz_status", "expected"),
    [
        ("requested", "queued"),
        ("accepted", "queued"),
        ("initializing", "in_progress"),
        ("preprocessing", "in_progress"),
        ("processing", "in_progress"),
        ("postprocessing", "in_progress"),
        ("complete", "completed"),
        ("canceled", "failed"),
        ("failed", "failed"),
    ],
)
def test_status_response_maps_every_topaz_state(topaz_status: str, expected: str):
    config = TopazVideoConfig()
    video = config.transform_video_status_retrieve_response(
        raw_response=_response({"status": topaz_status, "progress": 42}),
        logging_obj=None,
        custom_llm_provider="topaz",
    )
    assert video.status == expected


def test_status_response_surfaces_the_billed_lower_bound_credit_estimate():
    config = TopazVideoConfig()
    video = config.transform_video_status_retrieve_response(
        raw_response=_response({"status": "processing", "estimates": {"cost": [7, 9]}}),
        logging_obj=None,
        custom_llm_provider="topaz",
    )
    assert video.usage == {"topaz_credits": 7}


def test_status_response_carries_the_failure_message():
    config = TopazVideoConfig()
    video = config.transform_video_status_retrieve_response(
        raw_response=_response({"status": "failed", "error": {"message": "source unreadable"}}),
        logging_obj=None,
        custom_llm_provider="topaz",
    )
    assert video.status == "failed"
    assert video.error["message"] == "source unreadable"


def test_status_request_targets_the_topaz_status_path():
    config = TopazVideoConfig()
    url, params = config.transform_video_status_retrieve_request(
        video_id=REQUEST_ID, api_base=None, litellm_params=None, headers={}
    )
    assert url == f"https://api.topazlabs.com/video/{REQUEST_ID}/status"
    assert params == {}


def test_content_response_downloads_the_signed_url_from_the_status_payload():
    client = _FakeSyncClient(get_bytes=b"enhanced-mp4")
    config = TopazVideoConfig(sync_client=client)
    content = config.transform_video_content_response(
        raw_response=_response({"status": "complete", "download": {"url": "https://dl.example/out.mp4"}}),
        logging_obj=None,
    )
    assert content == b"enhanced-mp4"
    assert client.gets == ["https://dl.example/out.mp4"]


def test_content_response_refuses_while_the_enhancement_is_still_running():
    config = TopazVideoConfig(sync_client=_FakeSyncClient())
    with pytest.raises(Exception, match="not downloadable yet"):
        config.transform_video_content_response(raw_response=_response({"status": "processing"}), logging_obj=None)


def test_content_response_surfaces_a_failed_enhancement():
    config = TopazVideoConfig(sync_client=_FakeSyncClient())
    with pytest.raises(Exception, match="enhancement failed"):
        config.transform_video_content_response(raw_response=_response({"status": "failed"}), logging_obj=None)


def test_topaz_accepts_promptless_creates():
    assert TopazVideoConfig().supports_promptless_video_create(MODEL) is True


def test_supported_params_do_not_advertise_generation_only_controls():
    supported = TopazVideoConfig().get_supported_openai_params(MODEL)
    assert "resolution" in supported
    assert "input_reference" in supported
    for generation_only in ("prompt", "seed", "aspect_ratio", "generate_audio", "negative_prompt"):
        assert generation_only not in supported


def test_create_request_rejects_a_nonempty_prompt_instead_of_ignoring_it():
    config = TopazVideoConfig()
    with pytest.raises(litellm.BadRequestError, match="does not support `prompt`"):
        config.transform_video_create_request(
            model=MODEL,
            prompt="restore the colors",
            api_base=None,
            video_create_optional_request_params=_mapped(),
            litellm_params=None,
            headers={},
        )


@pytest.mark.parametrize("prompt", ["", "   ", None])
def test_create_request_still_accepts_an_empty_prompt(prompt):
    config = TopazVideoConfig()
    body, _files, _url = config.transform_video_create_request(
        model=MODEL,
        prompt=prompt,
        api_base=None,
        video_create_optional_request_params=_mapped(),
        litellm_params=None,
        headers={},
    )
    assert body["source"] == {"container": "mp4"}


def test_map_openai_params_rejects_a_prompt_smuggled_through_extra_body():
    with pytest.raises(litellm.BadRequestError, match="prompt"):
        _mapped(extra_body={"prompt": "restore the colors"})


def test_create_request_revalidates_a_container_supplied_through_extra_body():
    # VideoGenerationRequestUtils overlays raw extra_body onto the mapped params, so an
    # uppercased container reaches the create transform without having been normalized.
    config = TopazVideoConfig()
    body, _files, _url = config.transform_video_create_request(
        model=MODEL,
        prompt="",
        api_base=None,
        video_create_optional_request_params={**_mapped(), "container": "MOV"},
        litellm_params=None,
        headers={},
    )
    assert body["source"] == {"container": "mov"}


def test_create_response_labels_the_upload_with_the_revalidated_container():
    client = _FakeSyncClient()
    config = TopazVideoConfig(sync_client=client)
    config.transform_video_create_request(
        model=MODEL,
        prompt="",
        api_base=None,
        video_create_optional_request_params={**_mapped(), "container": "MOV"},
        litellm_params=None,
        headers={},
    )
    config.transform_video_create_response(
        model=MODEL,
        raw_response=_response({"requestId": REQUEST_ID, "uploadUrls": [UPLOAD_URL]}),
        logging_obj=None,
        custom_llm_provider="topaz",
        request_data=None,
    )
    assert client.puts[0][2]["Content-Type"] == "video/quicktime"


def test_create_request_rejects_an_unsupported_container_from_extra_body():
    config = TopazVideoConfig()
    with pytest.raises(litellm.BadRequestError, match="unsupported source container"):
        config.transform_video_create_request(
            model=MODEL,
            prompt="",
            api_base=None,
            video_create_optional_request_params={**_mapped(), "container": "avi"},
            litellm_params=None,
            headers={},
        )


def test_create_response_refuses_a_source_clip_above_the_relay_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(topaz_videos, "MAX_VIDEO_URL_DOWNLOAD_SIZE_MB", 1 / (1024 * 1024))
    client = _FakeSyncClient(get_bytes=b"more-than-one-byte")
    config = TopazVideoConfig(sync_client=client)
    with pytest.raises(litellm.BadRequestError, match="per-request limit"):
        _created(config, {"requestId": REQUEST_ID, "uploadUrls": [UPLOAD_URL]})
    assert client.puts == []


def test_create_response_refuses_a_source_url_that_targets_a_private_network(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "user_url_validation", True, raising=False)
    config = TopazVideoConfig(sync_client=_FakeSyncClient())
    config.transform_video_create_request(
        model=MODEL,
        prompt="",
        api_base=None,
        video_create_optional_request_params={
            **_mapped(),
            "input_reference": "http://169.254.169.254/latest/meta-data",
        },
        litellm_params=None,
        headers={},
    )
    with pytest.raises(SSRFError):
        config.transform_video_create_response(
            model=MODEL,
            raw_response=_response({"requestId": REQUEST_ID, "uploadUrls": [UPLOAD_URL]}),
            logging_obj=None,
            custom_llm_provider="topaz",
            request_data=None,
        )


class _RecordingHTTPHandler(HTTPHandler):
    """A real HTTPHandler so the shared video handler accepts it as the caller's client."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, str]] = []  # mutable-ok: test spy needs an append log

    def post(self, url: str, **kwargs: object) -> httpx.Response:
        self.calls.append(("POST", url))
        return _response({"requestId": REQUEST_ID, "uploadUrls": [UPLOAD_URL]}, url=url)

    def get(self, url: str, params: dict | None = None, headers: dict | None = None, **kwargs: object):
        self.calls.append(("GET", url))
        return httpx.Response(status_code=200, content=b"the-real-clip", request=httpx.Request("GET", url))

    def put(self, url: str, content: bytes | None = None, headers: dict | None = None, **kwargs: object):
        self.calls.append(("PUT", url))
        return httpx.Response(status_code=200, request=httpx.Request("PUT", url))


class _RecordingAsyncHTTPHandler(AsyncHTTPHandler):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, str]] = []  # mutable-ok: test spy needs an append log

    async def post(self, url: str, **kwargs: object) -> httpx.Response:
        self.calls.append(("POST", url))
        return _response({"requestId": REQUEST_ID, "uploadUrls": [UPLOAD_URL]}, url=url)

    async def get(self, url: str, params: dict | None = None, headers: dict | None = None, **kwargs: object):
        self.calls.append(("GET", url))
        return httpx.Response(status_code=200, content=b"the-real-clip", request=httpx.Request("GET", url))

    async def put(self, url: str, content: bytes | None = None, headers: dict | None = None, **kwargs: object):
        self.calls.append(("PUT", url))
        return httpx.Response(status_code=200, request=httpx.Request("PUT", url))


def test_video_generation_handler_relays_the_source_through_the_callers_client():
    # The source GET and presigned PUT must ride the caller's client, or a mock transport,
    # proxy or private CA applies to the create POST only.
    caller_client = _RecordingHTTPHandler()

    video = BaseLLMHTTPHandler().video_generation_handler(
        model=MODEL,
        prompt="",
        video_generation_provider_config=TopazVideoConfig(),
        video_generation_optional_request_params=_mapped(),
        custom_llm_provider="topaz",
        litellm_params=GenericLiteLLMParams(),
        logging_obj=Mock(),
        timeout=60.0,
        client=caller_client,
        api_key="test-key",
    )

    assert caller_client.calls == [
        ("POST", "https://api.topazlabs.com/video/express"),
        ("GET", "https://cdn.example/clip.mp4"),
        ("PUT", UPLOAD_URL),
    ]
    assert video.status == "queued"


@pytest.mark.asyncio
async def test_async_video_generation_handler_relays_the_source_through_the_callers_client():
    caller_client = _RecordingAsyncHTTPHandler()

    video = await BaseLLMHTTPHandler().async_video_generation_handler(
        model=MODEL,
        prompt="",
        video_generation_provider_config=TopazVideoConfig(),
        video_generation_optional_request_params=_mapped(),
        custom_llm_provider="topaz",
        litellm_params=GenericLiteLLMParams(),
        logging_obj=Mock(),
        timeout=60.0,
        client=caller_client,
        api_key="test-key",
    )

    assert caller_client.calls == [
        ("POST", "https://api.topazlabs.com/video/express"),
        ("GET", "https://cdn.example/clip.mp4"),
        ("PUT", UPLOAD_URL),
    ]
    assert video.status == "queued"


def test_status_request_percent_encodes_a_crafted_video_id():
    config = TopazVideoConfig()
    url, _params = config.transform_video_status_retrieve_request(
        video_id="../../account/v1/credits/balance?x=", api_base=None, litellm_params=None, headers={}
    )
    assert url == ("https://api.topazlabs.com/video/..%2F..%2Faccount%2Fv1%2Fcredits%2Fbalance%3Fx%3D/status")


def test_status_response_preserves_the_requested_video_id():
    config = TopazVideoConfig()
    encoded_id = encode_video_id_with_provider(REQUEST_ID, "topaz", MODEL)
    config.transform_video_status_retrieve_request(video_id=encoded_id, api_base=None, litellm_params=None, headers={})
    video = config.transform_video_status_retrieve_response(
        raw_response=_response({"status": "complete"}),
        logging_obj=None,
        custom_llm_provider="topaz",
    )
    assert video.id == encoded_id


def test_status_response_falls_back_to_the_id_in_the_status_url():
    config = TopazVideoConfig()
    video = config.transform_video_status_retrieve_response(
        raw_response=_response({"status": "processing"}, url=f"https://api.topazlabs.com/video/{REQUEST_ID}/status"),
        logging_obj=None,
        custom_llm_provider="topaz",
    )
    assert video.id == REQUEST_ID


def test_model_discovery_lists_the_video_enhancement_models():
    models = TopazModelInfo().get_models()
    assert "topaz/prob-4" in models
    assert "topaz/rhea-1" in models
    assert "topaz/Standard V2" in models
