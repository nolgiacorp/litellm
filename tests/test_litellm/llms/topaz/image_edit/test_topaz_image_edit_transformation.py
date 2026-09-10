"""
Topaz image enhancement, reachable through the proxy.

Topaz's image route already had a transformation in the fork, registered as an
image VARIATIONS config. Nothing could ever call it: `CallTypes` has no
`image_variation` member and the proxy mounts only `/images/generations` and
`/images/edits`, so the Topaz key we pay for sold video and nothing else.

`POST /image/v1/enhance` is multipart image-in / image-out with no prompt, which
is the image-EDIT shape, so this suite pins that surface: the engine codes the
live API actually accepts, the form field names it reads, and the flat credit it
bills. The engine set and the form contract below were verified against
api.topazlabs.com on 2026-09-10 - the tuple that shipped before this carried
"High Resolution V2", which the API answers with "Unknown model error".
"""

import base64
from unittest.mock import Mock

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.topaz.common_utils import TOPAZ_IMAGE_ENHANCE_MODELS, TopazException
from litellm.llms.topaz.image_edit.transformation import (
    TOPAZ_IMAGE_OUTPUT_FORMAT,
    TopazImageEditConfig,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

ENGINE = "Standard V2"
MODEL = f"topaz/{ENGINE}"
ENHANCE_URL = "https://api.topazlabs.com/image/v1/enhance"
SOURCE_PNG = b"\x89PNG\r\n\x1a\nsource-bytes"
ENHANCED_JPEG = b"\xff\xd8\xff\xe0enhanced-bytes"


def _enhanced_response(content: bytes = ENHANCED_JPEG, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        content=content,
        headers={"content-type": "image/jpeg"},
        request=httpx.Request("POST", ENHANCE_URL),
    )


class _RecordingHTTPHandler(HTTPHandler):
    """A real HTTPHandler so the shared image-edit handler accepts it as the caller's client."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []  # mutable-ok: test spy needs an append log

    def post(self, url: str, **kwargs: object) -> httpx.Response:
        self.calls.append({"url": url, **kwargs})
        return _enhanced_response()


class _RecordingAsyncHTTPHandler(AsyncHTTPHandler):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []  # mutable-ok: test spy needs an append log

    async def post(self, url: str, **kwargs: object) -> httpx.Response:
        self.calls.append({"url": url, **kwargs})
        return _enhanced_response()


def _transform(**overrides: object) -> tuple[dict, object]:
    request = {
        "model": ENGINE,
        "prompt": None,
        "image": SOURCE_PNG,
        "image_edit_optional_request_params": {"output_width": "4096", "output_height": "4096"},
        "litellm_params": GenericLiteLLMParams(),
        "headers": {},
        **overrides,
    }
    return TopazImageEditConfig().transform_image_edit_request(**request)


class TestReachability:
    def test_the_proxys_image_edit_route_resolves_a_topaz_config(self):
        """The bug this module exists to fix: /images/edits could not reach Topaz at all."""
        config = ProviderConfigManager.get_provider_image_edit_config(model=MODEL, provider=LlmProviders.TOPAZ)
        assert isinstance(config, TopazImageEditConfig)

    def test_enhance_is_the_route_topaz_serves_images_on(self):
        assert TopazImageEditConfig().get_complete_url(model=ENGINE, api_base=None, litellm_params={}) == ENHANCE_URL

    @pytest.mark.parametrize("source", ["api_base_argument", "litellm_params"])
    def test_a_configured_api_base_overrides_the_published_host(self, source: str):
        config = TopazImageEditConfig()
        url = config.get_complete_url(
            model=ENGINE,
            api_base="https://topaz.internal/" if source == "api_base_argument" else None,
            litellm_params={"api_base": "https://topaz.internal/"} if source == "litellm_params" else {},
        )
        assert url == "https://topaz.internal/image/v1/enhance"


class TestEngineCodes:
    @pytest.mark.parametrize("engine", TOPAZ_IMAGE_ENHANCE_MODELS)
    def test_every_advertised_engine_is_sent_verbatim(self, engine: str):
        """Topaz reads the engine off the `model` form field; a mangled code is an opaque 400 upstream."""
        data, _files = _transform(model=f"topaz/{engine}")
        assert data["model"] == engine

    def test_high_resolution_v2_is_refused_because_topaz_has_no_such_model(self):
        """The stale code this change removed. Left in, it would bill nothing and fail every call."""
        with pytest.raises(litellm.BadRequestError, match="not a Topaz image engine"):
            _transform(model="topaz/High Resolution V2")

    def test_an_unknown_engine_names_the_accepted_set_instead_of_deferring_to_topaz(self):
        with pytest.raises(litellm.BadRequestError, match="High Fidelity V2"):
            _transform(model="topaz/Wonder 3")


class TestRequestForm:
    def test_the_source_image_rides_the_image_part(self):
        _data, files = _transform()
        assert files == {"image": SOURCE_PNG}

    def test_a_list_of_one_image_is_unwrapped(self):
        """The proxy always hands `image` over as a list; Topaz enhances exactly one."""
        _data, files = _transform(image=[SOURCE_PNG])
        assert files == {"image": SOURCE_PNG}

    def test_the_requested_output_geometry_reaches_topaz(self):
        data, _files = _transform()
        assert data["output_width"] == "4096"
        assert data["output_height"] == "4096"

    def test_an_output_format_topaz_accepts_is_always_declared(self):
        data, _files = _transform()
        assert data["output_format"] == TOPAZ_IMAGE_OUTPUT_FORMAT
        assert TOPAZ_IMAGE_OUTPUT_FORMAT in ("jpeg", "jpg", "png", "tiff", "tif")

    def test_an_unsized_request_lets_topaz_pick_the_output(self):
        data, _files = _transform(image_edit_optional_request_params={})
        assert "output_width" not in data
        assert "output_height" not in data

    def test_a_missing_image_is_refused_rather_than_sent_as_a_bare_enhance(self):
        with pytest.raises(litellm.BadRequestError, match="requires one to be uploaded"):
            _transform(image=None)

    def test_a_prompt_is_refused_instead_of_billed_and_ignored(self):
        """Topaz has no prompt slot; enhancing while dropping the instruction bills a wrong render."""
        with pytest.raises(litellm.BadRequestError, match="does not support `prompt`"):
            _transform(prompt="make it look like a painting")

    @pytest.mark.parametrize("prompt", ["", "   ", None])
    def test_an_empty_prompt_still_enhances(self, prompt):
        data, _files = _transform(prompt=prompt)
        assert data["model"] == ENGINE


class TestOptionalParams:
    def test_size_becomes_the_output_geometry_topaz_bills(self):
        mapped = TopazImageEditConfig().map_openai_params(
            image_edit_optional_params={"size": "2048x1536"}, model=ENGINE, drop_params=False
        )
        assert mapped == {"output_width": "2048", "output_height": "1536"}

    def test_user_is_accepted_for_spend_grouping_but_never_sent_to_topaz(self):
        """The proxy always forwards `user`; rejecting it would 400 every catalog request."""
        assert "user" in TopazImageEditConfig().get_supported_openai_params(ENGINE)
        mapped = TopazImageEditConfig().map_openai_params(
            image_edit_optional_params={"user": "customer-1"}, model=ENGINE, drop_params=False
        )
        assert mapped == {}

    @pytest.mark.parametrize("size", ["1024", "widthxheight", "1024x", "1024*1024", "4096 x 4096"])
    def test_an_unreadable_size_is_refused_rather_than_defaulted(self, size: str):
        """Topaz bills the OUTPUT geometry, so a silently defaulted size bills a render nobody asked for."""
        with pytest.raises(litellm.BadRequestError, match="width>x<height"):
            TopazImageEditConfig().map_openai_params(
                image_edit_optional_params={"size": size}, model=ENGINE, drop_params=False
            )

    def test_generation_only_controls_are_not_advertised(self):
        supported = TopazImageEditConfig().get_supported_openai_params(ENGINE)
        for generation_only in ("prompt", "n", "quality", "background", "mask", "response_format"):
            assert generation_only not in supported


class TestAuth:
    def test_topaz_authenticates_with_its_own_header_not_a_bearer_token(self):
        headers = TopazImageEditConfig().validate_environment(headers={}, model=ENGINE, api_key="topaz-key")
        assert headers["X-API-Key"] == "topaz-key"
        assert "Authorization" not in headers

    def test_the_environment_key_is_used_when_the_deployment_carries_none(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("TOPAZ_API_KEY", "env-key")
        headers = TopazImageEditConfig().validate_environment(headers={}, model=ENGINE, api_key=None)
        assert headers["X-API-Key"] == "env-key"

    def test_a_missing_key_fails_before_the_request_is_sent(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("TOPAZ_API_KEY", raising=False)
        with pytest.raises(ValueError, match="TOPAZ_API_KEY"):
            TopazImageEditConfig().validate_environment(headers={}, model=ENGINE, api_key=None)

    def test_caller_headers_are_preserved_rather_than_replaced(self):
        headers = TopazImageEditConfig().validate_environment(
            headers={"X-Trace": "abc"}, model=ENGINE, api_key="topaz-key"
        )
        assert headers["X-Trace"] == "abc"


class TestResponse:
    def test_the_returned_image_bytes_become_the_openai_b64_payload(self):
        response = TopazImageEditConfig().transform_image_edit_response(
            model=ENGINE, raw_response=_enhanced_response(), logging_obj=Mock()
        )
        assert base64.b64decode(response.data[0].b64_json) == ENHANCED_JPEG

    def test_an_empty_body_is_an_error_rather_than_an_empty_image(self):
        with pytest.raises(TopazException):
            TopazImageEditConfig().transform_image_edit_response(
                model=ENGINE, raw_response=_enhanced_response(content=b""), logging_obj=Mock()
            )


class TestHandlerRoundTrip:
    def test_the_enhancement_posts_multipart_through_the_callers_client(self):
        caller_client = _RecordingHTTPHandler()

        response = BaseLLMHTTPHandler().image_edit_handler(
            model=ENGINE,
            image=[SOURCE_PNG],
            prompt=None,
            image_edit_provider_config=TopazImageEditConfig(),
            image_edit_optional_request_params={"output_width": "4096", "output_height": "4096"},
            custom_llm_provider=LlmProviders.TOPAZ.value,
            litellm_params=GenericLiteLLMParams(api_key="topaz-key"),
            logging_obj=Mock(),
            timeout=60.0,
            client=caller_client,
        )

        call = caller_client.calls[0]
        assert call["url"] == ENHANCE_URL
        # Topaz reads the engine and the geometry as form fields, never JSON.
        assert call["files"] == {"image": SOURCE_PNG}
        assert call["data"]["model"] == ENGINE
        assert call["data"]["output_width"] == "4096"
        assert call["headers"]["X-API-Key"] == "topaz-key"
        assert "json" not in call
        assert base64.b64decode(response.data[0].b64_json) == ENHANCED_JPEG

    @pytest.mark.asyncio
    async def test_the_async_enhancement_posts_multipart_through_the_callers_client(self):
        caller_client = _RecordingAsyncHTTPHandler()

        response = await BaseLLMHTTPHandler().async_image_edit_handler(
            model=ENGINE,
            image=[SOURCE_PNG],
            prompt=None,
            image_edit_provider_config=TopazImageEditConfig(),
            image_edit_optional_request_params={"output_width": "4096", "output_height": "4096"},
            custom_llm_provider=LlmProviders.TOPAZ.value,
            litellm_params=GenericLiteLLMParams(api_key="topaz-key"),
            logging_obj=Mock(),
            timeout=60.0,
            client=caller_client,
        )

        call = caller_client.calls[0]
        assert call["url"] == ENHANCE_URL
        assert call["files"] == {"image": SOURCE_PNG}
        assert call["data"]["model"] == ENGINE
        assert "json" not in call
        assert base64.b64decode(response.data[0].b64_json) == ENHANCED_JPEG


class TestCost:
    @pytest.fixture(autouse=True)
    def _bundled_cost_map(self, local_model_cost_map):
        """Price assertions must read this branch's map, not the network-fetched `main` copy."""

    @pytest.mark.parametrize("engine", TOPAZ_IMAGE_ENHANCE_MODELS)
    def test_every_advertised_engine_prices_at_one_topaz_credit(self, engine: str):
        """
        Gigapixel engines bill 1 credit per 24 MP of OUTPUT and the routed maximum is
        4096x4096 (16.8 MP), so one credit at the published $0.12 Starter rate covers
        any request this fork will send. An unpriced engine records $0 COGS, which is
        the regression NOL-519 exists to prevent.
        """
        assert litellm.get_model_info(model=f"topaz/{engine}").get("output_cost_per_image") == pytest.approx(0.12)

    @pytest.mark.parametrize("engine", TOPAZ_IMAGE_ENHANCE_MODELS)
    def test_the_cost_map_routes_each_engine_to_the_image_edit_surface(self, engine: str):
        info = litellm.get_model_info(model=f"topaz/{engine}")
        assert info.get("mode") == "image_edit"
        assert "/v1/images/edits" in (info.get("supported_endpoints") or ())

    def test_one_enhancement_costs_the_published_starter_rate(self):
        response = TopazImageEditConfig().transform_image_edit_response(
            model=ENGINE, raw_response=_enhanced_response(), logging_obj=Mock()
        )
        cost = litellm.completion_cost(
            completion_response=response,
            model=MODEL,
            custom_llm_provider=LlmProviders.TOPAZ.value,
            call_type="aimage_edit",
        )
        assert cost == pytest.approx(0.12)
