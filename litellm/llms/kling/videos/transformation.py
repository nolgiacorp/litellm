import base64
import re
from collections.abc import Mapping
from json import JSONDecodeError
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.litellm_core_utils.prompt_templates.common_utils import extract_file_data
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.llms.kling.auth import kling_auth_headers
from litellm.llms.kling.common_utils import (
    KLING_TASK_STATUS_MAP,
    resolve_kling_api_base,
    strip_kling_prefix,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
    extract_original_video_id,
)
from litellm.videos.capabilities import CapabilityParamSupport, DeclaredCapabilityParams

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

_TEXT_TO_VIDEO = "text2video"
_IMAGE_TO_VIDEO = "image2video"

_I2V_MODEL_MARKERS = ("image-to-video", "image2video")

_SIZE_TO_ASPECT_RATIO = {
    "1280x720": "16:9",
    "1920x1080": "16:9",
    "3840x2160": "16:9",
    "720x1280": "9:16",
    "1080x1920": "9:16",
    "2160x3840": "9:16",
    "1024x1024": "1:1",
    "1080x1080": "1:1",
}


_CAPABILITY_PARAMS = frozenset(
    (
        "input_reference",
        "image_url",
        "generate_audio",
    )
)

_KLING_RATE_LIMIT_STATUS = 429

# NOL-530. Kling's body-level error codes, mapped to the HTTP status that
# describes them. 1303 ("parallel task over resource pack limit") is the
# concurrency wall: the account is saturated and the request should be retried
# later, which is a 429 and not the 400 every body code used to become.
_KLING_BODY_CODE_STATUS: Mapping[int, int] = MappingProxyType(
    {  # mutable-ok: frozen constant lookup table
        1303: _KLING_RATE_LIMIT_STATUS,
    }
)

# Anything Kling reports that is not a known saturation code stays a client
# error: the request was understood and refused on its merits.
_KLING_DEFAULT_BODY_ERROR_STATUS = 400


def kling_error_response(status_code: int, error_message: str) -> httpx.Response:
    """
    A response whose BODY carries the vendor's message.

    `_handle_error` in the shared HTTP handler re-derives an error's text from
    `e.response.text` whenever the exception carries a response, and BOTH
    BaseLLMException and RateLimitError synthesise one with an EMPTY body. So
    any Kling error re-wrapped on that path - which is every error raised from a
    response transform - arrived at the caller with a BLANK message.

    That erases the vendor's own words, including the "parallel task over
    resource pack limit" text nolgia-api matches as its NOL-526 fallback. The
    429 status is now the primary signal, but a fallback that silently cannot
    fire is worse than no fallback. Putting the message in the body makes the
    handler's re-derivation a no-op instead of an erasure.

    Deliberately carries no headers: `_handle_error` prefers the exception's own
    headers and only falls back to the response's, and a vendor is not allowed
    to inject headers that a downstream serializer might forward to a client.
    """
    return httpx.Response(
        status_code=status_code,
        text=error_message,
        request=httpx.Request(method="POST", url=resolve_kling_api_base(None)),
    )


def kling_rate_limit_error(
    error_message: str,
    headers: Mapping[str, object] | httpx.Headers | None,
    model: str = "",
) -> litellm.RateLimitError:
    """
    Build the rate-limit error a saturated Kling deployment should raise.

    `Retry-After` rides through on `headers` when Kling sends it, which is the
    only way it reaches the client: RateLimitError deliberately does not copy
    response headers onto itself, so a header that is not passed explicitly is
    dropped.
    """
    resolved = dict(headers) if headers else {}  # mutable-ok: RateLimitError contract takes a dict of headers
    error = litellm.RateLimitError(
        message=error_message,
        llm_provider=litellm.LlmProviders.KLING.value,
        model=model,
        category=litellm.RateLimitErrorCategory.VENDOR_RATE_LIMIT,
        headers={str(key): str(value) for key, value in resolved.items()},
    )
    # Assigned after construction because RateLimitError.__init__ overwrites
    # self.response unconditionally, ignoring any response handed to it.
    error.response = kling_error_response(_KLING_RATE_LIMIT_STATUS, error_message)
    return error


class KlingVideoConfig(BaseVideoConfig):
    """
    Kling's classic /v1 API is a task API: POST to /v1/videos/text2video (or
    /v1/videos/image2video), then poll GET /v1/videos/{kind}/{task_id}. The poll
    response carries data.task_status (submitted|processing|succeed|failed) and, on
    success, data.task_result.videos[].url.

    The provider's public resolution knob (720p/1080p/4k) is translated to the
    classic API's mode field (std/pro/4k). A newer path-based Kling-3.0 API
    (POST /text-to-video/kling-3.0) accepts settings.resolution directly, but it
    rejects AccessKey:SecretKey credentials (requires a new-style single API key),
    so it is a future upgrade path rather than what we call today.
    """

    RESOLUTION_TO_MODE = {"720p": "std", "1080p": "pro", "4k": "4k"}
    DEFAULT_RESOLUTION = "1080p"

    def get_capability_param_support(self, model: str) -> CapabilityParamSupport:
        """
        Kling's direct API executes a start frame (image) and generate_audio (sound).

        negative_prompt is deliberately NOT declared, and this one is a judgement call
        rather than a documented fact. Kling's classic request table carries the field,
        but the only version-specific statement found says models 2.5, 2.6 and 3.0 do
        not honor it, and every model routed here is kling-v3. The vendor's own docs
        would settle it; they are not machine-readable from CI. Undeclared means a 400
        rather than a render that quietly ignored the exclusion, which is the failure
        mode this gate exists to remove, and it is a one-line change to flip once a
        live probe answers it. Note the fal-hosted kling-video/v3 twin does publish
        negative_prompt in its schema and IS declared, so this is another case of the
        route deciding capability rather than the vendor.

        It has NO end-frame surface here: Kling names that field image_tail, which
        this transformation never emits, so an end_image_url would be forwarded
        verbatim and ignored by the provider. Reference media, regeneration and
        bitrate are likewise unimplemented. Kling's fal-hosted twin does accept an
        end frame; that difference is exactly why advertisement has to follow the
        route a model is actually configured on rather than the vendor's catalog.
        """
        return DeclaredCapabilityParams(_CAPABILITY_PARAMS)

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "model",
            "prompt",
            "input_reference",
            "seconds",
            "size",
            "generate_audio",
            "user",
            "extra_headers",
            "extra_body",
        ]

    @classmethod
    def _mode_to_resolution(cls, mode: Any) -> str | None:
        """
        Inverse of _resolution_to_mode, for cost attribution (NOL-519).

        Returns the public resolution label (720p|1080p|4k) for Kling's classic
        mode field (std|pro|4k), or None when the mode is absent/unrecognised -
        the caller then omits video_resolution rather than guessing a tier, so a
        mispriced row is never invented.
        """
        if mode is None:
            return None
        key = str(mode).strip().lower()
        for resolution, mapped_mode in cls.RESOLUTION_TO_MODE.items():
            if mapped_mode == key:
                return resolution
        return None

    @classmethod
    def _resolution_to_mode(cls, resolution: Any) -> str:
        key = str(resolution).strip().lower()
        mode = cls.RESOLUTION_TO_MODE.get(key)
        if mode is None:
            raise ValueError(
                f"Unsupported Kling video resolution '{resolution}'. "
                f"Supported values are {sorted(cls.RESOLUTION_TO_MODE)}."
            )
        return mode

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:
        params: dict[str, Any] = dict(video_create_optional_params)
        extra_body = params.pop("extra_body", None)
        if isinstance(extra_body, dict):
            params = {**params, **extra_body}

        mapped: dict[str, Any] = {}

        seconds = params.get("seconds")
        if seconds is not None:
            mapped["duration"] = str(seconds)

        size = params.get("size")
        if isinstance(size, str):
            aspect = _SIZE_TO_ASPECT_RATIO.get(size)
            if aspect is not None:
                mapped["aspect_ratio"] = aspect
            elif "x" in size:
                mapped["aspect_ratio"] = size.replace("x", ":")

        resolution = params.get("resolution")
        if resolution is not None:
            mapped["mode"] = self._resolution_to_mode(resolution)

        # input_reference and image_url are the same start-frame slot under two names
        # and both are declared as executable, so both have to reach Kling's image
        # field; forwarding image_url verbatim would leave it ignored by the provider.
        start_image = self._coerce_start_image(params.get("input_reference") or params.get("image_url"))
        if start_image:
            mapped["image"] = start_image

        generate_audio = params.get("generate_audio")
        if generate_audio is not None:
            mapped["sound"] = "on" if generate_audio else "off"

        supported = self.get_supported_openai_params(model)
        handled = {"resolution", "image_url"}  # mutable-ok: local lookup set, never mutated
        for key, value in params.items():
            if key not in supported and key not in handled and key not in mapped:
                mapped[key] = value

        if self._is_image_to_video_model(model) and not mapped.get("image"):
            raise litellm.BadRequestError(
                message=(
                    f"Kling model '{model}' is an image-to-video variant, but no start image was provided. "
                    "Pass the start frame via input_reference as an image URL, a base64 string, or an uploaded "
                    "file; refusing to silently fall back to text-to-video, which would ignore the requested image "
                    "conditioning and return unrelated output."
                ),
                model=model,
                llm_provider=litellm.LlmProviders.KLING.value,
            )

        return mapped

    @staticmethod
    def _is_image_to_video_model(model: str) -> bool:
        normalized = model.lower()
        if any(marker in normalized for marker in _I2V_MODEL_MARKERS):
            return True
        return "i2v" in re.split(r"[/\-_.]+", normalized)

    @staticmethod
    def _coerce_start_image(input_reference: FileTypes | None) -> str | None:
        if input_reference is None:
            return None
        if isinstance(input_reference, str):
            stripped = input_reference.strip()
            return stripped or None
        extracted = extract_file_data(input_reference)
        return base64.b64encode(extracted["content"]).decode("utf-8")

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict:
        if litellm_params and litellm_params.api_key:
            api_key = api_key or litellm_params.api_key
        return {**headers, **kling_auth_headers(api_key)}

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        return resolve_kling_api_base(api_base)

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[dict, RequestFiles, str]:
        mapped: dict[str, Any] = dict(video_create_optional_request_params)
        mapped.pop("model", None)
        kind = _IMAGE_TO_VIDEO if mapped.get("image") else _TEXT_TO_VIDEO
        mapped.setdefault("mode", self._resolution_to_mode(self.DEFAULT_RESOLUTION))

        request_data = {
            key: value
            for key, value in {
                "model_name": strip_kling_prefix(model),
                "prompt": prompt,
                **mapped,
            }.items()
            if value is not None
        }
        return request_data, [], f"{api_base}/videos/{kind}"

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: dict | None = None,
    ) -> VideoObject:
        response_data = raw_response.json()
        self._raise_for_kling_error(response_data)

        data = response_data.get("data") or {}
        task_id = data.get("task_id")
        if not task_id:
            raise ValueError(f"Kling video submit response is missing data.task_id: {response_data}")

        status = KLING_TASK_STATUS_MAP.get(data.get("task_status", "submitted"), "queued")
        kind = _IMAGE_TO_VIDEO if request_data and request_data.get("image") else _TEXT_TO_VIDEO

        seconds: str | None = None
        size: str | None = None
        if request_data:
            if request_data.get("duration") is not None:
                seconds = str(request_data["duration"])
            if request_data.get("aspect_ratio") is not None:
                size = str(request_data["aspect_ratio"]).replace(":", "x")

        usage: dict[str, Any] = {}
        if seconds is not None:
            try:
                usage["duration_seconds"] = float(seconds)
            except (ValueError, TypeError):
                pass

        # NOL-519: Kling prices per SECOND and per RESOLUTION TIER, but one model
        # id (kling/kling-v3) serves all three tiers - the tier is a per-request
        # knob, not part of the model name. Without the resolution on usage the
        # shared video cost path has no way to pick between the 720p/1080p/4k
        # rates in the price map, so every generation resolved to $0.
        resolution = self._mode_to_resolution(request_data.get("mode") if request_data else None)
        if resolution is not None:
            usage["video_resolution"] = resolution

        video_obj = VideoObject(
            id=task_id,
            object="video",
            status=status,
            model=model,
            seconds=seconds,
            size=size,
            usage=usage,
        )

        if custom_llm_provider:
            video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, kind)
        return video_obj

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        return self._build_task_url(video_id, api_base), {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        self._raise_for_status(raw_response)
        try:
            response_data = raw_response.json()
        except (ValueError, JSONDecodeError):
            return VideoObject(id="", object="video", status="in_progress")

        data = response_data.get("data") or {}
        task_id = data.get("task_id", "")
        status = KLING_TASK_STATUS_MAP.get(data.get("task_status", "submitted"), "queued")

        error: dict[str, Any] | None = None
        if status == "failed":
            message = data.get("task_status_msg") or response_data.get("message") or "Video generation failed"
            error = {"code": "failed", "message": str(message)}

        video_obj = VideoObject(id=task_id, object="video", status=status, error=error)

        if custom_llm_provider and video_obj.id:
            kind = self._kind_from_request_url(raw_response)
            video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, kind)
        return video_obj

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        variant: str | None = None,
    ) -> tuple[str, dict]:
        return self._build_task_url(video_id, api_base), {}

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        self._raise_for_status(raw_response)
        video_url = self._extract_video_url(raw_response.json())
        httpx_client: HTTPHandler = _get_httpx_client()
        video_response = httpx_client.get(video_url)
        video_response.raise_for_status()
        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        self._raise_for_status(raw_response)
        video_url = self._extract_video_url(raw_response.json())
        async_client: AsyncHTTPHandler = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.KLING,
        )
        video_response = await async_client.get(video_url)
        video_response.raise_for_status()
        return video_response.content

    def _build_task_url(self, video_id: str, api_base: str) -> str:
        task_id, kind = self._extract_task_and_kind(video_id)
        encoded = encode_url_path_segment(task_id, field_name="video_id")
        return f"{api_base}/videos/{kind}/{encoded}"

    @staticmethod
    def _extract_task_and_kind(video_id: str) -> tuple[str, str]:
        decoded = decode_video_id_with_provider(video_id)
        task_id = decoded.get("video_id") or extract_original_video_id(video_id)
        kind = decoded.get("model_id")
        if kind not in (_TEXT_TO_VIDEO, _IMAGE_TO_VIDEO):
            raise ValueError(
                "Kling video status/content lookup requires the text2video/image2video "
                "kind encoded in the video_id. Use the id returned by video creation."
            )
        return task_id, kind

    @staticmethod
    def _kind_from_request_url(raw_response: httpx.Response) -> str | None:
        request = getattr(raw_response, "request", None)
        if request is None:
            return None
        path = request.url.path
        for kind in (_IMAGE_TO_VIDEO, _TEXT_TO_VIDEO):
            if f"/videos/{kind}/" in path or path.endswith(f"/videos/{kind}"):
                return kind
        return None

    @staticmethod
    def _extract_video_url(response_data: dict[str, Any]) -> str:
        data = response_data.get("data") or {}
        if data.get("task_status") == "failed":
            message = data.get("task_status_msg") or response_data.get("message")
            raise ValueError(f"Kling video generation failed: {message}")

        task_result = data.get("task_result") or {}
        videos = task_result.get("videos") or []
        if isinstance(videos, list) and videos and isinstance(videos[0], dict):
            url = videos[0].get("url")
            if isinstance(url, str) and url:
                return url

        raise ValueError("Video URL not found in Kling response. The job may still be processing.")

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[str, dict]:
        raise NotImplementedError("Video remix is not supported by the Kling API")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("Video remix is not supported by the Kling API")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        after: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        extra_query: dict[str, Any] | None = None,
    ) -> tuple[str, dict]:
        raise NotImplementedError("Video listing is not supported by the Kling API")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> dict[str, str]:
        raise NotImplementedError("Video listing is not supported by the Kling API")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        raise NotImplementedError("Video delete/cancel is not supported by the Kling API")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        raise NotImplementedError("Video delete/cancel is not supported by the Kling API")

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        """
        NOL-530. RETURNS the exception rather than raising it, matching every
        other provider's contract: the call sites, including the shared
        `_handle_error` path in llm_http_handler, already do
        `raise provider_config.get_error_class(...)`, so raising here made that
        `raise` unreachable and skipped the caller's own handling.

        A 429 becomes a real `litellm.RateLimitError`. Nothing else in the
        provider chain does this for Kling: `kling` is absent from
        `openai_compatible_providers` and `_openai_like_providers` and has no
        branch in `exception_type`, and a plain `BaseLLMException` is not in
        `LITELLM_EXCEPTION_TYPES`, so it used to fall all the way through to the
        terminal generic branch and surface as a 500 `APIConnectionError`. That
        is what made the 2026-08-05 concurrency incident answer 500.

        The consequence was worse than the wrong status. `APIConnectionError` is
        excluded from router cooldown by name -
        `cooldown_handlers._is_cooldown_required` skips any exception whose
        string contains it, and the message is literally prefixed with it - so a
        Kling rate limit could never cool the deployment down, and the router
        could not tell a saturated provider from a transport blip. Returning a
        RateLimitError takes the error OFF the APIConnectionError path
        entirely: it is in `LITELLM_EXCEPTION_TYPES`, so `exception_type`
        short-circuits and returns it untouched, and 429 is explicitly a
        cooldown status.
        """
        if status_code == _KLING_RATE_LIMIT_STATUS:
            # Deliberately NOT a BaseLLMException: only members of
            # LITELLM_EXCEPTION_TYPES survive exception_type() untouched, and
            # that passthrough is the entire point of this branch.
            rate_limited = kling_rate_limit_error(error_message, headers)
            return rate_limited  # pyright: ignore[reportReturnType]  # deliberate; see the comment above
        return BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
            response=kling_error_response(status_code, error_message),
        )

    def _raise_for_kling_error(self, response_data: dict[str, Any]) -> None:
        """
        Kling reports application-level failures in the response BODY, commonly
        under HTTP 200, so the body code is the only signal of what happened.

        NOL-530. Every such code used to become a hardcoded 400, which
        `litellm._should_retry(400)` refuses, so the concurrency wall
        (`{"code": 1303, "message": "parallel task over resource pack limit"}`)
        produced neither a retry nor a cooldown. It is a saturation signal, so
        it maps to 429 like any other.
        """
        code = response_data.get("code")
        if code is None or code == 0:
            return
        message = response_data.get("message")
        status_code = _KLING_BODY_CODE_STATUS.get(code, _KLING_DEFAULT_BODY_ERROR_STATUS)
        raise self.get_error_class(
            error_message=str(message) if message else "Kling API returned an error",
            status_code=status_code,
            headers={},  # mutable-ok: BaseLLMException contract takes a dict of headers
        )

    def _raise_for_status(self, raw_response: httpx.Response) -> None:
        if raw_response.is_success:
            return
        raise self.get_error_class(
            error_message=raw_response.text,
            status_code=raw_response.status_code,
            headers=raw_response.headers,
        )
