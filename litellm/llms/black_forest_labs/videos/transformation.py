import base64
import time
from collections.abc import Mapping
from json import JSONDecodeError
from types import MappingProxyType
from typing import TYPE_CHECKING, Any  # noqa: TID251  # base video ABC + OpenAI video TypedDict are Any-typed

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.litellm_core_utils.prompt_templates.common_utils import extract_file_data
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.black_forest_labs.common_utils import (
    EMPTY_MAP,
    FLUX_3_VIDEO_ENDPOINT,
    VIDEO_GENERATION_MODELS,
    BlackForestLabsError,
    assert_bfl_polling_url,
    bfl_auth_headers,
    resolve_bfl_api_base,
)
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    encode_video_id_with_provider,
    extract_original_video_id,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


_BFL_STATUS_MAP: Mapping[str, str] = MappingProxyType(
    {  # mutable-ok: frozen constant lookup table
        "Ready": "completed",
        "Pending": "in_progress",
        "Processing": "in_progress",
        "Queued": "queued",
        "Task Queued": "queued",
        "Error": "failed",
        "Failed": "failed",
        "Content Moderated": "failed",
        "Request Moderated": "failed",
        "Task not found": "failed",
    }
)

_FAILED_STATUSES = frozenset(("Error", "Failed", "Content Moderated", "Request Moderated", "Task not found"))

_RESULT_URL_FIELDS = ("sample", "video", "url", "sample_url")

_SIZE_TO_ASPECT_RATIO: Mapping[str, str] = MappingProxyType(
    {  # mutable-ok: frozen constant lookup table
        "1280x720": "16:9",
        "1920x1080": "16:9",
        "3840x2160": "16:9",
        "720x1280": "9:16",
        "1080x1920": "9:16",
        "2160x3840": "9:16",
        "1024x1024": "1:1",
        "1080x1080": "1:1",
    }
)

_SUPPORTED_OPENAI_PARAMS = (
    "model",
    "prompt",
    "input_reference",
    "image",
    "seconds",
    "size",
    "resolution",
    "aspect_ratio",
    "generate_audio",
    "safety_tolerance",
    "user",
    "extra_headers",
    "extra_body",
)

_OPENAI_ONLY_PARAMS = frozenset(
    (
        "model",
        "prompt",
        "user",
        "extra_headers",
        "extra_body",
        "input_reference",
        "image",
        "image_url",
        "image_urls",
        "end_image_url",
        "video_urls",
        "start_video",
        "keyframes",
        "seconds",
        "duration_seconds",
        "size",
        "resolution",
        "aspect_ratio",
        "generate_audio",
        "safety_tolerance",
    )
)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _duration_usage(seconds: str | None) -> dict:  # mutable-ok: VideoObject.usage expects a dict
    duration = _safe_float(seconds)
    if duration is None:
        return {}  # mutable-ok: VideoObject.usage expects a dict
    return {"duration_seconds": duration}  # mutable-ok: VideoObject.usage expects a dict


def _failure_error(bfl_status: Any) -> dict:  # mutable-ok: VideoObject.error expects a dict
    message = str(bfl_status or "Video generation failed")
    return {"code": "failed", "message": message}  # mutable-ok: VideoObject.error expects a dict


class BflVideoConfig(BaseVideoConfig):
    """
    Black Forest Labs flux-3-video is an async task API: POST /v1/flux-3-video returns
    {id, polling_url}; the caller polls polling_url until the result carries the video URL,
    then downloads it. flux-3-video produces synchronized native audio when generate_audio
    is true (the default), so the mp4 is downloaded as-is to keep that track intact.

    The polling_url is BFL's durable handle for a task and is carried through the encoded
    video_id. Every read validates it with assert_bfl_polling_url, which locks status and
    content GETs (they carry the x-key) to https hosts within the bfl.ai domain.
    """

    def __init__(
        self,
        sync_client: HTTPHandler | None = None,
        async_client: AsyncHTTPHandler | None = None,
    ) -> None:
        super().__init__()
        self._sync_client = sync_client
        self._async_client = async_client

    def _http_client(self) -> HTTPHandler:
        return self._sync_client or _get_httpx_client()

    def _async_http_client(self) -> AsyncHTTPHandler:
        return self._async_client or get_async_httpx_client(llm_provider=litellm.LlmProviders.BLACK_FOREST_LABS)

    def get_supported_openai_params(self, model: str) -> list:  # mutable-ok: BaseVideoConfig contract returns list
        return list(_SUPPORTED_OPENAI_PARAMS)  # mutable-ok: BaseVideoConfig contract returns list

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:  # mutable-ok: BaseVideoConfig contract returns dict
        params = self._merged_params(video_create_optional_params)
        seconds = params.get("seconds")
        if seconds is None:
            seconds = params.get("duration_seconds")
        resolution = params.get("resolution")
        generate_audio = params.get("generate_audio")
        start_video = self._start_video(params)
        keyframes = None if start_video is not None else self._keyframes(params, seconds)
        mapped = (
            ("duration", self._coerce_duration(seconds) if seconds is not None else None),
            ("resolution", str(resolution).strip().lower() if resolution is not None else None),
            ("aspect_ratio", self._resolve_aspect_ratio(params)),
            ("generate_audio", self._coerce_bool(generate_audio) if generate_audio is not None else None),
            ("safety_tolerance", params.get("safety_tolerance")),
            ("keyframes", keyframes),
            ("start_video", start_video),
            ("mode", self._mode(params, start_video=start_video, keyframes=keyframes)),
        )
        consumed = frozenset(key for key, value in mapped if value is not None)
        passthrough = tuple(
            (key, value) for key, value in params.items() if key not in _OPENAI_ONLY_PARAMS and key not in consumed
        )
        return {  # mutable-ok: BaseVideoConfig contract returns dict
            key: value for key, value in (*mapped, *passthrough) if value is not None
        }

    @staticmethod
    def _merged_params(video_create_optional_params: VideoCreateOptionalRequestParams) -> Mapping[str, Any]:
        extra_body = video_create_optional_params.get("extra_body")
        return {  # mutable-ok: one-shot merge, read only as a Mapping; extra_body wins over top-level params
            **video_create_optional_params,
            **(extra_body if isinstance(extra_body, Mapping) else EMPTY_MAP),
        }

    @staticmethod
    def _resolve_aspect_ratio(params: Mapping[str, Any]) -> str | None:
        aspect_ratio = params.get("aspect_ratio")
        if aspect_ratio is not None:
            return str(aspect_ratio)

        size = params.get("size")
        if not isinstance(size, str) or not size:
            return None
        derived = _SIZE_TO_ASPECT_RATIO.get(size)
        if derived is not None:
            return derived
        if "x" in size:
            return size.replace("x", ":")
        return None

    def _keyframes(self, params: Mapping[str, Any], seconds: object) -> tuple[object, ...] | None:
        existing = params.get("keyframes")
        if existing:
            return tuple(existing) if isinstance(existing, (list, tuple)) else (existing,)

        frames = tuple(
            frame
            for frame in (
                self._start_frame(params),
                *self._element_frames(params),
                self._coerce_media_ref(params.get("end_image_url")),
            )
            if frame is not None
        )
        if not frames:
            return None
        if len(frames) <= 2:
            return frames
        return self._timestamped_keyframes(frames, seconds)

    def _start_frame(self, params: Mapping[str, Any]) -> str | None:
        for source in ("image", "input_reference", "image_url"):
            coerced = self._coerce_media_ref(params.get(source))
            if coerced is not None:
                return coerced
        return None

    def _element_frames(self, params: Mapping[str, Any]) -> tuple[str, ...]:
        image_urls = params.get("image_urls")
        if not isinstance(image_urls, (list, tuple)):
            return ()
        return tuple(frame for frame in (self._coerce_media_ref(url) for url in image_urls) if frame is not None)

    @staticmethod
    def _timestamped_keyframes(frames: tuple[str, ...], seconds: object) -> tuple[object, ...]:
        duration = _safe_float(seconds)
        if duration is None or duration <= 0:
            return frames
        last = len(frames) - 1
        return tuple((round(index * duration / last, 2), frame) for index, frame in enumerate(frames))

    def _start_video(self, params: Mapping[str, Any]) -> str | None:
        native = params.get("start_video")
        if native:
            return self._coerce_media_ref(native)
        video_urls = params.get("video_urls")
        if isinstance(video_urls, (list, tuple)) and video_urls:
            return self._coerce_media_ref(video_urls[0])
        return None

    @staticmethod
    def _mode(params: Mapping[str, Any], start_video: str | None, keyframes: tuple[object, ...] | None) -> str | None:
        if start_video is not None:
            return "v2v"
        if keyframes:
            return "i2v"
        explicit = params.get("mode")
        return str(explicit) if explicit else None

    @staticmethod
    def _coerce_media_ref(ref: Any) -> str | None:
        if ref is None:
            return None
        if isinstance(ref, str):
            stripped = ref.strip()
            return stripped or None
        extracted = extract_file_data(ref)
        return base64.b64encode(extracted["content"]).decode("utf-8")

    @staticmethod
    def _coerce_duration(seconds: Any) -> int | str:
        if isinstance(seconds, bool):
            return int(seconds)
        if isinstance(seconds, (int, float)):
            return int(seconds)
        text = str(seconds).strip()
        return int(text) if text.isdigit() else text

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "on")
        return bool(value)

    def validate_environment(
        self,
        headers: Mapping[str, Any],
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict:  # mutable-ok: BaseVideoConfig contract returns dict
        if litellm_params and litellm_params.api_key:
            api_key = api_key or litellm_params.api_key
        return {**headers, **bfl_auth_headers(api_key)}  # mutable-ok: BaseVideoConfig contract returns dict

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: Mapping[str, Any],
    ) -> str:
        return resolve_bfl_api_base(api_base)

    @staticmethod
    def _video_endpoint(model: str) -> str:
        name = model.split("/")[-1].lower() if model else ""
        return VIDEO_GENERATION_MODELS.get(name, FLUX_3_VIDEO_ENDPOINT)

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Mapping[str, Any],
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, Any],
    ) -> tuple[dict, RequestFiles, str]:  # mutable-ok: BaseVideoConfig contract returns dict body
        mode = video_create_optional_request_params.get("mode") or "t2v"
        body = tuple(
            (key, value) for key, value in video_create_optional_request_params.items() if key not in ("model", "mode")
        )
        request_data = {  # mutable-ok: BaseVideoConfig contract returns dict body
            key: value for key, value in (("prompt", prompt), ("mode", mode), *body) if value is not None
        }
        return request_data, (), f"{api_base}{self._video_endpoint(model)}"

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: Mapping[str, Any] | None = None,
    ) -> VideoObject:
        response_data = raw_response.json()
        if "errors" in response_data:
            raise BlackForestLabsError(
                status_code=raw_response.status_code,
                message=f"BFL flux-3-video error: {response_data['errors']}",
            )

        polling_url = response_data.get("polling_url")
        task_id = response_data.get("id")
        if not polling_url or not task_id:
            raise ValueError(f"BFL flux-3-video submit response is missing id/polling_url: {response_data}")

        assert_bfl_polling_url(polling_url)

        duration = request_data.get("duration") if request_data else None
        aspect_ratio = request_data.get("aspect_ratio") if request_data else None
        seconds = str(duration) if duration is not None else None

        video_obj = VideoObject(
            id=str(task_id),
            object="video",
            status="queued",
            model=model,
            seconds=seconds,
            size=str(aspect_ratio).replace(":", "x") if aspect_ratio is not None else None,
            created_at=int(time.time()),
            usage=_duration_usage(seconds),
        )

        if custom_llm_provider:
            video_obj.id = encode_video_id_with_provider(polling_url, custom_llm_provider, model)
        return video_obj

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, Any],
    ) -> tuple[str, dict]:  # mutable-ok: BaseVideoConfig contract returns dict params
        return self._decode_polling_url(video_id), {}  # mutable-ok: BaseVideoConfig contract returns dict params

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

        bfl_status = response_data.get("status")
        status = _BFL_STATUS_MAP.get(bfl_status, "in_progress")

        video_obj = VideoObject(
            id=str(response_data.get("id") or ""),
            object="video",
            status=status,
            error=_failure_error(bfl_status) if status == "failed" else None,
        )

        if custom_llm_provider:
            polling_url = self._polling_url_from_request(raw_response)
            if polling_url:
                video_obj.id = encode_video_id_with_provider(polling_url, custom_llm_provider, None)
        return video_obj

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, Any],
        variant: str | None = None,
    ) -> tuple[str, dict]:  # mutable-ok: BaseVideoConfig contract returns dict params
        return self._decode_polling_url(video_id), {}  # mutable-ok: BaseVideoConfig contract returns dict params

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        self._raise_for_status(raw_response)
        video_url = self._extract_video_url(raw_response.json())
        video_response = self._http_client().get(video_url)
        video_response.raise_for_status()
        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        self._raise_for_status(raw_response)
        video_url = self._extract_video_url(raw_response.json())
        video_response = await self._async_http_client().get(video_url)
        video_response.raise_for_status()
        return video_response.content

    @staticmethod
    def _decode_polling_url(video_id: str) -> str:
        polling_url = extract_original_video_id(video_id)
        assert_bfl_polling_url(polling_url)
        return polling_url

    @staticmethod
    def _polling_url_from_request(raw_response: httpx.Response) -> str | None:
        request = getattr(raw_response, "request", None)
        if request is None:
            return None
        return str(request.url)

    @classmethod
    def _extract_video_url(cls, response_data: Mapping[str, Any]) -> str:
        status = response_data.get("status")
        if status in _FAILED_STATUSES:
            raise ValueError(f"flux-3-video generation failed: {status}")

        result = response_data.get("result")
        if isinstance(result, Mapping):
            for field in _RESULT_URL_FIELDS:
                value = result.get(field)
                if isinstance(value, str) and value:
                    return value

        raise ValueError("Video URL not found in BFL response. The job may still be processing.")

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, Any],
        extra_body: Mapping[str, Any] | None = None,
    ) -> tuple[str, dict]:  # mutable-ok: BaseVideoConfig contract returns dict params
        raise NotImplementedError("Video remix is not supported by the Black Forest Labs API")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("Video remix is not supported by the Black Forest Labs API")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, Any],
        after: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        extra_query: Mapping[str, Any] | None = None,
    ) -> tuple[str, dict]:  # mutable-ok: BaseVideoConfig contract returns dict params
        raise NotImplementedError("Video listing is not supported by the Black Forest Labs API")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseVideoConfig contract returns dict
        raise NotImplementedError("Video listing is not supported by the Black Forest Labs API")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, Any],
    ) -> tuple[str, dict]:  # mutable-ok: BaseVideoConfig contract returns dict params
        raise NotImplementedError("Video delete/cancel is not supported by the Black Forest Labs API")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        raise NotImplementedError("Video delete/cancel is not supported by the Black Forest Labs API")

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict | httpx.Headers,  # mutable-ok: BaseLLMException carries the raw response headers dict
    ) -> BaseLLMException:
        raise BlackForestLabsError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    def _raise_for_status(self, raw_response: httpx.Response) -> None:
        if raw_response.is_success:
            return
        raise BlackForestLabsError(
            status_code=raw_response.status_code,
            message=raw_response.text,
            headers=raw_response.headers,
        )
