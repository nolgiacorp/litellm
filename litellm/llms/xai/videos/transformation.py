from collections.abc import Mapping, Sequence
from json import JSONDecodeError
from typing import TYPE_CHECKING, Any

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.llms.xai.common_utils import XAIModelInfo
from litellm.types.router import GenericLiteLLMParams
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

XAI_VIDEO_STATUS_MAP = {
    "pending": "in_progress",
    "done": "completed",
    "failed": "failed",
    "expired": "failed",
}

_SIZE_TO_ASPECT_RATIO = {
    "1280x720": "16:9",
    "1920x1080": "16:9",
    "720x1280": "9:16",
    "1080x1920": "9:16",
    "1024x1024": "1:1",
    "1080x1080": "1:1",
    "1024x768": "4:3",
    "768x1024": "3:4",
}


def _resolve_xai_video_api_base(api_base: str | None) -> str:
    base = (XAIModelInfo.get_api_base(api_base) or "https://api.x.ai").rstrip("/")
    return base.removesuffix("/v1").rstrip("/")


def _voice_id(entry: Any) -> str | None:
    if isinstance(entry, str):
        return entry or None
    voice_id = entry.get("voice_id") if isinstance(entry, Mapping) else None
    return str(voice_id) if voice_id else None


def _normalize_reference_audios(raw_reference_audios: Any) -> Sequence[Mapping[str, str]]:
    """Normalize a voice_id string, a list of strings, or a list of {voice_id} objects to xAI's [{voice_id}] shape."""
    entries: tuple[Any, ...] = (
        (raw_reference_audios,)
        if isinstance(raw_reference_audios, str)
        else tuple(raw_reference_audios)
        if isinstance(raw_reference_audios, list)
        else ()
    )
    return [  # mutable-ok: xAI request payload, JSON-serialized as a list of {voice_id} objects
        {"voice_id": voice_id}  # mutable-ok: one reference_audios entry inside that payload list
        for voice_id in map(_voice_id, entries)
        if voice_id
    ]


_CAPABILITY_PARAMS = frozenset(
    (
        "input_reference",
        "image_url",
        "image_urls",
        "reference_audios",
    )
)


class XAIVideoConfig(BaseVideoConfig):
    def get_capability_param_support(self, model: str) -> CapabilityParamSupport:
        """
        xAI executes a start frame (image), reference_images and preset-voice
        reference_audios. It has no end-frame, reference-video, reference-audio-URL,
        regeneration, bitrate or generate_audio surface; grok audio is native and
        always on, so there is no soundtrack switch to honor.
        """
        return DeclaredCapabilityParams(_CAPABILITY_PARAMS)

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "model",
            "prompt",
            "input_reference",
            "seconds",
            "size",
            "user",
            "extra_headers",
            "extra_body",
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:
        extra_body = video_create_optional_params.get("extra_body")
        params: dict[str, Any] = {
            **{key: value for key, value in video_create_optional_params.items() if key != "extra_body"},
            **(extra_body if isinstance(extra_body, dict) else {}),
        }
        raw_duration = params.get("seconds") if params.get("seconds") is not None else params.get("duration")
        try:
            duration = int(float(raw_duration)) if raw_duration is not None else None
        except (TypeError, ValueError):
            raise ValueError(f"Unsupported xAI video duration '{raw_duration}'; expected a number of seconds.")
        size = params.get("size")
        aspect_ratio = (
            _SIZE_TO_ASPECT_RATIO.get(size) or (size.replace("x", ":") if "x" in size else None)
            if isinstance(size, str) and size
            else params.get("aspect_ratio")
        )
        input_reference = params.get("input_reference")
        image_url = params.get("image_url")
        image_source = (
            input_reference
            if isinstance(input_reference, str) and input_reference
            else image_url
            if isinstance(image_url, str) and image_url
            else None
        )
        image = {"url": image_source} if image_source else None
        raw_reference_images = params.get("reference_images") or params.get("image_urls")
        reference_image_urls = (
            [raw_reference_images]
            if isinstance(raw_reference_images, str) and raw_reference_images
            else [url for url in raw_reference_images if isinstance(url, str) and url]
            if isinstance(raw_reference_images, list)
            else []
        )
        reference_images = [{"url": url} for url in reference_image_urls]
        reference_audios = _normalize_reference_audios(params.get("reference_audios"))
        resolution = params.get("resolution")

        return {
            **({"duration": duration} if duration is not None else {}),
            **({"aspect_ratio": aspect_ratio} if aspect_ratio else {}),
            **({"resolution": resolution} if resolution else {}),
            # xAI requires image (start frame) to be mutually exclusive with reference_images / reference_audios.
            **({"image": image} if image and not reference_images and not reference_audios else {}),
            **({"reference_images": reference_images} if reference_images else {}),
            **({"reference_audios": reference_audios} if reference_audios else {}),  # mutable-ok: xAI request payload
        }

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict:
        if litellm_params and litellm_params.api_key:
            api_key = api_key or litellm_params.api_key
        final_api_key = XAIModelInfo.get_api_key(api_key)
        if not final_api_key:
            raise ValueError("XAI_API_KEY is not set")
        return {**headers, "Authorization": f"Bearer {final_api_key}"}

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        return _resolve_xai_video_api_base(api_base)

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
        if "reference_audios" in mapped:
            # get_optional_params_video_generation merges the raw extra_body over the mapped params, so the
            # caller's original shape can land back here; re-normalize to keep xAI's [{voice_id}] on the wire.
            mapped["reference_audios"] = _normalize_reference_audios(mapped["reference_audios"]) or None
        request_data = {
            key: value
            for key, value in {
                "model": model.removeprefix("xai/"),
                "prompt": prompt,
                **mapped,
            }.items()
            if value is not None
        }
        return request_data, [], f"{api_base}/v1/videos/generations"

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: dict | None = None,
    ) -> VideoObject:
        self._raise_for_status(raw_response)
        response_data = raw_response.json()
        request_id = response_data.get("request_id") or response_data.get("id")
        if not request_id:
            raise ValueError(f"xAI video submit response is missing request_id: {response_data}")

        seconds: str | None = None
        usage: dict[str, Any] = {}
        if request_data and request_data.get("duration") is not None:
            seconds = str(request_data["duration"])
            try:
                usage["duration_seconds"] = float(request_data["duration"])
            except (TypeError, ValueError):
                pass

        video_obj = VideoObject(
            id=request_id,
            object="video",
            status="queued",
            model=model,
            seconds=seconds,
            usage=usage,
        )
        if custom_llm_provider:
            video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, model.removeprefix("xai/"))
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

        request_id = response_data.get("request_id") or response_data.get("id") or ""
        raw_status = str(response_data.get("status", "pending")).lower()
        status = XAI_VIDEO_STATUS_MAP.get(raw_status, "in_progress")

        video_data = response_data.get("video") or {}
        usage: dict[str, Any] = {}
        duration = video_data.get("duration")
        if duration is not None:
            try:
                usage["duration_seconds"] = float(duration)
            except (TypeError, ValueError):
                pass

        error: dict[str, Any] | None = None
        if status == "failed":
            message = response_data.get("error") or response_data.get("detail") or f"Video generation {raw_status}"
            error = {"code": raw_status, "message": str(message)}

        video_obj = VideoObject(id=request_id, object="video", status=status, usage=usage, error=error)
        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, "")
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
            llm_provider=litellm.LlmProviders.XAI,
        )
        video_response = await async_client.get(video_url)
        video_response.raise_for_status()
        return video_response.content

    @staticmethod
    def _build_task_url(video_id: str, api_base: str) -> str:
        decoded = decode_video_id_with_provider(video_id)
        request_id = decoded.get("video_id") or extract_original_video_id(video_id)
        encoded = encode_url_path_segment(request_id, field_name="video_id")
        return f"{api_base}/v1/videos/{encoded}"

    @staticmethod
    def _extract_video_url(response_data: dict[str, Any]) -> str:
        raw_status = str(response_data.get("status", "")).lower()
        if raw_status in ("failed", "expired"):
            message = response_data.get("error") or response_data.get("detail") or raw_status
            raise ValueError(f"xAI video generation failed: {message}")

        video_data = response_data.get("video") or {}
        url = video_data.get("url")
        if isinstance(url, str) and url:
            return url

        raise ValueError("Video URL not found in xAI response. The job may still be processing.")

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[str, dict]:
        raise NotImplementedError("Video remix is not supported for xAI")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("Video remix is not supported for xAI")

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
        raise NotImplementedError("Video listing is not supported for xAI")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> dict[str, str]:
        raise NotImplementedError("Video listing is not supported for xAI")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        raise NotImplementedError("Video delete/cancel is not supported for xAI")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        raise NotImplementedError("Video delete/cancel is not supported for xAI")

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    def _raise_for_status(self, raw_response: httpx.Response) -> None:
        if raw_response.is_success:
            return
        raise self.get_error_class(
            error_message=raw_response.text,
            status_code=raw_response.status_code,
            headers=raw_response.headers,
        )
