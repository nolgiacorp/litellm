from json import JSONDecodeError
from typing import TYPE_CHECKING, Any

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
    extract_original_video_id,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


_OPENROUTER_DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
_OPENROUTER_PREFIX = "openrouter/"

# OpenRouter's normalized video status -> the OpenAI-shaped status LiteLLM exposes.
_OPENROUTER_STATUS_MAP = {
    "pending": "queued",
    "queued": "queued",
    "in_progress": "in_progress",
    "processing": "in_progress",
    "completed": "completed",
    "succeeded": "completed",
    "failed": "failed",
    "cancelled": "failed",
    "canceled": "failed",
}

# Params consumed here by name; anything else on the request is ignored rather
# than forwarded, since OpenRouter rejects unknown top-level video fields.
_PASSTHROUGH_PARAMS = (
    "resolution",
    "aspect_ratio",
    "seed",
    "generate_audio",
    "provider",
    "callback_url",
)


class OpenRouterVideoConfig(BaseVideoConfig):
    """
    OpenRouter exposes an async video API distinct from its chat surface: POST
    /api/v1/videos returns {"id", "polling_url", "status"}, then GET
    /api/v1/videos/{id} reports {"status", "usage", "error"} until status ==
    "completed". The finished MP4 is downloaded from the bearer-authenticated
    /api/v1/videos/{id}/content endpoint, mirroring the fal/Kling poll flow.

    A single normalized schema covers every OpenRouter video model. Seedance
    character consistency is driven by input_references[] (reference-to-video);
    start/end stills go through frame_images[] with frame_type first_frame /
    last_frame. map_openai_params accepts both the canonical OpenAI-shaped params
    and the fal-shaped names nolgia-api already sends (image_url / end_image_url /
    image_urls), so swapping Seedance from fal to OpenRouter needs no client change.
    """

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

    @staticmethod
    def _strip_openrouter_prefix(model: str) -> str:
        return model.removeprefix(_OPENROUTER_PREFIX)

    @staticmethod
    def _image_url_object(url: str) -> dict[str, Any]:
        return {"type": "image_url", "image_url": {"url": url}}

    @classmethod
    def _normalize_reference(cls, entry: Any) -> dict[str, Any] | None:
        if isinstance(entry, str) and entry:
            return cls._image_url_object(entry)
        if isinstance(entry, dict):
            if isinstance(entry.get("image_url"), dict):
                return entry
            url = entry.get("url")
            if isinstance(url, str) and url:
                return cls._image_url_object(url)
        return None

    @classmethod
    def _normalize_frame(cls, entry: Any, default_frame_type: str) -> dict[str, Any] | None:
        reference = cls._normalize_reference(entry)
        if reference is None:
            return None
        frame_type = entry.get("frame_type", default_frame_type) if isinstance(entry, dict) else default_frame_type
        return {**reference, "frame_type": frame_type}

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if value is None:
            return []
        return value if isinstance(value, list) else [value]

    @classmethod
    def _collect_frame_images(cls, params: dict[str, Any]) -> list[dict[str, Any]]:
        # Start/end stills for i2v. nolgia-api sends the fal-shaped image_url /
        # end_image_url; input_reference / frame_images are the canonical
        # OpenAI-shaped equivalents. image_url and input_reference are the same
        # start slot, so prefer whichever is set.
        start = params.get("input_reference") or params.get("image_url")
        candidates = [
            cls._normalize_frame(start, "first_frame"),
            cls._normalize_frame(params.get("end_image_url"), "last_frame"),
            *[cls._normalize_frame(entry, "first_frame") for entry in cls._as_list(params.get("frame_images"))],
        ]
        return [frame for frame in candidates if frame is not None]

    @classmethod
    def _collect_input_references(cls, params: dict[str, Any]) -> list[dict[str, Any]]:
        # Reference-to-video character images. nolgia-api sends them as the
        # fal-shaped image_urls; input_references is the canonical equivalent.
        entries = cls._as_list(params.get("input_references")) + cls._as_list(params.get("image_urls"))
        return [
            reference for reference in (cls._normalize_reference(entry) for entry in entries) if reference is not None
        ]

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

        duration = self._coerce_duration(params.get("seconds"), params.get("duration"))
        size = params.get("size")
        frame_images = self._collect_frame_images(params)
        input_references = self._collect_input_references(params)

        return {
            **({"duration": duration} if duration is not None else {}),
            **({"size": size} if isinstance(size, str) and size else {}),
            **{key: params[key] for key in _PASSTHROUGH_PARAMS if params.get(key) is not None},
            **({"frame_images": frame_images} if frame_images else {}),
            **({"input_references": input_references} if input_references else {}),
        }

    @staticmethod
    def _coerce_duration(seconds: Any, duration: Any) -> int | None:
        raw = seconds if seconds is not None else duration
        if raw is None:
            return None
        try:
            return int(float(raw))
        except (ValueError, TypeError):
            return None

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict:
        if litellm_params and litellm_params.api_key:
            api_key = api_key or litellm_params.api_key

        resolved_key = api_key or litellm.api_key or get_secret_str("OPENROUTER_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or pass api_key parameter."
            )

        headers.update(
            {
                "Authorization": f"Bearer {resolved_key}",
                "Content-Type": "application/json",
            }
        )
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        base = api_base or get_secret_str("OPENROUTER_API_BASE") or _OPENROUTER_DEFAULT_API_BASE
        return base.rstrip("/")

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[dict, RequestFiles, str]:
        mapped = dict(video_create_optional_request_params)
        mapped.pop("model", None)

        request_data = {
            key: value
            for key, value in {
                "model": self._strip_openrouter_prefix(model),
                "prompt": prompt,
                **mapped,
            }.items()
            if value is not None
        }
        return request_data, [], f"{api_base}/videos"

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

        job_id = response_data.get("id")
        if not job_id:
            raise ValueError(f"OpenRouter video submit response is missing 'id': {response_data}")

        status = _OPENROUTER_STATUS_MAP.get(str(response_data.get("status", "pending")).lower(), "queued")

        seconds = str(request_data["duration"]) if request_data and request_data.get("duration") is not None else None
        size = str(request_data["size"]) if request_data and request_data.get("size") is not None else None

        usage: dict[str, Any] = {}
        if seconds is not None:
            try:
                usage["duration_seconds"] = float(seconds)
            except (ValueError, TypeError):
                pass

        video_obj = VideoObject(
            id=job_id,
            object="video",
            status=status,
            model=model,
            seconds=seconds,
            size=size,
            usage=usage,
        )

        if custom_llm_provider:
            video_obj.id = encode_video_id_with_provider(
                video_obj.id, custom_llm_provider, self._strip_openrouter_prefix(model)
            )
        return video_obj

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        return self._build_job_url(video_id, api_base), {}

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

        job_id = response_data.get("id", "")
        status = _OPENROUTER_STATUS_MAP.get(str(response_data.get("status", "pending")).lower(), "queued")

        error: dict[str, Any] | None = None
        if status == "failed":
            message = response_data.get("error") or "Video generation failed"
            error = {"code": "failed", "message": str(message)}

        usage = response_data.get("usage") if isinstance(response_data.get("usage"), dict) else None

        video_obj = VideoObject(id=job_id, object="video", status=status, error=error, usage=usage)

        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, None)
        return video_obj

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        variant: str | None = None,
    ) -> tuple[str, dict]:
        return f"{self._build_job_url(video_id, api_base)}/content?index=0", {}

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        # OpenRouter's /videos/{id}/content endpoint returns the MP4 bytes directly,
        # bearer-authenticated by the handler. unsigned_urls from the poll body are
        # NOT publicly fetchable (they 401 without a session), so this dedicated
        # content route is the only reliable download path. The base async fallback
        # reuses this method, so no separate download hop is needed.
        self._raise_for_status(raw_response)
        return raw_response.content

    @staticmethod
    def _build_job_url(video_id: str, api_base: str) -> str:
        # Rebuild the poll URL from api_base + the decoded job id. The job id is
        # only base64-encoded inside video_id; never trust an embedded URL, which
        # a forged id could point at an arbitrary host to leak the bearer token.
        decoded = decode_video_id_with_provider(video_id)
        job_id = decoded.get("video_id") or extract_original_video_id(video_id)
        encoded = encode_url_path_segment(job_id, field_name="video_id")
        return f"{api_base}/videos/{encoded}"

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[str, dict]:
        raise NotImplementedError("Video remix is not supported by the OpenRouter video API")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("Video remix is not supported by the OpenRouter video API")

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
        raise NotImplementedError("Video listing is not supported by the OpenRouter video API")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> dict[str, str]:
        raise NotImplementedError("Video listing is not supported by the OpenRouter video API")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        raise NotImplementedError("Video delete/cancel is not supported by the OpenRouter video API")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        raise NotImplementedError("Video delete/cancel is not supported by the OpenRouter video API")

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
