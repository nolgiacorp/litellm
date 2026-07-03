import base64
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Optional

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.constants import DEFAULT_GOOGLE_VIDEO_DURATION_SECONDS
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.interactions import InteractionsAPIResponse
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    encode_video_id_with_provider,
    extract_original_video_id,
)

from .transformation import fetch_image_as_base64

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    from ...base_llm.chat.transformation import BaseLLMException as _BaseLLMException

    LiteLLMLoggingObj = _LiteLLMLoggingObj
    BaseLLMException = _BaseLLMException
else:
    LiteLLMLoggingObj = Any
    BaseLLMException = Any

INTERACTIONS_API_REVISION = "2026-05-20"

_OPENAI_VIDEO_SIZE_TO_ASPECT_RATIO: Mapping[str, str] = {
    "1280x720": "16:9",
    "1920x1080": "16:9",
    "720x1280": "9:16",
    "1080x1920": "9:16",
}

_SUPPORTED_ASPECT_RATIOS = frozenset({"16:9", "9:16"})

_TERMINAL_FAILURE_STATUSES = frozenset({"failed", "cancelled", "incomplete", "budget_exceeded"})


def _map_interaction_status(status: Optional[str]) -> str:
    if status == "completed":
        return "completed"
    if status in _TERMINAL_FAILURE_STATUSES:
        return "failed"
    return "processing"


def _find_video_part(interaction: InteractionsAPIResponse) -> Optional[dict[str, Any]]:
    steps = interaction.steps or interaction.outputs or []
    for step in reversed(steps):
        if step.get("type") != "model_output":
            continue
        for part in step.get("content") or []:
            if part.get("type") == "video":
                return part
    return None


class GeminiOmniVideoConfig(BaseVideoConfig):
    """
    Video generation for Gemini Omni models (e.g. gemini-omni-flash-preview).

    Unlike Veo, Omni models generate video through the Interactions API:
    1. POST /v1beta/interactions with background=true returns an interaction id
    2. Poll GET /v1beta/interactions/{id} until status is terminal
    3. The completed interaction carries the video inline as base64 in the
       model_output step (or as a files URI for large outputs)

    Omni has no explicit duration/negative-prompt parameters; per Google's
    prompt guide both are expressed in the prompt text, which is what
    transform_video_create_request does with ``seconds`` and ``negative_prompt``.
    """

    def get_supported_openai_params(self, model: str) -> list:
        return ["model", "prompt", "seconds", "size"]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, Any]:
        mapped_params: dict[str, Any] = {}

        size = video_create_optional_params.get("size")
        if size:
            aspect_ratio = _OPENAI_VIDEO_SIZE_TO_ASPECT_RATIO.get(size)
            if aspect_ratio:
                mapped_params["aspect_ratio"] = aspect_ratio

        for key, value in video_create_optional_params.items():
            if key not in {"model", "prompt", "size"} and key not in mapped_params:
                mapped_params[key] = value

        return mapped_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        litellm_params: Optional[GenericLiteLLMParams] = None,
    ) -> dict:
        if litellm_params and litellm_params.api_key:
            api_key = api_key or litellm_params.api_key

        api_key = api_key or litellm.api_key or get_secret_str("GOOGLE_API_KEY") or get_secret_str("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini Omni video generation. "
                "Set it via environment variable or pass it as api_key parameter."
            )

        headers.update(
            {
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
                "Api-Revision": INTERACTIONS_API_REVISION,
            }
        )
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        if api_base is None:
            api_base = get_secret_str("GEMINI_API_BASE") or "https://generativelanguage.googleapis.com"

        if not model or model == "":
            return api_base.rstrip("/")

        return f"{api_base.rstrip('/')}/v1beta/interactions"

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[dict, RequestFiles, str]:
        params = video_create_optional_request_params

        seconds = params.get("seconds") or params.get("duration_seconds")
        negative_prompt = params.get("negative_prompt")
        aspect_ratio = params.get("aspect_ratio")
        image_url = params.get("image_url")

        prompt_parts: list[str] = [prompt]
        if seconds:
            prompt_parts.append(f"The video must be exactly {seconds} seconds long.")
        if negative_prompt:
            prompt_parts.append(f"Do not include: {negative_prompt}.")
        full_prompt = " ".join(prompt_parts)

        response_format: dict[str, Any] = {"type": "video"}
        if aspect_ratio in _SUPPORTED_ASPECT_RATIOS:
            response_format["aspect_ratio"] = aspect_ratio

        request_data: dict[str, Any] = {
            "model": model.replace("gemini/", ""),
            "input": full_prompt,
            "response_format": response_format,
            "background": True,
        }

        if image_url:
            base64_data, mime_type = fetch_image_as_base64(image_url)
            request_data["input"] = [
                {"type": "image", "data": base64_data, "mime_type": mime_type},
                {"type": "text", "text": full_prompt},
            ]
            request_data["generation_config"] = {"video_config": {"task": "image_to_video"}}

        return request_data, [], api_base

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
        request_data: Optional[dict] = None,
    ) -> VideoObject:
        try:
            raw_json = raw_response.json()
        except ValueError as e:
            raise ValueError(f"Failed to parse interaction response: {e}")
        interaction = InteractionsAPIResponse(**raw_json)

        interaction_id = interaction.id
        if not interaction_id:
            raise ValueError(f"No interaction id in Gemini Omni response: {raw_response.text}")

        if custom_llm_provider:
            video_id = encode_video_id_with_provider(interaction_id, custom_llm_provider, model)
        else:
            video_id = interaction_id

        video_obj = VideoObject(
            id=video_id,
            object="video",
            status=_map_interaction_status(interaction.status),
            model=model,
        )
        video_obj.usage = {
            "duration_seconds": float(DEFAULT_GOOGLE_VIDEO_DURATION_SECONDS),
            "video_resolution": "720p",
        }
        return video_obj

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        interaction_id = extract_original_video_id(video_id)
        url = f"{api_base.rstrip('/')}/v1beta/interactions/{interaction_id}"
        return url, {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        response_json = raw_response.json()
        interaction = InteractionsAPIResponse(**response_json)

        interaction_id = interaction.id or ""
        if custom_llm_provider:
            video_id = encode_video_id_with_provider(interaction_id, custom_llm_provider, None)
        else:
            video_id = interaction_id

        status = _map_interaction_status(interaction.status)

        error_data: Optional[dict[str, Any]] = None
        if status == "failed":
            error_data = response_json.get("error") or {
                "code": "interaction_" + (interaction.status or "failed"),
                "message": f"Gemini Omni video generation ended with status {interaction.status!r}.",
            }

        return VideoObject(
            id=video_id,
            object="video",
            status=status,
            error=error_data,
        )

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        variant: Optional[str] = None,
    ) -> tuple[str, dict]:
        interaction_id = extract_original_video_id(video_id)
        url = f"{api_base.rstrip('/')}/v1beta/interactions/{interaction_id}"
        return url, {}

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        interaction = InteractionsAPIResponse(**raw_response.json())

        status = _map_interaction_status(interaction.status)
        if status == "processing":
            raise ValueError(
                "Video generation is not complete yet. Please check status with video_status() before downloading."
            )
        if status == "failed":
            raise ValueError(f"Gemini Omni video generation ended with status {interaction.status!r}.")

        video_part = _find_video_part(interaction)
        if video_part is None:
            raise ValueError("No video output in completed interaction.")

        inline_data = video_part.get("data")
        if inline_data:
            return base64.b64decode(inline_data)

        uri = video_part.get("uri")
        if uri:
            download_headers: dict[str, str] = {}
            api_key = raw_response.request.headers.get("x-goog-api-key")
            if api_key:
                download_headers["x-goog-api-key"] = api_key
            download_response = litellm.module_level_client.get(url=uri, headers=download_headers)
            download_response.raise_for_status()
            return download_response.content

        raise ValueError("Video output has neither inline data nor a download URI.")

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        extra_body: Optional[dict[str, Any]] = None,
    ) -> tuple[str, dict]:
        raise NotImplementedError("Video remix is not supported for Gemini Omni via the videos API.")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        raise NotImplementedError("Video remix is not supported for Gemini Omni via the videos API.")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        extra_query: Optional[dict[str, Any]] = None,
    ) -> tuple[str, dict]:
        raise NotImplementedError("Video list is not supported for Gemini Omni.")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> dict[str, str]:
        raise NotImplementedError("Video list is not supported for Gemini Omni.")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        raise NotImplementedError("Video delete is not supported for Gemini Omni.")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        raise NotImplementedError("Video delete is not supported for Gemini Omni.")

    def transform_video_create_character_request(self, name, video, api_base, litellm_params, headers):
        raise NotImplementedError("video create character is not supported for Gemini Omni")

    def transform_video_create_character_response(self, raw_response, logging_obj):
        raise NotImplementedError("video create character is not supported for Gemini Omni")

    def transform_video_get_character_request(self, character_id, api_base, litellm_params, headers):
        raise NotImplementedError("video get character is not supported for Gemini Omni")

    def transform_video_get_character_response(self, raw_response, logging_obj):
        raise NotImplementedError("video get character is not supported for Gemini Omni")

    def transform_video_edit_request(
        self,
        prompt,
        video_id,
        api_base,
        litellm_params,
        headers,
        extra_body=None,
        prefetched_source_data=None,
    ):
        raise NotImplementedError("video edit is not supported for Gemini Omni via the videos API")

    def transform_video_edit_response(
        self,
        raw_response,
        logging_obj,
        custom_llm_provider=None,
        request_data=None,
    ):
        raise NotImplementedError("video edit is not supported for Gemini Omni via the videos API")

    def transform_video_extension_request(
        self,
        prompt,
        video_id,
        seconds,
        api_base,
        litellm_params,
        headers,
        extra_body=None,
    ):
        raise NotImplementedError("video extension is not supported for Gemini Omni")

    def transform_video_extension_response(self, raw_response, logging_obj, custom_llm_provider=None):
        raise NotImplementedError("video extension is not supported for Gemini Omni")

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        from ..common_utils import GeminiError

        return GeminiError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
