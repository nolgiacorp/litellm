import base64
from collections.abc import Mapping
from json import JSONDecodeError
from types import MappingProxyType
from typing import TYPE_CHECKING, Any  # noqa: TID251  # base transformation contracts type these payloads as Any
from urllib.parse import quote

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.litellm_core_utils.prompt_templates.common_utils import extract_file_data
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.llms.minimax.common_utils import (
    EMPTY_MAP,
    drop_none_values,
    minimax_bearer_headers,
    resolve_minimax_media_api_base,
    strip_minimax_prefix,
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

MINIMAX_V2_VIDEO_STATUS_MAP: Mapping[str, str] = MappingProxyType(
    {  # mutable-ok: frozen constant lookup table
        "queued": "queued",
        "running": "in_progress",
        "succeeded": "completed",
        "failed": "failed",
        "cancelled": "failed",
        "expired": "failed",
    }
)

MINIMAX_V1_VIDEO_STATUS_MAP: Mapping[str, str] = MappingProxyType(
    {  # mutable-ok: frozen constant lookup table
        "preparing": "queued",
        "queueing": "queued",
        "processing": "in_progress",
        "success": "completed",
        "fail": "failed",
    }
)

_V1_ERROR_HTTP_STATUS: Mapping[int, int] = MappingProxyType(
    {1002: 429, 1004: 401, 1008: 402, 2049: 401}  # mutable-ok: frozen constant lookup table
)

_V1_ID_MARKER = "MiniMax-Hailuo"
_V2_ID_MARKER = "MiniMax-H3"

_V2_RATIOS = frozenset({"21:9", "16:9", "4:3", "1:1", "3:4", "9:16"})
_DEFAULT_T2V_RATIO = "16:9"
_DEFAULT_V2_DURATION = 6
_DEFAULT_V2_RESOLUTION = "2K"

# Regeneration is a separate endpoint with a narrower contract: role="base_video" is
# absent from /v2/video_generation's role enum (which is why it answered 2013 invalid
# params), resolution is an enum of one, and there is no duration field at all because
# the output inherits the source video's length.
_GENERATION_PATH = "/v2/video_generation"
_REGENERATION_PATH = "/v2/video_regeneration"
_REGENERATION_RESOLUTION = "2K"
# resolution is not omitted: _map_v2_params already pins it to 2K for regeneration,
# refusing anything else rather than silently upgrading it. duration is omitted from the
# body but not from billing: the declared source length is what the create call charges.
# ratio never reaches here once _map_v2_params refuses an explicit one, and stays listed
# so a raw extra_body ratio cannot slip into a body that has no such field.
_REGENERATION_UNSUPPORTED_KEYS = frozenset(("duration", "ratio"))
# An aspect ratio asked for by any name; regeneration inherits the source video's.
_REGENERATION_RATIO_KEYS = ("size", "aspect_ratio", "ratio")

_V2_MEDIA_KEYS = frozenset(
    {"first_frame", "last_frame", "reference_images", "reference_videos", "reference_audios", "base_video"}
)

# OpenAI-shaped aliases this config translates into MiniMax fields. The shared video handler merges raw
# extra_body over the mapped params, so these have to be stripped again before the request goes out.
_CONSUMED_ALIAS_KEYS = frozenset(
    {
        "seconds",
        "size",
        "aspect_ratio",
        "input_reference",
        "image_url",
        "end_image_url",
        "image_urls",
        "video_urls",
        "audio_urls",
        "base_video_url",
        "extra_body",
    }
)

_SIZE_TO_ASPECT_RATIO: Mapping[str, str] = MappingProxyType(
    {  # mutable-ok: frozen constant lookup table
        "1280x720": "16:9",
        "1920x1080": "16:9",
        "720x1280": "9:16",
        "1080x1920": "9:16",
        "1024x1024": "1:1",
        "1080x1080": "1:1",
        "1024x768": "4:3",
        "768x1024": "3:4",
    }
)


def _uses_legacy_video_api(model: str | None) -> bool:
    return "hailuo" in (model or "").lower()


def _coerce_media_url(value: FileTypes | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    extracted = extract_file_data(value)
    content_type = extracted.get("content_type") or "application/octet-stream"
    encoded = base64.b64encode(extracted["content"]).decode("utf-8")
    return f"data:{content_type};base64,{encoded}"


def _url_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    if isinstance(value, list):
        return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    return ()


def _safe_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _duration_usage(value: float | None) -> dict:  # mutable-ok: VideoObject.usage expects a dict
    return {"duration_seconds": value} if value is not None else {}


_LEGACY_CAPABILITY_PARAMS = frozenset(
    (
        "input_reference",
        "image_url",
    )
)

_V2_CAPABILITY_PARAMS = frozenset(
    (
        "input_reference",
        "image_url",
        "end_image_url",
        "image_urls",
        "audio_urls",
        "base_video_url",
    )
)


class MinimaxVideoConfig(BaseVideoConfig):
    def __init__(self) -> None:
        super().__init__()
        # Regeneration's body carries no duration, so the source length the caller declared
        # is held here between this request's transform and its response transform. A config
        # instance is built per call (ProviderConfigManager.get_provider_video_config), so
        # this never spans two requests.
        self._regeneration_seconds: Any = None

    def get_capability_param_support(self, model: str) -> CapabilityParamSupport:
        """
        Hailuo 3 (/v2) executes first/last frame conditioning, reference images and
        reference audio, plus base_video regeneration. Legacy Hailuo (/v1) has only a
        first_frame_image slot.

        Neither declares video_urls. H3 does recognize the param, but only to refuse
        it: a reference clip's length is unknown when the create call is billed, so
        accepting one would undercharge. Declaring it here would let the catalog
        advertise a reference-video surface no MiniMax route can serve.

        generate_audio is not declared on either: MiniMax audio is native and has no
        switch, so a flag would be discarded.

        negative_prompt is not declared on either. The legacy body is model, prompt,
        prompt_optimizer, fast_pretreatment, duration, resolution, callback_url and
        first_frame_image; the /v2 body is model, content, resolution, duration,
        ratio and callback_url, where prompt text is a single content item with no
        second negative channel. Folding the exclusion into the prompt would be a
        different feature, not support, so it is refused rather than guessed at.
        """
        return DeclaredCapabilityParams(
            _LEGACY_CAPABILITY_PARAMS if _uses_legacy_video_api(model) else _V2_CAPABILITY_PARAMS
        )

    def get_supported_openai_params(self, model: str) -> list:  # mutable-ok: BaseVideoConfig contract returns list
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
    ) -> dict:  # mutable-ok: BaseVideoConfig contract returns dict
        extra_body = video_create_optional_params.get("extra_body")
        params = {
            **video_create_optional_params,
            **(extra_body if isinstance(extra_body, dict) else EMPTY_MAP),
        }
        duration = self._parse_duration(params)
        resolution = params.get("resolution")
        image_url = params.get("image_url")
        first_frame = _coerce_media_url(params.get("input_reference")) or _coerce_media_url(
            image_url if isinstance(image_url, str) else None
        )
        if _uses_legacy_video_api(model):
            if params.get("base_video_url") is not None:
                raise litellm.BadRequestError(
                    message=(
                        "MiniMax base_video regeneration is a /v2 (Hailuo 3) flow; legacy Hailuo models do not "
                        "support base_video_url."
                    ),
                    model=model,
                    llm_provider=litellm.LlmProviders.MINIMAX.value,
                )
            return dict(self._map_legacy_params(params, duration, resolution, first_frame))
        return dict(
            self._map_v2_params(
                model=model,
                params=params,
                duration=duration,
                resolution=resolution,
                first_frame=first_frame,
            )
        )

    @staticmethod
    def _parse_duration(params: Mapping[str, Any]) -> int | None:
        raw_duration = params.get("seconds") if params.get("seconds") is not None else params.get("duration")
        try:
            return int(float(raw_duration)) if raw_duration is not None else None
        except (TypeError, ValueError):
            raise ValueError(f"Unsupported MiniMax video duration '{raw_duration}'; expected a number of seconds.")

    @staticmethod
    def _map_legacy_params(
        params: Mapping[str, Any],
        duration: int | None,
        resolution: Any,
        first_frame: str | None,
    ) -> Mapping[str, Any]:
        prompt_optimizer = params.get("prompt_optimizer")
        return drop_none_values(
            {
                "duration": duration,
                "resolution": str(resolution).upper() if resolution else None,
                "first_frame_image": first_frame,
                "prompt_optimizer": prompt_optimizer if isinstance(prompt_optimizer, bool) else None,
            }
        )

    def _map_v2_params(
        self,
        model: str,
        params: Mapping[str, Any],
        duration: int | None,
        resolution: Any,
        first_frame: str | None,
    ) -> Mapping[str, Any]:
        end_image_url = params.get("end_image_url")
        last_frame = end_image_url.strip() if isinstance(end_image_url, str) and end_image_url.strip() else None
        reference_images = _url_tuple(params.get("image_urls"))
        reference_videos = _url_tuple(params.get("video_urls"))
        reference_audios = _url_tuple(params.get("audio_urls"))
        base_video = self._v2_base_video(model, params)
        has_frames = bool(first_frame or last_frame)
        has_references = bool(reference_images or reference_videos or reference_audios)
        if last_frame and not first_frame:
            raise litellm.BadRequestError(
                message=(
                    "MiniMax H3 requires a first frame (input_reference) when an end frame (end_image_url) is "
                    "provided; a last_frame item cannot be sent alone."
                ),
                model=model,
                llm_provider=litellm.LlmProviders.MINIMAX.value,
            )
        if has_frames and has_references:
            raise litellm.BadRequestError(
                message=(
                    "MiniMax H3 first/last-frame conditioning (input_reference/end_image_url) and reference media "
                    "(image_urls/video_urls/audio_urls) are mutually exclusive; send one or the other."
                ),
                model=model,
                llm_provider=litellm.LlmProviders.MINIMAX.value,
            )
        if reference_videos:
            raise litellm.BadRequestError(
                message=(
                    "MiniMax H3 bills reference video input seconds (usage.input_seconds) on top of the generated "
                    "output seconds, and a reference clip's length is unknown when the create call is charged, so "
                    "reference videos (video_urls) would be undercharged and are not supported. Use reference images "
                    "(image_urls) or first/last-frame conditioning instead. To re-render a known-length source video "
                    "at a higher resolution, use base_video_url (regeneration)."
                ),
                model=model,
                llm_provider=litellm.LlmProviders.MINIMAX.value,
            )
        if base_video is not None and resolution is not None and str(resolution).upper() != _REGENERATION_RESOLUTION:
            raise litellm.BadRequestError(
                message=(
                    f"MiniMax H3 regeneration only renders at {_REGENERATION_RESOLUTION}; it re-renders a 768P source "
                    f"video at higher resolution and its request has no other resolution to ask for. Requested "
                    f"'{resolution}'. Drop the resolution or set it to {_REGENERATION_RESOLUTION}."
                ),
                model=model,
                llm_provider=litellm.LlmProviders.MINIMAX.value,
            )
        if base_video is not None:
            self._reject_regeneration_ratio(model, params)
            if duration is None:
                raise litellm.BadRequestError(
                    message=(
                        "MiniMax H3 regeneration requires seconds: the source video's length. Its request has no "
                        "duration field, since the output inherits the source's length, so seconds does not change "
                        "the output; it is the only number available to price the regeneration, and without it the "
                        "created video would be billed nothing."
                    ),
                    model=model,
                    llm_provider=litellm.LlmProviders.MINIMAX.value,
                )
        return drop_none_values(
            {
                "duration": duration if duration is not None else _DEFAULT_V2_DURATION,
                "resolution": str(resolution).upper() if resolution else _DEFAULT_V2_RESOLUTION,
                "ratio": self._v2_ratio(
                    params, has_frames=has_frames, has_references=has_references or base_video is not None
                ),
                "first_frame": first_frame,
                "last_frame": last_frame,
                "reference_images": reference_images or None,
                "reference_videos": reference_videos or None,
                "reference_audios": reference_audios or None,
                "base_video": (base_video,) if base_video else None,
            }
        )

    @staticmethod
    def _reject_regeneration_ratio(model: str, params: Mapping[str, Any]) -> None:
        requested = tuple(key for key in _REGENERATION_RATIO_KEYS if params.get(key) is not None)
        if not requested:
            return
        raise litellm.BadRequestError(
            message=(
                "MiniMax H3 regeneration re-renders the source video, so the output keeps the source's aspect ratio "
                f"and its request has no ratio field to ask for another. Requested {', '.join(requested)} cannot be "
                "honored; drop it, or crop the source video before regenerating."
            ),
            model=model,
            llm_provider=litellm.LlmProviders.MINIMAX.value,
        )

    @staticmethod
    def _v2_base_video(model: str, params: Mapping[str, Any]) -> str | None:
        raw = params.get("base_video_url")
        if raw is None:
            return None
        url = raw.strip() if isinstance(raw, str) else None
        if not url:
            raise litellm.BadRequestError(
                message=(
                    "MiniMax H3 regeneration requires base_video_url to be a single non-empty video URL pointing at "
                    "the source video to re-render."
                ),
                model=model,
                llm_provider=litellm.LlmProviders.MINIMAX.value,
            )
        return url

    @staticmethod
    def _v2_ratio(params: Mapping[str, Any], has_frames: bool, has_references: bool) -> str | None:
        size = params.get("size")
        ratio = (
            _SIZE_TO_ASPECT_RATIO.get(size) or (size.replace("x", ":") if "x" in size else None)
            if isinstance(size, str) and size
            else params.get("aspect_ratio")
        )
        if has_frames:
            return None
        if has_references:
            return ratio if isinstance(ratio, str) and ratio else None
        return ratio if ratio in _V2_RATIOS else _DEFAULT_T2V_RATIO

    def validate_environment(
        self,
        headers: Mapping[str, Any],
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict:  # mutable-ok: BaseVideoConfig contract returns dict
        if litellm_params and litellm_params.api_key:
            api_key = api_key or litellm_params.api_key
        return minimax_bearer_headers(headers, api_key)

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: Mapping[str, Any],
    ) -> str:
        return resolve_minimax_media_api_base(api_base)

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Mapping[str, Any],
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, Any],
    ) -> tuple[dict, RequestFiles, str]:  # mutable-ok: BaseVideoConfig contract returns dict body
        mapped = {
            key: value
            for key, value in video_create_optional_request_params.items()
            if key != "model" and key not in _CONSUMED_ALIAS_KEYS
        }
        model_name = strip_minimax_prefix(model)
        if _uses_legacy_video_api(model_name):
            request_data = dict(drop_none_values({"model": model_name, "prompt": prompt, **mapped}))
            return request_data, (), f"{api_base}/v1/video_generation"

        content = (
            {"type": "text", "text": prompt},
            *self._frame_items(mapped),
            *self._reference_items(mapped),
        )
        is_regeneration = bool(mapped.get("base_video"))
        self._regeneration_seconds = mapped.get("duration") if is_regeneration else None
        omitted = _V2_MEDIA_KEYS | (_REGENERATION_UNSUPPORTED_KEYS if is_regeneration else frozenset())
        passthrough = drop_none_values({key: value for key, value in mapped.items() if key not in omitted})
        request_data = {"model": model_name, "content": content, **passthrough}
        return request_data, (), f"{api_base}{_REGENERATION_PATH if is_regeneration else _GENERATION_PATH}"

    @staticmethod
    def _frame_items(mapped: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
        frames = (("first_frame", mapped.get("first_frame")), ("last_frame", mapped.get("last_frame")))
        return tuple(
            {"type": "image_url", "image_url": {"url": url}, "role": role}
            for role, url in frames
            if isinstance(url, str) and url
        )

    @staticmethod
    def _reference_items(mapped: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
        groups = (
            ("reference_images", "image_url", "reference_image"),
            ("reference_videos", "video_url", "reference_video"),
            ("reference_audios", "audio_url", "reference_audio"),
            ("base_video", "video_url", "base_video"),
        )
        return tuple(
            {"type": type_key, type_key: {"url": url}, "role": role}
            for key, type_key, role in groups
            for url in (mapped.get(key) or ())
            if isinstance(url, str) and url
        )

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: Mapping[str, Any] | None = None,
    ) -> VideoObject:
        self._raise_for_status(raw_response)
        response_data = raw_response.json()
        model_name = strip_minimax_prefix(model)
        if _uses_legacy_video_api(model_name):
            self._raise_for_base_resp_error(raw_response, response_data)
        task_id = response_data.get("task_id")
        if not task_id:
            raise ValueError(f"MiniMax video submit response is missing task_id: {response_data}")

        raw_duration = self._billed_duration(request_data)
        video_obj = VideoObject(
            id=str(task_id),
            object="video",
            status="queued",
            model=model,
            seconds=str(raw_duration) if raw_duration is not None else None,
            usage=_duration_usage(_safe_float(raw_duration)),
        )
        if custom_llm_provider:
            video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, model_name)
        return video_obj

    def _billed_duration(self, request_data: Mapping[str, Any] | None) -> Any:
        """
        The seconds the create call reports and bills.

        Regeneration has to read it off the config instead of the request: its
        endpoint has no duration field, so the declared source length was stripped
        from the body. Create time is the only point where video spend is priced,
        because polling is logged as CallTypes.video_retrieve and the video cost
        calculator prices create, edit and remix alone, so usage on the status
        response never becomes spend. Reporting nothing here would make every
        regeneration free rather than merely mispriced.
        """
        if self._regeneration_seconds is not None:
            return self._regeneration_seconds
        return request_data.get("duration") if request_data else None

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, Any],
    ) -> tuple[str, dict]:  # mutable-ok: BaseVideoConfig contract returns dict params
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
        if self._is_legacy_response(raw_response):
            return self._legacy_status_video_object(raw_response, response_data, custom_llm_provider)
        return self._v2_status_video_object(response_data, custom_llm_provider)

    def _legacy_status_video_object(
        self,
        raw_response: httpx.Response,
        response_data: Mapping[str, Any],
        custom_llm_provider: str | None,
    ) -> VideoObject:
        raw_status = str(response_data.get("status", "")).lower()
        if not raw_status:
            self._raise_for_base_resp_error(raw_response, response_data)
        status = MINIMAX_V1_VIDEO_STATUS_MAP.get(raw_status, "in_progress")
        base_resp = response_data.get("base_resp") or EMPTY_MAP
        video_obj = VideoObject(
            id=response_data.get("task_id", ""),
            object="video",
            status=status,
            error=(
                {"code": raw_status, "message": str(base_resp.get("status_msg") or "Video generation failed")}
                if status == "failed"
                else None
            ),
        )
        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, _V1_ID_MARKER)
        return video_obj

    @staticmethod
    def _v2_status_video_object(response_data: Mapping[str, Any], custom_llm_provider: str | None) -> VideoObject:
        task = response_data.get("task") or EMPTY_MAP
        raw_status = str(task.get("status", "")).lower()
        status = MINIMAX_V2_VIDEO_STATUS_MAP.get(raw_status, "in_progress")

        task_usage = task.get("usage") or EMPTY_MAP
        billed_seconds = _safe_float(task_usage.get("total_seconds") or task.get("duration"))
        task_error = task.get("error") or EMPTY_MAP
        video_obj = VideoObject(
            id=task.get("id", ""),
            object="video",
            status=status,
            usage=_duration_usage(billed_seconds),
            error=(
                {
                    "code": str(task_error.get("code") or raw_status or "failed"),
                    "message": str(task_error.get("message") or f"Video generation {raw_status or 'failed'}"),
                }
                if status == "failed"
                else None
            ),
        )
        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, _V2_ID_MARKER)
        return video_obj

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, Any],
        variant: str | None = None,
    ) -> tuple[str, dict]:  # mutable-ok: BaseVideoConfig contract returns dict params
        return self._build_task_url(video_id, api_base), {}

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        self._raise_for_status(raw_response)
        response_data = raw_response.json()
        httpx_client: HTTPHandler = _get_httpx_client()
        if self._is_legacy_response(raw_response):
            retrieve_url, retrieve_headers = self._legacy_file_retrieve_request(raw_response, response_data)
            retrieve_response = httpx_client.get(retrieve_url, headers=dict(retrieve_headers))
            retrieve_response.raise_for_status()
            download_url = self._legacy_extract_download_url(retrieve_response.json())
        else:
            download_url = self._extract_v2_video_url(response_data)
        video_response = httpx_client.get(download_url)
        video_response.raise_for_status()
        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        self._raise_for_status(raw_response)
        response_data = raw_response.json()
        async_client: AsyncHTTPHandler = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.MINIMAX,
        )
        if self._is_legacy_response(raw_response):
            retrieve_url, retrieve_headers = self._legacy_file_retrieve_request(raw_response, response_data)
            retrieve_response = await async_client.get(retrieve_url, headers=dict(retrieve_headers))
            retrieve_response.raise_for_status()
            download_url = self._legacy_extract_download_url(retrieve_response.json())
        else:
            download_url = self._extract_v2_video_url(response_data)
        video_response = await async_client.get(download_url)
        video_response.raise_for_status()
        return video_response.content

    def _build_task_url(self, video_id: str, api_base: str) -> str:
        task_id, model_name = self._decode_task(video_id)
        if _uses_legacy_video_api(model_name):
            return f"{api_base}/v1/query/video_generation?task_id={quote(task_id, safe='')}"
        encoded = encode_url_path_segment(task_id, field_name="video_id")
        return f"{api_base}/v2/query/video_generation/{encoded}"

    @staticmethod
    def _decode_task(video_id: str) -> tuple[str, str]:
        decoded = decode_video_id_with_provider(video_id)
        task_id = decoded.get("video_id") or extract_original_video_id(video_id)
        return task_id, decoded.get("model_id") or ""

    @staticmethod
    def _is_legacy_response(raw_response: httpx.Response) -> bool:
        request = getattr(raw_response, "request", None)
        return request is not None and request.url.path.startswith("/v1/")

    def _legacy_file_retrieve_request(
        self,
        raw_response: httpx.Response,
        response_data: Mapping[str, Any],
    ) -> tuple[str, Mapping[str, str]]:
        raw_status = str(response_data.get("status", "")).lower()
        if not raw_status:
            self._raise_for_base_resp_error(raw_response, response_data)
        if raw_status == "fail":
            base_resp = response_data.get("base_resp") or EMPTY_MAP
            raise ValueError(f"MiniMax video generation failed: {base_resp.get('status_msg') or response_data}")
        file_id = response_data.get("file_id")
        if not file_id:
            raise ValueError("Video file_id not found in MiniMax response. The job may still be processing.")
        api_base = str(raw_response.request.url).split("?")[0].rsplit("/v1/", 1)[0]
        url = f"{api_base}/v1/files/retrieve?file_id={quote(str(file_id), safe='')}"
        return url, {"Authorization": raw_response.request.headers.get("Authorization", "")}

    @staticmethod
    def _legacy_extract_download_url(retrieve_data: Mapping[str, Any]) -> str:
        base_resp = retrieve_data.get("base_resp") or EMPTY_MAP
        if base_resp.get("status_code") not in (None, 0):
            raise ValueError(f"MiniMax file retrieve failed: {base_resp.get('status_msg') or retrieve_data}")
        download_url = (retrieve_data.get("file") or EMPTY_MAP).get("download_url")
        if isinstance(download_url, str) and download_url:
            return download_url
        raise ValueError(f"MiniMax file retrieve response is missing download_url: {retrieve_data}")

    @staticmethod
    def _extract_v2_video_url(response_data: Mapping[str, Any]) -> str:
        task = response_data.get("task") or EMPTY_MAP
        raw_status = str(task.get("status", "")).lower()
        if raw_status in ("failed", "cancelled", "expired"):
            task_error = task.get("error") or EMPTY_MAP
            raise ValueError(
                f"MiniMax video generation {raw_status}: {task_error.get('message') or task_error or raw_status}"
            )
        url = (task.get("content") or EMPTY_MAP).get("url")
        if isinstance(url, str) and url:
            return url
        raise ValueError("Video URL not found in MiniMax response. The job may still be processing.")

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, Any],
        extra_body: Mapping[str, Any] | None = None,
    ) -> tuple[str, dict]:  # mutable-ok: BaseVideoConfig contract returns dict params
        raise NotImplementedError("Video remix is not supported for MiniMax")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("Video remix is not supported for MiniMax")

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
        raise NotImplementedError("Video listing is not supported for MiniMax")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseVideoConfig contract returns dict
        raise NotImplementedError("Video listing is not supported for MiniMax")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, Any],
    ) -> tuple[str, dict]:  # mutable-ok: BaseVideoConfig contract returns dict params
        raise NotImplementedError("Video delete/cancel is not supported for MiniMax")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        raise NotImplementedError("Video delete/cancel is not supported for MiniMax")

    def _raise_for_status(self, raw_response: httpx.Response) -> None:
        if raw_response.is_success:
            return
        raise self.get_error_class(
            error_message=self._error_message_from_body(raw_response),
            status_code=raw_response.status_code,
            headers=raw_response.headers,
        )

    @staticmethod
    def _error_message_from_body(raw_response: httpx.Response) -> str:
        try:
            body = raw_response.json()
        except (ValueError, JSONDecodeError):
            return raw_response.text
        error = body.get("error") if isinstance(body, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        return message if isinstance(message, str) and message else raw_response.text

    def _raise_for_base_resp_error(self, raw_response: httpx.Response, response_data: Mapping[str, Any]) -> None:
        base_resp = response_data.get("base_resp") or EMPTY_MAP
        status_code = base_resp.get("status_code")
        if status_code in (None, 0):
            return
        message = base_resp.get("status_msg") or "MiniMax API returned an error"
        raise self.get_error_class(
            error_message=f"{message} ({status_code})",
            status_code=_V1_ERROR_HTTP_STATUS.get(status_code, 400),
            headers=raw_response.headers,
        )
