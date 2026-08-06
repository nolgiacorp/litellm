import time
from collections.abc import Mapping
from dataclasses import dataclass
from json import JSONDecodeError
from types import MappingProxyType
from typing import TYPE_CHECKING, Any  # noqa: TID251  # base video ABC + OpenAI video TypedDict are Any-typed

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.litellm_core_utils.prompt_templates.common_utils import extract_file_data
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.llms.topaz.common_utils import TopazException, TopazModelInfo
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import encode_video_id_with_provider, extract_original_video_id

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


_SUPPORTED_OPENAI_PARAMS = (
    "model",
    "prompt",
    "input_reference",
    "seconds",
    "size",
    "resolution",
    "user",
    "extra_headers",
    "extra_body",
)

_FILTER_PARAMS = frozenset(
    (
        "videoType",
        "auto",
        "fieldOrder",
        "focusFixLevel",
        "compression",
        "details",
        "prenoise",
        "noise",
        "halo",
        "preblur",
        "blur",
        "grain",
        "grainSize",
        "recoverOriginalDetailValue",
    )
)

_OUTPUT_PARAMS = frozenset(
    (
        "frameRate",
        "audioCodec",
        "audioTransfer",
        "videoEncoder",
        "videoProfile",
        "videoBitrate",
        "audiobitrate",
        "codecId",
        "cropToFit",
        "dynamicCompressionLevel",
    )
)

_CONSUMED_PARAMS = frozenset(
    (
        "model",
        "prompt",
        "user",
        "extra_headers",
        "extra_body",
        "input_reference",
        "seconds",
        "duration_seconds",
        "size",
        "resolution",
        "container",
    )
)

_CONTAINER_MIME: Mapping[str, str] = MappingProxyType(
    {  # mutable-ok: frozen constant lookup table
        "mp4": "video/mp4",
        "mov": "video/quicktime",
        "mkv": "video/x-matroska",
    }
)

TOPAZ_STATUS_MAP: Mapping[str, str] = MappingProxyType(
    {  # mutable-ok: frozen constant lookup table
        "requested": "queued",
        "accepted": "queued",
        "initializing": "in_progress",
        "preprocessing": "in_progress",
        "processing": "in_progress",
        "postprocessing": "in_progress",
        "canceling": "in_progress",
        "complete": "completed",
        "canceled": "failed",
        "failed": "failed",
    }
)

TOPAZ_TERMINAL_FAILURES = frozenset(("canceled", "failed"))

SOURCE_CONTAINERS = frozenset(("mp4", "mov", "mkv"))

RESOLUTION_ALIASES: Mapping[str, tuple[int, int]] = MappingProxyType(
    {  # mutable-ok: frozen constant lookup table
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "1440p": (2560, 1440),
        "2160p": (3840, 2160),
        "4k": (3840, 2160),
        "4320p": (7680, 4320),
        "8k": (7680, 4320),
    }
)

UPSCALE_MODEL_CODES = frozenset(
    (
        "aaa-9",
        "aaa-10",
        "ahq-12",
        "aion-1",
        "alq-13",
        "alqs-2",
        "amq-13",
        "amqs-2",
        "color-1",
        "ddv-3",
        "dtd-4",
        "dtds-2",
        "dtv-4",
        "dtvs-2",
        "ganim-1",
        "gcg-5",
        "ghq-5",
        "hyp-1",
        "hyp-2",
        "iris-2",
        "iris-3",
        "nxf-1",
        "nxl-1",
        "nyx-3",
        "pnat-1",
        "prob-4",
        "rhea-1",
        "sl-1",
        "slf-1",
        "slf-2",
        "slhq-1",
        "slm-1",
        "slp-2",
        "slp-2.5",
        "thd-3",
        "thf-4",
        "thm-2",
        "wonder-1",
    )
)


def resolve_topaz_api_base(api_base: str | None) -> str:
    base = TopazModelInfo.get_api_base(api_base) or "https://api.topazlabs.com"
    return base.rstrip("/")


def topaz_auth_headers(api_key: str | None) -> Mapping[str, str]:
    resolved = TopazModelInfo.get_api_key(api_key)
    if not resolved:
        raise ValueError("TOPAZ_API_KEY is not set")
    return {"X-API-Key": resolved, "Content-Type": "application/json"}


def strip_topaz_prefix(model: str) -> str:
    return model.split("/", 1)[1] if model.startswith("topaz/") else model


@dataclass(frozen=True, slots=True)
class _PendingUpload:
    source: object
    container: str
    seconds: object


def _safe_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None  # pyright: ignore[reportArgumentType]  # guarded by except
    except (TypeError, ValueError):
        return None


def _billed_credits(estimates: object) -> int | None:
    if not isinstance(estimates, Mapping):
        return None
    cost = estimates.get("cost")
    if not isinstance(cost, (list, tuple)) or not cost:
        return None
    lower = _safe_float(cost[0])
    return int(lower) if lower is not None else None


class TopazVideoConfig(BaseVideoConfig):
    """
    Topaz Labs video enhancement is a create -> upload -> poll -> download API, and it is an
    upscaler rather than a generator: it takes mandatory source footage and no prompt.

    POST /video/express returns {requestId, uploadId, uploadUrls}; the source bytes are then
    PUT to the single presigned upload URL, which is what starts processing. Because the
    express create and the byte upload are two separate HTTP calls but LiteLLM's create leg
    issues exactly one, the upload is performed in the create RESPONSE transform, where the
    presigned URL first becomes known. The source to relay is captured off the request
    transform on this per-request config instance.

    GET /video/{requestId}/status carries the status, the credit estimate and, once complete,
    a signed download URL. Topaz bills the LOWER bound of estimates.cost, which is surfaced on
    the status object as usage.topaz_credits so callers can reconcile real COGS.
    """

    def __init__(
        self,
        sync_client: HTTPHandler | None = None,
        async_client: AsyncHTTPHandler | None = None,
    ) -> None:
        super().__init__()
        self._sync_client = sync_client
        self._async_client = async_client
        self._pending_upload: _PendingUpload | None = None

    def _http_client(self) -> HTTPHandler:
        return self._sync_client or _get_httpx_client()

    def _async_http_client(self) -> AsyncHTTPHandler:
        return self._async_client or get_async_httpx_client(llm_provider=litellm.LlmProviders.TOPAZ)

    def get_supported_openai_params(self, model: str) -> list:  # mutable-ok: BaseVideoConfig contract returns list
        return list(_SUPPORTED_OPENAI_PARAMS)  # mutable-ok: BaseVideoConfig contract returns list

    def supports_promptless_video_create(self, model: str) -> bool:
        return True

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:  # mutable-ok: BaseVideoConfig contract returns dict
        params = self._merged_params(video_create_optional_params)
        self._reject_unsupported(params, model)
        seconds = params.get("seconds") if params.get("seconds") is not None else params.get("duration_seconds")
        width, height = self._resolution(params, model)
        carried = tuple((key, value) for key, value in params.items() if key in _FILTER_PARAMS or key in _OUTPUT_PARAMS)
        mapped = (
            ("input_reference", params.get("input_reference")),
            ("container", self._container(params, model)),
            ("resolution", f"{width}x{height}"),
            ("seconds", seconds),
        )
        return {  # mutable-ok: BaseVideoConfig contract returns dict
            key: value for key, value in (*mapped, *carried) if value is not None
        }

    @staticmethod
    def _merged_params(params: Mapping[str, Any]) -> Mapping[str, Any]:
        extra_body = params.get("extra_body")
        if not isinstance(extra_body, Mapping):
            return params
        return {**params, **extra_body}

    @staticmethod
    def _reject_unsupported(params: Mapping[str, Any], model: str) -> None:
        unsupported = tuple(
            key
            for key, value in params.items()
            if value is not None
            and key not in _CONSUMED_PARAMS
            and key not in _FILTER_PARAMS
            and key not in _OUTPUT_PARAMS
        )
        if not unsupported:
            return
        raise litellm.BadRequestError(
            message=(
                f"Topaz model '{model}' does not support the following parameters: {', '.join(sorted(unsupported))}. "
                "Topaz is a footage upscaler: it takes source video plus an output resolution, and it accepts no "
                "prompt, seed, aspect ratio or audio controls. Silently dropping them would bill an enhancement "
                "that ignored the caller's request."
            ),
            model=model,
            llm_provider=litellm.LlmProviders.TOPAZ.value,
        )

    @classmethod
    def _resolution(cls, params: Mapping[str, Any], model: str) -> tuple[int, int]:
        requested = params.get("resolution") if params.get("resolution") is not None else params.get("size")
        if requested is None:
            raise litellm.BadRequestError(
                message=(
                    f"Topaz model '{model}' requires a target output resolution. Pass `resolution` as one of "
                    f"{', '.join(sorted(RESOLUTION_ALIASES))} or as an explicit WIDTHxHEIGHT."
                ),
                model=model,
                llm_provider=litellm.LlmProviders.TOPAZ.value,
            )
        text = str(requested).strip().lower()
        alias = RESOLUTION_ALIASES.get(text)
        if alias is not None:
            return alias
        return cls._explicit_resolution(text, model)

    @staticmethod
    def _explicit_resolution(text: str, model: str) -> tuple[int, int]:
        width, _, height = text.partition("x")
        if width.isdigit() and height.isdigit():
            return int(width), int(height)
        raise litellm.BadRequestError(
            message=(
                f"Topaz model '{model}' received an unusable resolution {text!r}. Use one of "
                f"{', '.join(sorted(RESOLUTION_ALIASES))} or an explicit WIDTHxHEIGHT."
            ),
            model=model,
            llm_provider=litellm.LlmProviders.TOPAZ.value,
        )

    @staticmethod
    def _container(params: Mapping[str, Any], model: str) -> str:
        requested = params.get("container")
        if requested is None:
            return "mp4"
        container = str(requested).strip().lower()
        if container in SOURCE_CONTAINERS:
            return container
        raise litellm.BadRequestError(
            message=(
                f"Topaz model '{model}' received an unsupported source container {container!r}. "
                f"Topaz accepts {', '.join(sorted(SOURCE_CONTAINERS))}."
            ),
            model=model,
            llm_provider=litellm.LlmProviders.TOPAZ.value,
        )

    def validate_environment(
        self,
        headers: Mapping[str, Any],
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict:  # mutable-ok: BaseVideoConfig contract returns dict
        if litellm_params and litellm_params.api_key:
            api_key = api_key or litellm_params.api_key
        return {**headers, **topaz_auth_headers(api_key)}  # mutable-ok: BaseVideoConfig contract returns dict

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: Mapping[str, Any],
    ) -> str:
        return resolve_topaz_api_base(api_base)

    @staticmethod
    def _model_code(model: str) -> str:
        code = strip_topaz_prefix(model).strip().lower()
        if code in UPSCALE_MODEL_CODES:
            return code
        raise litellm.BadRequestError(
            message=(
                f"Unknown Topaz enhancement model {code!r}. Supported models: {', '.join(sorted(UPSCALE_MODEL_CODES))}."
            ),
            model=model,
            llm_provider=litellm.LlmProviders.TOPAZ.value,
        )

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Mapping[str, Any],
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, Any],
    ) -> tuple[dict, RequestFiles, str]:  # mutable-ok: BaseVideoConfig contract returns dict body
        params = video_create_optional_request_params
        source = params.get("input_reference")
        if source is None:
            raise litellm.BadRequestError(
                message=(
                    f"Topaz model '{model}' requires source footage. Pass the clip to enhance as `input_reference`; "
                    "Topaz upscales existing video and cannot generate from a prompt."
                ),
                model=model,
                llm_provider=litellm.LlmProviders.TOPAZ.value,
            )
        container = str(params.get("container") or "mp4")
        width, height = self._resolution(params, model)
        upscale_filter = {  # mutable-ok: request body fragment
            "model": self._model_code(model),
            **{key: value for key, value in params.items() if key in _FILTER_PARAMS},
        }
        output = {  # mutable-ok: request body fragment
            "resolution": {"width": width, "height": height},
            **{key: value for key, value in params.items() if key in _OUTPUT_PARAMS},
        }
        body = {  # mutable-ok: BaseVideoConfig contract returns dict body
            "source": {"container": container},
            "filters": [upscale_filter],
            "output": output,
        }
        self._pending_upload = _PendingUpload(source=source, container=container, seconds=params.get("seconds"))
        return body, (), f"{resolve_topaz_api_base(api_base)}/video/express"

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: Mapping[str, Any] | None = None,
    ) -> VideoObject:
        request_id, upload_url = self._accepted_create(raw_response)
        pending = self._take_pending_upload()
        content = self._source_bytes(pending.source)
        response = self._http_client().put(
            upload_url,
            content=content,
            headers={"Content-Type": _CONTAINER_MIME.get(pending.container, "video/mp4")},
        )
        self._raise_for_status(response)
        return self._created_video_object(model, request_id, custom_llm_provider, pending.seconds)

    async def async_transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: Mapping[str, Any] | None = None,
    ) -> VideoObject:
        request_id, upload_url = self._accepted_create(raw_response)
        pending = self._take_pending_upload()
        content = await self._async_source_bytes(pending.source)
        response = await self._async_http_client().put(
            upload_url,
            content=content,
            headers={"Content-Type": _CONTAINER_MIME.get(pending.container, "video/mp4")},
        )
        self._raise_for_status(response)
        return self._created_video_object(model, request_id, custom_llm_provider, pending.seconds)

    def _take_pending_upload(self) -> _PendingUpload:
        pending = self._pending_upload
        if pending is None:
            raise ValueError("Topaz create response reached without a captured source upload")
        self._pending_upload = None
        return pending

    def _accepted_create(self, raw_response: httpx.Response) -> tuple[str, str]:
        self._raise_for_status(raw_response)
        payload = raw_response.json()
        request_id = payload.get("requestId")
        upload_urls = payload.get("uploadUrls")
        if not request_id:
            raise TopazException(
                status_code=raw_response.status_code, message=f"Topaz create response has no requestId: {payload}"
            )
        if not isinstance(upload_urls, (list, tuple)) or len(upload_urls) != 1:
            raise TopazException(
                status_code=raw_response.status_code,
                message=(
                    "Topaz express create must return exactly one upload URL; got "
                    f"{len(upload_urls) if isinstance(upload_urls, (list, tuple)) else 0}. Uploading only the first "
                    "part would silently truncate the source footage."
                ),
            )
        return str(request_id), str(upload_urls[0])

    @staticmethod
    def _created_video_object(
        model: str,
        request_id: str,
        custom_llm_provider: str | None,
        seconds: object,
    ) -> VideoObject:
        duration = _safe_float(seconds)
        video_obj = VideoObject(
            id=request_id,
            object="video",
            status="queued",
            model=model,
            seconds=str(seconds) if seconds is not None else None,
            created_at=int(time.time()),
            usage={"duration_seconds": duration} if duration is not None else {},  # mutable-ok: usage expects a dict
        )
        if custom_llm_provider:
            video_obj.id = encode_video_id_with_provider(request_id, custom_llm_provider, model)
        return video_obj

    def _source_bytes(self, source: object) -> bytes:
        if isinstance(source, str):
            response = self._http_client().get(source)
            self._raise_for_status(response)
            return response.content
        return extract_file_data(source)["content"]  # pyright: ignore[reportArgumentType]  # FileTypes union

    async def _async_source_bytes(self, source: object) -> bytes:
        if isinstance(source, str):
            response = await self._async_http_client().get(source)
            self._raise_for_status(response)
            return response.content
        return extract_file_data(source)["content"]  # pyright: ignore[reportArgumentType]  # FileTypes union

    @staticmethod
    def _status_url(video_id: str, api_base: str) -> str:
        request_id = extract_original_video_id(video_id)
        return f"{resolve_topaz_api_base(api_base)}/video/{request_id}/status"

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, Any],
    ) -> tuple[str, dict]:  # mutable-ok: BaseVideoConfig contract returns dict params
        return self._status_url(video_id, api_base), {}  # mutable-ok: BaseVideoConfig contract returns dict params

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        self._raise_for_status(raw_response)
        try:
            payload = raw_response.json()
        except (ValueError, JSONDecodeError):
            return VideoObject(id="", object="video", status="in_progress")
        topaz_status = str(payload.get("status") or "")
        status = TOPAZ_STATUS_MAP.get(topaz_status, "in_progress")
        credits = _billed_credits(payload.get("estimates"))
        return VideoObject(
            id="",
            object="video",
            status=status,
            progress=payload.get("progress"),
            error=self._failure_error(payload) if topaz_status in TOPAZ_TERMINAL_FAILURES else None,
            usage={"topaz_credits": credits} if credits is not None else {},  # mutable-ok: usage expects a dict
        )

    @staticmethod
    def _failure_error(payload: Mapping[str, Any]) -> dict:  # mutable-ok: VideoObject.error expects a dict
        error = payload.get("error")
        message = error.get("message") if isinstance(error, Mapping) else None
        return {  # mutable-ok: VideoObject.error expects a dict
            "code": str(payload.get("status") or "failed"),
            "message": str(message or "Topaz video enhancement failed"),
        }

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, Any],
        variant: str | None = None,
    ) -> tuple[str, dict]:  # mutable-ok: BaseVideoConfig contract returns dict params
        return self._status_url(video_id, api_base), {}  # mutable-ok: BaseVideoConfig contract returns dict params

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        download_url = self._download_url(raw_response)
        response = self._http_client().get(download_url)
        self._raise_for_status(response)
        return response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        download_url = self._download_url(raw_response)
        response = await self._async_http_client().get(download_url)
        self._raise_for_status(response)
        return response.content

    def _download_url(self, raw_response: httpx.Response) -> str:
        self._raise_for_status(raw_response)
        payload = raw_response.json()
        topaz_status = str(payload.get("status") or "")
        if topaz_status in TOPAZ_TERMINAL_FAILURES:
            raise TopazException(status_code=502, message=f"Topaz video enhancement failed: {topaz_status}")
        download = payload.get("download")
        url = download.get("url") if isinstance(download, Mapping) else None
        if not url:
            raise TopazException(
                status_code=409,
                message=f"Topaz enhanced video is not downloadable yet (status {topaz_status!r})",
            )
        return str(url)

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        raise TopazException(status_code=response.status_code, message=response.text)

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Mapping[str, Any] | httpx.Headers,
    ) -> TopazException:
        return TopazException(status_code=status_code, message=error_message)

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, Any],
        extra_body: Mapping[str, Any] | None = None,
    ) -> tuple[str, dict]:  # mutable-ok: BaseVideoConfig contract returns dict params
        raise NotImplementedError("Video remix is not supported by the Topaz Labs API")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("Video remix is not supported by the Topaz Labs API")

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
        raise NotImplementedError("Video list is not supported by the Topaz Labs API")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> dict:  # mutable-ok: BaseVideoConfig contract returns dict
        raise NotImplementedError("Video list is not supported by the Topaz Labs API")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, Any],
    ) -> tuple[str, dict]:  # mutable-ok: BaseVideoConfig contract returns dict params
        raise NotImplementedError("Video delete is not supported by the Topaz Labs API")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("Video delete is not supported by the Topaz Labs API")
