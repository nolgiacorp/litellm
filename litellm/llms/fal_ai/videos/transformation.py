from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from json import JSONDecodeError, loads
from typing import TYPE_CHECKING, Any, Literal, assert_never

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.constants import FAL_AI_DEFAULT_API_BASE
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.llms.fal_ai.utils import normalize_fal_model_id as _normalize_fal_model_id
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


_FAL_AI_STATUS_MAP = {
    "IN_QUEUE": "queued",
    "IN_PROGRESS": "in_progress",
    "COMPLETED": "completed",
    "FAILED": "failed",
    "CANCELLED": "failed",
}

_SIZE_TO_ASPECT_RATIO = {
    "1280x720": "16:9",
    "1920x1080": "16:9",
    "720x1280": "9:16",
    "1080x1920": "9:16",
    "1024x1024": "1:1",
    "1280x1280": "1:1",
}

_MISSING_VIDEO_URL_MESSAGE = "Video URL not found in fal.ai response. The job may still be processing."
_UNREADABLE_RESULT_MESSAGE = "fal.ai returned an unreadable video result payload"
_FAL_ERROR_KEYS = ("detail", "error")
_MAX_ERROR_UNWRAP_DEPTH = 5
_RETRYABLE_CLIENT_STATUS_CODES = frozenset((408, 425))
_RESULT_VERDICT_STATUS_CODES = frozenset((200, 422))


@dataclass(frozen=True, slots=True)
class _QueuePending:
    status: Literal["queued", "in_progress"]
    queue_position: int | None


@dataclass(frozen=True, slots=True)
class _QueueSettled:
    pass


@dataclass(frozen=True, slots=True)
class _QueueRejected:
    message: str


_QueueState = _QueuePending | _QueueSettled | _QueueRejected


@dataclass(frozen=True, slots=True)
class _GeneratedVideo:
    url: str


@dataclass(frozen=True, slots=True)
class _GenerationFailed:
    message: str


_GenerationOutcome = _GeneratedVideo | _GenerationFailed


def _fal_error_field(loc: object) -> str | None:
    if not isinstance(loc, Sequence) or isinstance(loc, (str, bytes)):
        return None
    segments = tuple(segment for segment in loc if isinstance(segment, str) and segment != "body")
    return segments[-1] if segments else None


def _entry_reason(entry: object) -> str | None:
    if isinstance(entry, str):
        return entry.strip() or None
    if not isinstance(entry, Mapping):
        return None

    message = entry.get("msg") or entry.get("message")
    if not isinstance(message, str) or not message.strip():
        return None

    field = _fal_error_field(entry.get("loc"))
    return f"{message.strip()} (field: {field})" if field else message.strip()


def _unwrap_error_container(payload: object) -> object:
    container = payload
    for _ in range(_MAX_ERROR_UNWRAP_DEPTH):
        if not isinstance(container, Mapping) or _entry_reason(container) is not None:
            return container
        nested = next(
            (container[key] for key in _FAL_ERROR_KEYS if container.get(key) is not None),
            None,
        )
        if nested is None:
            return None
        container = nested
    return None


def _fal_failure_reason(payload: object) -> str | None:
    container = _unwrap_error_container(payload)
    if isinstance(container, Sequence) and not isinstance(container, (str, bytes)):
        reasons = tuple(reason for reason in (_entry_reason(item) for item in container) if reason)
        return "; ".join(reasons) or None
    return _entry_reason(container)


def _video_url_from_payload(payload: Mapping[str, object]) -> str | None:
    video = payload.get("video")
    if isinstance(video, dict):
        url = video.get("url")
        if isinstance(url, str) and url:
            return url

    top_level = payload.get("url")
    return top_level if isinstance(top_level, str) and top_level else None


def _classify_result_payload(payload: object) -> _GenerationOutcome:
    if not isinstance(payload, dict):
        return _GenerationFailed(_UNREADABLE_RESULT_MESSAGE)

    failure = _fal_failure_reason(payload)
    if failure is not None:
        return _GenerationFailed(failure)

    url = _video_url_from_payload(payload)
    if url is None:
        return _GenerationFailed(_MISSING_VIDEO_URL_MESSAGE)
    return _GeneratedVideo(url)


def _request_id_from(payload: Mapping[str, object] | None) -> str:
    if payload is None:
        return ""
    request_id = payload.get("request_id")
    return request_id if isinstance(request_id, str) else ""


def _parse_queue_state(payload: Mapping[str, object]) -> _QueueState:
    rejection = _fal_failure_reason(payload)
    if rejection is not None:
        return _QueueRejected(rejection)

    status_raw = payload.get("status")
    normalized = status_raw.upper() if isinstance(status_raw, str) else "IN_QUEUE"
    queue_position = payload.get("queue_position")
    position = queue_position if isinstance(queue_position, int) else None

    if normalized in ("FAILED", "CANCELLED"):
        return _QueueRejected(f"fal.ai queue reported {normalized.lower()}")
    if normalized == "COMPLETED":
        return _QueueSettled()
    if normalized == "IN_PROGRESS":
        return _QueuePending("in_progress", position)
    return _QueuePending("queued", position)


class FalAIVideoConfig(BaseVideoConfig):
    """
    fal.ai uses a queue API: POST to /{model_id}, then poll
    /{model_id}/requests/{id}/status and GET /{model_id}/requests/{id} for the
    result. Video models return {"video": {"url": ...}}.

    A queue status of COMPLETED only means the queue request finished; a run that
    errored also reports COMPLETED, and the failure is visible only in the result
    payload. Terminal status therefore resolves against the result payload before
    reporting success.
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
        return self._async_client or get_async_httpx_client(llm_provider=litellm.LlmProviders.FAL_AI)

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
    def _image_url_field_for_model(model: str) -> str:
        # Kling v3 image-to-video requires `start_image_url`; Seedance uses `image_url`.
        normalized = model.lower()
        if "kling-video/v3" in normalized:
            return "start_image_url"
        return "image_url"

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:
        mapped: dict[str, Any] = {}

        seconds = video_create_optional_params.get("seconds")
        if seconds is not None:
            mapped["duration"] = str(seconds)

        size = video_create_optional_params.get("size")
        if isinstance(size, str):
            aspect = _SIZE_TO_ASPECT_RATIO.get(size)
            if aspect is not None:
                mapped["aspect_ratio"] = aspect
            elif "x" in size:
                mapped["aspect_ratio"] = size.replace("x", ":")

        input_reference = video_create_optional_params.get("input_reference")
        if isinstance(input_reference, str) and input_reference:
            mapped[self._image_url_field_for_model(model)] = input_reference

        supported = self.get_supported_openai_params(model)
        for key, value in video_create_optional_params.items():
            if key not in supported:
                mapped[key] = value

        extra_body = video_create_optional_params.get("extra_body")
        if isinstance(extra_body, dict):
            mapped.update(extra_body)
            mapped.pop("extra_body", None)

        return mapped

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict:
        if litellm_params and litellm_params.api_key:
            api_key = api_key or litellm_params.api_key

        resolved_key = api_key or litellm.api_key or get_secret_str("FAL_AI_API_KEY") or get_secret_str("FAL_KEY")

        if not resolved_key:
            raise ValueError(
                "fal.ai API key is required. Set FAL_AI_API_KEY (or FAL_KEY) "
                "environment variable or pass api_key parameter."
            )

        headers.update(
            {
                "Authorization": f"Key {resolved_key}",
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
        base = api_base or get_secret_str("FAL_AI_API_BASE") or FAL_AI_DEFAULT_API_BASE
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
        model_id = _normalize_fal_model_id(model)

        request_data: dict[str, Any] = {"prompt": prompt}
        request_data.update(video_create_optional_request_params)
        request_data.pop("model", None)

        return request_data, [], f"{api_base}/{model_id}"

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: dict | None = None,
    ) -> VideoObject:
        response_data = raw_response.json()
        model_id = _normalize_fal_model_id(model)

        video_data: dict[str, Any] = {
            "id": response_data.get("request_id", ""),
            "object": "video",
            "status": _FAL_AI_STATUS_MAP.get(response_data.get("status", "IN_QUEUE").upper(), "queued"),
            "model": model,
        }

        if request_data:
            if "duration" in request_data:
                video_data["seconds"] = str(request_data["duration"])
            if "aspect_ratio" in request_data:
                video_data["size"] = str(request_data["aspect_ratio"]).replace(":", "x")

        video_obj = VideoObject(**video_data)  # type: ignore[arg-type]

        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(
                video_obj.id,
                custom_llm_provider,
                model_id,
            )

        usage: dict[str, Any] = {}
        if video_obj.seconds:
            try:
                usage["duration_seconds"] = float(video_obj.seconds)
            except (ValueError, TypeError):
                pass
        video_obj.usage = usage

        return video_obj

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        original_id, model_id = self._extract_request_and_model_id(video_id)
        encoded = encode_url_path_segment(original_id, field_name="video_id")
        namespace = self._queue_request_namespace(model_id)
        return f"{api_base}/{namespace}/requests/{encoded}/status", {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        payload = self._status_payload(raw_response)
        state = _parse_queue_state(payload) if payload is not None else _QueuePending("in_progress", None)
        request_id = _request_id_from(payload)

        result_url = self._settled_result_url(state, raw_response)
        if result_url is None:
            return self._video_object_for_state(state, request_id, raw_response, custom_llm_provider)

        result_response = self._http_client().get(
            url=result_url,
            headers=self._forwarded_auth_headers(raw_response),
        )
        return self._video_object_for_outcome(
            self._classify_result_response(result_response),
            request_id,
            raw_response,
            custom_llm_provider,
        )

    async def async_transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        payload = self._status_payload(raw_response)
        state = _parse_queue_state(payload) if payload is not None else _QueuePending("in_progress", None)
        request_id = _request_id_from(payload)

        result_url = self._settled_result_url(state, raw_response)
        if result_url is None:
            return self._video_object_for_state(state, request_id, raw_response, custom_llm_provider)

        result_response = await self._async_http_client().get(
            url=result_url,
            headers=self._forwarded_auth_headers(raw_response),
        )
        return self._video_object_for_outcome(
            self._classify_result_response(result_response),
            request_id,
            raw_response,
            custom_llm_provider,
        )

    def _status_payload(self, raw_response: httpx.Response) -> Mapping[str, object] | None:
        self._raise_for_status(raw_response)
        try:
            payload = raw_response.json()
        except (ValueError, JSONDecodeError):
            return None
        return payload if isinstance(payload, Mapping) else None

    @classmethod
    def _settled_result_url(cls, state: _QueueState, raw_response: httpx.Response) -> str | None:
        if not isinstance(state, _QueueSettled):
            return None
        return cls._result_url_from_status_request(raw_response)

    def _classify_result_response(self, result_response: httpx.Response) -> _GenerationOutcome:
        if result_response.status_code not in _RESULT_VERDICT_STATUS_CODES:
            raise self.get_error_class(
                error_message=result_response.text,
                status_code=result_response.status_code,
                headers=result_response.headers,
            )
        try:
            payload = result_response.json()
        except (ValueError, JSONDecodeError):
            raise self.get_error_class(
                error_message=_UNREADABLE_RESULT_MESSAGE,
                status_code=502,
                headers=result_response.headers,
            )
        return _classify_result_payload(payload)

    def _video_object_for_state(
        self,
        state: _QueueState,
        request_id: str,
        raw_response: httpx.Response,
        custom_llm_provider: str | None,
    ) -> VideoObject:
        match state:
            case _QueuePending(status, queue_position):
                return self._build_video_object(
                    request_id, raw_response, custom_llm_provider, status, queue_position, None
                )
            case _QueueRejected(message):
                return self._build_video_object(request_id, raw_response, custom_llm_provider, "failed", None, message)
            case _QueueSettled():
                return self._build_video_object(request_id, raw_response, custom_llm_provider, "completed", None, None)
            case _:
                assert_never(state)

    def _video_object_for_outcome(
        self,
        outcome: _GenerationOutcome,
        request_id: str,
        raw_response: httpx.Response,
        custom_llm_provider: str | None,
    ) -> VideoObject:
        match outcome:
            case _GeneratedVideo():
                return self._build_video_object(request_id, raw_response, custom_llm_provider, "completed", None, None)
            case _GenerationFailed(message):
                return self._build_video_object(request_id, raw_response, custom_llm_provider, "failed", None, message)
            case _:
                assert_never(outcome)

    def _build_video_object(
        self,
        request_id: str,
        raw_response: httpx.Response,
        custom_llm_provider: str | None,
        status: str,
        queue_position: int | None,
        failure_message: str | None,
    ) -> VideoObject:
        video_obj = VideoObject(
            id=request_id,
            object="video",
            status=status,
            progress=queue_position,
            error=None if failure_message is None else {"code": "generation_failed", "message": failure_message},
        )

        if custom_llm_provider and video_obj.id:
            model_id = self._model_id_from_request_url(raw_response)
            video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, model_id)

        return video_obj

    @staticmethod
    def _result_url_from_status_request(raw_response: httpx.Response) -> str | None:
        request = getattr(raw_response, "request", None)
        if request is None:
            return None
        url = str(request.url)
        return url.removesuffix("/status") if url.endswith("/status") else None

    @staticmethod
    def _forwarded_auth_headers(raw_response: httpx.Response) -> httpx.Headers:
        request = getattr(raw_response, "request", None)
        authorization = None if request is None else request.headers.get("Authorization")
        if not authorization:
            return httpx.Headers()
        return httpx.Headers((("Authorization", authorization),))

    @staticmethod
    def _model_id_from_request_url(raw_response: httpx.Response) -> str | None:
        request = getattr(raw_response, "request", None)
        if request is None:
            return None
        path = request.url.path
        head = path.split("/requests/", 1)[0].strip("/")
        return head or None

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        variant: str | None = None,
    ) -> tuple[str, dict]:
        original_id, model_id = self._extract_request_and_model_id(video_id)
        encoded = encode_url_path_segment(original_id, field_name="video_id")
        namespace = self._queue_request_namespace(model_id)
        return f"{api_base}/{namespace}/requests/{encoded}", {}

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        self._raise_for_status(raw_response)
        video_url = self._video_url_or_raise(raw_response)
        video_response = self._http_client().get(video_url)
        video_response.raise_for_status()
        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        self._raise_for_status(raw_response)
        video_url = self._video_url_or_raise(raw_response)
        video_response = await self._async_http_client().get(video_url)
        video_response.raise_for_status()
        return video_response.content

    def _video_url_or_raise(self, raw_response: httpx.Response) -> str:
        outcome = self._classify_result_response(raw_response)
        match outcome:
            case _GeneratedVideo(url):
                return url
            case _GenerationFailed(message):
                raise self.get_error_class(
                    error_message=message,
                    status_code=424,
                    headers=raw_response.headers,
                )
            case _:
                assert_never(outcome)

    @staticmethod
    def _queue_request_namespace(model_id: str) -> str:
        # Queue submits accept full model subpaths (fal-ai/kling-video/v3/pro/
        # image-to-video), but request status/result routes only exist under the
        # owner/app prefix; deeper paths answer 405 Method Not Allowed.
        segments = [segment for segment in model_id.split("/") if segment]
        return "/".join(segments[:2])

    @staticmethod
    def _extract_request_and_model_id(video_id: str) -> tuple[str, str]:
        # Queue URLs are always rebuilt from api_base + model_id + request id, never
        # taken from the (caller-supplied, only base64-encoded) video_id. Trusting an
        # embedded URL would let a forged id redirect fal-authenticated requests to an
        # arbitrary host and leak the API key.
        decoded = decode_video_id_with_provider(video_id)
        original_id = decoded.get("video_id") or extract_original_video_id(video_id)
        model_id = decoded.get("model_id")

        if not model_id:
            raise ValueError(
                "fal.ai video status/content lookup requires a model id encoded "
                "in the video_id. Use the id returned by video creation."
            )

        return original_id, model_id

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[str, dict]:
        raise NotImplementedError("Video remix is not supported by the fal.ai queue API")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("Video remix is not supported by the fal.ai queue API")

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
        raise NotImplementedError("Video listing is not supported by the fal.ai queue API")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> dict[str, str]:
        raise NotImplementedError("Video listing is not supported by the fal.ai queue API")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        # fal cancels jobs via PUT /requests/{id}/cancel, not the DELETE the shared handler issues.
        raise NotImplementedError("Video delete/cancel is not supported by the fal.ai queue API via LiteLLM")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        raise NotImplementedError("Video delete/cancel is not supported by the fal.ai queue API via LiteLLM")

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        if self._is_content_policy_rejection(error_message):
            raise litellm.ContentPolicyViolationError(
                message=error_message,
                model="",
                llm_provider=litellm.LlmProviders.FAL_AI.value,
            )
        customer_message = self._customer_facing_error_message(error_message)
        provider = litellm.LlmProviders.FAL_AI.value

        if status_code == 401:
            raise litellm.AuthenticationError(message=customer_message, model="", llm_provider=provider)
        if status_code == 403:
            raise litellm.PermissionDeniedError(
                message=customer_message,
                model="",
                llm_provider=provider,
                response=httpx.Response(status_code, request=httpx.Request("GET", FAL_AI_DEFAULT_API_BASE)),
            )
        if status_code == 429:
            raise litellm.RateLimitError(message=customer_message, model="", llm_provider=provider)
        if 400 <= status_code < 500 and status_code not in _RETRYABLE_CLIENT_STATUS_CODES:
            raise litellm.BadRequestError(message=customer_message, model="", llm_provider=provider)

        raise BaseLLMException(
            status_code=status_code,
            message=customer_message,
            headers=headers,
        )

    @staticmethod
    def _customer_facing_error_message(error_message: str) -> str:
        try:
            payload = loads(error_message)
        except (ValueError, JSONDecodeError):
            return error_message
        reason = _fal_failure_reason(payload)
        return f"fal.ai video generation failed: {reason}" if reason else error_message

    @staticmethod
    def _is_content_policy_rejection(error_message: str) -> bool:
        normalized_message = error_message.lower()
        return "content_policy_violation" in normalized_message or "partner_validation_failed" in normalized_message

    def _raise_for_status(self, raw_response: httpx.Response) -> None:
        if raw_response.is_success:
            return
        raise self.get_error_class(
            error_message=raw_response.text,
            status_code=raw_response.status_code,
            headers=raw_response.headers,
        )
