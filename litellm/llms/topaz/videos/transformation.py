import time
from collections.abc import Mapping
from dataclasses import dataclass
from json import JSONDecodeError
from types import MappingProxyType
from typing import TYPE_CHECKING, Any  # noqa: TID251  # base video ABC + OpenAI video TypedDict are Any-typed
from urllib.parse import unquote

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.constants import MAX_VIDEO_URL_DOWNLOAD_SIZE_MB
from litellm.litellm_core_utils.prompt_templates.common_utils import extract_file_data
from litellm.litellm_core_utils.url_utils import (
    async_safe_get,
    encode_url_path_segment,
    safe_get,
)
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.llms.topaz.common_utils import TOPAZ_VIDEO_MODELS, TopazException, TopazModelInfo
from litellm.llms.topaz.cost_calculator import cost_calculator as topaz_cost_calculator
from litellm.llms.topaz.video_geometry import SourceGeometry, parse_video_geometry
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import encode_video_id_with_provider, extract_original_video_id
from litellm.videos.capabilities import CapabilityParamSupport, DeclaredCapabilityParams

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


# NOL-519. The create leg prices a restore against Topaz's FREE `POST /video/`
# estimate endpoint.
#
# THE PREVIOUS APPROACH IS NOT TUNABLE, AND WAS REVERTED. It read the quote off
# the job's own status object, where `estimates` appears only once the job
# reaches `preprocessing`. That transition is QUEUE-DEPENDENT, not a fixed cost:
# measured on the live API with identical inputs, one restore reached it in 2.6s
# and the next was still `initializing` at 68s. No window acceptable on a
# customer's create request covers a 68s tail, so the poll captured the quote
# only SOMETIMES while adding ~7s to every create.
#
# Intermittent capture is worse than recording nothing. A partially populated
# COGS ledger looks healthy while understating by an unpredictable amount, and
# it permanently mutes the NOL-535 guard, which fires only when a model has
# NEVER recorded a cost - one lucky capture silences it forever. A clean zero is
# detectable; a partial one is not.
#
# `POST /video/` has none of that timing risk because it prices SUPPLIED
# GEOMETRY rather than a queued job: per Topaz's documentation it "does NOT
# consume credits" and "does NOT start processing", so it is free, synchronous
# and off the queue, and it answers on the first call or not at all. The
# geometry it needs is read from the source bytes this leg has already
# downloaded for the upload PUT, so the quote costs no extra network I/O.
#
# The timeout is short and deliberate: this sits inline on the customer's create
# request, and a slow vendor endpoint must yield "no quote" rather than hold the
# leg open. httpx applies a scalar timeout per socket operation rather than to
# the whole request, so the bound is enforced by the timeout AND by there being
# exactly one attempt - there is nothing to retry, since a deterministic
# endpoint that failed once will fail again for the same input.
_ESTIMATE_TIMEOUT_SECS = 6.0

# Topaz's estimate schema requires a source byte size, but it provably does not
# affect the quote - verified against the live endpoint, where 1MB, 24MB and
# 500MB sources returned an identical cost for the same geometry. The real
# length is sent when known (it always is here, since the bytes are in hand),
# and this is the declared fallback.
_NOMINAL_SOURCE_BYTES = 32 << 20

# Topaz requires an audio codec and transfer mode on `output`. The upscale lane
# does not re-encode audio, so the transfer is None; neither field moves the
# per-frame cost.
_ESTIMATE_AUDIO_CODEC = "AAC"
_ESTIMATE_AUDIO_TRANSFER = "None"

# Source geometry a caller may state explicitly, overriding what this proxy
# reads from the footage. Present for the containers the ISO-BMFF reader does
# not parse (mkv) and for callers that already measured their own source, e.g.
# nolgia-api, which computes exactly these to resolve a restore tier against the
# source aspect ratio. Consumed for the quote only: none of it reaches Topaz's
# create body, which describes the job rather than the footage.
_SOURCE_GEOMETRY_PARAMS = frozenset(
    (
        "source_width",
        "source_height",
        "source_frame_rate",
        "source_duration_seconds",
    )
)

_SUPPORTED_OPENAI_PARAMS = (
    "model",
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
        "user",
        "extra_headers",
        "extra_body",
        "input_reference",
        "seconds",
        "duration_seconds",
        "size",
        "resolution",
        "container",
        "source_width",
        "source_height",
        "source_frame_rate",
        "source_duration_seconds",
    )
)

_CAPABILITY_PARAMS = frozenset(("input_reference",))

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

UPSCALE_MODEL_CODES = TOPAZ_VIDEO_MODELS


def resolve_topaz_api_base(api_base: str | None) -> str:
    base = TopazModelInfo.get_api_base(api_base) or "https://api.topazlabs.com"
    return base.rstrip("/")


def topaz_auth_headers(api_key: str | None) -> Mapping[str, str]:
    resolved = TopazModelInfo.get_api_key(api_key)
    if not resolved:
        raise ValueError("TOPAZ_API_KEY is not set")
    return {"X-API-Key": resolved, "Content-Type": "application/json"}  # mutable-ok: returned as a Mapping view


def strip_topaz_prefix(model: str) -> str:
    return model.split("/", 1)[1] if model.startswith("topaz/") else model


@dataclass(frozen=True, slots=True)
class _PendingUpload:
    """
    What the create RESPONSE leg needs but only the REQUEST leg knows.

    The upload target is the source and container; the rest is the job
    description Topaz's estimate endpoint prices, carried across so the quote
    can be built without re-deriving it from the request body.
    """

    source: object
    container: str
    seconds: object
    model_code: str
    output_width: int
    output_height: int
    output_frame_rate: float | None
    declared_geometry: SourceGeometry | None


def _safe_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None  # pyright: ignore[reportArgumentType]  # guarded by except
    except (TypeError, ValueError):
        return None


def _progress_percent(value: object) -> int | None:
    """
    Topaz reports progress as a fractional percent (63.92405063291139); VideoObject.progress is an
    int, and pydantic refuses a float with a fractional part outright, so passing it through raises
    a ValidationError that surfaces to the caller as a 500 on every mid-render poll.

    Truncated rather than rounded: 99.6 must not read as a finished render while the status is
    still in_progress.
    """
    percent = _safe_float(value)
    if percent is None:
        return None
    return int(percent)


def _source_too_large(size_bytes: int, model: str) -> Exception:
    return litellm.BadRequestError(
        message=(
            f"Topaz source footage is {size_bytes / (1024 * 1024):.1f}MB, above the "
            f"{MAX_VIDEO_URL_DOWNLOAD_SIZE_MB}MB per-request limit for relayed source video. Raise "
            "MAX_VIDEO_URL_DOWNLOAD_SIZE_MB if this proxy is provisioned for larger masters."
        ),
        model=model,
        llm_provider=litellm.LlmProviders.TOPAZ.value,
    )


def _request_id_from_status_url(raw_response: httpx.Response) -> str:
    """Recover the Topaz request id from a `/video/{id}/status` URL."""
    request: httpx.Request | None = getattr(raw_response, "request", None)
    if request is None:
        return ""
    segments = tuple(segment for segment in request.url.path.split("/") if segment)
    if len(segments) < 2 or segments[-1] != "status":
        return ""
    return unquote(segments[-2])


def _billed_credits(estimates: object) -> float | None:
    # The lower bound is kept as a float: Topaz quotes fractional credits for
    # small jobs (0.25 for a 5s 960x720 restore), so truncating to int would
    # record $0 for exactly the cheap end of the range this pricing exists to
    # capture, and shave the fraction off every larger quote.
    if not isinstance(estimates, Mapping):
        return None
    cost = estimates.get("cost")
    if not isinstance(cost, (list, tuple)) or not cost:
        return None
    return _safe_float(cost[0])


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
        self._requested_video_id: str | None = None

    def set_status_lookup_client(self, client: HTTPHandler | AsyncHTTPHandler) -> None:
        # The source GET, the presigned PUT and the enhanced download must ride the same
        # client as the leg the handler issued, or a caller's mock transport, proxy,
        # private CA or ssl_verify setting applies to only part of the request.
        if isinstance(client, AsyncHTTPHandler):
            self._async_client = client
        else:
            self._sync_client = client

    def _http_client(self) -> HTTPHandler:
        return self._sync_client or _get_httpx_client()

    def _async_http_client(self) -> AsyncHTTPHandler:
        return self._async_client or get_async_httpx_client(llm_provider=litellm.LlmProviders.TOPAZ)

    def get_supported_openai_params(self, model: str) -> list:  # mutable-ok: BaseVideoConfig contract returns list
        return list(_SUPPORTED_OPENAI_PARAMS)  # mutable-ok: BaseVideoConfig contract returns list

    def supports_promptless_video_create(self, model: str) -> bool:
        return True

    def get_capability_param_support(self, model: str) -> CapabilityParamSupport:
        """
        Topaz enhances mandatory source footage and nothing else: input_reference carries that
        clip, and the rest of the request is the engine choice and the output frame.

        Every other member of the vocabulary is genuinely absent rather than merely unmapped.
        There is no prompt to negate, no soundtrack to render, and no slot for a start or end
        frame, reference media or a base video, so declaring any of them would let a caller be
        billed for an enhancement that ignored what they attached. image_url is NOT declared
        alongside input_reference here, unlike on the generators where the two are the same
        start-frame slot: this input is the source video, and a still passed to an upscaler is
        not footage it can enhance.
        """
        return DeclaredCapabilityParams(_CAPABILITY_PARAMS)

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
        return {**params, **extra_body}  # mutable-ok: returned as a Mapping view

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

    @staticmethod
    def _reject_prompt(prompt: str | None, model: str) -> None:
        if not prompt or not str(prompt).strip():
            return
        raise litellm.BadRequestError(
            message=(
                f"Topaz model '{model}' does not support `prompt`. Topaz is a footage upscaler driven by the source "
                "clip and the requested output resolution alone; billing an enhancement that ignored the caller's "
                "instructions would be worse than refusing it."
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
        self._reject_prompt(prompt, model)
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
        # extra_body is overlaid onto the mapped params after map_openai_params runs, so an
        # extra_body container never passed through _container and is revalidated here.
        container = self._container(params, model)
        width, height = self._resolution(params, model)
        upscale_filter = {  # mutable-ok: request body fragment
            "model": self._model_code(model),
            **{  # mutable-ok: request body fragment
                key: value for key, value in params.items() if key in _FILTER_PARAMS
            },
        }
        output = {  # mutable-ok: request body fragment
            "resolution": {"width": width, "height": height},  # mutable-ok: request body fragment
            **{  # mutable-ok: request body fragment
                key: value for key, value in params.items() if key in _OUTPUT_PARAMS
            },
        }
        body = {  # mutable-ok: BaseVideoConfig contract returns dict body
            "source": {"container": container},  # mutable-ok: request body fragment
            "filters": [upscale_filter],  # mutable-ok: request body fragment
            "output": output,
        }
        self._pending_upload = _PendingUpload(
            source=source,
            container=container,
            seconds=params.get("seconds"),
            model_code=self._model_code(model),
            output_width=width,
            output_height=height,
            output_frame_rate=_safe_float(params.get("frameRate")),
            declared_geometry=self._declared_geometry(params),
        )
        return body, (), f"{resolve_topaz_api_base(api_base)}/video/express"

    @staticmethod
    def _declared_geometry(params: Mapping[str, Any]) -> SourceGeometry | None:
        """
        Source geometry the caller stated explicitly, if it stated all of it.

        Partial geometry is refused rather than completed with defaults: a
        guessed frame rate or duration produces a plausible-looking quote that
        is quietly wrong, which is the failure mode this whole change exists to
        avoid. Duration falls back to `seconds` because that is the same
        quantity under the name the OpenAI video surface already uses.
        """
        width = _safe_float(params.get("source_width"))
        height = _safe_float(params.get("source_height"))
        frame_rate = _safe_float(params.get("source_frame_rate"))
        duration = _safe_float(params.get("source_duration_seconds"))
        if duration is None:
            duration = _safe_float(params.get("seconds"))
        if width is None or height is None or frame_rate is None or duration is None:
            return None
        if width <= 0 or height <= 0 or frame_rate <= 0 or duration <= 0:
            return None
        return SourceGeometry(
            width=int(width),
            height=int(height),
            duration_seconds=duration,
            frame_rate=frame_rate,
        )

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
        content = self._source_bytes(pending.source, model)
        response = self._http_client().put(
            upload_url,
            content=content,
            headers={  # mutable-ok: httpx expects a dict of headers
                "Content-Type": _CONTAINER_MIME.get(pending.container, "video/mp4")
            },
        )
        self._raise_for_status(response)
        # Quoted AFTER the upload so the customer's job starts first: the
        # estimate is bookkeeping and must never delay the work being paid for.
        credits = self._estimate_billed_credits(raw_response, pending, content)
        return self._created_video_object(model, request_id, custom_llm_provider, pending.seconds, credits)

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
        content = await self._async_source_bytes(pending.source, model)
        response = await self._async_http_client().put(
            upload_url,
            content=content,
            headers={  # mutable-ok: httpx expects a dict of headers
                "Content-Type": _CONTAINER_MIME.get(pending.container, "video/mp4")
            },
        )
        self._raise_for_status(response)
        credits = await self._async_estimate_billed_credits(raw_response, pending, content)
        return self._created_video_object(model, request_id, custom_llm_provider, pending.seconds, credits)

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
        topaz_credits: float | None = None,
    ) -> VideoObject:
        duration = _safe_float(seconds)
        usage: dict[str, Any] = {}  # mutable-ok: usage expects a dict
        if duration is not None:
            usage["duration_seconds"] = duration
        if topaz_credits is not None:
            usage["topaz_credits"] = topaz_credits

        video_obj = VideoObject(
            id=request_id,
            object="video",
            status="queued",
            model=model,
            seconds=str(seconds) if seconds is not None else None,
            created_at=int(time.time()),
            usage=usage,
        )

        # NOL-519. The create leg is the ONLY leg that writes a spend row, and
        # Topaz cost cannot be derived from anything on it: it bills credits for
        # frames processed, non-monotonically in source/output geometry. So the
        # cost is computed here from the credits Topaz itself quoted and handed
        # over as an explicit response_cost, which the logging path prefers over
        # its own per-second calculation. Without this the shared video cost path
        # sees only duration_seconds and records $0.
        if topaz_credits is not None:
            cost = topaz_cost_calculator(model=model, topaz_credits=topaz_credits)
            if cost > 0:
                video_obj._hidden_params = {"response_cost": cost}  # mutable-ok: hidden params expects a dict

        if custom_llm_provider:
            video_obj.id = encode_video_id_with_provider(request_id, custom_llm_provider, model)
        return video_obj

    @staticmethod
    def _estimate_url(raw_response: httpx.Response) -> str:
        """
        The estimate endpoint, derived from the create URL so any
        TOPAZ_API_BASE override is preserved.
        """
        create_url = str(raw_response.request.url)
        marker = "/video/express"
        # rsplit returns the input unchanged when the marker is absent, which
        # would post the quote to the wrong path. Fall back to the resolved
        # base rather than guessing from an unexpected create URL.
        base = create_url.rsplit(marker, 1)[0] if marker in create_url else resolve_topaz_api_base(None)
        return f"{base}/video/"

    @staticmethod
    def _estimate_headers(raw_response: httpx.Response) -> dict:  # mutable-ok: httpx expects a dict of headers
        api_key = raw_response.request.headers.get("X-API-Key", "")
        if not api_key:
            return {}  # mutable-ok: httpx expects a dict of headers
        return {  # mutable-ok: httpx expects a dict of headers
            "X-API-Key": api_key,
            "Content-Type": "application/json",
        }

    @staticmethod
    def _estimate_body(
        pending: _PendingUpload,
        geometry: SourceGeometry,
        source_bytes: int,
    ) -> dict:  # mutable-ok: httpx expects a dict body
        """
        Topaz's `POST /video/` request shape.

        The output frame rate defaults to the source's: an upscale that does not
        ask for interpolation processes exactly the frames it was given, and
        quoting a different rate would price a job we are not running.
        """
        output_frame_rate = pending.output_frame_rate if pending.output_frame_rate else geometry.frame_rate
        return {  # mutable-ok: httpx expects a dict body
            "source": {  # mutable-ok: request body fragment
                "container": pending.container,
                "size": source_bytes if source_bytes > 0 else _NOMINAL_SOURCE_BYTES,
                "duration": geometry.duration_seconds,
                "frameCount": geometry.frame_count,
                "frameRate": geometry.frame_rate,
                "resolution": {"width": geometry.width, "height": geometry.height},  # mutable-ok: request body fragment
            },
            "filters": [{"model": pending.model_code}],  # mutable-ok: request body fragment
            "output": {  # mutable-ok: request body fragment
                "resolution": {  # mutable-ok: request body fragment
                    "width": pending.output_width,
                    "height": pending.output_height,
                },
                "frameRate": output_frame_rate,
                "audioCodec": _ESTIMATE_AUDIO_CODEC,
                "audioTransfer": _ESTIMATE_AUDIO_TRANSFER,
            },
        }

    @staticmethod
    def _resolve_geometry(pending: _PendingUpload, content: bytes) -> SourceGeometry | None:
        """
        The footage description to quote against.

        What was MEASURED from the bytes being uploaded wins over what the
        caller declared. The declaration is a convenience for containers this
        proxy cannot parse, not a pricing input to be trusted: a caller that
        understated its geometry would understate our own recorded COGS, and a
        ledger that errs low is the failure this ticket exists to fix.
        """
        parsed = parse_video_geometry(content)
        if parsed is not None:
            return parsed
        return pending.declared_geometry

    def _credits_from_estimate(self, response: httpx.Response) -> float | None:
        if response.status_code != 200:
            return None
        try:
            payload = response.json()
        except (ValueError, JSONDecodeError):
            return None
        # A 200 carrying valid JSON that is not an object (null, a list, a
        # string from an intermediary) must read as a failed quote: .get() on it
        # would raise past the fail-silent handling and turn a bookkeeping
        # hiccup into a create error for footage Topaz has already accepted.
        if not isinstance(payload, Mapping):
            return None
        return _billed_credits(payload.get("estimates"))

    def _estimate_billed_credits(
        self,
        raw_response: httpx.Response,
        pending: _PendingUpload,
        content: bytes,
    ) -> float | None:
        """
        Quote this restore against Topaz's free estimate endpoint.

        One attempt, by design. The endpoint is deterministic over its inputs,
        so a failure is not a race worth re-running; and this sits on the
        customer's create request, which must not be held open for bookkeeping.

        Silent on failure for the same reason the old poll was: a slow or
        unhappy vendor endpoint must never fail a job whose footage Topaz has
        already accepted. No quote means no cost recorded, which is the state
        the NOL-535 ledger guard is built to catch.
        """
        geometry = self._resolve_geometry(pending, content)
        if geometry is None:
            return None
        try:
            response = self._http_client().post(
                self._estimate_url(raw_response),
                json=self._estimate_body(pending, geometry, len(content)),
                headers=self._estimate_headers(raw_response),
                timeout=_ESTIMATE_TIMEOUT_SECS,
            )
        except (httpx.HTTPError, litellm.Timeout):
            # The handler re-raises a non-2xx as MaskedHTTPStatusError (an
            # httpx.HTTPError) but converts a read timeout into litellm.Timeout,
            # which is not one. Both mean "no quote", and neither may escape:
            # Topaz has already accepted the footage by this point.
            return None
        return self._credits_from_estimate(response)

    async def _async_estimate_billed_credits(
        self,
        raw_response: httpx.Response,
        pending: _PendingUpload,
        content: bytes,
    ) -> float | None:
        """Async twin of _estimate_billed_credits; see that docstring."""
        geometry = self._resolve_geometry(pending, content)
        if geometry is None:
            return None
        try:
            response = await self._async_http_client().post(
                self._estimate_url(raw_response),
                json=self._estimate_body(pending, geometry, len(content)),
                headers=self._estimate_headers(raw_response),
                timeout=_ESTIMATE_TIMEOUT_SECS,
            )
        except (httpx.HTTPError, litellm.Timeout):
            # The handler re-raises a non-2xx as MaskedHTTPStatusError (an
            # httpx.HTTPError) but converts a read timeout into litellm.Timeout,
            # which is not one. Both mean "no quote", and neither may escape:
            # Topaz has already accepted the footage by this point.
            return None
        return self._credits_from_estimate(response)

    def _source_bytes(self, source: object, model: str) -> bytes:
        if isinstance(source, str):
            # Caller-supplied URL: safe_get validates DNS and every redirect hop so the
            # proxy cannot be pointed at loopback, private-network or metadata endpoints.
            response: httpx.Response = safe_get(  # pyright: ignore[reportAny]  # safe_get is Any-in/Any-out; it returns the httpx response
                self._http_client(), source
            )
            self._raise_for_status(response)
            return self._bounded_source_content(response, model)
        return extract_file_data(source)["content"]  # pyright: ignore[reportArgumentType]  # FileTypes union

    async def _async_source_bytes(self, source: object, model: str) -> bytes:
        if isinstance(source, str):
            response: httpx.Response = await async_safe_get(  # pyright: ignore[reportAny]  # async_safe_get is Any-in/Any-out
                self._async_http_client(), source
            )
            self._raise_for_status(response)
            return self._bounded_source_content(response, model)
        return extract_file_data(source)["content"]  # pyright: ignore[reportArgumentType]  # FileTypes union

    @staticmethod
    def _bounded_source_content(response: httpx.Response, model: str) -> bytes:
        """
        Cap the source clip a single request may relay.

        An unbounded remote response would let one delivery-grade clip consume gigabytes of a
        shared proxy, so the declared length is refused before the body is touched and the
        body itself is refused when the sender understated it.
        """
        max_bytes = int(MAX_VIDEO_URL_DOWNLOAD_SIZE_MB * 1024 * 1024)
        declared = str(response.headers.get("content-length") or "")
        if declared.isdigit() and int(declared) > max_bytes:
            raise _source_too_large(int(declared), model)
        content = response.content
        if len(content) > max_bytes:
            raise _source_too_large(len(content), model)
        return content

    @staticmethod
    def _status_url(video_id: str, api_base: str) -> str:
        # The decoded Topaz request id is caller-controlled, so it is percent-encoded before
        # it reaches this credential-bearing URL; `../`, `?` and `#` must not repoint the path.
        request_id = encode_url_path_segment(extract_original_video_id(video_id), field_name="video_id")
        return f"{resolve_topaz_api_base(api_base)}/video/{request_id}/status"

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, Any],
    ) -> tuple[str, dict]:  # mutable-ok: BaseVideoConfig contract returns dict params
        self._requested_video_id = video_id
        return self._status_url(video_id, api_base), {}  # mutable-ok: BaseVideoConfig contract returns dict params

    def _status_video_id(self, raw_response: httpx.Response) -> str:
        # Topaz status payloads carry no id, so the requested one is retained: callers that
        # correlate or persist jobs from a status response need it to reach the content flow.
        if self._requested_video_id:
            return self._requested_video_id
        return _request_id_from_status_url(raw_response)

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        self._raise_for_status(raw_response)
        video_id = self._status_video_id(raw_response)
        try:
            payload = raw_response.json()
        except (ValueError, JSONDecodeError):
            return VideoObject(id=video_id, object="video", status="in_progress")
        topaz_status = str(payload.get("status") or "")
        status = TOPAZ_STATUS_MAP.get(topaz_status, "in_progress")
        credits = _billed_credits(payload.get("estimates"))
        return VideoObject(
            id=video_id,
            object="video",
            status=status,
            progress=_progress_percent(payload.get("progress")),
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
