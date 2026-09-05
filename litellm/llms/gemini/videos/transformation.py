import base64
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.constants import DEFAULT_GOOGLE_VIDEO_DURATION_SECONDS
from litellm.images.utils import ImageEditRequestUtils
from litellm.litellm_core_utils.token_counter import get_image_type
from litellm.litellm_core_utils.url_utils import safe_get
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.gemini import (
    GeminiLongRunningOperationResponse,
    GeminiVideoGenerationInstance,
    GeminiVideoGenerationParameters,
    GeminiVideoGenerationRequest,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    encode_video_id_with_provider,
    extract_original_video_id,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.videos.capabilities import CapabilityParamSupport as _CapabilityParamSupport

    from ...base_llm.chat.transformation import BaseLLMException as _BaseLLMException

    LiteLLMLoggingObj = _LiteLLMLoggingObj
    BaseLLMException = _BaseLLMException
    CapabilityParamSupport = _CapabilityParamSupport
else:
    LiteLLMLoggingObj = Any
    BaseLLMException = Any
    CapabilityParamSupport = Any


_MAX_REFERENCE_IMAGES = 3

_VEO_3X_MODEL = re.compile(r"veo-3(?![0-9])")

_AUDIO_PARAM_KEYS = ("generate_audio", "generateAudio")

_PERSON_GENERATION_IMAGE_VALUE = "allow_adult"


def _is_veo_3x(model: str) -> bool:
    """Whether ``model`` names a Veo 3.x variant (3.0/3.1, fast/lite, preview or GA)."""
    return _VEO_3X_MODEL.search(model.lower()) is not None


def _audio_preference(params: Mapping[str, Any]) -> bool | None:
    """
    Return the audio track the caller asked for, or None if it did not ask.

    The Gemini video surface has no audio toggle: Veo 3.x always generates
    audio natively ("Always on" per https://ai.google.dev/gemini-api/docs/veo),
    and Google's own SDK rejects the flag outright with "generate_audio
    parameter is not supported in Gemini API". Only Vertex exposes a
    ``generateAudio`` boolean. So the flag is consumed by the transform rather
    than forwarded; forwarding it would make Google reject the request.
    """
    requested = tuple(params[key] for key in _AUDIO_PARAM_KEYS if params.get(key) is not None)
    return bool(requested[0]) if requested else None


def _reject_unrenderable_audio(model: str, wants_audio: bool | None) -> None:
    """
    Refuse an audio preference this model cannot render.

    The flag is consumed by the transform rather than forwarded, so an accepted value
    the model cannot honor would bill a generation whose soundtrack silently differs
    from what was asked for. Veo 3.x always renders audio and exposes no field to
    disable it; every other Veo renders silent video and has no field to enable it.
    """
    if wants_audio is None:
        return
    if wants_audio is False and _is_veo_3x(model):
        raise ValueError(
            "generate_audio=false is not supported for Veo 3.x on the Gemini video route: "
            "audio is generated natively and always on, and the Gemini API exposes no field "
            "to disable it. Route to a model that renders silent video if you need no audio track."
        )
    if wants_audio is True and not _is_veo_3x(model):
        raise ValueError(
            f"generate_audio=true is not supported for '{model}' on the Gemini video route: "
            "only Veo 3.x renders audio, and this model has no audio field to enable. "
            "Route to a Veo 3.x model if you need an audio track."
        )


def _person_generation_for_request(
    model: str,
    instance: GeminiVideoGenerationInstance,
    params: Mapping[str, Any],
) -> str | None:
    """
    Resolve ``personGeneration`` for an image-bearing Veo 3.x request.

    Per https://ai.google.dev/gemini-api/docs/veo the accepted value is
    mode-scoped: text-to-video and extension take "allow_all" only, while
    image-to-video, interpolation, and reference-image runs take
    "allow_adult" only. "allow_adult" is therefore not a permissive default
    we picked; it is the single value Google accepts for this request shape,
    and it is the stricter of the two (adults only, no minors).

    Text-to-video is deliberately left unset so the provider default applies,
    since that path works today and the Gemini surface documents no default.
    """
    if not _is_veo_3x(model):
        return None
    carries_image = "image" in instance or "referenceImages" in instance or params.get("lastFrame") is not None
    if not carries_image:
        return None
    return _PERSON_GENERATION_IMAGE_VALUE


def _image_mime_type_from_response(response: httpx.Response) -> str:
    """
    Resolve the mimeType Veo receives, preferring the bytes over the header.

    Signed URLs frequently answer with a generic ``application/octet-stream``
    or an outright wrong image type, and Veo rejects a reference image whose
    mimeType disagrees with its payload.
    """
    sniffed = get_image_type(response.content[:100])
    if sniffed is not None:
        return f"image/{sniffed}"
    header_type = str(response.headers.get("content-type", "")).split(";")[0].strip()
    return header_type if header_type.startswith("image/") else "image/jpeg"


def fetch_image_as_base64(image_url: str) -> tuple[str, str]:
    """
    Download an image URL and return (base64_data, mime_type).

    Used for image-to-video: callers pass a signed URL, while the Gemini
    APIs want inline base64 bytes. The URL is caller-controlled, so it is
    fetched via ``safe_get``, which validates every redirect hop against the
    SSRF block list.
    """
    response: httpx.Response = safe_get(  # pyright: ignore[reportAny]  # safe_get is declared Any-in/Any-out; it returns the httpx response
        litellm.module_level_client, image_url
    )
    response.raise_for_status()
    return base64.b64encode(response.content).decode("utf-8"), _image_mime_type_from_response(response)


def _convert_image_to_gemini_format(image_file) -> dict[str, str]:
    """
    Convert image file to Gemini format with base64 encoding and MIME type.

    Args:
        image_file: File-like object opened in binary mode (e.g., open("path", "rb"))

    Returns:
        Dict with bytesBase64Encoded and mimeType
    """
    mime_type: Final = ImageEditRequestUtils.get_image_content_type(image_file)

    if hasattr(image_file, "seek"):
        image_file.seek(0)
    image_bytes: Final = image_file.read()
    base64_encoded: Final = base64.b64encode(image_bytes).decode("utf-8")

    return {"bytesBase64Encoded": base64_encoded, "mimeType": mime_type}


def _json_payload(raw_response: httpx.Response) -> object:
    """Read an HTTP response body as an opaque JSON payload."""
    return raw_response.json()


def _usage_video_resolution_from_parameters(
    parameters: Mapping[str, object],
) -> str | None:
    """Normalize Veo ``parameters.resolution`` for usage and cost tracking."""
    res: Final = parameters.get("resolution")
    if res is None or res == "":
        return None
    return str(res).strip().lower()


_CAPABILITY_PARAMS = frozenset(
    (
        "input_reference",
        "image_url",
        "image_urls",
        "generate_audio",
        "negative_prompt",
    )
)


class GeminiVideoConfig(BaseVideoConfig):
    """
    Configuration class for Gemini (Veo) video generation.

    Veo uses a long-running operation model:
    1. POST to :predictLongRunning returns operation name
    2. Poll operation until done=true
    3. Extract video URI from response
    4. Download video using file API
    """

    _OPENAI_VIDEO_SIZE_TO_ASPECT_RATIO: dict[str, str] = {
        "1280x720": "16:9",
        "1920x1080": "16:9",
        "720x1280": "9:16",
        "1080x1920": "9:16",
    }

    def __init__(self):
        super().__init__()

    def get_capability_param_support(self, model: str) -> "CapabilityParamSupport":
        """
        Veo executes a start frame (image / image_url), up to three reference
        images ("ingredients", mapped to referenceImages on the instance) and
        negative_prompt, which every Veo model carries as parameters.negativePrompt
        and which map_openai_params normalizes onto that camelCase name below.

        generate_audio is declared for every Veo model because the transform consumes
        and validates it per model rather than dropping it: on Veo 3.x audio is native
        and always on, so True is satisfied and False is refused, while Veo 2.x renders
        silent video, so False is satisfied and True is refused. Only the value a given
        model cannot produce is rejected, which keeps a caller that sends a uniform
        request shape working instead of 4xx-ing a flag the model already honors.

        It has no end-frame, reference-video, reference-audio, regeneration or
        bitrate surface.
        """
        from litellm.videos.capabilities import DeclaredCapabilityParams

        return DeclaredCapabilityParams(_CAPABILITY_PARAMS)

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get the list of supported OpenAI parameters for Veo video generation.
        Veo supports minimal parameters compared to OpenAI.
        """
        return ["model", "prompt", "input_reference", "seconds", "size"]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:
        """
        Map OpenAI-style parameters to Veo format.

        Mappings:
        - prompt → prompt
        - input_reference → image
        - size → aspectRatio (e.g., "1280x720" → "16:9")
        - size → resolution when inferable ("1280x720"/"720x1280" → "720p",
          "1920x1080"/"1080x1920" → "1080p"); skipped if ``resolution`` is already set
        - seconds → durationSeconds (defaults to 4 seconds if not provided)

        All other params are passed through as-is to support Gemini-specific parameters.
        """
        mapped_params: Final[dict[str, object]] = {}

        # Get supported OpenAI params (exclude "model" and "prompt" which are handled separately)
        supported_openai_params: Final = self.get_supported_openai_params(model)
        openai_params_to_map: Final = {param for param in supported_openai_params if param not in {"model", "prompt"}}

        # Map input_reference to image
        if "input_reference" in video_create_optional_params:
            mapped_params["image"] = video_create_optional_params["input_reference"]

        # Map size to aspectRatio
        if "size" in video_create_optional_params:
            size: Final = video_create_optional_params["size"]
            if size is not None:
                aspect_ratio: Final = self._convert_size_to_aspect_ratio(size)
                if aspect_ratio:
                    mapped_params["aspectRatio"] = aspect_ratio
                if not video_create_optional_params.get("resolution"):
                    inferred_resolution: Final = self._convert_size_to_resolution(size)
                    if inferred_resolution is not None:
                        mapped_params["resolution"] = inferred_resolution

        # Map seconds to durationSeconds, default to 4 seconds (matching OpenAI)
        if "seconds" in video_create_optional_params:
            seconds: Final = video_create_optional_params["seconds"]
            try:
                duration: Final = int(seconds) if isinstance(seconds, str) else seconds
                if duration is not None:
                    mapped_params["durationSeconds"] = duration
            except (ValueError, TypeError):
                # If conversion fails, use default
                pass

        # Pass through any other params that weren't mapped (Gemini-specific params)
        for key, value in video_create_optional_params.items():
            if key not in openai_params_to_map and key not in mapped_params:
                mapped_params[key] = value

        # Normalize snake_case aliases onto the camelCase fields Veo expects.
        # GeminiVideoGenerationParameters silently ignores undeclared keys
        # (pydantic extra="ignore"), so an un-normalized aspect_ratio was
        # dropped on the floor and every text-to-video render came out 16:9
        # regardless of the requested ratio.
        for snake, camel in (
            ("aspect_ratio", "aspectRatio"),
            ("negative_prompt", "negativePrompt"),
            ("person_generation", "personGeneration"),
        ):
            if snake in mapped_params:
                value = mapped_params.pop(snake)
                if value is not None and camel not in mapped_params:
                    mapped_params[camel] = value

        return mapped_params

    def _convert_size_to_aspect_ratio(self, size: str) -> str | None:
        """
        Convert OpenAI size format to Veo aspectRatio format.

        https://cloud.google.com/vertex-ai/generative-ai/docs/image/generate-videos

        Supported aspect ratios: 9:16 (portrait), 16:9 (landscape)
        """
        if not size:
            return None

        return self._OPENAI_VIDEO_SIZE_TO_ASPECT_RATIO.get(size, "16:9")

    def _convert_size_to_resolution(self, size: str) -> str | None:
        """
        Map OpenAI ``size`` (WxH) to Veo ``resolution`` for presets in
        ``_OPENAI_VIDEO_SIZE_TO_ASPECT_RATIO`` (720p / 1080p from the smaller edge).

        Unknown sizes return None so the API default applies (no forced resolution).
        """
        if not size or size not in self._OPENAI_VIDEO_SIZE_TO_ASPECT_RATIO:
            return None
        try:
            w_str, h_str = size.split("x", 1)
            smaller: Final = min(int(w_str), int(h_str))
        except (ValueError, TypeError):
            return None
        if smaller == 720:
            return "720p"
        if smaller == 1080:
            return "1080p"
        return None

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict:
        """
        Validate environment and add Gemini API key to headers.
        Gemini uses x-goog-api-key header for authentication.
        """
        # Use api_key from litellm_params if available, otherwise fall back to other sources
        if litellm_params and litellm_params.api_key:
            api_key = api_key or litellm_params.api_key

        api_key = api_key or litellm.api_key or get_secret_str("GOOGLE_API_KEY") or get_secret_str("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY is required for Veo video generation. "
                "Set it via environment variable or pass it as api_key parameter."
            )

        headers.update(
            {
                "x-goog-api-key": api_key,
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
        """
        Get the complete URL for Veo video generation.
        For video creation: returns full URL with :predictLongRunning
        For status/delete: returns base URL only
        """
        if api_base is None:
            api_base = get_secret_str("GEMINI_API_BASE") or "https://generativelanguage.googleapis.com"

        if not model or model == "":
            return api_base.rstrip("/")

        model_name: Final = model.replace("gemini/", "")
        url: Final = f"{api_base.rstrip('/')}/v1beta/models/{model_name}:predictLongRunning"

        return url

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[dict, RequestFiles, str]:
        """
        Transform the video creation request for Veo API.

        Veo expects:
        {
            "instances": [
                {
                    "prompt": "A cat playing with a ball of yarn",
                    "image": {
                        "bytesBase64Encoded": "...",
                        "mimeType": "image/jpeg"
                    }
                }
            ],
            "parameters": {
                "aspectRatio": "16:9",
                "durationSeconds": 8,
                "resolution": "720p"
            }
        }
        """
        instance: Final[GeminiVideoGenerationInstance] = {"prompt": prompt}

        params_copy: Final = video_create_optional_request_params.copy()

        if "image" in params_copy:
            image: Final = params_copy.pop("image")
            if image is not None:
                if isinstance(image, dict):
                    instance["image"] = image
                elif isinstance(image, str):
                    if not image.startswith(("http://", "https://")):
                        raise ValueError(
                            "Unsupported string image input for Gemini video generation; "
                            f"expected an http(s) image URL, got: {image[:100]}"
                        )
                    params_copy["image_url"] = image
                else:
                    instance["image"] = _convert_image_to_gemini_format(image)

        if "image_url" in params_copy:
            image_url = params_copy.pop("image_url")
            if image_url and "image" not in instance:
                base64_data, mime_type = fetch_image_as_base64(image_url)
                instance["image"] = {"bytesBase64Encoded": base64_data, "mimeType": mime_type}

        # Veo 3.1 reference images ("ingredients", subject/character
        # consistency): callers send fal-shaped image_urls; Veo wants up to
        # three inline-base64 referenceImages ON THE INSTANCE (mirroring the
        # image/lastFrame placement — NOT the parameters block).
        if "image_urls" in params_copy:
            image_urls = params_copy.pop("image_urls")
            if image_urls:
                reference_images = []
                for reference_url in list(image_urls)[:_MAX_REFERENCE_IMAGES]:
                    if not reference_url:
                        continue
                    base64_data, mime_type = fetch_image_as_base64(reference_url)
                    reference_images.append(
                        {"image": {"bytesBase64Encoded": base64_data, "mimeType": mime_type}, "referenceType": "asset"}
                    )
                if reference_images:
                    instance["referenceImages"] = reference_images

        wants_audio = _audio_preference(params_copy)
        for audio_key in _AUDIO_PARAM_KEYS:
            params_copy.pop(audio_key, None)
        _reject_unrenderable_audio(model, wants_audio)

        params_copy["personGeneration"] = params_copy.get("personGeneration") or _person_generation_for_request(
            model, instance, params_copy
        )

        parameters: Final = GeminiVideoGenerationParameters(**params_copy)

        request_body_obj: Final = GeminiVideoGenerationRequest(instances=[instance], parameters=parameters)

        request_data: Final = request_body_obj.model_dump(exclude_none=True)

        return request_data, [], api_base

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: dict | None = None,
    ) -> VideoObject:
        """
        Transform the Veo video creation response.

        Veo returns:
        {
            "name": "operations/generate_1234567890",
            "metadata": {...},
            "done": false,
            "error": {...}
        }

        We return this as a VideoObject with:
        - id: operation name (used for polling)
        - status: "processing"
        - usage: includes duration_seconds and optional video_resolution for cost calculation
        """
        response_data: Final = _json_payload(raw_response)

        # Parse response using Pydantic model for type safety
        try:
            operation_response: Final = GeminiLongRunningOperationResponse.model_validate(response_data)
        except Exception as e:
            raise ValueError(f"Failed to parse operation response: {e}")

        operation_name: Final = operation_response.name
        if not operation_name:
            raise ValueError(f"No operation name in Veo response: {response_data}")

        if custom_llm_provider:
            video_id = encode_video_id_with_provider(operation_name, custom_llm_provider, model)
        else:
            video_id = operation_name

        video_obj: Final = VideoObject(
            id=video_id,
            object="video",
            status="processing",
            model=model,
        )

        usage_data: Final[dict[str, float | str]] = {}
        if request_data:
            parameters: Final = request_data.get("parameters", {})
            duration: Final = parameters.get("durationSeconds") or DEFAULT_GOOGLE_VIDEO_DURATION_SECONDS
            if duration is not None:
                try:
                    usage_data["duration_seconds"] = float(duration)
                except (ValueError, TypeError):
                    pass
            video_resolution: Final = _usage_video_resolution_from_parameters(parameters)
            if video_resolution is not None:
                usage_data["video_resolution"] = video_resolution

        video_obj.usage = usage_data
        return video_obj

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        """
        Transform the video status retrieve request for Veo API.

        Veo polls operations at:
        GET https://generativelanguage.googleapis.com/v1beta/{operation_name}
        """
        operation_name: Final = extract_original_video_id(video_id)
        url: Final = f"{api_base.rstrip('/')}/v1beta/{operation_name}"
        params: Final[dict[str, object]] = {}

        return url, params

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        """
        Transform the Veo operation status response.

        Veo returns:
        {
            "name": "operations/generate_1234567890",
            "done": false  # or true when complete
        }

        When done=true:
        {
            "name": "operations/generate_1234567890",
            "done": true,
            "response": {
                "generateVideoResponse": {
                    "generatedSamples": [
                        {
                            "video": {
                                "uri": "files/abc123..."
                            }
                        }
                    ]
                }
            }
        }
        """
        response_data: Final = _json_payload(raw_response)
        # Parse response using Pydantic model for type safety
        operation_response: Final = GeminiLongRunningOperationResponse.model_validate(response_data)

        operation_name: Final = operation_response.name
        is_done: Final = operation_response.done

        if custom_llm_provider:
            video_id = encode_video_id_with_provider(operation_name, custom_llm_provider, None)
        else:
            video_id = operation_name

        error_data = operation_response.error
        if is_done and error_data is None:
            generate_video_response = (
                operation_response.response.generateVideoResponse if operation_response.response else None
            )
            if generate_video_response is not None:
                if (
                    not generate_video_response.generatedSamples
                    and (generate_video_response.raiMediaFilteredCount or 0) > 0
                ):
                    reasons = generate_video_response.raiMediaFilteredReasons or []
                    error_data = {
                        "code": "rai_media_filtered",
                        "message": "Video generation failed: all samples were filtered by Responsible AI policies. "
                        + " ".join(reasons),
                    }

        if error_data:
            status = "failed"
        elif is_done:
            status = "completed"
        else:
            status = "processing"

        video_obj = VideoObject(
            id=video_id,
            object="video",
            status=status,
            error=error_data,
        )
        return video_obj

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        variant: str | None = None,
    ) -> tuple[str, dict]:
        """
        Transform the video content request for Veo API.

        For Veo, we need to:
        1. Get operation status to extract video URI
        2. Return download URL for the video
        """
        operation_name: Final = extract_original_video_id(video_id)

        status_url: Final = f"{api_base.rstrip('/')}/v1beta/{operation_name}"
        client: Final = litellm.module_level_client
        status_response: Final = client.get(url=status_url, headers=headers)
        status_response.raise_for_status()
        response_data: Final = _json_payload(status_response)

        operation_response: Final = GeminiLongRunningOperationResponse.model_validate(response_data)

        if not operation_response.done:
            raise ValueError(
                "Video generation is not complete yet. Please check status with video_status() before downloading."
            )

        if not operation_response.response:
            raise ValueError("No response data in completed operation")

        generate_video_response = operation_response.response.generateVideoResponse
        generated_samples = generate_video_response.generatedSamples
        if not generated_samples:
            reasons = generate_video_response.raiMediaFilteredReasons or []
            raise ValueError("No generated samples in completed operation. " + " ".join(reasons))
        download_url = generated_samples[0].video.uri

        params: Final[dict[str, object]] = {}

        return download_url, params

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        """
        Transform the Veo video content download response.
        Returns the video bytes directly.
        """
        return raw_response.content

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        extra_body: Mapping[str, object] | None = None,
    ) -> tuple[str, dict]:
        """
        Video remix is not supported by Veo API.
        """
        raise NotImplementedError(
            "Video remix is not supported by Google Veo. Please use video_generation() to create new videos."
        )

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        """Video remix is not supported."""
        raise NotImplementedError("Video remix is not supported by Google Veo.")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        after: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        extra_query: Mapping[str, object] | None = None,
    ) -> tuple[str, dict]:
        """
        Video list is not supported by Veo API.
        """
        raise NotImplementedError(
            "Video list is not supported by Google Veo. "
            "Use the operations endpoint directly if you need to list operations."
        )

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> dict[str, str]:
        """Video list is not supported."""
        raise NotImplementedError("Video list is not supported by Google Veo.")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        """
        Video delete is not supported by Veo API.
        """
        raise NotImplementedError(
            "Video delete is not supported by Google Veo. Videos are automatically cleaned up by Google."
        )

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        """Video delete is not supported."""
        raise NotImplementedError("Video delete is not supported by Google Veo.")

    def transform_video_create_character_request(self, name, video: object, api_base, litellm_params, headers):
        raise NotImplementedError("video create character is not supported for Gemini")

    def transform_video_create_character_response(self, raw_response, logging_obj):
        raise NotImplementedError("video create character is not supported for Gemini")

    def transform_video_get_character_request(self, character_id, api_base, litellm_params, headers):
        raise NotImplementedError("video get character is not supported for Gemini")

    def transform_video_get_character_response(self, raw_response, logging_obj):
        raise NotImplementedError("video get character is not supported for Gemini")

    def transform_video_edit_request(
        self,
        prompt,
        video_id,
        api_base,
        litellm_params,
        headers,
        video_file=None,
        extra_body=None,
        prefetched_source_data=None,
    ):
        raise NotImplementedError("video edit is not supported for Gemini")

    def transform_video_edit_response(
        self,
        raw_response,
        logging_obj,
        custom_llm_provider=None,
        request_data=None,
    ):
        raise NotImplementedError("video edit is not supported for Gemini")

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
        raise NotImplementedError("video extension is not supported for Gemini")

    def transform_video_extension_response(self, raw_response, logging_obj, custom_llm_provider=None):
        raise NotImplementedError("video extension is not supported for Gemini")

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        from ..common_utils import GeminiError

        return GeminiError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
