from collections.abc import Mapping, Sequence
from math import gcd
from types import MappingProxyType
from typing import TYPE_CHECKING, Any  # noqa: TID251  # base transformation contracts type these payloads as Any

import httpx

from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.llms.minimax.common_utils import (
    EMPTY_MAP,
    drop_none_values,
    minimax_bearer_headers,
    resolve_minimax_media_api_base,
    strip_minimax_prefix,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import ImageObject, ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

MINIMAX_IMAGE_PASSTHROUGH_PARAMS = frozenset(
    {"aspect_ratio", "width", "height", "seed", "prompt_optimizer", "subject_reference", "image_url"}
)

_V1_ERROR_HTTP_STATUS: Mapping[int, int] = MappingProxyType(
    {1002: 429, 1004: 401, 1008: 402, 2049: 401}  # mutable-ok: frozen constant lookup table
)

_SIZE_TO_ASPECT_RATIO: Mapping[str, str] = MappingProxyType(
    {  # mutable-ok: frozen constant lookup table
        "1024x1024": "1:1",
        "1280x720": "16:9",
        "1920x1080": "16:9",
        "720x1280": "9:16",
        "1080x1920": "9:16",
        "1152x864": "4:3",
        "864x1152": "3:4",
        "1248x832": "3:2",
        "832x1248": "2:3",
        "1344x576": "21:9",
    }
)

_RESERVED_REQUEST_KEYS = frozenset({"model", "prompt", "user", "size", "extra_body"})


def _aspect_ratio_from_size(size: str) -> str:
    mapped = _SIZE_TO_ASPECT_RATIO.get(size)
    if mapped:
        return mapped
    width, _, height = size.partition("x")
    try:
        parsed_width, parsed_height = int(width), int(height)
    except ValueError:
        return size.replace("x", ":")
    if parsed_width <= 0 or parsed_height <= 0:
        return size.replace("x", ":")
    divisor = gcd(parsed_width, parsed_height)
    return f"{parsed_width // divisor}:{parsed_height // divisor}"


class MinimaxImageGenerationConfig(BaseImageGenerationConfig):
    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIImageGenerationOptionalParams]:  # mutable-ok: BaseImageGenerationConfig contract returns list
        return ["n", "response_format", "size"]

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, Any],
        litellm_params: Mapping[str, Any],
        stream: bool | None = None,
    ) -> str:
        return f"{resolve_minimax_media_api_base(api_base)}/v1/image_generation"

    def validate_environment(
        self,
        headers: Mapping[str, Any],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, Any],
        litellm_params: Mapping[str, Any],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:  # mutable-ok: BaseImageGenerationConfig contract returns dict
        return minimax_bearer_headers(headers, api_key)

    def map_openai_params(
        self,
        non_default_params: Mapping[str, Any],
        optional_params: Mapping[str, Any],
        model: str,
        drop_params: bool,
    ) -> dict:  # mutable-ok: BaseImageGenerationConfig contract returns dict
        supported = self.get_supported_openai_params(model)
        unsupported = tuple(
            key
            for key in non_default_params
            if key != "size" and key not in supported and key not in MINIMAX_IMAGE_PASSTHROUGH_PARAMS
        )
        if unsupported and not drop_params:
            raise ValueError(
                f"Parameters {list(unsupported)} are not supported for model {model}. Supported "
                f"parameters are {supported}. Set drop_params=True to drop unsupported parameters."
            )
        size = non_default_params.get("size")
        aspect_ratio = _aspect_ratio_from_size(size) if isinstance(size, str) and size else None
        return {
            **optional_params,
            **{key: value for key, value in non_default_params.items() if key != "size" and key not in unsupported},
            **({"aspect_ratio": aspect_ratio} if aspect_ratio else EMPTY_MAP),
        }

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: Mapping[str, Any],
        litellm_params: Mapping[str, Any],
        headers: Mapping[str, Any],
    ) -> dict:  # mutable-ok: BaseImageGenerationConfig contract returns dict
        extra_body = optional_params.get("extra_body")
        params = {
            **optional_params,
            **(extra_body if isinstance(extra_body, dict) else EMPTY_MAP),
        }
        forwarded = drop_none_values(
            {key: value for key, value in params.items() if key not in _RESERVED_REQUEST_KEYS and key != "image_url"}
        )
        image_url = params.get("image_url")
        reference = (
            ({"type": "character", "image_file": image_url.strip()},)  # mutable-ok: MiniMax wire shape is a JSON object
            if "subject_reference" not in forwarded and isinstance(image_url, str) and image_url.strip()
            else None
        )
        return dict(
            drop_none_values(
                {
                    "model": strip_minimax_prefix(model),
                    "prompt": prompt,
                    **forwarded,
                    "subject_reference": reference if reference is not None else forwarded.get("subject_reference"),
                }
            )
        )

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: Mapping[str, Any],
        optional_params: Mapping[str, Any],
        litellm_params: Mapping[str, Any],
        encoding: Any,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        self._raise_for_status(raw_response)
        response_data = self._parse_json(raw_response)
        self._raise_for_minimax_error(raw_response, response_data)
        data = response_data.get("data") or EMPTY_MAP
        urls = data.get("image_urls") or ()
        b64_images = data.get("image_base64") or ()
        images = [
            *(ImageObject(url=url, b64_json=None) for url in urls if isinstance(url, str) and url),
            *(ImageObject(url=None, b64_json=b64) for b64 in b64_images if isinstance(b64, str) and b64),
        ]
        if not images:
            raise ValueError(f"MiniMax image generation returned no images: {response_data}")
        model_response.data = images
        return model_response

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
        except Exception:
            return raw_response.text
        if not isinstance(body, dict):
            return raw_response.text
        error = body.get("error")
        message = error.get("message") if isinstance(error, dict) else None
        if not (isinstance(message, str) and message):
            base_resp = body.get("base_resp")
            message = base_resp.get("status_msg") if isinstance(base_resp, dict) else None
        return message if isinstance(message, str) and message else raw_response.text

    @staticmethod
    def _parse_json(raw_response: httpx.Response) -> Mapping[str, Any]:
        try:
            return raw_response.json()
        except Exception as e:
            raise ValueError(f"Error parsing MiniMax image generation response: {e}") from e

    def _raise_for_minimax_error(self, raw_response: httpx.Response, response_data: Mapping[str, Any]) -> None:
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
