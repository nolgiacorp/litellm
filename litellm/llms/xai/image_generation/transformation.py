from typing import TYPE_CHECKING, Any

import httpx

from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.llms.xai.common_utils import XAIModelInfo
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

_SIZE_TO_ASPECT_RATIO = {
    "1024x1024": "1:1",
    "1280x720": "16:9",
    "1920x1080": "16:9",
    "720x1280": "9:16",
    "1080x1920": "9:16",
    "1024x768": "4:3",
    "768x1024": "3:4",
}

_PASSTHROUGH_PARAMS = frozenset({"aspect_ratio", "resolution"})


class XAIImageGenerationConfig(BaseImageGenerationConfig):
    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
        return ["n", "response_format"]

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        base = (XAIModelInfo.get_api_base(api_base) or "https://api.x.ai").rstrip("/")
        base = base.removesuffix("/v1").rstrip("/")
        return f"{base}/v1/images/generations"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        final_api_key = XAIModelInfo.get_api_key(api_key)
        if not final_api_key:
            raise ValueError("XAI_API_KEY is not set")
        return {**headers, "Authorization": f"Bearer {final_api_key}"}

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported = self.get_supported_openai_params(model)
        result: dict[str, Any] = dict(optional_params)
        for key, value in non_default_params.items():
            if key == "size" and isinstance(value, str):
                result["aspect_ratio"] = _SIZE_TO_ASPECT_RATIO.get(value, value.replace("x", ":"))
            elif key in supported or key in _PASSTHROUGH_PARAMS:
                result[key] = value
            elif drop_params:
                continue
            else:
                raise ValueError(
                    f"Parameter {key} is not supported for model {model}. Supported "
                    f"parameters are {supported}. Set drop_params=True to drop unsupported "
                    f"parameters."
                )
        return result

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        params: dict[str, Any] = dict(optional_params)
        extra_body = params.pop("extra_body", None)
        merged = {**params, **extra_body} if isinstance(extra_body, dict) else params
        reserved = {"model", "prompt", "user"}
        forwarded = {key: value for key, value in merged.items() if key not in reserved and value is not None}
        return {
            "model": model.removeprefix("xai/"),
            "prompt": prompt,
            **forwarded,
        }

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        try:
            response_data = raw_response.json()
        except ValueError as e:
            raise self.get_error_class(
                error_message=f"Error parsing xAI image generation response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )
        if not model_response.data:
            model_response.data = []
        images = response_data.get("data", [])
        if isinstance(images, list):
            for image_data in images:
                if isinstance(image_data, dict):
                    model_response.data.append(
                        ImageObject(
                            url=image_data.get("url", None),
                            b64_json=image_data.get("b64_json", None),
                        )
                    )
        return model_response
