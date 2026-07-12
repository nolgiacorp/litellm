from typing import Any, List, Optional

import httpx

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import OpenAIImageGenerationOptionalParams
from litellm.types.utils import ImageObject, ImageResponse

from .transformation import FalAIBaseConfig, LiteLLMLoggingObj


class FalAIClarityUpscalerConfig(FalAIBaseConfig):
    """
    Configuration for Fal AI's Clarity Upscaler (fal-ai/clarity-upscaler).

    An image-to-image super-resolution model: it takes an image_url plus an
    upscale_factor and returns a single upscaled image. Unlike the text-to-image
    fal models it responds with a single `image` object rather than an `images`
    array, so the response transform is overridden.

    Documentation: https://fal.ai/models/fal-ai/clarity-upscaler
    """

    IGNORED_OPENAI_PARAMS: frozenset[str] = frozenset({"response_format", "n", "size"})

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        base_url = (api_base or get_secret_str("FAL_AI_API_BASE") or self.DEFAULT_BASE_URL).rstrip("/")
        endpoint = model if model.startswith("fal-ai/") else f"fal-ai/{model}"
        return f"{base_url}/{endpoint}"

    def get_supported_openai_params(self, model: str) -> List[OpenAIImageGenerationOptionalParams]:
        return ["response_format", "n", "size"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        return {
            **optional_params,
            **{
                key: value
                for key, value in non_default_params.items()
                if key not in optional_params and key not in self.IGNORED_OPENAI_PARAMS
            },
        }

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        return {
            **({"prompt": prompt} if prompt else {}),
            **optional_params,
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
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ImageResponse:
        try:
            response_data = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Error transforming image generation response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        single = response_data.get("image")
        images = (
            [single]
            if isinstance(single, dict)
            else [image for image in response_data.get("images", []) if isinstance(image, dict)]
        )
        model_response.data = [
            *(model_response.data or []),
            *(ImageObject(url=image.get("url"), b64_json=image.get("b64_json")) for image in images),
        ]
        return model_response
