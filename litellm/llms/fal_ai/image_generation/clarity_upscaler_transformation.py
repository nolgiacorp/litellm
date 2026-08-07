from collections.abc import Mapping
from typing import TYPE_CHECKING

import httpx

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import OpenAIImageGenerationOptionalParams
from litellm.types.utils import ImageObject, ImageResponse

from .transformation import FalAIBaseConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = object


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
        api_base: "str | None",
        api_key: "str | None",
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: "bool | None" = None,
    ) -> str:
        base_url = (api_base or get_secret_str("FAL_AI_API_BASE") or self.DEFAULT_BASE_URL).rstrip("/")
        endpoint = model if model.startswith("fal-ai/") else f"fal-ai/{model}"
        return f"{base_url}/{endpoint}"

    def get_supported_openai_params(self, model: str) -> "list[OpenAIImageGenerationOptionalParams]":
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
        encoding: object,
        api_key: "str | None" = None,
        json_mode: "bool | None" = None,
    ) -> ImageResponse:
        try:
            response_data = raw_response.json()
        except ValueError as e:
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
            *(
                ImageObject(
                    url=image.get("url"),
                    b64_json=image.get("b64_json"),
                    provider_specific_fields=self._dimension_fields(image),
                )
                for image in images
            ),
        ]
        return model_response

    @staticmethod
    def _dimension_fields(
        image: "Mapping[str, object]",
    ) -> "dict | None":  # mutable-ok: ImageObject.provider_specific_fields expects a dict
        """
        Delivered image dimensions, carried so the fal cost calculator can price
        the upscale at fal's published per-megapixel rate (NOL-535). Output size
        depends on the input image and upscale_factor, so it is only knowable
        from the response.
        """
        width = image.get("width")
        height = image.get("height")
        if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
            return {"width": width, "height": height}  # mutable-ok: ImageObject.provider_specific_fields expects a dict
        return None
