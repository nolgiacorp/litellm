"""
Topaz Labs image enhancement over the OpenAI image-edit surface.

The provider's `POST /image/v1/enhance` route is multipart image-in / image-out
with no prompt, which is the image-EDIT shape rather than the image-variation
shape its sibling module models. `CallTypes` has no `image_variation` member and
the proxy only mounts `/images/generations` and `/images/edits`, so the
variations config is unreachable through the proxy and this one serves the route.
"""

import base64
import time
from typing import TYPE_CHECKING, Final

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes, ImageObject, ImageResponse

from ..common_utils import TOPAZ_IMAGE_ENHANCE_MODELS, TopazException, TopazModelInfo

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


# Topaz accepts jpeg/jpg/png/tiff/tif. JPEG is the only one whose worst case
# stays inside the base64 response budget its callers hold: a 4096x4096 PNG
# measured 9.6MB on a synthetic source, and a dense photographic one runs
# several times that, which base64 pushes past a 48MB reader cap.
TOPAZ_IMAGE_OUTPUT_FORMAT: Final = "jpeg"

_TOPAZ_SIZE_FIELDS: Final = ("output_width", "output_height")


class TopazImageEditConfig(BaseImageEditConfig):
    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: BaseImageEditConfig signature
        return ["n", "size", "user"]  # mutable-ok: contract returns a list

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:  # mutable-ok: BaseImageEditConfig signature
        self._reject_multiple_images(image_edit_optional_params.get("n"), model)
        size: Final = image_edit_optional_params.get("size")
        if size is None:
            return {}  # mutable-ok: contract returns a dict
        width, separator, height = str(size).partition("x")
        if not separator or not width.isdigit() or not height.isdigit():
            raise litellm.BadRequestError(
                message=(
                    f"Topaz model '{model}' takes `size` as '<width>x<height>' in pixels, got {size!r}. "
                    "Topaz bills the OUTPUT geometry, so an unreadable size cannot be defaulted."
                ),
                model=model,
                llm_provider=litellm.LlmProviders.TOPAZ.value,
            )
        return dict(zip(_TOPAZ_SIZE_FIELDS, (width, height)))  # mutable-ok: contract returns a dict

    def validate_environment(
        self,
        headers: dict,  # mutable-ok: BaseImageEditConfig signature
        model: str,
        api_key: str | None = None,
        litellm_params: dict | None = None,  # mutable-ok: BaseImageEditConfig signature
        api_base: str | None = None,
    ) -> dict:  # mutable-ok: BaseImageEditConfig signature
        resolved_api_key: Final = TopazModelInfo.get_api_key(api_key)
        if not resolved_api_key:
            raise ValueError("TOPAZ_API_KEY is not set. Set it in the environment or pass `api_key=..`")
        return {  # mutable-ok: the shared handler updates the returned headers in place
            **headers,
            "X-API-Key": resolved_api_key,
            "Accept": f"image/{TOPAZ_IMAGE_OUTPUT_FORMAT}",
        }

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,  # mutable-ok: BaseImageEditConfig signature
    ) -> str:
        resolved_api_base: Final = TopazModelInfo.get_api_base(api_base or litellm_params.get("api_base"))
        if not resolved_api_base:
            raise ValueError("Topaz api_base could not be resolved")
        return f"{resolved_api_base.rstrip('/')}/image/v1/enhance"

    def transform_image_edit_request(
        self,
        model: str,
        prompt: str | None,
        image: FileTypes | None,
        image_edit_optional_request_params: dict,  # mutable-ok: BaseImageEditConfig signature
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: BaseImageEditConfig signature
    ) -> tuple[dict, RequestFiles]:  # mutable-ok: BaseImageEditConfig signature
        engine: Final = model.removeprefix(f"{litellm.LlmProviders.TOPAZ.value}/")
        self._reject_unknown_engine(engine, model)
        self._reject_prompt(prompt, model)
        source: Final = image[0] if isinstance(image, list) else image
        if source is None:
            raise litellm.BadRequestError(
                message=(
                    f"Topaz model '{model}' enhances an existing image and requires one to be uploaded as `image`."
                ),
                model=model,
                llm_provider=litellm.LlmProviders.TOPAZ.value,
            )
        sizing: Final = {  # mutable-ok: multipart form fields, sent as a dict
            key: value
            for key, value in image_edit_optional_request_params.items()
            if key in _TOPAZ_SIZE_FIELDS and value is not None
        }
        form_fields: Final = {  # mutable-ok: httpx sends the multipart fields as a dict
            "model": engine,
            "output_format": TOPAZ_IMAGE_OUTPUT_FORMAT,
            **sizing,
        }
        return form_fields, {"image": source}  # mutable-ok: httpx takes the file parts as a dict

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> ImageResponse:
        enhanced: Final = raw_response.content
        if not enhanced:
            raise self.get_error_class(
                error_message="Topaz returned an empty image",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )
        return ImageResponse(
            created=int(time.time()),
            data=[  # mutable-ok: ImageResponse.data is a list
                ImageObject(
                    b64_json=base64.b64encode(enhanced).decode("utf-8"),
                    url=None,
                    revised_prompt=None,
                )
            ],
        )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict | httpx.Headers,  # mutable-ok: BaseImageEditConfig signature
    ) -> BaseLLMException:
        return TopazException(status_code=status_code, message=error_message, headers=headers)

    @staticmethod
    def _reject_unknown_engine(engine: str, model: str) -> None:
        if engine in TOPAZ_IMAGE_ENHANCE_MODELS:
            return
        raise litellm.BadRequestError(
            message=(
                f"Topaz model '{model}' is not a Topaz image engine. Supported engines are: "
                f"{', '.join(TOPAZ_IMAGE_ENHANCE_MODELS)}. Topaz answers an unknown code with an opaque "
                "'Unknown model error', so the check is done here where the accepted set can be named."
            ),
            model=model,
            llm_provider=litellm.LlmProviders.TOPAZ.value,
        )

    @staticmethod
    def _reject_multiple_images(n: object, model: str) -> None:
        if n is None:
            return
        requested: Final = str(n).strip()
        if requested in ("1", "1.0"):
            return
        raise litellm.BadRequestError(
            message=(
                f"Topaz model '{model}' enhances the uploaded image and returns exactly one result, so `n` must be "
                f"1, got {n!r}. Returning fewer images than were asked for, and billing for them, would be worse "
                "than refusing the request."
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
                f"Topaz model '{model}' does not support `prompt`. Topaz enhances the uploaded image from its own "
                "pixels and the requested output size alone; billing an enhancement that ignored the caller's "
                "instructions would be worse than refusing it."
            ),
            model=model,
            llm_provider=litellm.LlmProviders.TOPAZ.value,
        )
