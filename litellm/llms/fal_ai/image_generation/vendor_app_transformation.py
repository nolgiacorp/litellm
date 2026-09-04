import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import PositiveInt, TypeAdapter, ValidationError

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import OpenAIImageGenerationOptionalParams

from .transformation import FalAIBaseConfig

_WIDTH_X_HEIGHT = re.compile(r"^\s*(\d+)\s*x\s*(\d+)\s*$")

_REVE_ASPECT_RATIOS: tuple[str, ...] = (
    "4:1",
    "3:1",
    "21:9",
    "2:1",
    "17:9",
    "16:9",
    "3:2",
    "4:3",
    "5:4",
    "1:1",
    "4:5",
    "3:4",
    "2:3",
    "9:16",
    "1:2",
    "1:3",
    "1:4",
)


@dataclass(frozen=True, slots=True)
class _ImageSize:
    width: PositiveInt
    height: PositiveInt


_IMAGE_SIZE = TypeAdapter(_ImageSize)


def _preset_dimensions(alias: str) -> tuple[int, int] | None:
    match alias:
        case "square_hd":
            return (1024, 1024)
        case "square":
            return (512, 512)
        case "portrait_4_3":
            return (768, 1024)
        case "portrait_16_9":
            return (576, 1024)
        case "landscape_4_3":
            return (1024, 768)
        case "landscape_16_9":
            return (1024, 576)
        case _:
            return None


def dimensions(size: object) -> tuple[int, int] | None:
    """
    Pixel dimensions of an OpenAI `WxH` size, a fal image_size alias, or a fal
    ImageSize object; None when the value carries no usable dimensions.
    """
    if isinstance(size, str):
        matched = _WIDTH_X_HEIGHT.match(size)
        if matched is None:
            return _preset_dimensions(size)
        width, height = int(matched.group(1)), int(matched.group(2))
        return (width, height) if width > 0 and height > 0 else None
    try:
        parsed = _IMAGE_SIZE.validate_python(size)
    except ValidationError:
        return None
    return (parsed.width, parsed.height)


def fal_image_size(size: object) -> object:
    """
    OpenAI `WxH` becomes fal's ImageSize object. Anything else (a fal alias such
    as square_hd or auto_2K, or an ImageSize object) is already fal's dialect.
    """
    if not isinstance(size, str):
        return size
    matched = _WIDTH_X_HEIGHT.match(size)
    if matched is None:
        return size
    width, height = int(matched.group(1)), int(matched.group(2))
    return {"width": width, "height": height}  # mutable-ok: fal's ImageSize is a JSON object


def _ratio(aspect_ratio: str) -> float:
    width, _, height = aspect_ratio.partition(":")
    return int(width) / int(height)


def reve_aspect_ratio(size: object) -> str:
    """
    The Reve aspect_ratio nearest to an OpenAI `WxH` size, a fal image_size
    alias, or a fal ImageSize object. A value that is already a Reve ratio
    passes through; one without usable dimensions leaves the choice to Reve.
    """
    if isinstance(size, str) and (size == "auto" or size in _REVE_ASPECT_RATIOS):
        return size
    dims = dimensions(size)
    if dims is None:
        return "auto"
    target = math.log(dims[0] / dims[1])
    return min(_REVE_ASPECT_RATIOS, key=lambda ratio: abs(math.log(_ratio(ratio)) - target))


class FalAIVendorAppConfig(FalAIBaseConfig):
    """
    Shared configuration for fal apps published under their vendor's own
    namespace (reve/..., bytedance/seedream/v5/..., ideogram/v4, alibaba/...).

    The endpoint is the model id itself, so one config serves every route of a
    family (text-to-image and edit) instead of pinning a single fal-ai/... path.
    Params fal understands natively (image_url, image_urls, seed, negative_prompt,
    image_size aliases, ...) reach the app verbatim through litellm's
    provider-specific passthrough; only the OpenAI-shaped ones are translated.
    """

    APP_OWNER: str = ""
    SUPPORTED_OPENAI_PARAMS: tuple[OpenAIImageGenerationOptionalParams, ...] = ("n", "response_format", "size")

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        base_url = (api_base or get_secret_str("FAL_AI_API_BASE") or self.DEFAULT_BASE_URL).rstrip("/")
        endpoint = model if model.startswith(f"{self.APP_OWNER}/") else f"{self.APP_OWNER}/{model}"
        return f"{base_url}/{endpoint}"

    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIImageGenerationOptionalParams]:  # mutable-ok: BaseImageGenerationConfig declares a list
        return [*self.SUPPORTED_OPENAI_PARAMS]  # mutable-ok: BaseImageGenerationConfig declares a list

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: BaseImageGenerationConfig declares a dict
        mapped = (self._openai_param(key, value, model, drop_params) for key, value in non_default_params.items())
        translated = dict(  # mutable-ok: merged into the dict below
            pair for pair in mapped if pair is not None and pair[0] not in optional_params
        )
        return {**optional_params, **translated}  # mutable-ok: BaseImageGenerationConfig declares a dict

    def _openai_param(self, key: str, value: object, model: str, drop_params: bool) -> tuple[str, object] | None:
        if key not in self.SUPPORTED_OPENAI_PARAMS:
            if drop_params:
                return None
            raise ValueError(
                f"Parameter {key} is not supported for model {model}. "
                f"Supported parameters are {self.SUPPORTED_OPENAI_PARAMS}. "
                "Set drop_params=True to drop unsupported parameters."
            )
        match key:
            case "n":
                return ("num_images", value)
            case "size":
                return self._size_param(value)
            case _:
                return None

    def _size_param(self, size: object) -> tuple[str, object]:
        return ("image_size", fal_image_size(size))

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, str],
    ) -> dict[str, object]:  # mutable-ok: the request body the http handler serialises
        body = dict(self._body_params(optional_params))  # mutable-ok: the request body the http handler serialises
        return {"prompt": prompt, **body}  # mutable-ok: the request body the http handler serialises

    def _body_params(self, optional_params: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
        return tuple(optional_params.items())


class FalAIReveConfig(FalAIVendorAppConfig):
    """
    Reve 2.1: reve/2.1/text-to-image and reve/2.1/edit (edit takes image_url
    alongside the prompt).

    Reve sizes by aspect_ratio rather than image_size, so an OpenAI `size` and a
    passthrough fal image_size alias or object both land on the nearest ratio
    Reve accepts. fal bills $0.25 per image on both routes.

    Documentation: https://fal.ai/models/reve/2.1/text-to-image
    """

    APP_OWNER: str = "reve"

    def _size_param(self, size: object) -> tuple[str, object]:
        return ("aspect_ratio", reve_aspect_ratio(size))

    def _body_params(self, optional_params: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
        rest = tuple(item for item in optional_params.items() if item[0] != "image_size")
        if "image_size" not in optional_params or "aspect_ratio" in optional_params:
            return rest
        return (*rest, ("aspect_ratio", reve_aspect_ratio(optional_params["image_size"])))


class FalAISeedreamV5Config(FalAIVendorAppConfig):
    """
    Seedream 5.0 Pro: bytedance/seedream/v5/pro/text-to-image and
    bytedance/seedream/v5/pro/edit (edit takes image_urls, up to 10).

    image_size accepts the fal aliases, auto_1K / auto_2K, or a width/height
    object; total pixels must fall between 1024x1024 and 2048x2048. fal bills
    $0.0675 per image up to 1536x1536 and $0.135 up to 2048x2048.

    Documentation: https://fal.ai/models/bytedance/seedream/v5/pro/text-to-image
    """

    APP_OWNER: str = "bytedance"


def _rendering_speed(quality: object, model: str) -> str | None:
    match quality:
        case "low":
            return "TURBO"
        case "medium":
            return "BALANCED"
        case "high":
            return "QUALITY"
        case "auto":
            return None
        case _:
            raise ValueError(f"quality {quality!r} is not supported for model {model}. Use low, medium, high or auto.")


class FalAIIdeogramV4Config(FalAIVendorAppConfig):
    """
    Ideogram v4: ideogram/v4 and ideogram/v4/image-to-image (image-to-image
    takes image_url and strength).

    OpenAI `quality` low / medium / high selects rendering_speed TURBO /
    BALANCED / QUALITY; expansion_model, seed and image_size aliases pass
    through. fal bills per output megapixel: $0.0075 TURBO, $0.015 BALANCED,
    $0.025 QUALITY.

    Documentation: https://fal.ai/models/ideogram/v4
    """

    APP_OWNER: str = "ideogram"
    SUPPORTED_OPENAI_PARAMS: tuple[OpenAIImageGenerationOptionalParams, ...] = (
        "n",
        "quality",
        "response_format",
        "size",
    )

    def _openai_param(self, key: str, value: object, model: str, drop_params: bool) -> tuple[str, object] | None:
        if key != "quality":
            return super()._openai_param(key, value, model, drop_params)
        rendering_speed = _rendering_speed(value, model)
        return None if rendering_speed is None else ("rendering_speed", rendering_speed)


class FalAIQwenImage3Config(FalAIVendorAppConfig):
    """
    Qwen Image 3: alibaba/qwen-image-3/text-to-image and
    alibaba/qwen-image-3/edit (edit takes image_urls, 1 to 3).

    image_size accepts the fal aliases or a width/height object between
    512x512 and 2048x2048 total pixels; negative_prompt, seed and
    enable_prompt_expansion pass through. fal bills $0.04 per image at 1K and
    $0.075 at 2K.

    Documentation: https://fal.ai/models/alibaba/qwen-image-3/text-to-image
    """

    APP_OWNER: str = "alibaba"
