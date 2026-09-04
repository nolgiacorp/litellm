from collections.abc import Mapping

import litellm
from litellm.llms.fal_ai.image_generation.vendor_app_transformation import dimensions
from litellm.types.utils import ImageObject, ImageResponse
from litellm.utils import _get_model_cost_key


def _cost_entry(model: str) -> "dict | None":  # mutable-ok: litellm.model_cost stores raw dict entries
    """
    Price-map entry for a fal model, read directly from litellm.model_cost.

    get_model_info() cannot serve the per-pixel path: it materialises a
    ModelInfoBase whose fields are enumerated explicitly, and output_cost_per_pixel
    is not among them, so the rate would be silently dropped (NOL-535). Reading
    the map directly also keeps the degrade-to-0.0 behaviour on a miss - this
    runs after the image has been generated and paid for, so raising would turn
    a pricing gap into a failed generation for the caller.

    Keys are resolved through _get_model_cost_key so a differently cased but
    otherwise valid model name (fal's config selector and endpoint are
    case-insensitive) still finds its lowercase price-map entry, as the previous
    get_model_info() path did.
    """
    provider = litellm.LlmProviders.FAL_AI.value
    for cost_key in (f"{provider}/{model}", model, model.split("/")[-1]):
        matched_key = _get_model_cost_key(cost_key)
        entry = litellm.model_cost.get(matched_key) if matched_key is not None else None
        if entry:
            return entry
    return None


def _image_cost(image: ImageObject, output_cost_per_pixel: float, output_cost_per_image: float) -> float:
    fields = image.provider_specific_fields
    width = fields.get("width") if fields else None
    height = fields.get("height") if fields else None
    if output_cost_per_pixel and isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
        return output_cost_per_pixel * width * height
    return output_cost_per_image


def _is_2k(size: object) -> bool:
    if isinstance(size, str) and size.lower() == "auto_2k":
        return True
    resolved = dimensions(size)
    return resolved is not None and resolved[0] * resolved[1] > 1536 * 1536


def _vendor_image_rate(model: str, optional_params: Mapping[str, object], default_rate: float) -> float:
    normalized_model = model.lower()
    if "bytedance/seedream/v5/pro/" in normalized_model:
        return 0.135 if _is_2k(optional_params.get("image_size") or optional_params.get("size")) else 0.0675
    if normalized_model.startswith("ideogram/v4"):
        rendering_speed = optional_params.get("rendering_speed")
        if rendering_speed is None:
            rendering_speed = {  # mutable-ok: local lookup table is not exposed or mutated
                "low": "TURBO",
                "high": "QUALITY",
            }.get(str(optional_params.get("quality") or "medium").lower(), "BALANCED")
        per_megapixel = {  # mutable-ok: local lookup table is not exposed or mutated
            "TURBO": 0.0075,
            "QUALITY": 0.025,
        }.get(str(rendering_speed).upper(), 0.015)
        resolved = dimensions(optional_params.get("image_size") or optional_params.get("size"))
        megapixels = resolved[0] * resolved[1] / 1_000_000 if resolved is not None else 1.0
        return per_megapixel * megapixels
    if "alibaba/qwen-image-3/" in normalized_model:
        return 0.075 if _is_2k(optional_params.get("image_size") or optional_params.get("size")) else 0.04
    return default_rate


def _seedream_input_surcharge(model: str, optional_params: Mapping[str, object]) -> float:
    if "bytedance/seedream/v5/pro/edit" not in model.lower():
        return 0.0
    image_urls = optional_params.get("image_urls")
    return max(len(image_urls) - 1, 0) * 0.0045 if isinstance(image_urls, (list, tuple)) else 0.0


def cost_calculator(
    model: str,
    image_response: object,
    optional_params: Mapping[str, object] | None = None,
) -> float:
    """
    fal.ai image generation cost calculator.

    Most fal image models bill a flat rate per image (output_cost_per_image).
    Upscalers bill per output megapixel (output_cost_per_pixel, NOL-535): the
    delivered dimensions arrive on each image's provider_specific_fields, and an
    image missing them falls back to the flat per-image rate (0.0 when the entry
    declares none) rather than a guessed size.
    """
    if not isinstance(image_response, ImageResponse):
        raise TypeError(f"image_response must be of type ImageResponse got type={type(image_response)}")

    entry = _cost_entry(model)
    if entry is None:
        return 0.0
    output_cost_per_pixel: float = entry.get("output_cost_per_pixel") or 0.0
    params = optional_params or {}  # mutable-ok: empty fallback is read-only
    output_cost_per_image = _vendor_image_rate(model, params, entry.get("output_cost_per_image") or 0.0)
    output_cost = sum(
        _image_cost(image, output_cost_per_pixel, output_cost_per_image) for image in (image_response.data or ())
    )
    return output_cost + _seedream_input_surcharge(model, params)
