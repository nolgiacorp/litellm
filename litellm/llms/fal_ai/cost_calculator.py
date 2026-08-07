import litellm
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


def cost_calculator(
    model: str,
    image_response: object,
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
    output_cost_per_image: float = entry.get("output_cost_per_image") or 0.0
    return sum(
        _image_cost(image, output_cost_per_pixel, output_cost_per_image) for image in (image_response.data or ())
    )
