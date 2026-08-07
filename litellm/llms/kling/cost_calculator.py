import litellm
from litellm.types.utils import ImageResponse


def cost_calculator(
    model: str,
    image_response: ImageResponse,
) -> float:
    """
    Kling image generation cost.

    NOL-519: this used to `return 0.0` unconditionally behind a TODO saying
    Kling billed in opaque prepaid "Units" with no published USD price. Kling
    does publish one - it just prices IMAGE Units separately from VIDEO Units
    (1 image Unit = $0.0035, against $0.14 for video), which is why reading the
    single video Unit price made image pricing look unresolvable. Kling Image
    3.0 bills a flat 8 U/image at both 1k and 2k, i.e. $0.028/image.

    The rate now comes from the model cost map like every other provider, so a
    rate change is a data edit rather than a code change.

    The map is read directly rather than through get_model_info() because that
    helper raises for an unmapped model. A missing price must degrade to 0.0
    here, exactly as it did before: this function runs after the image has
    already been generated and paid for, so raising would turn a pricing gap
    into a failed generation for the caller.
    """
    if not isinstance(image_response, ImageResponse):
        raise TypeError(f"image_response must be of type ImageResponse got type={type(image_response)}")

    provider = litellm.LlmProviders.KLING.value
    bare_model = model.split("/")[-1]
    cost_entry = (
        litellm.model_cost.get(f"{provider}/{bare_model}")
        or litellm.model_cost.get(model)
        or litellm.model_cost.get(bare_model)
        or {}
    )

    output_cost_per_image: float = cost_entry.get("output_cost_per_image") or 0.0
    num_images: int = len(image_response.data) if image_response.data else 0
    return output_cost_per_image * num_images
