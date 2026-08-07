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
    deployment-level override or a future rate change is a data edit rather
    than a code change. Mirrors litellm/llms/fal_ai/cost_calculator.py.
    """
    if not isinstance(image_response, ImageResponse):
        raise ValueError(f"image_response must be of type ImageResponse got type={type(image_response)}")

    try:
        model_info = litellm.get_model_info(
            model=model,
            custom_llm_provider=litellm.LlmProviders.KLING.value,
        )
    except Exception:
        # An unmapped Kling image model must not blow up the request; it falls
        # back to 0.0 the way it always did, but now that is a genuine "no entry"
        # rather than a hardcoded refusal to price anything.
        return 0.0

    output_cost_per_image: float = model_info.get("output_cost_per_image") or 0.0
    num_images: int = len(image_response.data) if image_response.data else 0
    return output_cost_per_image * num_images
