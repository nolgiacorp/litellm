from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from litellm.types.utils import ImageResponse


def image_cost_calculator(
    model: str,
    image_response: "ImageResponse",
) -> float:
    import litellm
    from litellm.types.utils import ImageResponse as _ImageResponse

    if not isinstance(image_response, _ImageResponse):
        raise ValueError(f"image_response must be of type ImageResponse got type={type(image_response)}")

    model_info = litellm.get_model_info(
        model=model,
        custom_llm_provider=litellm.LlmProviders.BLACK_FOREST_LABS.value,
    )
    output_cost_per_image: float = model_info.get("output_cost_per_image") or 0.0
    num_images = len(image_response.data) if image_response.data else 0
    return output_cost_per_image * num_images
