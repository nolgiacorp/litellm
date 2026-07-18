from litellm.types.utils import ImageResponse


def cost_calculator(
    model: str,
    image_response: ImageResponse,
) -> float:
    # TODO(kling): Kling bills image and video generation in prepaid "Units"/credits
    # rather than a published per-image USD price, so there is no reliable USD figure
    # to charge yet. Return 0.0 until Kling per-generation pricing is finalized and
    # wired into the model cost map (mirroring the fal_ai output_cost_per_image path).
    return 0.0
