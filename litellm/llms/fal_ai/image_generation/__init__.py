from collections.abc import Callable

from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .bria_transformation import FalAIBriaConfig
from .bytedance_transformation import (
    FalAIBytedanceDreaminaV31Config,
    FalAIBytedanceSeedreamV3Config,
)
from .clarity_upscaler_transformation import FalAIClarityUpscalerConfig
from .flux_pro_v11_transformation import FalAIFluxProV11Config
from .flux_pro_v11_ultra_transformation import FalAIFluxProV11UltraConfig
from .flux_schnell_transformation import FalAIFluxSchnellConfig
from .ideogram_v3_transformation import FalAIIdeogramV3Config
from .imagen4_transformation import FalAIImagen4Config
from .nano_banana_transformation import FalAINanoBananaConfig
from .recraft_v3_transformation import FalAIRecraftV3Config
from .stable_diffusion_transformation import FalAIStableDiffusionConfig
from .transformation import FalAIBaseConfig, FalAIImageGenerationConfig
from .vendor_app_transformation import (
    FalAIIdeogramV4Config,
    FalAIQwenImage3Config,
    FalAIReveConfig,
    FalAISeedreamV5Config,
    FalAIVendorAppConfig,
)

__all__ = [
    "FalAIBaseConfig",
    "FalAIBriaConfig",
    "FalAIBytedanceDreaminaV31Config",
    "FalAIBytedanceSeedreamV3Config",
    "FalAIClarityUpscalerConfig",
    "FalAIFluxProV11Config",
    "FalAIFluxProV11UltraConfig",
    "FalAIFluxSchnellConfig",
    "FalAIIdeogramV3Config",
    "FalAIIdeogramV4Config",
    "FalAIImageGenerationConfig",
    "FalAIImagen4Config",
    "FalAINanoBananaConfig",
    "FalAIQwenImage3Config",
    "FalAIRecraftV3Config",
    "FalAIReveConfig",
    "FalAISeedreamV5Config",
    "FalAIStableDiffusionConfig",
    "FalAIVendorAppConfig",
]

_CONFIG_BY_SUBSTRINGS: tuple[tuple[tuple[str, ...], Callable[[], BaseImageGenerationConfig]], ...] = (
    (("clarity-upscaler",), FalAIClarityUpscalerConfig),
    (("clarity_upscaler",), FalAIClarityUpscalerConfig),
    (("nano-banana",), FalAINanoBananaConfig),
    (("gemini-25-flash-image",), FalAINanoBananaConfig),
    (("imagen4",), FalAIImagen4Config),
    (("imagen-4",), FalAIImagen4Config),
    (("recraft",), FalAIRecraftV3Config),
    (("bria",), FalAIBriaConfig),
    (("flux-pro", "ultra"), FalAIFluxProV11UltraConfig),
    (("flux-pro",), FalAIFluxProV11Config),
    (("schnell",), FalAIFluxSchnellConfig),
    (("reve/",), FalAIReveConfig),
    (("qwen-image-3",), FalAIQwenImage3Config),
    (("seedream/v5",), FalAISeedreamV5Config),
    (("bytedance/seedream",), FalAIBytedanceSeedreamV3Config),
    (("bytedance/dreamina",), FalAIBytedanceDreaminaV31Config),
    (("ideogram/v4",), FalAIIdeogramV4Config),
    (("ideogram",), FalAIIdeogramV3Config),
    (("stable-diffusion",), FalAIStableDiffusionConfig),
)


def get_fal_ai_image_generation_config(model: str) -> BaseImageGenerationConfig:
    """
    Get the appropriate Fal AI image generation configuration based on the model.

    Args:
        model: The Fal AI model name (e.g., "fal-ai/imagen4/preview", "reve/2.1/edit")

    Returns:
        The appropriate configuration class for the specified model. Entries are
        matched in order, so a versioned family (seedream/v5, ideogram/v4) precedes
        its broader match.
    """
    model_lower = model.lower()
    return next(
        (
            config()
            for substrings, config in _CONFIG_BY_SUBSTRINGS
            if all(substring in model_lower for substring in substrings)
        ),
        FalAIImageGenerationConfig(),
    )
