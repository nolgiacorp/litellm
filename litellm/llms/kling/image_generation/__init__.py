from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .transformation import KlingImageGenerationConfig

__all__ = [
    "KlingImageGenerationConfig",
    "get_kling_image_generation_config",
]


def get_kling_image_generation_config(model: str) -> BaseImageGenerationConfig:
    return KlingImageGenerationConfig()
