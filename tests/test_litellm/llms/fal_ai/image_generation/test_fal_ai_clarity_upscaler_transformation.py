import os
import sys
from unittest.mock import MagicMock

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import litellm

litellm.model_cost = litellm.get_model_cost_map(url="")
from litellm.llms.fal_ai.image_generation import (
    FalAIClarityUpscalerConfig,
    FalAIImagen4Config,
    get_fal_ai_image_generation_config,
)
from litellm.types.utils import ImageResponse


@pytest.mark.parametrize(
    "model",
    ["fal-ai/clarity-upscaler", "clarity-upscaler", "clarity_upscaler"],
)
def test_clarity_upscaler_config_selected(model):
    assert isinstance(get_fal_ai_image_generation_config(model), FalAIClarityUpscalerConfig)


def test_non_upscaler_model_does_not_route_to_upscaler():
    assert not isinstance(
        get_fal_ai_image_generation_config("fal-ai/imagen4/preview"),
        FalAIClarityUpscalerConfig,
    )
    assert isinstance(
        get_fal_ai_image_generation_config("fal-ai/imagen4/preview"),
        FalAIImagen4Config,
    )


def test_get_complete_url_derives_endpoint_from_model():
    url = FalAIClarityUpscalerConfig().get_complete_url(
        api_base=None,
        api_key="test-key",
        model="clarity-upscaler",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://fal.run/fal-ai/clarity-upscaler"


def test_map_params_forwards_upscale_inputs_and_drops_inapplicable_openai_params():
    optional_params = FalAIClarityUpscalerConfig().map_openai_params(
        non_default_params={
            "image_url": "https://cdn/native.png",
            "upscale_factor": 4,
            "response_format": "b64_json",
            "n": 2,
            "size": "1024x1024",
        },
        optional_params={},
        model="fal-ai/clarity-upscaler",
        drop_params=False,
    )
    assert optional_params == {
        "image_url": "https://cdn/native.png",
        "upscale_factor": 4,
    }


def test_map_params_preserves_existing_optional_params():
    optional_params = FalAIClarityUpscalerConfig().map_openai_params(
        non_default_params={"upscale_factor": 2},
        optional_params={"image_url": "https://cdn/native.png"},
        model="fal-ai/clarity-upscaler",
        drop_params=False,
    )
    assert optional_params == {
        "image_url": "https://cdn/native.png",
        "upscale_factor": 2,
    }


def test_transform_request_carries_image_url_and_factor():
    request = FalAIClarityUpscalerConfig().transform_image_generation_request(
        model="fal-ai/clarity-upscaler",
        prompt="a luxury watch",
        optional_params={"image_url": "https://cdn/native.png", "upscale_factor": 4},
        litellm_params={},
        headers={},
    )
    assert request == {
        "prompt": "a luxury watch",
        "image_url": "https://cdn/native.png",
        "upscale_factor": 4,
    }


def test_transform_request_omits_empty_prompt():
    request = FalAIClarityUpscalerConfig().transform_image_generation_request(
        model="fal-ai/clarity-upscaler",
        prompt="",
        optional_params={"image_url": "https://cdn/native.png", "upscale_factor": 2},
        litellm_params={},
        headers={},
    )
    assert "prompt" not in request
    assert request == {"image_url": "https://cdn/native.png", "upscale_factor": 2}


def _raw_response(payload: dict) -> httpx.Response:
    return httpx.Response(status_code=200, json=payload, request=httpx.Request("POST", "https://fal.run"))


def test_response_parses_single_image_object():
    """Clarity Upscaler returns a single `image` (not an `images` array); the base
    array-only parser would drop it, so this override is the crux of the feature."""
    response = FalAIClarityUpscalerConfig().transform_image_generation_response(
        model="fal-ai/clarity-upscaler",
        raw_response=_raw_response(
            {"image": {"url": "https://cdn/upscaled-4k.png", "width": 3840, "height": 2160}, "seed": 7}
        ),
        model_response=ImageResponse(),
        logging_obj=MagicMock(),
        request_data={},
        optional_params={},
        litellm_params={},
        encoding=None,
    )
    assert [image.url for image in response.data] == ["https://cdn/upscaled-4k.png"]


def test_response_falls_back_to_images_array():
    response = FalAIClarityUpscalerConfig().transform_image_generation_response(
        model="fal-ai/clarity-upscaler",
        raw_response=_raw_response({"images": [{"url": "https://cdn/a.png"}, {"url": "https://cdn/b.png"}]}),
        model_response=ImageResponse(),
        logging_obj=MagicMock(),
        request_data={},
        optional_params={},
        litellm_params={},
        encoding=None,
    )
    assert [image.url for image in response.data] == ["https://cdn/a.png", "https://cdn/b.png"]


def test_response_empty_when_no_image():
    response = FalAIClarityUpscalerConfig().transform_image_generation_response(
        model="fal-ai/clarity-upscaler",
        raw_response=_raw_response({"seed": 7}),
        model_response=ImageResponse(),
        logging_obj=MagicMock(),
        request_data={},
        optional_params={},
        litellm_params={},
        encoding=None,
    )
    assert response.data == []
