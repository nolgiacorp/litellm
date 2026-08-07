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


def test_response_carries_delivered_dimensions_for_costing():
    """NOL-535: fal bills clarity per output megapixel, and the output size is only
    knowable from the response, so the transform must carry the delivered
    dimensions or every upscale prices at $0."""
    response = FalAIClarityUpscalerConfig().transform_image_generation_response(
        model="fal-ai/clarity-upscaler",
        raw_response=_raw_response(
            {"image": {"url": "https://cdn/upscaled-4k.png", "width": 4096, "height": 4096}}
        ),
        model_response=ImageResponse(),
        logging_obj=MagicMock(),
        request_data={},
        optional_params={},
        litellm_params={},
        encoding=None,
    )
    assert response.data[0].provider_specific_fields == {"width": 4096, "height": 4096}


@pytest.mark.parametrize(
    "image",
    [
        {"url": "https://cdn/a.png"},
        {"url": "https://cdn/a.png", "width": 4096},
        {"url": "https://cdn/a.png", "width": "4096", "height": "4096"},
        {"url": "https://cdn/a.png", "width": 0, "height": 4096},
    ],
)
def test_response_missing_or_bogus_dimensions_carry_no_fields(image):
    """Absent or malformed dimensions must yield nothing rather than a guess."""
    response = FalAIClarityUpscalerConfig().transform_image_generation_response(
        model="fal-ai/clarity-upscaler",
        raw_response=_raw_response({"image": image}),
        model_response=ImageResponse(),
        logging_obj=MagicMock(),
        request_data={},
        optional_params={},
        litellm_params={},
        encoding=None,
    )
    assert response.data[0].provider_specific_fields is None


class TestClarityUpscalerCogs:
    """NOL-535: clarity-upscaler is the upscale pass behind every 2k/4k image
    tier, billed to customers, yet its COGS recorded $0 - it had no price-map
    entry and the fal cost calculator only understood flat per-image rates."""

    def test_upscale_cost_is_output_megapixels_times_published_rate(self):
        from litellm.llms.fal_ai.cost_calculator import cost_calculator
        from litellm.types.utils import ImageObject

        resp = ImageResponse(
            data=[
                ImageObject(
                    url="https://cdn/upscaled-2k.png",
                    provider_specific_fields={"width": 2048, "height": 2048},
                )
            ]
        )
        cost = cost_calculator(model="fal-ai/clarity-upscaler", image_response=resp)
        assert cost == pytest.approx(2048 * 2048 * 3e-08)
        assert cost, "clarity upscales still price at $0 - this is the NOL-535 defect"

    def test_tiers_are_actually_distinguished(self):
        from litellm.llms.fal_ai.cost_calculator import cost_calculator
        from litellm.types.utils import ImageObject

        def _cost(side: int) -> float:
            resp = ImageResponse(
                data=[ImageObject(url="https://cdn/u.png", provider_specific_fields={"width": side, "height": side})]
            )
            return cost_calculator(model="fal-ai/clarity-upscaler", image_response=resp)

        assert _cost(4096) == pytest.approx(4 * _cost(2048))

    def test_missing_dimensions_degrade_to_zero_rather_than_a_guess(self):
        from litellm.llms.fal_ai.cost_calculator import cost_calculator
        from litellm.types.utils import ImageObject

        resp = ImageResponse(data=[ImageObject(url="https://cdn/u.png")])
        assert cost_calculator(model="fal-ai/clarity-upscaler", image_response=resp) == 0.0

    def test_flat_rate_models_still_price_per_image(self):
        from litellm.llms.fal_ai.cost_calculator import cost_calculator
        from litellm.types.utils import ImageObject

        resp = ImageResponse(data=[ImageObject(url=f"https://cdn/{i}.png") for i in range(2)])
        assert cost_calculator(model="fal-ai/flux/schnell", image_response=resp) == pytest.approx(0.006)

    def test_unmapped_model_degrades_to_zero(self):
        from litellm.llms.fal_ai.cost_calculator import cost_calculator
        from litellm.types.utils import ImageObject

        resp = ImageResponse(data=[ImageObject(url="https://cdn/u.png")])
        assert cost_calculator(model="fal-ai/not-a-real-model", image_response=resp) == 0.0

    def test_non_image_response_raises(self):
        from litellm.llms.fal_ai.cost_calculator import cost_calculator

        with pytest.raises(TypeError, match="ImageResponse"):
            cost_calculator(model="fal-ai/clarity-upscaler", image_response={"data": []})
