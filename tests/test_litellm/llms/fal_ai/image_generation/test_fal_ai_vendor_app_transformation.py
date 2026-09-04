import os
import sys
from unittest.mock import MagicMock

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import litellm

litellm.model_cost = litellm.get_model_cost_map(url="")
from litellm.llms.fal_ai.cost_calculator import cost_calculator
from litellm.llms.fal_ai.image_generation import (
    FalAIBytedanceDreaminaV31Config,
    FalAIBytedanceSeedreamV3Config,
    FalAIIdeogramV3Config,
    FalAIIdeogramV4Config,
    FalAIQwenImage3Config,
    FalAIReveConfig,
    FalAISeedreamV5Config,
    get_fal_ai_image_generation_config,
)
from litellm.types.utils import ImageObject, ImageResponse

REVE_T2I = "reve/2.1/text-to-image"
REVE_EDIT = "reve/2.1/edit"
SEEDREAM_T2I = "bytedance/seedream/v5/pro/text-to-image"
SEEDREAM_EDIT = "bytedance/seedream/v5/pro/edit"
IDEOGRAM_T2I = "ideogram/v4"
IDEOGRAM_I2I = "ideogram/v4/image-to-image"
QWEN_T2I = "alibaba/qwen-image-3/text-to-image"
QWEN_EDIT = "alibaba/qwen-image-3/edit"

ROUTES = {
    REVE_T2I: FalAIReveConfig,
    REVE_EDIT: FalAIReveConfig,
    SEEDREAM_T2I: FalAISeedreamV5Config,
    SEEDREAM_EDIT: FalAISeedreamV5Config,
    IDEOGRAM_T2I: FalAIIdeogramV4Config,
    IDEOGRAM_I2I: FalAIIdeogramV4Config,
    QWEN_T2I: FalAIQwenImage3Config,
    QWEN_EDIT: FalAIQwenImage3Config,
}

PRICES = {
    REVE_T2I: 0.25,
    REVE_EDIT: 0.25,
    SEEDREAM_T2I: 0.0675,
    SEEDREAM_EDIT: 0.0675,
    IDEOGRAM_T2I: 0.015,
    IDEOGRAM_I2I: 0.015,
    QWEN_T2I: 0.04,
    QWEN_EDIT: 0.04,
}

IMAGE_SIZE_CONFIGS = [FalAISeedreamV5Config(), FalAIIdeogramV4Config(), FalAIQwenImage3Config()]
ALL_CONFIGS = [FalAIReveConfig(), *IMAGE_SIZE_CONFIGS]


def _url(config, model, api_base=None):
    return config.get_complete_url(
        api_base=api_base, api_key="test-key", model=model, optional_params={}, litellm_params={}
    )


def _map(config, non_default_params, optional_params=None, drop_params=False, model="m"):
    return config.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params or {},
        model=model,
        drop_params=drop_params,
    )


def _request(config, model, optional_params, prompt="a lighthouse at dusk"):
    return config.transform_image_generation_request(
        model=model, prompt=prompt, optional_params=optional_params, litellm_params={}, headers={}
    )


class TestDispatch:
    @pytest.mark.parametrize("model,expected", list(ROUTES.items()))
    def test_vendor_apps_select_their_config(self, model, expected):
        assert type(get_fal_ai_image_generation_config(model)) is expected

    def test_dispatch_is_case_insensitive(self):
        assert type(get_fal_ai_image_generation_config("Bytedance/Seedream/V5/Pro/Edit")) is FalAISeedreamV5Config

    def test_seedream_v5_no_longer_falls_through_to_v3(self):
        """`bytedance/seedream` matched first, so every v5 request was posted to
        the hardcoded v3 endpoint while claiming to be Seedream 5."""
        config = get_fal_ai_image_generation_config(SEEDREAM_T2I)
        assert not isinstance(config, FalAIBytedanceSeedreamV3Config)
        assert _url(config, SEEDREAM_T2I) == "https://fal.run/bytedance/seedream/v5/pro/text-to-image"

    def test_seedream_v3_and_dreamina_keep_their_configs(self):
        assert (
            type(get_fal_ai_image_generation_config("fal-ai/bytedance/seedream/v3/text-to-image"))
            is FalAIBytedanceSeedreamV3Config
        )
        assert (
            type(get_fal_ai_image_generation_config("fal-ai/bytedance/dreamina/v3.1/text-to-image"))
            is FalAIBytedanceDreaminaV31Config
        )

    def test_ideogram_v4_no_longer_falls_through_to_v3(self):
        config = get_fal_ai_image_generation_config(IDEOGRAM_I2I)
        assert not isinstance(config, FalAIIdeogramV3Config)
        assert _url(config, IDEOGRAM_I2I) == "https://fal.run/ideogram/v4/image-to-image"

    def test_ideogram_v3_keeps_its_config(self):
        assert type(get_fal_ai_image_generation_config("fal-ai/ideogram/v3")) is FalAIIdeogramV3Config


class TestEndpoint:
    @pytest.mark.parametrize("model", list(ROUTES))
    def test_endpoint_is_the_model_id(self, model):
        assert _url(get_fal_ai_image_generation_config(model), model) == f"https://fal.run/{model}"

    @pytest.mark.parametrize(
        "config,model,expected",
        [
            (FalAIReveConfig(), "2.1/edit", "https://fal.run/reve/2.1/edit"),
            (FalAISeedreamV5Config(), "seedream/v5/pro/edit", "https://fal.run/bytedance/seedream/v5/pro/edit"),
            (FalAIIdeogramV4Config(), "v4", "https://fal.run/ideogram/v4"),
            (FalAIQwenImage3Config(), "qwen-image-3/edit", "https://fal.run/alibaba/qwen-image-3/edit"),
        ],
    )
    def test_owner_is_prefixed_when_missing(self, config, model, expected):
        assert _url(config, model) == expected

    def test_owner_casing_is_normalized_without_duplicate_prefix(self):
        assert _url(FalAISeedreamV5Config(), "Bytedance/Seedream/V5/Pro/Edit") == (
            "https://fal.run/bytedance/seedream/v5/pro/edit"
        )

    def test_api_base_override(self):
        assert _url(FalAIReveConfig(), REVE_EDIT, api_base="https://proxy.internal/") == (
            "https://proxy.internal/reve/2.1/edit"
        )


class TestOpenAIParams:
    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_n_maps_to_num_images(self, config):
        assert _map(config, {"n": 3}) == {"num_images": 3}

    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_response_format_is_ignored(self, config):
        assert _map(config, {"response_format": "b64_json"}) == {}

    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_existing_optional_params_win(self, config):
        assert _map(config, {"n": 4}, optional_params={"num_images": 1}) == {"num_images": 1}

    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_unsupported_param_raises_without_drop_params(self, config):
        with pytest.raises(ValueError, match="style"):
            _map(config, {"style": "vivid"})

    @pytest.mark.parametrize("config", ALL_CONFIGS)
    def test_unsupported_param_dropped_with_drop_params(self, config):
        assert _map(config, {"style": "vivid", "n": 2}, drop_params=True) == {"num_images": 2}

    @pytest.mark.parametrize("config", IMAGE_SIZE_CONFIGS)
    def test_size_becomes_fal_image_size_object(self, config):
        assert _map(config, {"size": "1536x1024"}) == {"image_size": {"width": 1536, "height": 1024}}

    @pytest.mark.parametrize("config", IMAGE_SIZE_CONFIGS)
    @pytest.mark.parametrize("size", ["square_hd", "auto_2K", {"width": 2048, "height": 2048}])
    def test_fal_dialect_size_passes_through(self, config, size):
        assert _map(config, {"size": size}) == {"image_size": size}

    @pytest.mark.parametrize("config", [FalAIReveConfig(), FalAISeedreamV5Config(), FalAIQwenImage3Config()])
    def test_supported_params(self, config):
        assert config.get_supported_openai_params("m") == ["n", "response_format", "size"]

    def test_ideogram_supported_params_include_quality(self):
        assert FalAIIdeogramV4Config().get_supported_openai_params("m") == ["n", "quality", "response_format", "size"]


class TestReveAspectRatio:
    @pytest.mark.parametrize(
        "size,ratio",
        [
            ("1024x1024", "1:1"),
            ("1792x1024", "16:9"),
            ("1024x1792", "9:16"),
            ("1536x1024", "3:2"),
            ("1024x1536", "2:3"),
            ("2048x876", "21:9"),
            ("1024x768", "4:3"),
            ("768x1024", "3:4"),
            ("1280x1024", "5:4"),
            ("4096x1024", "4:1"),
            ("1024x4096", "1:4"),
        ],
    )
    def test_size_becomes_nearest_reve_ratio(self, size, ratio):
        assert _map(FalAIReveConfig(), {"size": size}) == {"aspect_ratio": ratio}

    @pytest.mark.parametrize("size", ["21:9", "auto"])
    def test_reve_ratio_passes_through(self, size):
        assert _map(FalAIReveConfig(), {"size": size}) == {"aspect_ratio": size}

    @pytest.mark.parametrize("size", ["huge", "0x100", None, {"width": "wide"}])
    def test_unusable_size_leaves_the_ratio_to_reve(self, size):
        assert _map(FalAIReveConfig(), {"size": size}) == {"aspect_ratio": "auto"}

    @pytest.mark.parametrize(
        "image_size,ratio",
        [
            ("square_hd", "1:1"),
            ("square", "1:1"),
            ("landscape_4_3", "4:3"),
            ("portrait_4_3", "3:4"),
            ("landscape_16_9", "16:9"),
            ("portrait_16_9", "9:16"),
            ({"width": 2048, "height": 1024}, "2:1"),
            ("1024x1536", "2:3"),
        ],
    )
    def test_passthrough_image_size_is_translated_in_the_request(self, image_size, ratio):
        """Callers that speak fal's image_size aliases to every fal_ai model keep
        working: Reve has no image_size, so the alias becomes its aspect_ratio."""
        request = _request(FalAIReveConfig(), REVE_T2I, {"image_size": image_size, "num_images": 1})
        assert request == {"prompt": "a lighthouse at dusk", "num_images": 1, "aspect_ratio": ratio}

    def test_explicit_aspect_ratio_wins_over_image_size(self):
        request = _request(FalAIReveConfig(), REVE_T2I, {"image_size": "square_hd", "aspect_ratio": "16:9"})
        assert request == {"prompt": "a lighthouse at dusk", "aspect_ratio": "16:9"}


class TestIdeogramQuality:
    @pytest.mark.parametrize("quality,speed", [("low", "TURBO"), ("medium", "BALANCED"), ("high", "QUALITY")])
    def test_quality_selects_rendering_speed(self, quality, speed):
        assert _map(FalAIIdeogramV4Config(), {"quality": quality, "n": 2}) == {
            "rendering_speed": speed,
            "num_images": 2,
        }

    def test_auto_quality_leaves_fal_default(self):
        assert _map(FalAIIdeogramV4Config(), {"quality": "auto"}) == {}

    def test_unknown_quality_raises(self):
        with pytest.raises(ValueError, match="ultra"):
            _map(FalAIIdeogramV4Config(), {"quality": "ultra"})

    @pytest.mark.parametrize("config", [FalAIReveConfig(), FalAISeedreamV5Config(), FalAIQwenImage3Config()])
    def test_quality_is_not_a_knob_on_the_others(self, config):
        with pytest.raises(ValueError, match="quality"):
            _map(config, {"quality": "high"})
        assert _map(config, {"quality": "high"}, drop_params=True) == {}


class TestRequestBody:
    def test_reve_edit_carries_image_url(self):
        params = {"image_url": "https://cdn/ref.png", "num_images": 2, "aspect_ratio": "auto"}
        assert _request(FalAIReveConfig(), REVE_EDIT, params) == {"prompt": "a lighthouse at dusk", **params}

    def test_seedream_edit_carries_image_urls_and_native_size(self):
        params = {"image_urls": ["https://cdn/a.png", "https://cdn/b.png"], "image_size": "auto_2K", "num_images": 1}
        assert _request(FalAISeedreamV5Config(), SEEDREAM_EDIT, params) == {"prompt": "a lighthouse at dusk", **params}

    def test_ideogram_carries_native_knobs_verbatim(self):
        params = {"rendering_speed": "QUALITY", "expansion_model": "None", "seed": 7, "image_size": "square_hd"}
        assert _request(FalAIIdeogramV4Config(), IDEOGRAM_T2I, params) == {"prompt": "a lighthouse at dusk", **params}

    def test_ideogram_image_to_image_carries_image_url_and_strength(self):
        params = {"image_url": "https://cdn/ref.png", "strength": 0.6, "image_size": "auto"}
        assert _request(FalAIIdeogramV4Config(), IDEOGRAM_I2I, params) == {"prompt": "a lighthouse at dusk", **params}

    def test_qwen_edit_carries_image_urls_and_native_knobs(self):
        params = {
            "image_urls": ["https://cdn/a.png"],
            "negative_prompt": "text, watermark",
            "seed": 3,
            "enable_prompt_expansion": False,
            "image_size": {"width": 1024, "height": 1024},
        }
        assert _request(FalAIQwenImage3Config(), QWEN_EDIT, params) == {"prompt": "a lighthouse at dusk", **params}


def _raw_response(payload):
    return httpx.Response(status_code=200, json=payload, request=httpx.Request("POST", "https://fal.run"))


@pytest.mark.parametrize(
    "config,model,payload,urls",
    [
        (
            FalAIReveConfig(),
            REVE_T2I,
            {"images": [{"url": "https://cdn/reve.png", "content_type": "image/png", "width": 1024, "height": 1024}]},
            ["https://cdn/reve.png"],
        ),
        (
            FalAISeedreamV5Config(),
            SEEDREAM_T2I,
            {"images": [{"url": "https://cdn/a.jpeg", "width": 2048, "height": 2048}, {"url": "https://cdn/b.jpeg"}]},
            ["https://cdn/a.jpeg", "https://cdn/b.jpeg"],
        ),
        (
            FalAIIdeogramV4Config(),
            IDEOGRAM_T2I,
            {
                "images": [{"url": "https://cdn/i.jpeg"}],
                "seed": 42,
                "has_nsfw_concepts": [False],
                "prompt": "p",
                "timings": {},
            },
            ["https://cdn/i.jpeg"],
        ),
        (
            FalAIQwenImage3Config(),
            QWEN_T2I,
            {"images": [{"url": "https://cdn/q.png"}], "seed": 9},
            ["https://cdn/q.png"],
        ),
    ],
)
def test_response_images_become_image_objects(config, model, payload, urls):
    response = config.transform_image_generation_response(
        model=model,
        raw_response=_raw_response(payload),
        model_response=ImageResponse(),
        logging_obj=MagicMock(),
        request_data={},
        optional_params={},
        litellm_params={},
        encoding=None,
    )
    assert [image.url for image in response.data] == urls


class TestPricing:
    @pytest.mark.parametrize("model,rate", list(PRICES.items()))
    def test_pricing_registered(self, model, rate):
        info = litellm.get_model_info(model=model, custom_llm_provider=litellm.LlmProviders.FAL_AI.value)
        assert info["output_cost_per_image"] == rate
        assert info["mode"] == "image_generation"

    @pytest.mark.parametrize("model,rate", list(PRICES.items()))
    def test_cost_scales_with_image_count(self, model, rate):
        image_response = ImageResponse(data=[ImageObject(url="https://x/1.png"), ImageObject(url="https://x/2.png")])
        assert cost_calculator(model=model, image_response=image_response) == pytest.approx(2 * rate)
        assert cost_calculator(model=model, image_response=image_response) > 0

    @pytest.mark.parametrize("model", [SEEDREAM_T2I, SEEDREAM_EDIT])
    def test_seedream_2k_uses_upper_tier(self, model):
        response = ImageResponse(data=[ImageObject(url="https://x/1.png")])
        assert cost_calculator(model, response, {"image_size": "auto_2K"}) == pytest.approx(0.135)

    def test_seedream_edit_charges_for_extra_input_images(self):
        response = ImageResponse(data=[ImageObject(url="https://x/1.png")])
        params = {"image_urls": ["https://x/a.png", "https://x/b.png", "https://x/c.png"]}
        assert cost_calculator(SEEDREAM_EDIT, response, params) == pytest.approx(0.0675 + 2 * 0.0045)

    @pytest.mark.parametrize("speed,rate", [("TURBO", 0.0075), ("BALANCED", 0.015), ("QUALITY", 0.025)])
    def test_ideogram_rendering_speed_selects_rate(self, speed, rate):
        response = ImageResponse(data=[ImageObject(url="https://x/1.png")])
        assert cost_calculator(IDEOGRAM_T2I, response, {"rendering_speed": speed}) == pytest.approx(rate)

    @pytest.mark.parametrize("quality,rate", [("low", 0.0075), ("medium", 0.015), ("high", 0.025)])
    def test_ideogram_openai_quality_selects_rate(self, quality, rate):
        response = ImageResponse(data=[ImageObject(url="https://x/1.png")])
        assert cost_calculator(IDEOGRAM_T2I, response, {"quality": quality}) == pytest.approx(rate)

    def test_ideogram_rate_scales_with_requested_megapixels(self):
        response = ImageResponse(data=[ImageObject(url="https://x/1.png")])
        params = {"rendering_speed": "QUALITY", "image_size": {"width": 2048, "height": 1024}}
        assert cost_calculator(IDEOGRAM_T2I, response, params) == pytest.approx(0.025 * 2.097152)

    @pytest.mark.parametrize("model", [QWEN_T2I, QWEN_EDIT])
    def test_qwen_2k_uses_upper_tier(self, model):
        response = ImageResponse(data=[ImageObject(url="https://x/1.png")])
        assert cost_calculator(model, response, {"image_size": "auto_2K"}) == pytest.approx(0.075)
