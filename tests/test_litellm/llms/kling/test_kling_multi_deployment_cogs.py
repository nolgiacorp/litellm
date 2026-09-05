"""
NOL-519: the direct `kling/` provider logged $0 COGS on all 206 production
generations, and the two obvious fixes both fail for reasons worth pinning.

Kling is priced per SECOND and per RESOLUTION TIER, but a single model id
(`kling/kling-v3`) serves 720p, 1080p, 4K *and* image generation - the tier is
a per-request knob, not part of the model name. Production declares six video
deployments that differ only by the `resolution` litellm_param.

Attempt 1 (shipped, then found inert): put `output_cost_per_second` on each
deployment's `model_info` in litellm-config.yaml. Verified live on the prod
proxy - `/model/info` served the right rates, a real 3s generation still logged
$0, and the proxy warned "No cost information found for video model
kling/kling-v3". Deployment model_info does not reach this cost path.

The real fix is therefore where the video cost path actually looks:
  1. a `kling/kling-v3` price-map entry carrying the tiered per-second rates
     (same mechanism NOL-107 used for grok-imagine-video), and
  2. `usage.video_resolution` emitted by the Kling video transform, without
     which the shared cost path cannot choose between the tiers.
Both are required; either alone still prices at $0 or prices every tier the
same.
"""

from unittest.mock import patch

import httpx
import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.kling.videos.transformation import KlingVideoConfig

# Prices come from the bundled map: the network-fetched copy is upstream's and has no kling entries.
pytestmark = pytest.mark.usefixtures("local_model_cost_map")

# Kling's published direct API rates, audio-on at 720p/1080p (audio is on by
# default on this route), flat at 4K where Kling charges no uplift.
RATE_720P = 0.126
RATE_1080P = 0.168
RATE_4K = 0.42

TIERS = [
    ("kling-v3", "720p"),
    ("kling-v3-i2v", "720p"),
    ("kling-v3-pro", "1080p"),
    ("kling-v3-pro-i2v", "1080p"),
    ("kling-v3-master", "4k"),
    ("kling-v3-master-i2v", "4k"),
]


class _CostCapture(CustomLogger):
    """Keyed by model group so a success event still draining from a previous test cannot be
    mistaken for this test's own; the callback list is process-global."""

    def __init__(self):
        super().__init__()
        self.costs: dict = {}

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        slp = kwargs.get("standard_logging_object") or {}
        self.costs[slp.get("model_group")] = slp.get("response_cost")


def _submit(*args, **kwargs) -> httpx.Response:
    return httpx.Response(
        200,
        json={"code": 0, "message": "SUCCESS", "data": {"task_id": "t-1", "task_status": "submitted"}},
        request=httpx.Request("POST", "https://api.klingai.com/v1/videos/text2video"),
    )


def _prod_shaped_router() -> litellm.Router:
    """Mirrors envs/prod/litellm-config.yaml: 6 deployments, one backend model."""
    return litellm.Router(
        model_list=[
            {
                "model_name": name,
                "litellm_params": {"model": "kling/kling-v3", "api_key": "ak:sk", "resolution": resolution},
                "model_info": {"mode": "video_generation"},
            }
            for name, resolution in TIERS
        ]
    )


async def _cost_of(model_name: str, seconds: int):
    import asyncio

    capture = _CostCapture()
    litellm.callbacks = [capture]
    router = _prod_shaped_router()
    with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post", side_effect=_submit):
        await router.avideo_generation(model=model_name, prompt="a drop of water on black glass", seconds=seconds)

    # the success callback is fired off-thread; poll rather than race a fixed sleep
    for _ in range(50):
        if model_name in capture.costs:
            break
        await asyncio.sleep(0.05)
    assert model_name in capture.costs, f"no success event logged for {model_name}: {capture.costs}"
    return capture.costs[model_name]


class TestKlingResolutionIsReportedForCosting:
    """Half 1: the transform must tell the cost path which tier ran."""

    @pytest.mark.parametrize("mode,expected", [("std", "720p"), ("pro", "1080p"), ("4k", "4k")])
    def test_mode_maps_back_to_public_resolution(self, mode, expected):
        assert KlingVideoConfig._mode_to_resolution(mode) == expected

    @pytest.mark.parametrize("mode", [None, "", "bogus"])
    def test_unknown_mode_yields_no_resolution_rather_than_a_guess(self, mode):
        """A wrong tier would silently mis-price; absence falls back to the base rate."""
        assert KlingVideoConfig._mode_to_resolution(mode) is None

    def test_create_response_puts_resolution_on_usage(self):
        resp = httpx.Response(
            200,
            json={"code": 0, "message": "SUCCESS", "data": {"task_id": "t-1", "task_status": "submitted"}},
            request=httpx.Request("POST", "https://api.klingai.com/v1/videos/text2video"),
        )
        video = KlingVideoConfig().transform_video_create_response(
            model="kling/kling-v3",
            raw_response=resp,
            logging_obj=None,
            custom_llm_provider="kling",
            request_data={"duration": "5", "mode": "4k"},
        )
        assert video.usage["duration_seconds"] == 5.0
        assert video.usage["video_resolution"] == "4k"


class TestKlingVideoCOGS:
    """Half 2: end to end through the real router, in prod's deployment shape."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model_name,seconds,expected",
        [
            # the exact generation run against prod to verify this ticket
            ("kling-v3", 3, round(RATE_720P * 3, 6)),
            ("kling-v3", 10, round(RATE_720P * 10, 6)),
            ("kling-v3-i2v", 5, round(RATE_720P * 5, 6)),
            ("kling-v3-pro", 5, round(RATE_1080P * 5, 6)),
            ("kling-v3-pro-i2v", 10, round(RATE_1080P * 10, 6)),
            ("kling-v3-master", 5, round(RATE_4K * 5, 6)),
            ("kling-v3-master-i2v", 4, round(RATE_4K * 4, 6)),
        ],
    )
    async def test_each_tier_records_its_own_nonzero_rate(self, model_name, seconds, expected):
        cost = await _cost_of(model_name, seconds)
        assert cost, f"{model_name} still logs $0 - this is the NOL-519 defect"
        assert abs(cost - expected) < 1e-6, f"expected ${expected} for {seconds}s of {model_name}, got ${cost}"

    @pytest.mark.asyncio
    async def test_tiers_are_actually_distinguished(self):
        """
        Guards the failure mode where a single flat rate makes every tier look
        priced while 4K is under-recorded by 3.3x.
        """
        import asyncio

        capture = _CostCapture()
        litellm.callbacks = [capture]
        router = _prod_shaped_router()

        with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post", side_effect=_submit):
            for name in ("kling-v3", "kling-v3-pro", "kling-v3-master"):
                await router.avideo_generation(model=name, prompt="a drop of water", seconds=5)

        names = ("kling-v3", "kling-v3-pro", "kling-v3-master")
        for _ in range(50):
            if all(name in capture.costs for name in names):
                break
            await asyncio.sleep(0.05)

        assert all(name in capture.costs for name in names), f"expected 3 cost events, got {capture.costs}"
        cheap, mid, dear = (capture.costs[name] for name in names)
        assert cheap < mid < dear, f"tiers not distinguished: {cheap} / {mid} / {dear}"
        assert (cheap, mid, dear) == pytest.approx((0.63, 0.84, 2.10))


class TestKlingImageCOGS:
    """The image path had a hardcoded `return 0.0`, unreachable by any config."""

    def test_image_cost_is_no_longer_hardcoded_zero(self):
        from litellm.llms.kling.cost_calculator import cost_calculator
        from litellm.types.utils import ImageObject, ImageResponse

        resp = ImageResponse(data=[ImageObject(url="https://cdn.kling.test/a.png")])
        assert cost_calculator(model="kling-v3", image_response=resp) == pytest.approx(0.028)

    def test_image_cost_scales_with_image_count(self):
        from litellm.llms.kling.cost_calculator import cost_calculator
        from litellm.types.utils import ImageObject, ImageResponse

        resp = ImageResponse(data=[ImageObject(url=f"https://cdn.kling.test/{i}.png") for i in range(3)])
        assert cost_calculator(model="kling-v3", image_response=resp) == pytest.approx(0.084)
