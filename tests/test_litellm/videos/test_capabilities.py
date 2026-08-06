"""
Regression tests for the video capability-param gate (litellm/videos/capabilities.py).

The bug being locked out: a capability-bearing param that a provider cannot execute
was accepted and silently discarded, so the caller was billed for a video that
ignored the frames, reference media or soundtrack they asked for. Each test below
fails if the gate stops refusing, if it starts refusing something a provider does
handle, or if the scoping widens past the closed vocabulary.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm.llms.black_forest_labs.videos.transformation import BflVideoConfig
from litellm.llms.fal_ai.videos.transformation import FalAIVideoConfig
from litellm.llms.kling.videos.transformation import KlingVideoConfig
from litellm.llms.minimax.videos.transformation import MinimaxVideoConfig
from litellm.llms.openai.videos.transformation import OpenAIVideoConfig
from litellm.llms.openrouter.videos.transformation import OpenRouterVideoConfig
from litellm.llms.xai.videos.transformation import XAIVideoConfig
from litellm.videos.capabilities import (
    CAPABILITY_PARAMS,
    DeclaredCapabilityParams,
    UndeclaredCapabilityParams,
    check_capability_params,
)
from litellm.videos.utils import VideoGenerationRequestUtils


def _map(config, model, params, provider="test-provider"):
    return VideoGenerationRequestUtils.get_optional_params_video_generation(
        model=model,
        video_generation_provider_config=config,
        video_generation_optional_params=dict(params),
        custom_llm_provider=provider,
    )


# (config, model, param, value) that the provider genuinely cannot execute, and that
# was previously accepted and dropped on the floor.
SILENTLY_DROPPED_BEFORE = (
    (XAIVideoConfig(), "grok-imagine-video-1.5", "end_image_url", "https://example.com/end.png"),
    (XAIVideoConfig(), "grok-imagine-video-1.5", "video_urls", ["https://example.com/ref.mp4"]),
    (XAIVideoConfig(), "grok-imagine-video-1.5", "generate_audio", False),
    (BflVideoConfig(), "flux-3-video", "reference_audios", [{"voice_id": "eve"}]),
    (BflVideoConfig(), "flux-3-video", "audio_urls", ["https://example.com/ref.mp3"]),
    (BflVideoConfig(), "flux-3-video", "bitrate_mode", "high"),
    (BflVideoConfig(), "flux-3-video", "base_video_url", "https://example.com/src.mp4"),
    # Kling's direct API names the end frame image_tail; end_image_url reaches the
    # provider verbatim and is ignored, which is why the catalog must not promise it
    # for a kling-routed model.
    (KlingVideoConfig(), "kling/kling-v3", "end_image_url", "https://example.com/end.png"),
    # OpenRouter's normalized schema has no reference-video or reference-audio slot,
    # and this transformation drops unknown fields rather than forwarding them.
    (OpenRouterVideoConfig(), "openrouter/bytedance/seedance-2.0", "video_urls", ["https://example.com/r.mp4"]),
    (OpenRouterVideoConfig(), "openrouter/bytedance/seedance-2.0", "audio_urls", ["https://example.com/r.mp3"]),
    (OpenRouterVideoConfig(), "openrouter/bytedance/seedance-2.0", "bitrate_mode", "high"),
    (MinimaxVideoConfig(), "MiniMax-Hailuo-2.3", "end_image_url", "https://example.com/end.png"),
    (MinimaxVideoConfig(), "MiniMax-Hailuo-2.3", "image_urls", ["https://example.com/a.png"]),
    (FalAIVideoConfig(), "fal_ai/bytedance/seedance-2.0/text-to-video", "end_image_url", "https://e.com/e.png"),
)


@pytest.mark.parametrize("config,model,param,value", SILENTLY_DROPPED_BEFORE)
def test_unsupported_capability_param_is_refused_not_dropped(config, model, param, value):
    with pytest.raises(litellm.BadRequestError) as excinfo:
        _map(config, model, {param: value})

    message = str(excinfo.value)
    assert param in message, "the refusal must name the offending param"
    assert model in message, "the refusal must name the model"


# (config, model, params) the provider does execute; refusing these would be a
# regression that breaks working generations.
EXECUTED_CAPABILITIES = (
    (XAIVideoConfig(), "grok-imagine-video-1.5", {"reference_audios": [{"voice_id": "eve"}]}),
    (XAIVideoConfig(), "grok-imagine-video-1.5", {"image_urls": ["https://example.com/a.png"]}),
    (XAIVideoConfig(), "grok-imagine-video-1.5", {"input_reference": "https://example.com/s.png"}),
    (
        BflVideoConfig(),
        "flux-3-video",
        {
            "input_reference": "https://example.com/s.png",
            "end_image_url": "https://example.com/e.png",
            "image_urls": ["https://example.com/a.png"],
            "generate_audio": True,
        },
    ),
    (BflVideoConfig(), "flux-3-video", {"video_urls": ["https://example.com/src.mp4"]}),
    (KlingVideoConfig(), "kling/kling-v3", {"input_reference": "https://e.com/s.png", "generate_audio": True}),
    (
        OpenRouterVideoConfig(),
        "openrouter/bytedance/seedance-2.0",
        {"image_urls": ["https://example.com/a.png"], "end_image_url": "https://example.com/e.png"},
    ),
    # H3 makes frame conditioning and reference media mutually exclusive provider-side,
    # so they are exercised as separate requests.
    (
        MinimaxVideoConfig(),
        "MiniMax-H3",
        {"input_reference": "https://example.com/s.png", "end_image_url": "https://example.com/e.png"},
    ),
    (
        MinimaxVideoConfig(),
        "MiniMax-H3",
        {"image_urls": ["https://example.com/a.png"], "audio_urls": ["https://example.com/a.mp3"]},
    ),
    (MinimaxVideoConfig(), "MiniMax-H3", {"base_video_url": "https://example.com/src.mp4"}),
    # fal's kling twin does take an end frame, unlike the direct kling route.
    (FalAIVideoConfig(), "fal_ai/fal-ai/kling-video/v3/pro/image-to-video", {"end_image_url": "https://e.com/e.png"}),
    (
        FalAIVideoConfig(),
        "fal_ai/bytedance/seedance-2.0/reference-to-video",
        {
            "image_urls": ["https://example.com/a.png"],
            "video_urls": ["https://example.com/r.mp4"],
            "audio_urls": ["https://example.com/r.mp3"],
            "bitrate_mode": "high",
        },
    ),
)


@pytest.mark.parametrize("config,model,params", EXECUTED_CAPABILITIES)
def test_supported_capability_params_still_pass(config, model, params):
    _map(config, model, params)


@pytest.mark.parametrize(
    "value",
    (None, "", [], (), {}),
    ids=("none", "empty-string", "empty-list", "empty-tuple", "empty-dict"),
)
def test_absent_capability_param_does_not_trip_the_gate(value):
    """An omitted or empty capability param asks for nothing, so there is nothing to refuse."""
    _map(XAIVideoConfig(), "grok-imagine-video-1.5", {"end_image_url": value, "video_urls": value})


def test_gate_is_scoped_to_the_capability_vocabulary():
    """
    Params outside the vocabulary keep today's behavior. Widening the gate to every
    unsupported param would turn drop_params into a global strict-mode flip and 4xx
    live traffic that works.
    """
    assert "seed" not in CAPABILITY_PARAMS
    assert "safety_tolerance" not in CAPABILITY_PARAMS

    mapped = _map(BflVideoConfig(), "flux-3-video", {"seed": 42, "safety_tolerance": 2, "some_future_param": "x"})
    assert mapped["seed"] == 42


def test_drop_params_does_not_suppress_the_gate(monkeypatch):
    """
    drop_params governs tuning knobs. Letting it silence a capability refusal would
    reinstate exactly the failure this gate exists to prevent.
    """
    monkeypatch.setattr(litellm, "drop_params", True)
    with pytest.raises(litellm.BadRequestError):
        _map(XAIVideoConfig(), "grok-imagine-video-1.5", {"end_image_url": "https://example.com/e.png"})


def test_undeclared_provider_behavior_is_unchanged():
    """
    A provider that has not opted in must keep behaving exactly as before; silence
    means "not audited", never "supports nothing".
    """
    config = OpenAIVideoConfig()
    assert isinstance(config.get_capability_param_support("sora-2"), UndeclaredCapabilityParams)
    assert (
        check_capability_params(
            model="sora-2",
            custom_llm_provider="openai",
            support=config.get_capability_param_support("sora-2"),
            requested_params={"end_image_url": "https://example.com/e.png"},
        )
        is None
    )


def test_refusal_lists_the_params_the_model_does_support():
    failure = check_capability_params(
        model="grok-imagine-video-1.5",
        custom_llm_provider="xai",
        support=DeclaredCapabilityParams(frozenset(("input_reference", "reference_audios"))),
        requested_params={"end_image_url": "https://e.com/e.png", "video_urls": ["https://e.com/v.mp4"]},
    )
    assert failure is not None
    assert failure.requested == ("end_image_url", "video_urls")
    assert failure.supported == ("input_reference", "reference_audios")


def test_declared_support_never_claims_params_outside_the_vocabulary():
    """
    A declaration is a promise the catalog reads. Anything it names outside the
    vocabulary is unenforceable, so the report must not surface it as executable.
    """
    failure = check_capability_params(
        model="m",
        custom_llm_provider="p",
        support=DeclaredCapabilityParams(frozenset(("input_reference", "not_a_capability_param"))),
        requested_params={"end_image_url": "https://e.com/e.png"},
    )
    assert failure is not None
    assert "not_a_capability_param" not in failure.supported
