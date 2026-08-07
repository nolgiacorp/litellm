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
from litellm.llms.gemini.videos.transformation import GeminiVideoConfig
from litellm.llms.kling.videos.transformation import KlingVideoConfig
from litellm.llms.minimax.videos.transformation import MinimaxVideoConfig
from litellm.llms.openai.videos.transformation import OpenAIVideoConfig
from litellm.llms.openrouter.videos.transformation import OpenRouterVideoConfig
from litellm.llms.xai.videos.transformation import XAIVideoConfig
from litellm.proxy.video_endpoints.capabilities import build_video_capability_report
from litellm.types.router import GenericLiteLLMParams
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
    # negative_prompt. None of these surfaces has a negative channel: xAI's published
    # OpenAPI schema for /v1/videos/generations does not contain the string
    # "negative"; MiniMax's legacy body and its /v2 content-item shape both carry a
    # single prompt; OpenRouter's normalized schema has no slot and this
    # transformation drops what it does not recognize; and no seedance-2.0 or
    # seedvr schema on fal exposes one.
    (XAIVideoConfig(), "grok-imagine-video-1.5", "negative_prompt", "blurry, low quality"),
    (MinimaxVideoConfig(), "MiniMax-H3", "negative_prompt", "blurry, low quality"),
    (MinimaxVideoConfig(), "MiniMax-Hailuo-2.3", "negative_prompt", "blurry, low quality"),
    (OpenRouterVideoConfig(), "openrouter/bytedance/seedance-2.0", "negative_prompt", "blurry, low quality"),
    (FalAIVideoConfig(), "fal_ai/bytedance/seedance-2.0/text-to-video", "negative_prompt", "blurry"),
    (FalAIVideoConfig(), "fal_ai/fal-ai/seedvr/upscale/video", "negative_prompt", "blurry"),
    # Same fal family as the declared kling lanes, but the turbo schemas expose only
    # prompt, aspect_ratio and duration, so the family marker alone is too coarse.
    (FalAIVideoConfig(), "fal_ai/fal-ai/kling-video/v3/pro/turbo/text-to-video", "negative_prompt", "blurry"),
    # Direct Kling routes every model here as kling-v3, and the only version-specific
    # statement available says 2.5/2.6/3.0 do not honor negative_prompt. Its fal twin
    # does publish the field, which is why they differ.
    (KlingVideoConfig(), "kling/kling-v3", "negative_prompt", "blurry, low quality"),
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
    # Regeneration carries seconds as the source video's declared length: it is a price
    # input, not a control, and the request is refused without it.
    (
        MinimaxVideoConfig(),
        "MiniMax-H3",
        {"base_video_url": "https://example.com/src.mp4", "seconds": 6},
    ),
    # fal's kling twin does take an end frame, unlike the direct kling route.
    (FalAIVideoConfig(), "fal_ai/fal-ai/kling-video/v3/pro/image-to-video", {"end_image_url": "https://e.com/e.png"}),
    # negative_prompt on the surfaces that do carry one. fal's non-turbo
    # kling-video/v3 schemas publish it in both directions (default
    # "blur, distort, and low quality"), unlike the direct kling route below.
    (FalAIVideoConfig(), "fal_ai/fal-ai/kling-video/v3/pro/text-to-video", {"negative_prompt": "blurry"}),
    (FalAIVideoConfig(), "fal_ai/fal-ai/kling-video/v3/pro/image-to-video", {"negative_prompt": "blurry"}),
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


def test_unaudited_fal_app_stays_undeclared():
    """
    fal is a gateway onto arbitrary app schemas. Declaring an app we have never read
    as exhaustively known would 400 a param that app accepts under a name this
    transformation has not seen, replacing a working passthrough with a refusal.
    """
    config = FalAIVideoConfig()
    assert isinstance(
        config.get_capability_param_support("fal_ai/some-vendor/mystery-app"),
        UndeclaredCapabilityParams,
    )
    _map(config, "fal_ai/some-vendor/mystery-app", {"end_image_url": "https://e.com/e.png"})


def test_fal_upscale_app_declares_only_the_media_slot_it_reads():
    """The restore lane takes a video and restore controls; it renders no soundtrack."""
    model = "fal_ai/fal-ai/seedvr/upscale/video"
    _map(FalAIVideoConfig(), model, {"input_reference": "https://e.com/src.mp4"})
    with pytest.raises(litellm.BadRequestError):
        _map(FalAIVideoConfig(), model, {"generate_audio": True})


def test_kling_image_url_alias_sets_the_start_frame():
    """
    image_url is declared as executable, so it has to land in Kling's image field.
    Forwarding it verbatim would leave an i2v request rejected as frameless and a
    t2v request silently unconditioned.
    """
    mapped = _map(KlingVideoConfig(), "kling/kling-v3-i2v", {"image_url": "https://e.com/s.png"})
    assert mapped["image"] == "https://e.com/s.png"
    assert "image_url" not in mapped, "the alias must not also reach Kling as an unknown field"


def test_gemini_omni_honors_input_reference_as_the_start_frame(monkeypatch):
    """
    input_reference is declared as executable, so an Omni request carrying it must
    become an image_to_video interaction. Sending it as text-to-video would bill for a
    generation that ignored the requested frame.
    """
    from litellm.llms.gemini.videos import omni_transformation

    monkeypatch.setattr(omni_transformation, "fetch_image_as_base64", lambda url: ("Zm9v", "image/png"))

    request_data, _, _ = omni_transformation.GeminiOmniVideoConfig().transform_video_create_request(
        model="gemini/gemini-omni-flash-preview",
        prompt="a cat",
        api_base="https://generativelanguage.googleapis.com/v1beta/interactions",
        video_create_optional_request_params={"input_reference": "https://e.com/s.png"},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert request_data["generation_config"] == {"video_config": {"task": "image_to_video"}}
    assert request_data["input"][0] == {"type": "image", "data": "Zm9v", "mime_type": "image/png"}


def _veo_request(model: str, params: dict):
    return GeminiVideoConfig().transform_video_create_request(
        model=model,
        prompt="a cat",
        api_base="https://generativelanguage.googleapis.com",
        video_create_optional_request_params=params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )


def test_veo_2_refuses_a_soundtrack_it_cannot_render():
    """
    Only Veo 3.x renders audio. The flag is consumed rather than forwarded, so
    accepting generate_audio=true on Veo 2 would bill a silent video for a request
    that asked for sound.
    """
    with pytest.raises(ValueError, match="generate_audio=true is not supported"):
        _veo_request("gemini/veo-2.0-generate-001", {"generate_audio": True})


@pytest.mark.parametrize(
    "model,params",
    (
        # Veo 2 renders silent video, so no-audio is exactly what it delivers.
        ("gemini/veo-2.0-generate-001", {"generate_audio": False}),
        ("gemini/veo-3.0-generate-001", {"generate_audio": True}),
    ),
)
def test_veo_audio_values_a_model_does_deliver_still_pass(model, params):
    _veo_request(model, params)


def _deployment(model_name: str, model: str, custom_llm_provider: str | None = None):
    litellm_params = {"model": model}
    if custom_llm_provider is not None:
        litellm_params["custom_llm_provider"] = custom_llm_provider
    return {"model_name": model_name, "litellm_params": litellm_params}


def _entry(report, model_name: str):
    return next((entry for entry in report["data"] if entry["model"] == model_name), None)


def test_report_intersects_capabilities_across_deployments_of_one_name():
    """
    The router may route to any healthy deployment under a name, so advertising the
    first one's surface would promise a capability that 400s once routing lands on the
    other. Only what every route executes is safe to publish.
    """
    report = build_video_capability_report(
        (
            _deployment("kling-v3-i2v", "kling/kling-v3-i2v"),
            _deployment("kling-v3-i2v", "fal_ai/fal-ai/kling-video/v3/pro/image-to-video"),
        ),
        visible_models=frozenset({"kling-v3-i2v"}),
    )

    entry = _entry(report, "kling-v3-i2v")
    assert entry is not None
    assert entry["custom_llm_providers"] == ["fal_ai", "kling"]
    # fal's twin takes an end frame; the direct Kling route has no field for one.
    assert "end_image_url" not in entry["capability_params"]
    assert "input_reference" in entry["capability_params"]


def test_report_uses_the_deployments_explicit_provider():
    """
    A deployment routes on litellm_params.custom_llm_provider, so resolving from the
    model string alone would report a surface the route never serves.
    """
    report = build_video_capability_report(
        (_deployment("veo-3", "veo-3.0-generate-001", custom_llm_provider="gemini"),),
        visible_models=frozenset({"veo-3"}),
    )

    entry = _entry(report, "veo-3")
    assert entry is not None
    assert entry["custom_llm_providers"] == ["gemini"]
    assert entry["declared"] is True


def test_report_omits_models_the_caller_cannot_route_to():
    """A key scoped to a subset of models must not learn the rest exist."""
    report = build_video_capability_report(
        (
            _deployment("visible-video", "kling/kling-v3"),
            _deployment("restricted-video", "kling/kling-v3"),
        ),
        visible_models=frozenset({"visible-video"}),
    )

    assert [entry["model"] for entry in report["data"]] == ["visible-video"]


# --- negative_prompt ------------------------------------------------------


def test_bfl_negative_prompt_is_refused_rather_than_forwarded_into_a_422():
    """
    BFL is the one provider where forwarding was worse than dropping. Every
    /v1/flux-3-video mode schema sets additionalProperties false, so a passed-through
    negative_prompt comes back as 422 extra_forbidden and the whole generation fails
    with a provider-shaped error that names no capability. The gate turns it into a
    400 that names the param and the model before the request is ever sent.
    """
    with pytest.raises(litellm.BadRequestError) as excinfo:
        _map(BflVideoConfig(), "flux-3-video", {"negative_prompt": "blurry"})

    assert "negative_prompt" in str(excinfo.value)
    assert "flux-3-video" in str(excinfo.value)


def test_declared_negative_prompt_actually_reaches_the_fal_provider():
    """
    Passing the gate is not the same as being executed. fal carries negative_prompt on
    its verbatim passthrough rather than through an explicit mapping, so a change to
    what that loop forwards would leave the param declared and silently dropped, which
    is the exact failure this ticket is about.
    """
    mapped = _map(FalAIVideoConfig(), "fal_ai/fal-ai/kling-video/v3/pro/text-to-video", {"negative_prompt": "blurry"})

    assert mapped["negative_prompt"] == "blurry"


def test_veo_negative_prompt_reaches_the_provider_as_negative_prompt():
    """
    Veo names the field negativePrompt. The normalization lives in map_openai_params,
    so this goes through the same choke point a real request does; asserting against
    transform_video_create_request alone would skip the rename and pass on a body that
    never carried the exclusion.
    """
    mapped = _map(GeminiVideoConfig(), "gemini/veo-3.0-generate-001", {"negative_prompt": "blurry, low quality"})

    assert mapped["negativePrompt"] == "blurry, low quality"
    assert "negative_prompt" not in mapped

    request_data, _, _ = _veo_request("gemini/veo-3.0-generate-001", mapped)
    assert request_data["parameters"]["negativePrompt"] == "blurry, low quality"


def test_gemini_omni_negative_prompt_is_folded_into_the_prompt():
    """
    Omni has no negative channel, so the exclusion is carried as prompt text. That is
    why it is declared: it constrains the render rather than being discarded.
    """
    from litellm.llms.gemini.videos import omni_transformation

    request_data, _, _ = omni_transformation.GeminiOmniVideoConfig().transform_video_create_request(
        model="gemini/gemini-omni-flash-preview",
        prompt="a cat",
        api_base="https://generativelanguage.googleapis.com/v1beta/interactions",
        video_create_optional_request_params={"negative_prompt": "blurry, low quality"},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert request_data["input"] == "a cat Do not include: blurry, low quality."


def test_negative_prompt_is_in_the_vocabulary_but_carries_no_catalog_flag():
    """
    Every other member of the vocabulary backs a published GET /models capability
    flag. negative_prompt does not; it is published unconditionally on the video
    request, which is precisely why nothing caught the silent drop.
    """
    assert "negative_prompt" in CAPABILITY_PARAMS
