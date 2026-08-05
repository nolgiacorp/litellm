"""
Regression tests for litellm/types/videos/utils.py video-id encoding.

The managed video id embeds the provider's async retrieve handle (for BFL flux-3-video
a get_result URL) inside a base64 envelope. The id has to survive being placed in the
GET /v1/videos/{id} path, so the base64 must be URL-safe: no '/', '+', or '=' that would
break path routing (a standard-base64 id 404'd because the embedded URL's '/' split the
path). These lock URL-safety, a full round-trip, and backward-compatible decoding of ids
minted by the previous standard-base64 encoder.
"""

import base64

from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
    extract_original_video_id,
)

PROVIDER = "black_forest_labs"
MODEL_ID = "flux-3-video"
BFL_VIDEO_ID = "https://api.us6.bfl.ai/v1/get_result?id=1d6f6b1e-6c1a-4a2e-8b3a-9f0c2d4e5a6b"


def _assemble(video_id: str) -> str:
    return f"litellm:custom_llm_provider:{PROVIDER};model_id:{MODEL_ID};video_id:{video_id}"


def test_encoded_video_id_is_url_path_safe():
    encoded = encode_video_id_with_provider(BFL_VIDEO_ID, PROVIDER, MODEL_ID)
    assert encoded.startswith("video_")
    body = encoded.removeprefix("video_")
    assert "/" not in body
    assert "+" not in body
    assert "=" not in body
    assert "/" in base64.b64encode(_assemble(BFL_VIDEO_ID).encode("utf-8")).decode("utf-8")


def test_encode_drops_slash_and_plus_for_a_payload_that_produces_both():
    # this video_id makes the standard-base64 envelope carry both '+' and '/', the two
    # characters the URL-safe alphabet replaces with '-' and '_'
    video_id = "https://api.us6.bfl.ai/v1/get_result?id=࠾"
    standard = base64.b64encode(_assemble(video_id).encode("utf-8")).decode("utf-8")
    assert "+" in standard
    assert "/" in standard

    encoded = encode_video_id_with_provider(video_id, PROVIDER, MODEL_ID)
    body = encoded.removeprefix("video_")
    assert set(body) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")
    assert decode_video_id_with_provider(encoded)["video_id"] == video_id


def test_encode_decode_round_trips_bfl_polling_url():
    encoded = encode_video_id_with_provider(BFL_VIDEO_ID, PROVIDER, MODEL_ID)
    decoded = decode_video_id_with_provider(encoded)
    assert decoded["custom_llm_provider"] == PROVIDER
    assert decoded["model_id"] == MODEL_ID
    assert decoded["video_id"] == BFL_VIDEO_ID
    assert extract_original_video_id(encoded) == BFL_VIDEO_ID


def test_encode_is_idempotent_on_an_already_encoded_id():
    once = encode_video_id_with_provider(BFL_VIDEO_ID, PROVIDER, MODEL_ID)
    twice = encode_video_id_with_provider(once, PROVIDER, MODEL_ID)
    assert twice == once


def test_decodes_legacy_standard_base64_id():
    legacy = "video_" + base64.b64encode(_assemble(BFL_VIDEO_ID).encode("utf-8")).decode("utf-8")
    decoded = decode_video_id_with_provider(legacy)
    assert decoded["custom_llm_provider"] == PROVIDER
    assert decoded["model_id"] == MODEL_ID
    assert decoded["video_id"] == BFL_VIDEO_ID
