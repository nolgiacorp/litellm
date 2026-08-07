"""
NOL-521 / NOL-499: fal-ai/minimax-music/v2.6 fails every job with a 422 because
fal requires `lyrics` whenever `is_instrumental` is false (both default to
false/empty), and the fal audio transform only ever sent {"text", "prompt"}.

The documented escape hatch is `lyrics_optimizer: true`, which makes fal write
lyrics from the prompt so the model still produces the vocal output it is
chosen for (setting `is_instrumental: true` would also clear the 422 but
silently changes the product).

These tests pin the plumbing that lets a deployment carry that flag WITHOUT a
fork code change: an `extra_body` mapping on the router deployment's
`litellm_params` must survive all the way into the JSON body POSTed to fal.

Chain under test:
    router deployment litellm_params
      -> Router._aspeech spreads litellm_params into litellm.aspeech(**kwargs)
      -> main.speech() leaves unknown kwargs in **kwargs
      -> FalAIAudioConfig.dispatch_text_to_speech lifts kwargs["extra_body"]
      -> transform_text_to_speech_request merges extra_body into the body
"""

import json

import httpx
import pytest
import respx

import litellm
from litellm.llms.fal_ai.audio.transformation import FalAIAudioConfig

MODEL = "fal_ai/fal-ai/minimax-music/v2.6"
SUBMIT_URL = "https://queue.fal.run/fal-ai/minimax-music/v2.6"
STATUS_URL = "https://queue.fal.run/fal-ai/minimax-music/v2.6/requests/rid-1/status"
RESULT_URL = "https://queue.fal.run/fal-ai/minimax-music/v2.6/requests/rid-1"
AUDIO_URL = "https://cdn.fal.test/song.mp3"


def _mock_fal_happy_path(mock: respx.MockRouter) -> list:
    """Mock fal's queue API end to end and capture every submitted body."""
    submitted: list = []

    def _on_submit(request: httpx.Request) -> httpx.Response:
        submitted.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "request_id": "rid-1",
                "status_url": STATUS_URL,
                "response_url": RESULT_URL,
            },
        )

    mock.post(SUBMIT_URL).mock(side_effect=_on_submit)
    mock.get(STATUS_URL).mock(return_value=httpx.Response(200, json={"status": "COMPLETED"}))
    mock.get(RESULT_URL).mock(return_value=httpx.Response(200, json={"audio": {"url": AUDIO_URL}}))
    mock.get(AUDIO_URL).mock(return_value=httpx.Response(200, content=b"ID3fake-mp3-bytes"))
    return submitted


class TestFalAudioExtraBodyReachesRequest:
    def test_transform_merges_extra_body(self):
        """Unit: extra_body keys land in the fal body."""
        data = FalAIAudioConfig().transform_text_to_speech_request(
            model="fal-ai/minimax-music/v2.6",
            input="a dreamy synthwave track about the coast at night",
            voice=None,
            optional_params={"extra_body": {"lyrics_optimizer": True}},
            litellm_params={},
            headers={},
        )
        body = data["dict_body"]
        assert body["lyrics_optimizer"] is True
        # the prompt fal generates lyrics FROM must still be present
        assert body["prompt"] == "a dreamy synthwave track about the coast at night"
        # we must NOT silently switch the product to instrumental
        assert "is_instrumental" not in body

    def test_extra_body_is_not_leaked_as_a_literal_key(self):
        """`extra_body` itself must be unwrapped, never sent as a field."""
        data = FalAIAudioConfig().transform_text_to_speech_request(
            model="fal-ai/minimax-music/v2.6",
            input="prompt",
            voice=None,
            optional_params={"extra_body": {"lyrics_optimizer": True}},
            litellm_params={},
            headers={},
        )
        assert "extra_body" not in data["dict_body"]

    @respx.mock
    def test_speech_call_sends_lyrics_optimizer(self, monkeypatch):
        """Integration: litellm.speech(extra_body=...) reaches fal's wire body."""
        monkeypatch.setenv("FAL_AI_API_KEY", "test-key")
        submitted = _mock_fal_happy_path(respx.mock)

        litellm.speech(
            model=MODEL,
            input="a dreamy synthwave track about the coast at night",
            extra_body={"lyrics_optimizer": True},
        )

        assert len(submitted) == 1
        assert submitted[0]["lyrics_optimizer"] is True
        assert submitted[0]["prompt"]

    def test_router_spreads_deployment_extra_body_into_speech_kwargs(self, monkeypatch):
        """
        Integration: a deployment declaring extra_body in litellm_params (i.e. a
        pure litellm-config.yaml change) reaches litellm.aspeech as a kwarg.

        This is the step that makes the fix config-only rather than a fork edit.
        """
        import asyncio

        captured: dict = {}

        async def _fake_aspeech(**kwargs):
            captured.update(kwargs)
            return "ok"

        monkeypatch.setattr(litellm, "aspeech", _fake_aspeech)

        router = litellm.Router(
            model_list=[
                {
                    "model_name": "music-minimax-v2.6",
                    "litellm_params": {
                        "model": MODEL,
                        "api_key": "test-key",
                        "extra_body": {"lyrics_optimizer": True},
                    },
                    "model_info": {"mode": "audio_speech"},
                }
            ]
        )

        asyncio.run(router.aspeech(model="music-minimax-v2.6", input="a moody trip-hop song", voice=None))

        assert captured.get("extra_body") == {"lyrics_optimizer": True}, (
            f"deployment extra_body did not reach litellm.aspeech; got keys={sorted(captured)}"
        )


class TestFalAudioRegressionGuard:
    def test_without_extra_body_the_flag_is_absent(self):
        """Pins the pre-fix behaviour so the test proves the flag is what changed."""
        data = FalAIAudioConfig().transform_text_to_speech_request(
            model="fal-ai/minimax-music/v2.6",
            input="prompt",
            voice=None,
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert "lyrics_optimizer" not in data["dict_body"]
