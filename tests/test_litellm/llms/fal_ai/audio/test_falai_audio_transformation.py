from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from litellm.llms.fal_ai.audio.transformation import (
    FalAIAudioConfig,
    _normalize_fal_model_id,
)

ELEVEN_V3 = "fal_ai/fal-ai/elevenlabs/tts/eleven-v3"
ELEVEN_V3_ID = "fal-ai/elevenlabs/tts/eleven-v3"
FAL_API_BASE = "https://queue.fal.run"
SUBMIT_PAYLOAD = {
    "request_id": "test-rid",
    "status_url": f"{FAL_API_BASE}/fal-ai/elevenlabs/requests/test-rid/status",
    "response_url": f"{FAL_API_BASE}/fal-ai/elevenlabs/requests/test-rid",
}
RESULT_PAYLOAD = {
    "audio": {
        "url": "https://v3b.fal.media/files/x/output.mp3",
        "content_type": "audio/mpeg",
    }
}


def _resp(json_payload=None, content=b"", status_code=200):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_payload
    resp.content = content
    resp.raise_for_status = MagicMock()
    return resp


class TestFalAIAudioBasics:
    def setup_method(self):
        self.config = FalAIAudioConfig()

    def test_validate_environment_uses_fal_ai_api_key(self, monkeypatch):
        monkeypatch.setenv("FAL_AI_API_KEY", "key-123")
        headers = self.config.validate_environment(headers={}, model=ELEVEN_V3)
        assert headers["Authorization"] == "Key key-123"
        assert headers["Content-Type"] == "application/json"

    def test_validate_environment_falls_back_to_fal_key(self, monkeypatch):
        monkeypatch.delenv("FAL_AI_API_KEY", raising=False)
        monkeypatch.setenv("FAL_KEY", "fallback")
        headers = self.config.validate_environment(headers={}, model=ELEVEN_V3)
        assert headers["Authorization"] == "Key fallback"

    def test_validate_environment_raises_when_missing(self, monkeypatch):
        monkeypatch.delenv("FAL_AI_API_KEY", raising=False)
        monkeypatch.delenv("FAL_KEY", raising=False)
        with pytest.raises(ValueError, match="fal.ai API key is required"):
            self.config.validate_environment(headers={}, model=ELEVEN_V3)

    def test_get_complete_url_default(self, monkeypatch):
        monkeypatch.delenv("FAL_AI_API_BASE", raising=False)
        assert (
            self.config.get_complete_url(
                model=ELEVEN_V3, api_base=None, litellm_params={}
            )
            == FAL_API_BASE
        )

    def test_get_complete_url_strips_trailing_slash(self):
        assert (
            self.config.get_complete_url(
                model=ELEVEN_V3,
                api_base="https://custom.example.com/",
                litellm_params={},
            )
            == "https://custom.example.com"
        )

    def test_normalize_model_id_strips_prefix(self):
        assert _normalize_fal_model_id(ELEVEN_V3) == ELEVEN_V3_ID
        assert _normalize_fal_model_id(ELEVEN_V3_ID) == ELEVEN_V3_ID

    def test_normalize_model_id_rejects_empty(self):
        with pytest.raises(ValueError, match="empty after stripping"):
            _normalize_fal_model_id("fal_ai/")

    def test_transform_request_carries_text_voice_and_extras(self):
        request = self.config.transform_text_to_speech_request(
            model=ELEVEN_V3,
            input="hello",
            voice="Aria",
            optional_params={
                "stability": 0.5,
                "extra_body": {"language_code": "en"},
                "response_format": "mp3",
            },
            litellm_params={},
            headers={},
        )
        body = request["dict_body"]
        assert body["text"] == "hello"
        assert body["prompt"] == "hello"
        assert body["voice"] == "Aria"
        assert body["stability"] == 0.5
        assert body["language_code"] == "en"
        assert "response_format" not in body
        assert "extra_body" not in body

    def test_extract_audio_url_supports_known_shapes(self):
        assert (
            self.config._extract_audio_url({"audio": {"url": "x"}}) == "x"
        )
        assert self.config._extract_audio_url({"audio_url": "y"}) == "y"
        assert (
            self.config._extract_audio_url({"audio_file": {"url": "z"}}) == "z"
        )

    def test_extract_audio_url_raises_when_missing(self):
        with pytest.raises(ValueError, match="missing audio url"):
            self.config._extract_audio_url({"other": "shape"})


class TestFalAIAudioDispatch:
    def setup_method(self):
        self.config = FalAIAudioConfig()

    def test_sync_dispatch_submits_polls_downloads(self, monkeypatch):
        monkeypatch.setenv("FAL_AI_API_KEY", "key-123")

        binary_payload = b"audio-bytes-12345"
        binary_resp = _resp(content=binary_payload)
        result_resp = _resp(json_payload=RESULT_PAYLOAD)
        in_progress_status = _resp(json_payload={"status": "IN_PROGRESS"})
        completed_status = _resp(json_payload={"status": "COMPLETED"})
        submit_resp = _resp(json_payload=SUBMIT_PAYLOAD)

        client = MagicMock()
        client.post.return_value = submit_resp
        client.get.side_effect = [
            in_progress_status,
            completed_status,
            result_resp,
            binary_resp,
        ]

        monkeypatch.setattr(
            "litellm.llms.fal_ai.audio.transformation._get_httpx_client",
            lambda: client,
        )
        monkeypatch.setattr(
            "litellm.llms.fal_ai.audio.transformation.time.sleep", lambda _s: None
        )

        out = self.config._sync_dispatch(
            model=ELEVEN_V3,
            input="hello",
            voice="Aria",
            optional_params={},
            litellm_params_dict={},
            extra_headers=None,
            api_base=None,
            api_key="key-123",
        )

        assert out.response.content == binary_payload
        post_args = client.post.call_args
        assert post_args.kwargs["url"] == f"{FAL_API_BASE}/{ELEVEN_V3_ID}"
        assert post_args.kwargs["json"] == {
            "text": "hello",
            "prompt": "hello",
            "voice": "Aria",
        }
        assert post_args.kwargs["headers"]["Authorization"] == "Key key-123"
        get_urls = [c.kwargs.get("url") or c.args[0] for c in client.get.call_args_list]
        assert get_urls[0] == SUBMIT_PAYLOAD["status_url"]
        assert get_urls[1] == SUBMIT_PAYLOAD["status_url"]
        assert get_urls[2] == SUBMIT_PAYLOAD["response_url"]
        assert get_urls[3] == RESULT_PAYLOAD["audio"]["url"]

    def test_sync_dispatch_pulls_extra_body_from_kwargs(self, monkeypatch):
        monkeypatch.setenv("FAL_AI_API_KEY", "key-123")

        binary_resp = _resp(content=b"audio")
        result_resp = _resp(json_payload=RESULT_PAYLOAD)
        completed_status = _resp(json_payload={"status": "COMPLETED"})
        submit_resp = _resp(json_payload=SUBMIT_PAYLOAD)
        client = MagicMock()
        client.post.return_value = submit_resp
        client.get.side_effect = [completed_status, result_resp, binary_resp]

        monkeypatch.setattr(
            "litellm.llms.fal_ai.audio.transformation._get_httpx_client",
            lambda: client,
        )
        monkeypatch.setattr(
            "litellm.llms.fal_ai.audio.transformation.time.sleep", lambda _s: None
        )

        self.config.dispatch_text_to_speech(
            model=ELEVEN_V3,
            input="hello",
            voice="Aria",
            optional_params={},
            litellm_params_dict={},
            logging_obj=MagicMock(),
            timeout=30.0,
            extra_headers=None,
            base_llm_http_handler=None,
            aspeech=False,
            api_base=None,
            api_key="key-123",
            extra_body={"is_instrumental": True},
        )
        post_args = client.post.call_args
        assert post_args.kwargs["json"]["is_instrumental"] is True

    def test_sync_dispatch_raises_on_failed_status(self, monkeypatch):
        monkeypatch.setenv("FAL_AI_API_KEY", "key-123")
        submit_resp = _resp(json_payload=SUBMIT_PAYLOAD)
        failed = _resp(json_payload={"status": "FAILED"})

        client = MagicMock()
        client.post.return_value = submit_resp
        client.get.side_effect = [failed]
        monkeypatch.setattr(
            "litellm.llms.fal_ai.audio.transformation._get_httpx_client",
            lambda: client,
        )

        with pytest.raises(RuntimeError, match="status=FAILED"):
            self.config._sync_dispatch(
                model=ELEVEN_V3,
                input="hi",
                voice=None,
                optional_params={},
                litellm_params_dict={},
                extra_headers=None,
                api_base=None,
                api_key="key-123",
            )

    @pytest.mark.asyncio
    async def test_async_dispatch_submits_polls_downloads(self, monkeypatch):
        monkeypatch.setenv("FAL_AI_API_KEY", "key-123")

        binary_payload = b"audio-bytes-async"
        binary_resp = _resp(content=binary_payload)
        result_resp = _resp(json_payload=RESULT_PAYLOAD)
        in_progress_status = _resp(json_payload={"status": "IN_PROGRESS"})
        completed_status = _resp(json_payload={"status": "COMPLETED"})
        submit_resp = _resp(json_payload=SUBMIT_PAYLOAD)

        client = MagicMock()
        client.post = AsyncMock(return_value=submit_resp)
        get_responses = iter(
            [in_progress_status, completed_status, result_resp, binary_resp]
        )
        client.get = AsyncMock(side_effect=lambda **_: next(get_responses))

        monkeypatch.setattr(
            "litellm.llms.fal_ai.audio.transformation.get_async_httpx_client",
            lambda llm_provider: client,
        )

        async def _no_sleep(_s):
            return None

        monkeypatch.setattr(
            "litellm.llms.fal_ai.audio.transformation.asyncio.sleep", _no_sleep
        )

        out = await self.config._async_dispatch(
            model=ELEVEN_V3,
            input="hello",
            voice="Aria",
            optional_params={},
            litellm_params_dict={},
            extra_headers=None,
            api_base=None,
            api_key="key-123",
        )
        assert out.response.content == binary_payload


FAL_AI_AUDIO_MODELS = [
    "fal_ai/fal-ai/elevenlabs/tts/eleven-v3",
    "fal_ai/fal-ai/elevenlabs/tts/turbo-v2.5",
    "fal_ai/fal-ai/elevenlabs/tts/multilingual-v2",
    "fal_ai/fal-ai/minimax/speech-2.8-hd",
    "fal_ai/fal-ai/minimax/speech-2.8-turbo",
    "fal_ai/fal-ai/kokoro/american-english",
    "fal_ai/fal-ai/orpheus-tts",
    "fal_ai/fal-ai/dia-tts",
    "fal_ai/fal-ai/inworld-tts",
    "fal_ai/fal-ai/elevenlabs/music",
    "fal_ai/fal-ai/lyria3/pro",
    "fal_ai/fal-ai/minimax-music/v2.6",
    "fal_ai/fal-ai/stable-audio-25/text-to-audio",
    "fal_ai/fal-ai/elevenlabs/sound-effects/v2",
    "fal_ai/fal-ai/mmaudio-v2/text-to-audio",
    "fal_ai/fal-ai/stable-audio-3/medium/text-to-audio",
]


@pytest.mark.parametrize("model_id", FAL_AI_AUDIO_MODELS)
def test_fal_ai_audio_model_registered(model_id):
    from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

    backup = GetModelCostMap.load_local_model_cost_map()
    entry = backup.get(model_id)
    assert entry is not None, f"{model_id} missing from local backup model cost map"
    assert entry["litellm_provider"] == "fal_ai"
    assert entry["mode"] == "audio_speech"
    assert "/v1/audio/speech" in entry["supported_endpoints"]
    assert entry["supported_output_modalities"] == ["audio"]
    assert isinstance(entry["output_cost_per_second"], (int, float))


def test_provider_config_manager_returns_fal_ai_audio_config():
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_text_to_speech_config(
        model=ELEVEN_V3, provider=LlmProviders.FAL_AI
    )
    assert isinstance(config, FalAIAudioConfig)
