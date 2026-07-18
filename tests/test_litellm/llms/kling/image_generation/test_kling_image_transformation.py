from unittest.mock import Mock

import httpx
import pytest

from litellm.llms.kling.image_generation.transformation import (
    KlingImageGenerationConfig,
)
from litellm.types.utils import ImageResponse

MODEL = "kling/kling-v3"
API_BASE = "https://api-singapore.klingai.com/v1"


class _FakeClient:
    """Deterministic stand-in for the httpx handler; returns queued responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get(self, url, headers=None):
        self.calls.append(url)
        payload = self._responses.pop(0)
        response = Mock(spec=httpx.Response)
        response.raise_for_status.return_value = None
        response.json.return_value = payload
        return response


class TestKlingImageTransformation:
    def setup_method(self):
        self.config = KlingImageGenerationConfig()
        self.logging_obj = Mock()

    def test_validate_environment_sets_bearer_jwt(self):
        headers = self.config.validate_environment(
            headers={},
            model=MODEL,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="A" * 32 + ":" + "S" * 32,
        )
        assert headers["Authorization"].startswith("Bearer ")

    def test_get_complete_url_targets_images_generations(self, monkeypatch):
        monkeypatch.delenv("KLING_API_BASE", raising=False)
        url = self.config.get_complete_url(
            api_base=None,
            api_key=None,
            model=MODEL,
            optional_params={},
            litellm_params={},
        )
        assert url == f"{API_BASE}/images/generations"

    @pytest.mark.parametrize("resolution", ["1k", "2k"])
    def test_request_resolution_reaches_body(self, resolution):
        body = self.config.transform_image_generation_request(
            model=MODEL,
            prompt="a red fox",
            optional_params={"resolution": resolution, "n": 2, "aspect_ratio": "16:9"},
            litellm_params={},
            headers={},
        )
        assert body["model_name"] == "kling-v3"
        assert body["prompt"] == "a red fox"
        assert body["resolution"] == resolution
        assert body["n"] == 2
        assert body["aspect_ratio"] == "16:9"

    def test_request_defaults_resolution_to_1k(self):
        body = self.config.transform_image_generation_request(
            model=MODEL,
            prompt="x",
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert body["resolution"] == "1k"

    def test_request_reads_resolution_from_extra_body(self):
        body = self.config.transform_image_generation_request(
            model=MODEL,
            prompt="x",
            optional_params={"extra_body": {"resolution": "2k", "negative_prompt": "blurry"}},
            litellm_params={},
            headers={},
        )
        assert body["resolution"] == "2k"
        assert body["negative_prompt"] == "blurry"

    def test_map_openai_params_size_to_aspect_and_resolution_passthrough(self):
        mapped = self.config.map_openai_params(
            non_default_params={"size": "1024x1024", "resolution": "2k", "n": 3},
            optional_params={},
            model="kling-v3",
            drop_params=False,
        )
        assert mapped["aspect_ratio"] == "1:1"
        assert mapped["resolution"] == "2k"
        assert mapped["n"] == 3

    def test_map_openai_params_raises_on_unsupported(self):
        with pytest.raises(ValueError, match="is not supported"):
            self.config.map_openai_params(
                non_default_params={"totally_unknown": 1},
                optional_params={},
                model="kling-v3",
                drop_params=False,
            )

    def test_poll_returns_images_on_succeed(self):
        client = _FakeClient(
            [
                {
                    "code": 0,
                    "data": {
                        "task_status": "succeed",
                        "task_result": {"images": [{"url": "https://cdn/i.png"}]},
                    },
                }
            ]
        )
        result = self.config._poll_task_sync(
            poll_url=f"{API_BASE}/images/generations/task-1",
            headers={"Authorization": "Bearer x"},
            timeout_secs=30,
            client=client,
        )
        assert result["data"]["task_result"]["images"][0]["url"] == "https://cdn/i.png"
        assert client.calls == [f"{API_BASE}/images/generations/task-1"]

    def test_poll_raises_on_failed(self):
        client = _FakeClient([{"code": 0, "data": {"task_status": "failed", "task_status_msg": "nsfw"}}])
        with pytest.raises(ValueError, match="Kling image generation failed: nsfw"):
            self.config._poll_task_sync(
                poll_url=f"{API_BASE}/images/generations/task-1",
                headers={},
                timeout_secs=30,
                client=client,
            )

    def test_poll_times_out(self):
        client = _FakeClient([{"code": 0, "data": {"task_status": "processing"}}])
        with pytest.raises(TimeoutError, match="timed out"):
            self.config._poll_task_sync(
                poll_url=f"{API_BASE}/images/generations/task-1",
                headers={},
                timeout_secs=-1,
                client=client,
            )

    def test_transform_images_populates_response(self):
        model_response = ImageResponse()
        out = self.config._transform_images(
            {
                "data": {
                    "task_status": "succeed",
                    "task_result": {
                        "images": [
                            {"url": "https://cdn/a.png"},
                            {"url": "https://cdn/b.png"},
                        ]
                    },
                }
            },
            model_response,
        )
        assert [img.url for img in out.data] == [
            "https://cdn/a.png",
            "https://cdn/b.png",
        ]

    def test_response_raises_on_error_code(self):
        response = Mock(spec=httpx.Response)
        response.status_code = 401
        response.headers = {}
        response.json.return_value = {"code": 1002, "message": "AK/SK not supported"}
        with pytest.raises(Exception, match="AK/SK not supported"):
            self.config.transform_image_generation_response(
                model=MODEL,
                raw_response=response,
                model_response=ImageResponse(),
                logging_obj=self.logging_obj,
                request_data={},
                optional_params={},
                litellm_params={},
                encoding=None,
            )


def test_provider_config_manager_returns_kling_image_config():
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_image_generation_config(model="kling-v3", provider=LlmProviders.KLING)
    assert isinstance(config, KlingImageGenerationConfig)
