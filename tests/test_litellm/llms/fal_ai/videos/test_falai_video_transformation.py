from unittest.mock import Mock

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.exception_mapping_utils import exception_type
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.fal_ai.videos.transformation import (
    FalAIVideoConfig,
    _classify_result_payload,
    _GeneratedVideo,
    _GenerationFailed,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
)

SORA_2_MODEL = "fal_ai/fal-ai/sora-2/text-to-video"
KLING_MODEL = "fal_ai/fal-ai/kling-video/v2.5-turbo/pro/text-to-video"
KLING_MODEL_ID = "fal-ai/kling-video/v2.5-turbo/pro/text-to-video"
KLING_QUEUE_NAMESPACE = "fal-ai/kling-video"
FAL_API_BASE = "https://queue.fal.run"
FAL_CONTENT_POLICY_BODY = (
    '{"detail":[{"loc":["body","image_urls"],"type":"content_policy_violation",'
    '"ctx":{"extra_info":{"reason":"partner_validation_failed"}},'
    '"msg":"The images or videos provided may contain likenesses of real people or other private information '
    'that cannot be processed."}]}'
)


FAL_FILE_DOWNLOAD_ERROR_RESULT = {
    "detail": [
        {
            "loc": ["body", "image_urls"],
            "msg": "Failed to download the file. Please check if the URL is accessible and try again.",
            "type": "file_download_error",
            "url": "https://docs.fal.ai/errors#file_download_error",
            "input": ["https://storage.example.com/missing-reference.jpg"],
        }
    ]
}

FAL_QUEUE_COMPLETED_STATUS = {
    "status": "COMPLETED",
    "request_id": "abc-123",
    "response_url": f"{FAL_API_BASE}/{KLING_QUEUE_NAMESPACE}/requests/abc-123",
    "status_url": f"{FAL_API_BASE}/{KLING_QUEUE_NAMESPACE}/requests/abc-123/status",
    "cancel_url": f"{FAL_API_BASE}/{KLING_QUEUE_NAMESPACE}/requests/abc-123/cancel",
    "logs": None,
    "metrics": {"inference_time": 0.5173070430755615},
}


def _fal_status_response(payload, request_id="abc-123", status_code=200):
    request = httpx.Request(
        "GET", f"{FAL_API_BASE}/{KLING_MODEL_ID}/requests/{request_id}/status"
    )
    return httpx.Response(status_code, json=payload, request=request)


def _fal_result_response(payload, request_id="abc-123", status_code=200):
    request = httpx.Request(
        "GET", f"{FAL_API_BASE}/{KLING_MODEL_ID}/requests/{request_id}"
    )
    return httpx.Response(status_code, json=payload, request=request)


class _RecordingClient:
    """Stands in for the injected httpx handler; records the follow-up result lookup."""

    def __init__(self, response):
        self._response = response
        self.calls = []

    def get(self, url, headers=None, **kwargs):
        self.calls.append((url, headers))
        return self._response


class _RecordingAsyncClient(_RecordingClient):
    async def get(self, url, headers=None, **kwargs):
        self.calls.append((url, headers))
        return self._response


class TestFalAIVideoTransformation:
    def setup_method(self):
        self.config = FalAIVideoConfig()
        self.mock_logging_obj = Mock()

    def test_get_error_class_raises_content_policy_violation(self) -> None:
        with pytest.raises(litellm.ContentPolicyViolationError) as exc_info:
            self.config.get_error_class(
                error_message=FAL_CONTENT_POLICY_BODY,
                status_code=422,
                headers=httpx.Headers(),
            )

        assert exc_info.value.status_code == 400
        assert FAL_CONTENT_POLICY_BODY in exc_info.value.message
        assert "likenesses of real people" in str(exc_info.value)

    def test_content_policy_violation_remains_terminal_after_exception_mapping(self) -> None:
        with pytest.raises(litellm.ContentPolicyViolationError) as exc_info:
            self.config.get_error_class(
                error_message=FAL_CONTENT_POLICY_BODY,
                status_code=422,
                headers=httpx.Headers(),
            )

        mapped_exception = exception_type(
            model="fal_ai/bytedance/seedance-2.0/reference-to-video",
            original_exception=exc_info.value,
            custom_llm_provider="fal_ai",
        )

        assert isinstance(mapped_exception, litellm.ContentPolicyViolationError)
        assert mapped_exception.status_code == 400
        assert litellm._should_retry(400) is False

    def test_get_error_class_keeps_generic_errors_as_base_llm_exception(self) -> None:
        with pytest.raises(BaseLLMException) as exc_info:
            self.config.get_error_class(
                error_message="internal server error",
                status_code=500,
                headers=httpx.Headers(),
            )

        assert type(exc_info.value) is BaseLLMException
        assert exc_info.value.status_code == 500
        assert not isinstance(exc_info.value, litellm.ContentPolicyViolationError)

    def test_content_policy_detection_supports_partner_validation_signature(self) -> None:
        assert self.config._is_content_policy_rejection("partner_validation_failed")

    def test_validate_environment_uses_fal_ai_api_key(self, monkeypatch):
        monkeypatch.setenv("FAL_AI_API_KEY", "test-key-123")
        headers = self.config.validate_environment(
            headers={},
            model=SORA_2_MODEL,
        )
        assert headers["Authorization"] == "Key test-key-123"
        assert headers["Content-Type"] == "application/json"

    def test_validate_environment_falls_back_to_fal_key(self, monkeypatch):
        monkeypatch.delenv("FAL_AI_API_KEY", raising=False)
        monkeypatch.setenv("FAL_KEY", "fallback-key")
        headers = self.config.validate_environment(headers={}, model=SORA_2_MODEL)
        assert headers["Authorization"] == "Key fallback-key"

    def test_validate_environment_raises_when_missing(self, monkeypatch):
        monkeypatch.delenv("FAL_AI_API_KEY", raising=False)
        monkeypatch.delenv("FAL_KEY", raising=False)
        with pytest.raises(ValueError, match="fal.ai API key is required"):
            self.config.validate_environment(headers={}, model=SORA_2_MODEL)

    def test_get_complete_url_uses_default_base(self, monkeypatch):
        monkeypatch.delenv("FAL_AI_API_BASE", raising=False)
        url = self.config.get_complete_url(
            model=SORA_2_MODEL, api_base=None, litellm_params={}
        )
        assert url == FAL_API_BASE

    def test_get_complete_url_strips_trailing_slash(self):
        url = self.config.get_complete_url(
            model=SORA_2_MODEL,
            api_base="https://custom.example.com/",
            litellm_params={},
        )
        assert url == "https://custom.example.com"

    def test_map_openai_params_converts_seconds_and_size(self):
        params = self.config.map_openai_params(
            video_create_optional_params={"seconds": 5, "size": "1280x720"},
            model=KLING_MODEL,
            drop_params=False,
        )
        assert params["duration"] == "5"
        assert params["aspect_ratio"] == "16:9"

    def test_map_openai_params_falls_back_to_colon_replacement(self):
        params = self.config.map_openai_params(
            video_create_optional_params={"size": "640x480"},
            model=KLING_MODEL,
            drop_params=False,
        )
        assert params["aspect_ratio"] == "640:480"

    def test_map_openai_params_unpacks_extra_body(self):
        params = self.config.map_openai_params(
            video_create_optional_params={
                "extra_body": {"negative_prompt": "blurry", "cfg_scale": 0.5}
            },
            model=KLING_MODEL,
            drop_params=False,
        )
        assert params["negative_prompt"] == "blurry"
        assert params["cfg_scale"] == 0.5
        assert "extra_body" not in params

    def test_transform_video_create_request_builds_queue_url(self):
        data, files, url = self.config.transform_video_create_request(
            model=KLING_MODEL,
            prompt="A demo video",
            api_base=FAL_API_BASE,
            video_create_optional_request_params={
                "duration": "5",
                "aspect_ratio": "16:9",
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == f"{FAL_API_BASE}/{KLING_MODEL_ID}"
        assert data["prompt"] == "A demo video"
        assert data["duration"] == "5"
        assert data["aspect_ratio"] == "16:9"
        assert "model" not in data
        assert files == []

    def test_transform_video_create_response_encodes_model_into_video_id(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "request_id": "abc-123",
            "status": "IN_QUEUE",
        }

        video_obj = self.config.transform_video_create_response(
            model=KLING_MODEL,
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="fal_ai",
            request_data={"duration": "5", "aspect_ratio": "16:9"},
        )

        assert isinstance(video_obj, VideoObject)
        assert video_obj.status == "queued"
        assert video_obj.id.startswith("video_")

        decoded = decode_video_id_with_provider(video_obj.id)
        assert decoded.get("video_id") == "abc-123"
        assert decoded.get("custom_llm_provider") == "fal_ai"
        assert decoded.get("model_id") == KLING_MODEL_ID

        assert video_obj.seconds == "5"
        assert video_obj.size == "16x9"

    def test_transform_video_status_retrieve_request_builds_status_url(self):
        encoded_id = encode_video_id_with_provider("abc-123", "fal_ai", KLING_MODEL_ID)
        url, params = self.config.transform_video_status_retrieve_request(
            video_id=encoded_id,
            api_base=FAL_API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == f"{FAL_API_BASE}/{KLING_QUEUE_NAMESPACE}/requests/abc-123/status"
        assert params == {}

    def test_transform_video_status_retrieve_request_reconstructs_from_model_id(self):
        encoded_id = encode_video_id_with_provider("abc-123", "fal_ai", KLING_MODEL_ID)
        url, params = self.config.transform_video_status_retrieve_request(
            video_id=encoded_id,
            api_base="https://attacker.example.com",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert (
            url
            == f"https://attacker.example.com/{KLING_QUEUE_NAMESPACE}/requests/abc-123/status"
        )
        assert params == {}

    def test_status_and_content_urls_use_owner_app_namespace(self):
        seedance_id = "fal-ai/bytedance/seedance/v2/pro/text-to-video"
        encoded_id = encode_video_id_with_provider("abc-123", "fal_ai", seedance_id)

        status_url, _ = self.config.transform_video_status_retrieve_request(
            video_id=encoded_id,
            api_base=FAL_API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        content_url, _ = self.config.transform_video_content_request(
            video_id=encoded_id,
            api_base=FAL_API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert status_url == f"{FAL_API_BASE}/fal-ai/bytedance/requests/abc-123/status"
        assert content_url == f"{FAL_API_BASE}/fal-ai/bytedance/requests/abc-123"

    def test_queue_namespace_keeps_two_segment_model_ids(self):
        encoded_id = encode_video_id_with_provider("abc-123", "fal_ai", "fal-ai/sora-2")
        url, _ = self.config.transform_video_status_retrieve_request(
            video_id=encoded_id,
            api_base=FAL_API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert url == f"{FAL_API_BASE}/fal-ai/sora-2/requests/abc-123/status"

    def test_transform_video_status_request_url_path_segment_is_encoded(self):
        encoded_id = encode_video_id_with_provider(
            "../../../etc/passwd", "fal_ai", KLING_MODEL_ID
        )
        url, _ = self.config.transform_video_status_retrieve_request(
            video_id=encoded_id,
            api_base=FAL_API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert "/requests/..%2F..%2F..%2Fetc%2Fpasswd/status" in url

    def test_transform_video_status_response_maps_in_progress(self):
        mock_response = _fal_status_response(
            {
                "request_id": "abc-123",
                "status": "IN_PROGRESS",
                "queue_position": 2,
            }
        )
        status_obj = self.config.transform_video_status_retrieve_response(
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="fal_ai",
        )
        assert status_obj.status == "in_progress"
        assert status_obj.progress == 2

    def test_transform_video_status_response_maps_failed_with_error(self):
        mock_response = _fal_status_response(
            {
                "request_id": "abc-123",
                "status": "FAILED",
                "error": "model timed out",
            }
        )
        status_obj = self.config.transform_video_status_retrieve_response(
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="fal_ai",
        )
        assert status_obj.status == "failed"
        assert status_obj.error is not None
        assert status_obj.error["message"] == "model timed out"

    def test_queue_completed_with_failed_generation_reports_failed_status(self):
        result_client = _RecordingClient(
            _fal_result_response(FAL_FILE_DOWNLOAD_ERROR_RESULT, status_code=422)
        )
        config = FalAIVideoConfig(sync_client=result_client)

        status_obj = config.transform_video_status_retrieve_response(
            raw_response=_fal_status_response(FAL_QUEUE_COMPLETED_STATUS),
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="fal_ai",
        )

        assert status_obj.status == "failed"
        assert status_obj.error is not None
        assert "Failed to download the file" in status_obj.error["message"]
        assert "image_urls" in status_obj.error["message"]

        assert result_client.calls, "terminal queue status must be resolved against the result payload"
        result_url, _ = result_client.calls[0]
        assert result_url == f"{FAL_API_BASE}/{KLING_MODEL_ID}/requests/abc-123"

    @pytest.mark.asyncio
    async def test_async_queue_completed_with_failed_generation_reports_failed_status(self):
        result_client = _RecordingAsyncClient(
            _fal_result_response(FAL_FILE_DOWNLOAD_ERROR_RESULT, status_code=422)
        )
        config = FalAIVideoConfig(async_client=result_client)

        status_obj = await config.async_transform_video_status_retrieve_response(
            raw_response=_fal_status_response(FAL_QUEUE_COMPLETED_STATUS),
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="fal_ai",
        )

        assert status_obj.status == "failed"
        assert status_obj.error is not None
        assert "Failed to download the file" in status_obj.error["message"]

    def test_status_lookup_forwards_authorization_to_result_endpoint(self):
        result_client = _RecordingClient(
            _fal_result_response({"video": {"url": "https://cdn.example.com/v.mp4"}})
        )
        config = FalAIVideoConfig(sync_client=result_client)
        request = httpx.Request(
            "GET",
            f"{FAL_API_BASE}/{KLING_MODEL_ID}/requests/abc-123/status",
            headers={"Authorization": "Key secret-token"},
        )
        status_response = httpx.Response(
            200, json=FAL_QUEUE_COMPLETED_STATUS, request=request
        )

        status_obj = config.transform_video_status_retrieve_response(
            raw_response=status_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="fal_ai",
        )

        assert status_obj.status == "completed"
        _, headers = result_client.calls[0]
        assert headers == {"Authorization": "Key secret-token"}

    def test_queue_completed_with_video_reports_completed_status(self):
        result_client = _RecordingClient(
            _fal_result_response({"video": {"url": "https://cdn.example.com/v.mp4"}})
        )
        config = FalAIVideoConfig(sync_client=result_client)

        status_obj = config.transform_video_status_retrieve_response(
            raw_response=_fal_status_response(FAL_QUEUE_COMPLETED_STATUS),
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="fal_ai",
        )

        assert status_obj.status == "completed"
        assert status_obj.error is None

    def test_content_on_failed_job_raises_instead_of_serving_json_as_media(self):
        config = FalAIVideoConfig()
        failed_result = _fal_result_response(
            FAL_FILE_DOWNLOAD_ERROR_RESULT, status_code=200
        )

        with pytest.raises(litellm.BadRequestError) as exc_info:
            config.transform_video_content_response(
                raw_response=failed_result,
                logging_obj=self.mock_logging_obj,
            )

        assert exc_info.value.status_code == 400
        assert "Failed to download the file" in exc_info.value.message
        assert litellm._should_retry(exc_info.value.status_code) is False

    def test_customer_facing_error_drops_raw_fal_envelope(self):
        import json

        with pytest.raises(litellm.BadRequestError) as exc_info:
            self.config.get_error_class(
                error_message=json.dumps(FAL_FILE_DOWNLOAD_ERROR_RESULT),
                status_code=422,
                headers=httpx.Headers(),
            )

        message = exc_info.value.message
        assert "Failed to download the file" in message
        assert "file_download_error" not in message
        assert "loc" not in message
        assert "docs.fal.ai" not in message

    @pytest.mark.parametrize("transient_status", [429, 500, 502, 503])
    def test_transient_result_lookup_failure_is_not_reported_as_generation_failure(
        self, transient_status
    ):
        result_client = _RecordingClient(
            _fal_result_response({"detail": "upstream unavailable"}, status_code=transient_status)
        )
        config = FalAIVideoConfig(sync_client=result_client)

        with pytest.raises(Exception) as exc_info:
            config.transform_video_status_retrieve_response(
                raw_response=_fal_status_response(FAL_QUEUE_COMPLETED_STATUS),
                logging_obj=self.mock_logging_obj,
                custom_llm_provider="fal_ai",
            )

        assert not isinstance(exc_info.value, litellm.BadRequestError)
        assert exc_info.value.status_code == transient_status

    def test_rate_limited_result_lookup_stays_retryable(self):
        result_client = _RecordingClient(
            _fal_result_response({"detail": "slow down"}, status_code=429)
        )
        config = FalAIVideoConfig(sync_client=result_client)

        with pytest.raises(litellm.RateLimitError) as exc_info:
            config.transform_video_status_retrieve_response(
                raw_response=_fal_status_response(FAL_QUEUE_COMPLETED_STATUS),
                logging_obj=self.mock_logging_obj,
                custom_llm_provider="fal_ai",
            )

        assert exc_info.value.status_code == 429
        assert litellm._should_retry(429) is True

    @pytest.mark.parametrize(
        "status_code,expected,expected_status,retryable",
        [
            (401, litellm.AuthenticationError, 401, False),
            (403, litellm.PermissionDeniedError, 403, False),
            (429, litellm.RateLimitError, 429, True),
            (408, BaseLLMException, 408, True),
            (422, litellm.BadRequestError, 400, False),
        ],
    )
    def test_get_error_class_preserves_status_categories(
        self, status_code, expected, expected_status, retryable
    ):
        with pytest.raises(expected) as exc_info:
            self.config.get_error_class(
                error_message="upstream said no",
                status_code=status_code,
                headers=httpx.Headers(),
            )

        assert exc_info.value.status_code == expected_status
        assert litellm._should_retry(exc_info.value.status_code) is retryable

    def test_unreadable_result_body_does_not_claim_generation_failed(self):
        request = httpx.Request(
            "GET", f"{FAL_API_BASE}/{KLING_MODEL_ID}/requests/abc-123"
        )
        result_client = _RecordingClient(
            httpx.Response(200, content=b"<html>gateway</html>", request=request)
        )
        config = FalAIVideoConfig(sync_client=result_client)

        with pytest.raises(BaseLLMException) as exc_info:
            config.transform_video_status_retrieve_response(
                raw_response=_fal_status_response(FAL_QUEUE_COMPLETED_STATUS),
                logging_obj=self.mock_logging_obj,
                custom_llm_provider="fal_ai",
            )

        assert exc_info.value.status_code == 502

    def test_transform_video_status_response_tolerates_non_json_body(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.side_effect = ValueError(
            "Expecting value: line 1 column 1 (char 0)"
        )

        status_obj = self.config.transform_video_status_retrieve_response(
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="fal_ai",
        )

        assert status_obj.status == "in_progress"

    def test_transform_video_content_request_builds_result_url(self):
        encoded_id = encode_video_id_with_provider("abc-123", "fal_ai", KLING_MODEL_ID)
        url, params = self.config.transform_video_content_request(
            video_id=encoded_id,
            api_base=FAL_API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert url == f"{FAL_API_BASE}/{KLING_QUEUE_NAMESPACE}/requests/abc-123"
        assert params == {}

    def test_transform_video_content_request_reconstructs_from_model_id(self):
        encoded_id = encode_video_id_with_provider("abc-123", "fal_ai", KLING_MODEL_ID)
        url, params = self.config.transform_video_content_request(
            video_id=encoded_id,
            api_base="https://attacker.example.com",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert (
            url
            == f"https://attacker.example.com/{KLING_QUEUE_NAMESPACE}/requests/abc-123"
        )
        assert params == {}

    def test_classify_result_handles_video_object(self):
        outcome = _classify_result_payload(
            {"video": {"url": "https://cdn.example.com/v.mp4"}}
        )
        assert outcome == _GeneratedVideo("https://cdn.example.com/v.mp4")

    def test_classify_result_handles_top_level_url(self):
        outcome = _classify_result_payload({"url": "https://cdn.example.com/v.mp4"})
        assert outcome == _GeneratedVideo("https://cdn.example.com/v.mp4")

    def test_classify_result_fails_when_url_missing(self):
        outcome = _classify_result_payload({"status": "IN_PROGRESS"})
        assert isinstance(outcome, _GenerationFailed)
        assert "Video URL not found" in outcome.message

    def test_classify_result_fails_on_fal_error_envelope(self):
        outcome = _classify_result_payload(FAL_FILE_DOWNLOAD_ERROR_RESULT)
        assert outcome == _GenerationFailed(
            "Failed to download the file. Please check if the URL is accessible and try again. (field: image_urls)"
        )

    def test_status_request_requires_model_id_in_video_id(self):
        plain_id = encode_video_id_with_provider("abc-123", "fal_ai", None)
        with pytest.raises(ValueError, match="model id encoded"):
            self.config.transform_video_status_retrieve_request(
                video_id=plain_id,
                api_base=FAL_API_BASE,
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

    def test_transform_video_delete_request_raises_not_implemented(self):
        encoded_id = encode_video_id_with_provider("abc-123", "fal_ai", KLING_MODEL_ID)
        with pytest.raises(NotImplementedError, match="delete/cancel is not supported"):
            self.config.transform_video_delete_request(
                video_id=encoded_id,
                api_base=FAL_API_BASE,
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

    def test_transform_video_delete_response_raises_not_implemented(self):
        mock_response = Mock(spec=httpx.Response)
        with pytest.raises(NotImplementedError, match="delete/cancel is not supported"):
            self.config.transform_video_delete_response(
                raw_response=mock_response,
                logging_obj=self.mock_logging_obj,
            )

    def test_remix_and_list_raise_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self.config.transform_video_remix_request(
                video_id="x",
                prompt="p",
                api_base=FAL_API_BASE,
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )
        with pytest.raises(NotImplementedError):
            self.config.transform_video_list_request(
                api_base=FAL_API_BASE,
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

    def test_full_video_workflow(self):
        result_client = _RecordingClient(
            _fal_result_response(
                {"video": {"url": "https://cdn.example.com/v.mp4"}},
                request_id="queued-id-1",
            )
        )
        config = FalAIVideoConfig(sync_client=result_client)
        mock_logging_obj = Mock()

        data, _, url = config.transform_video_create_request(
            model=KLING_MODEL,
            prompt="A high quality demo of LiteLLM video gateway",
            api_base=FAL_API_BASE,
            video_create_optional_request_params={
                "duration": "5",
                "aspect_ratio": "16:9",
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert url.endswith(KLING_MODEL_ID)

        create_response = Mock(spec=httpx.Response)
        create_response.json.return_value = {
            "request_id": "queued-id-1",
            "status": "IN_QUEUE",
        }
        video_obj = config.transform_video_create_response(
            model=KLING_MODEL,
            raw_response=create_response,
            logging_obj=mock_logging_obj,
            custom_llm_provider="fal_ai",
            request_data=data,
        )
        assert video_obj.status == "queued"
        assert video_obj.id.startswith("video_")

        status_url, _ = config.transform_video_status_retrieve_request(
            video_id=video_obj.id,
            api_base=FAL_API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert status_url.endswith("/requests/queued-id-1/status")

        completed_response = _fal_status_response(
            {
                "request_id": "queued-id-1",
                "status": "COMPLETED",
            },
            request_id="queued-id-1",
        )
        completed_obj = config.transform_video_status_retrieve_response(
            raw_response=completed_response,
            logging_obj=mock_logging_obj,
            custom_llm_provider="fal_ai",
        )
        assert completed_obj.status == "completed"


class _RecordingHTTPHandler(HTTPHandler):
    """A real HTTPHandler so the shared video handler accepts it as the caller's client."""

    def __init__(self, responses):
        super().__init__()
        self._responses = responses
        self.calls = []

    def get(self, url, params=None, headers=None, **kwargs):
        self.calls.append(url)
        return self._responses[url]


class _RecordingAsyncHTTPHandler(AsyncHTTPHandler):
    def __init__(self, responses):
        super().__init__()
        self._responses = responses
        self.calls = []

    async def get(self, url, params=None, headers=None, **kwargs):
        self.calls.append(url)
        return self._responses[url]


def _handler_poll_responses(result_payload, result_status_code=200):
    status_url = f"{FAL_API_BASE}/{KLING_QUEUE_NAMESPACE}/requests/abc-123/status"
    result_url = f"{FAL_API_BASE}/{KLING_QUEUE_NAMESPACE}/requests/abc-123"
    responses = {
        status_url: httpx.Response(
            200, json=FAL_QUEUE_COMPLETED_STATUS, request=httpx.Request("GET", status_url)
        ),
        result_url: httpx.Response(
            result_status_code, json=result_payload, request=httpx.Request("GET", result_url)
        ),
    }
    return status_url, result_url, responses


def test_video_status_handler_resolves_result_through_the_callers_client():
    # A caller-supplied client (or one built from ssl_verify) must carry the follow-up
    # result lookup too, or half the poll escapes its transport and CA settings.
    status_url, result_url, responses = _handler_poll_responses(
        FAL_FILE_DOWNLOAD_ERROR_RESULT, result_status_code=422
    )
    caller_client = _RecordingHTTPHandler(responses)

    video_obj = BaseLLMHTTPHandler().video_status_handler(
        video_id=encode_video_id_with_provider("abc-123", "fal_ai", KLING_MODEL_ID),
        video_status_provider_config=FalAIVideoConfig(),
        custom_llm_provider="fal_ai",
        litellm_params=GenericLiteLLMParams(api_base=FAL_API_BASE),
        logging_obj=Mock(),
        client=caller_client,
        api_key="test-key",
    )

    assert caller_client.calls == [status_url, result_url]
    assert video_obj.status == "failed"
    assert "Failed to download the file" in video_obj.error["message"]


@pytest.mark.asyncio
async def test_async_video_status_handler_resolves_result_through_the_callers_client():
    status_url, result_url, responses = _handler_poll_responses(
        {"video": {"url": "https://cdn.example.com/v.mp4"}}
    )
    caller_client = _RecordingAsyncHTTPHandler(responses)

    video_obj = await BaseLLMHTTPHandler().async_video_status_handler(
        video_id=encode_video_id_with_provider("abc-123", "fal_ai", KLING_MODEL_ID),
        video_status_provider_config=FalAIVideoConfig(),
        custom_llm_provider="fal_ai",
        litellm_params=GenericLiteLLMParams(api_base=FAL_API_BASE),
        logging_obj=Mock(),
        client=caller_client,
        api_key="test-key",
    )

    assert caller_client.calls == [status_url, result_url]
    assert video_obj.status == "completed"


def test_provider_config_manager_returns_fal_ai_video_config():
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_video_config(
        model=SORA_2_MODEL, provider=LlmProviders.FAL_AI
    )
    assert isinstance(config, FalAIVideoConfig)


@pytest.mark.parametrize(
    "model_id,expected_modalities",
    [
        ("fal_ai/fal-ai/kling-video/v3/standard/text-to-video", ("text",)),
        ("fal_ai/fal-ai/kling-video/v3/pro/text-to-video", ("text",)),
        ("fal_ai/bytedance/seedance-2.0/text-to-video", ("text",)),
        ("fal_ai/fal-ai/veo3.1/fast", ("text",)),
        ("fal_ai/fal-ai/kling-video/v3/standard/image-to-video", ("text", "image")),
        ("fal_ai/fal-ai/kling-video/v3/pro/image-to-video", ("text", "image")),
        ("fal_ai/bytedance/seedance-2.0/image-to-video", ("text", "image")),
    ],
)
def test_fal_ai_video_model_registered_with_video_endpoint(
    model_id: str, expected_modalities: tuple
):
    from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

    backup = GetModelCostMap.load_local_model_cost_map()
    entry = backup.get(model_id)
    assert entry is not None, f"{model_id} missing from local backup model cost map"
    assert entry["litellm_provider"] == "fal_ai"
    assert entry["mode"] == "video_generation"
    assert "/v1/videos" in entry["supported_endpoints"]
    assert tuple(entry["supported_modalities"]) == expected_modalities
    assert entry["supported_output_modalities"] == ["video"]
    assert isinstance(entry["output_cost_per_video_per_second"], (int, float))
