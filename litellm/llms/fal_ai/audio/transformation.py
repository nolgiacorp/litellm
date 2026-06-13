import asyncio
import time
from typing import TYPE_CHECKING, Any, Coroutine, Dict, Optional, Tuple, Union

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.constants import FAL_AI_DEFAULT_API_BASE, FAL_AI_POLLING_TIMEOUT
from litellm.llms.base_llm.text_to_speech.transformation import (
    BaseTextToSpeechConfig,
    TextToSpeechRequestData,
)
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.secret_managers.main import get_secret_str

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.types.llms.openai import (
        HttpxBinaryResponseContent as _HttpxBinaryResponseContent,
    )

    LiteLLMLoggingObj = _LiteLLMLoggingObj
    HttpxBinaryResponseContent = _HttpxBinaryResponseContent
else:
    LiteLLMLoggingObj = Any
    HttpxBinaryResponseContent = Any


_TERMINAL_OK = "COMPLETED"
_TERMINAL_FAIL = {"FAILED", "CANCELLED"}
_POLL_INTERVAL_SECS = 1.5


def _normalize_fal_model_id(model: str) -> str:
    stripped = model
    if stripped.startswith("fal_ai/"):
        stripped = stripped[len("fal_ai/") :]
    stripped = stripped.strip("/")
    if not stripped:
        raise ValueError("fal.ai model id is empty after stripping provider prefix")
    return stripped


class FalAIAudioConfig(BaseTextToSpeechConfig):
    """
    fal.ai audio (TTS / music / SFX) via its queue API.

    POST {api_base}/{model_id}     -> {request_id, status_url, response_url}
    GET  status_url                -> {status: IN_QUEUE | IN_PROGRESS | COMPLETED | FAILED}
    GET  response_url              -> {"audio": {"url": "..."}}
    GET  audio.url                 -> binary audio bytes

    Routed through main.py.aspeech / speech as a queue-style provider:
    ``dispatch_text_to_speech`` performs the whole submit->poll->download
    cycle and returns the binary content directly.
    """

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "input",
            "voice",
            "response_format",
            "speed",
            "extra_headers",
            "extra_body",
        ]

    def map_openai_params(
        self,
        model: str,
        optional_params: Dict,
        voice: Optional[Union[str, Dict]] = None,
        drop_params: bool = False,
        kwargs: Dict = {},
    ) -> Tuple[Optional[str], Dict]:
        return (voice if isinstance(voice, str) else None), dict(optional_params)

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        resolved_key = (
            api_key
            or litellm.api_key
            or get_secret_str("FAL_AI_API_KEY")
            or get_secret_str("FAL_KEY")
        )
        if not resolved_key:
            raise ValueError(
                "fal.ai API key is required. Set FAL_AI_API_KEY (or FAL_KEY) "
                "environment variable or pass api_key parameter."
            )
        headers.update(
            {
                "Authorization": f"Key {resolved_key}",
                "Content-Type": "application/json",
            }
        )
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        base = api_base or get_secret_str("FAL_AI_API_BASE") or FAL_AI_DEFAULT_API_BASE
        return base.rstrip("/")

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: Optional[str],
        optional_params: Dict,
        litellm_params: Dict,
        headers: dict,
    ) -> TextToSpeechRequestData:
        body: Dict[str, Any] = {"text": input}
        if voice is not None:
            body["voice"] = voice
        for key, value in optional_params.items():
            if key in ("response_format", "speed", "extra_headers", "extra_body"):
                continue
            body[key] = value
        extra_body = optional_params.get("extra_body")
        if isinstance(extra_body, dict):
            body.update(extra_body)
        return TextToSpeechRequestData(dict_body=body, headers={})

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> "HttpxBinaryResponseContent":
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        return HttpxBinaryResponseContent(response=raw_response)

    def dispatch_text_to_speech(
        self,
        model: str,
        input: str,
        voice: Optional[Union[str, Dict]],
        optional_params: Dict,
        litellm_params_dict: Dict,
        logging_obj: "LiteLLMLoggingObj",
        timeout: Union[float, httpx.Timeout],
        extra_headers: Optional[Dict[str, Any]],
        base_llm_http_handler: Any,
        aspeech: bool,
        api_base: Optional[str],
        api_key: Optional[str],
        **kwargs: Any,
    ) -> Union[
        "HttpxBinaryResponseContent",
        Coroutine[Any, Any, "HttpxBinaryResponseContent"],
    ]:
        if aspeech:
            return self._async_dispatch(
                model=model,
                input=input,
                voice=voice,
                optional_params=optional_params,
                litellm_params_dict=litellm_params_dict,
                extra_headers=extra_headers,
                api_base=api_base,
                api_key=api_key,
            )
        return self._sync_dispatch(
            model=model,
            input=input,
            voice=voice,
            optional_params=optional_params,
            litellm_params_dict=litellm_params_dict,
            extra_headers=extra_headers,
            api_base=api_base,
            api_key=api_key,
        )

    def _build_request(
        self,
        model: str,
        input: str,
        voice: Optional[Union[str, Dict]],
        optional_params: Dict,
        litellm_params_dict: Dict,
        extra_headers: Optional[Dict[str, Any]],
        api_base: Optional[str],
        api_key: Optional[str],
    ) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
        base = self.get_complete_url(
            model=model, api_base=api_base, litellm_params=litellm_params_dict
        )
        model_id = _normalize_fal_model_id(model)
        submit_url = f"{base}/{model_id}"

        headers: Dict[str, str] = {}
        if extra_headers:
            headers.update(extra_headers)
        headers = self.validate_environment(
            headers=headers, model=model, api_key=api_key, api_base=api_base
        )

        voice_str = voice if isinstance(voice, str) else None
        request = self.transform_text_to_speech_request(
            model=model,
            input=input,
            voice=voice_str,
            optional_params=optional_params,
            litellm_params=litellm_params_dict,
            headers=headers,
        )
        body = request.get("dict_body") or {}
        return submit_url, headers, body

    def _sync_dispatch(
        self,
        model: str,
        input: str,
        voice: Optional[Union[str, Dict]],
        optional_params: Dict,
        litellm_params_dict: Dict,
        extra_headers: Optional[Dict[str, Any]],
        api_base: Optional[str],
        api_key: Optional[str],
    ) -> "HttpxBinaryResponseContent":
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        submit_url, headers, body = self._build_request(
            model=model,
            input=input,
            voice=voice,
            optional_params=optional_params,
            litellm_params_dict=litellm_params_dict,
            extra_headers=extra_headers,
            api_base=api_base,
            api_key=api_key,
        )
        client = _get_httpx_client()
        submit_resp = client.post(url=submit_url, headers=headers, json=body)
        submit_resp.raise_for_status()
        submit_payload = submit_resp.json()

        status_url = submit_payload.get("status_url")
        response_url = submit_payload.get("response_url")
        if not status_url or not response_url:
            raise ValueError(
                "fal.ai queue submit response missing status_url/response_url"
            )

        verbose_logger.debug(
            "fal.ai audio polling: rid=%s",
            submit_payload.get("request_id"),
        )
        self._poll_until_complete_sync(
            status_url=status_url,
            headers=headers,
            client=client,
        )

        result_resp = client.get(url=response_url, headers=headers)
        result_resp.raise_for_status()
        audio_url = self._extract_audio_url(result_resp.json())

        binary_resp = client.get(url=audio_url, headers={})
        binary_resp.raise_for_status()
        return HttpxBinaryResponseContent(response=binary_resp)

    async def _async_dispatch(
        self,
        model: str,
        input: str,
        voice: Optional[Union[str, Dict]],
        optional_params: Dict,
        litellm_params_dict: Dict,
        extra_headers: Optional[Dict[str, Any]],
        api_base: Optional[str],
        api_key: Optional[str],
    ) -> "HttpxBinaryResponseContent":
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        submit_url, headers, body = self._build_request(
            model=model,
            input=input,
            voice=voice,
            optional_params=optional_params,
            litellm_params_dict=litellm_params_dict,
            extra_headers=extra_headers,
            api_base=api_base,
            api_key=api_key,
        )
        client = get_async_httpx_client(llm_provider=litellm.LlmProviders.FAL_AI)
        submit_resp = await client.post(url=submit_url, headers=headers, json=body)
        submit_resp.raise_for_status()
        submit_payload = submit_resp.json()

        status_url = submit_payload.get("status_url")
        response_url = submit_payload.get("response_url")
        if not status_url or not response_url:
            raise ValueError(
                "fal.ai queue submit response missing status_url/response_url"
            )

        verbose_logger.debug(
            "fal.ai audio polling (async): rid=%s",
            submit_payload.get("request_id"),
        )
        await self._poll_until_complete_async(
            status_url=status_url,
            headers=headers,
            client=client,
        )

        result_resp = await client.get(url=response_url, headers=headers)
        result_resp.raise_for_status()
        audio_url = self._extract_audio_url(result_resp.json())

        binary_resp = await client.get(url=audio_url, headers={})
        binary_resp.raise_for_status()
        return HttpxBinaryResponseContent(response=binary_resp)

    def _poll_until_complete_sync(
        self,
        status_url: str,
        headers: Dict[str, str],
        client: Any,
        timeout_secs: int = FAL_AI_POLLING_TIMEOUT,
    ) -> None:
        deadline = time.monotonic() + timeout_secs
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"fal.ai audio job did not complete within {timeout_secs}s"
                )
            resp = client.get(url=status_url, headers=headers)
            resp.raise_for_status()
            status = (resp.json().get("status") or "").upper()
            if status == _TERMINAL_OK:
                return
            if status in _TERMINAL_FAIL:
                raise RuntimeError(f"fal.ai audio job ended with status={status}")
            time.sleep(_POLL_INTERVAL_SECS)

    async def _poll_until_complete_async(
        self,
        status_url: str,
        headers: Dict[str, str],
        client: Any,
        timeout_secs: int = FAL_AI_POLLING_TIMEOUT,
    ) -> None:
        deadline = time.monotonic() + timeout_secs
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"fal.ai audio job did not complete within {timeout_secs}s"
                )
            resp = await client.get(url=status_url, headers=headers)
            resp.raise_for_status()
            status = (resp.json().get("status") or "").upper()
            if status == _TERMINAL_OK:
                return
            if status in _TERMINAL_FAIL:
                raise RuntimeError(f"fal.ai audio job ended with status={status}")
            await asyncio.sleep(_POLL_INTERVAL_SECS)

    @staticmethod
    def _extract_audio_url(result_payload: Dict[str, Any]) -> str:
        audio = result_payload.get("audio")
        if isinstance(audio, dict) and isinstance(audio.get("url"), str):
            return audio["url"]
        audio_url = result_payload.get("audio_url")
        if isinstance(audio_url, str):
            return audio_url
        audio_file = result_payload.get("audio_file")
        if isinstance(audio_file, dict) and isinstance(audio_file.get("url"), str):
            return audio_file["url"]
        raise ValueError(
            "fal.ai audio result missing audio url; got keys: "
            f"{list(result_payload.keys())}"
        )
