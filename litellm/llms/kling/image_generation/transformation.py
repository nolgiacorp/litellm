import time
from typing import TYPE_CHECKING, Any

import httpx

from litellm.constants import KLING_POLLING_TIMEOUT
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.llms.custom_httpx.http_handler import HTTPHandler, _get_httpx_client
from litellm.llms.kling.auth import kling_auth_headers
from litellm.llms.kling.common_utils import resolve_kling_api_base, strip_kling_prefix
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import ImageObject, ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

_KLING_IMAGE_POLL_INTERVAL_SECONDS = 2

_SIZE_TO_ASPECT_RATIO = {
    "1024x1024": "1:1",
    "1280x720": "16:9",
    "1920x1080": "16:9",
    "720x1280": "9:16",
    "1080x1920": "9:16",
    "1024x768": "4:3",
    "768x1024": "3:4",
}


class KlingImageGenerationConfig(BaseImageGenerationConfig):
    """
    Kling's /v1/images/generations is a task API: POST returns data.task_id, then
    poll GET /v1/images/generations/{task_id} until data.task_status is succeed.
    The finished result carries data.task_result.images[].url.

    resolution (1k|2k) is a native Kling image field and passes straight through.
    """

    DEFAULT_RESOLUTION = "1k"
    KLING_IMAGE_PASSTHROUGH_PARAMS = {"resolution", "aspect_ratio", "negative_prompt"}

    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
        return ["n", "size", "response_format"]

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        return f"{resolve_kling_api_base(api_base)}/images/generations"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        return {**headers, **kling_auth_headers(api_key)}

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported = self.get_supported_openai_params(model)
        result: dict[str, Any] = dict(optional_params)
        for key, value in non_default_params.items():
            if key == "size" and isinstance(value, str):
                result["aspect_ratio"] = _SIZE_TO_ASPECT_RATIO.get(value, value.replace("x", ":"))
            elif key in supported or key in self.KLING_IMAGE_PASSTHROUGH_PARAMS:
                result[key] = value
            elif drop_params:
                continue
            else:
                raise ValueError(
                    f"Parameter {key} is not supported for model {model}. Supported "
                    f"parameters are {supported}. Set drop_params=True to drop unsupported "
                    f"parameters."
                )
        return result

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        params: dict[str, Any] = dict(optional_params)
        extra_body = params.pop("extra_body", None)
        if isinstance(extra_body, dict):
            params = {**params, **extra_body}

        reserved = {
            "model_name",
            "prompt",
            "resolution",
            "size",
            "response_format",
            "model",
            "user",
        }
        forwarded = {key: value for key, value in params.items() if key not in reserved and value is not None}
        return {
            "model_name": strip_kling_prefix(model),
            "prompt": prompt,
            "resolution": str(params.get("resolution", self.DEFAULT_RESOLUTION)).lower(),
            **forwarded,
        }

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        submit_data = self._parse_json(raw_response)
        self._raise_for_kling_error(raw_response, submit_data)

        task_id = (submit_data.get("data") or {}).get("task_id")
        if not task_id:
            raise ValueError(f"Kling image submit response is missing data.task_id: {submit_data}")

        poll_url = f"{str(raw_response.request.url).rstrip('/')}/{task_id}"
        poll_headers = {"Authorization": raw_response.request.headers.get("Authorization", "")}
        final_data = self._poll_task_sync(
            poll_url=poll_url,
            headers=poll_headers,
            timeout_secs=KLING_POLLING_TIMEOUT,
        )
        return self._transform_images(final_data, model_response)

    def _poll_task_sync(
        self,
        poll_url: str,
        headers: dict[str, str],
        timeout_secs: float,
        client: HTTPHandler | None = None,
    ) -> dict[str, Any]:
        client = client or _get_httpx_client()
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout_secs:
                raise TimeoutError(f"Kling image task polling timed out after {timeout_secs} seconds")
            response = client.get(url=poll_url, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            data = response_data.get("data") or {}
            status = data.get("task_status")
            if status == "succeed":
                return response_data
            if status == "failed":
                message = data.get("task_status_msg") or response_data.get("message")
                raise ValueError(f"Kling image generation failed: {message}")
            time.sleep(_KLING_IMAGE_POLL_INTERVAL_SECONDS)

    @staticmethod
    def _transform_images(response_data: dict[str, Any], model_response: ImageResponse) -> ImageResponse:
        if not model_response.data:
            model_response.data = []
        task_result = (response_data.get("data") or {}).get("task_result") or {}
        images = task_result.get("images") or []
        if isinstance(images, list):
            for image in images:
                if isinstance(image, dict):
                    model_response.data.append(ImageObject(url=image.get("url"), b64_json=image.get("b64_json")))
                elif isinstance(image, str):
                    model_response.data.append(ImageObject(url=image, b64_json=None))
        return model_response

    @staticmethod
    def _parse_json(raw_response: httpx.Response) -> dict[str, Any]:
        try:
            return raw_response.json()
        except Exception as e:
            raise ValueError(f"Error parsing Kling image generation response: {e}") from e

    def _raise_for_kling_error(self, raw_response: httpx.Response, response_data: dict[str, Any]) -> None:
        code = response_data.get("code")
        if code is not None and code != 0:
            message = response_data.get("message") or "Kling API returned an error"
            raise self.get_error_class(
                error_message=str(message),
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )
