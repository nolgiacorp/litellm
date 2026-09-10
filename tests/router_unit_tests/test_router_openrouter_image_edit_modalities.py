"""
Router -> OpenRouter image edit: `modalities` is decided per deployment.

OpenRouter matches a chat body's `modalities` against the model's advertised
output_modalities and answers 404 "No endpoints found that support the
requested output modalities: image, text" for every image-only model (the
Microsoft MAI-Image family, Sourceful Riverflow, ByteDance Seedream, Qwen
Image 3 Pro all advertise ["image"] alone). The hardcoded ["image", "text"]
the edit config used to send failed every edit on them, and the failed edit
cooled the deployment down, taking text-to-image with it.

These tests drive the real Router through `aimage_edit` against a respx
transport, so they cover the whole path a proxy request takes: the
deployment's `model_info` is stamped into the call by the router, reaches
`GenericLiteLLMParams.model_info`, and the OpenRouter edit config reads
`supported_output_modalities` off it.
"""

import json
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm import Router

OPENROUTER_CHAT_URL: Final = "https://openrouter.ai/api/v1/chat/completions"
PNG_BYTES: Final = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32

OPENROUTER_EDIT_RESPONSE: Final = {
    "id": "gen-1",
    "model": "microsoft/mai-image-2.6",
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "images": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1n"}}],
            }
        }
    ],
    "usage": {
        "prompt_tokens": 1030,
        "completion_tokens": 1024,
        "total_tokens": 2054,
        "prompt_tokens_details": {"image_tokens": 1024},
        "completion_tokens_details": {"image_tokens": 1024},
        "cost": 0.05,
    },
}


def _router(model_info: dict | None) -> Router:
    deployment: Final = {
        "model_name": "mai-image-2.6",
        "litellm_params": {"model": "openrouter/microsoft/mai-image-2.6", "api_key": "sk-test"},
        **({"model_info": model_info} if model_info is not None else {}),
    }
    return Router(model_list=[deployment])


@pytest.fixture(autouse=True)
def _httpx_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)


async def _edit_body(router: Router) -> dict:
    with respx.mock(assert_all_called=True) as mock:
        route: Final = mock.post(OPENROUTER_CHAT_URL).mock(
            return_value=httpx.Response(200, json=OPENROUTER_EDIT_RESPONSE)
        )
        response: Final = await router.aimage_edit(model="mai-image-2.6", image=PNG_BYTES, prompt="make it night")
    assert response.data, "the mocked OpenRouter reply carries one image"
    request: Final = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer sk-test"
    return json.loads(request.content)


@pytest.mark.asyncio
async def test_router_image_edit_omits_modalities_for_a_plain_deployment():
    """Every image route in the proxy config declares nothing about output modalities and must not send the field."""
    body: Final = await _edit_body(_router(model_info={"mode": "image_generation"}))
    assert "modalities" not in body
    assert body["model"] == "microsoft/mai-image-2.6"
    content: Final = body["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[1] == {"type": "text", "text": "make it night"}


@pytest.mark.asyncio
async def test_router_image_edit_omits_modalities_for_an_image_only_declaration():
    body: Final = await _edit_body(
        _router(model_info={"mode": "image_generation", "supported_output_modalities": ["image"]})
    )
    assert "modalities" not in body


@pytest.mark.asyncio
async def test_router_image_edit_sends_modalities_for_a_text_capable_declaration():
    body: Final = await _edit_body(
        _router(model_info={"mode": "image_generation", "supported_output_modalities": ["image", "text"]})
    )
    assert body["modalities"] == ["image", "text"]


@pytest.mark.asyncio
async def test_router_image_edit_without_model_info_omits_modalities():
    body: Final = await _edit_body(_router(model_info=None))
    assert "modalities" not in body
