"""
Tests for Black Forest Labs common_utils — specifically assert_bfl_polling_url.

BFL uses regional subdomains (e.g. gateway.bfl.ai) for polling URLs that
differ from the submission host (api.bfl.ai). These tests verify that the
domain-aware check accepts legitimate BFL subdomains while still rejecting
off-domain and non-HTTPS URLs.
"""

import pytest

from litellm.llms.black_forest_labs.common_utils import (
    IMAGE_EDIT_MODELS,
    IMAGE_GENERATION_MODELS,
    BlackForestLabsError,
    assert_bfl_polling_url,
)


class TestAssertBflPollingUrl:
    # --- should pass ---

    def test_exact_registered_domain(self):
        assert_bfl_polling_url("https://bfl.ai/v1/get_result?id=abc")

    def test_api_subdomain(self):
        assert_bfl_polling_url("https://api.bfl.ai/v1/get_result?id=abc")

    def test_gateway_subdomain(self):
        # BFL uses gateway.bfl.ai for polling — this was the original bug trigger
        assert_bfl_polling_url("https://gateway.bfl.ai/v1/get_result?id=abc")

    def test_regional_subdomain(self):
        assert_bfl_polling_url("https://eu.api.bfl.ai/v1/get_result?id=abc")

    def test_deep_subdomain(self):
        assert_bfl_polling_url("https://region.gateway.bfl.ai/poll?id=xyz")

    # --- should raise BlackForestLabsError ---

    def test_rejects_http_scheme(self):
        # HTTP must be rejected — x-key would be forwarded in plaintext
        with pytest.raises(BlackForestLabsError, match="scheme must be https"):
            assert_bfl_polling_url("http://api.bfl.ai/v1/get_result?id=abc")

    def test_rejects_off_domain(self):
        with pytest.raises(BlackForestLabsError, match="host is not within"):
            assert_bfl_polling_url("https://evil.com/steal-key")

    def test_rejects_lookalike_domain(self):
        with pytest.raises(BlackForestLabsError, match="host is not within"):
            assert_bfl_polling_url("https://notbfl.ai/v1/get_result?id=abc")

    def test_rejects_bfl_ai_as_suffix_only(self):
        # "fakebfl.ai" must not match — the check is on registered domain boundary
        with pytest.raises(BlackForestLabsError, match="host is not within"):
            assert_bfl_polling_url("https://fakebfl.ai/v1/get_result?id=abc")

    def test_rejects_bfl_in_path(self):
        with pytest.raises(BlackForestLabsError, match="host is not within"):
            assert_bfl_polling_url("https://evil.com/bfl.ai/steal")

    def test_rejects_ftp_scheme(self):
        with pytest.raises(BlackForestLabsError, match="scheme must be https"):
            assert_bfl_polling_url("ftp://api.bfl.ai/v1/get_result?id=abc")

    def test_rejects_javascript_scheme(self):
        with pytest.raises(BlackForestLabsError, match="scheme must be https"):
            assert_bfl_polling_url("javascript://api.bfl.ai/alert(1)")


class TestFlux2ModelMaps:
    @pytest.mark.parametrize(
        "model",
        ["flux-2-pro", "flux-2-max", "flux-2-flex", "flux-2-klein-9b", "flux-2-klein-4b"],
    )
    def test_flux2_models_registered_for_generation_and_edit(self, model):
        assert IMAGE_GENERATION_MODELS[model] == f"/v1/{model}"
        assert IMAGE_EDIT_MODELS[model] == f"/v1/{model}"

    @pytest.mark.parametrize(
        "model",
        ["flux-2-pro", "flux-2-max", "flux-2-flex", "flux-2-klein-9b", "flux-2-klein-4b"],
    )
    def test_flux2_models_have_published_pricing(self, model, monkeypatch):
        import litellm

        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        litellm.model_cost = litellm.get_model_cost_map(url="")
        info = litellm.get_model_info(model=model, custom_llm_provider="black_forest_labs")
        assert info.get("output_cost_per_image", 0) > 0
