"""Regression + self-test for the duplicate-key guard on the model-price maps.

Upstream has twice handed us a price map with the same model key written twice
(`gemini-omni-flash-preview` in NOL-79, `jp.anthropic.claude-sonnet-4-6` in
NOL-90). `json.load` keeps the LAST occurrence, so both parsed fine and both
shipped an entry that did not match the one a reader would find first in the
file. `jq empty` -- the only validation the price maps had -- accepts duplicates,
so nothing caught either case.

Two things are pinned here:

1. the tracked price maps carry no duplicate keys, and
2. the guard actually FAILS on a duplicate. A guard that has only ever been
   observed passing is not a guard, so the negative cases are asserted too.
"""

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = REPO_ROOT / "scripts" / "check_model_prices_duplicate_keys.py"
_spec = importlib.util.spec_from_file_location(
    "check_model_prices_duplicate_keys", _MODULE_PATH
)
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)


PRICE_MAPS = [
    "model_prices_and_context_window.json",
    "litellm/model_prices_and_context_window_backup.json",
]


@pytest.mark.parametrize("relative_path", PRICE_MAPS)
def test_price_map_has_no_duplicate_keys(relative_path):
    """The shipped price maps must not repeat a key inside any object."""
    path = REPO_ROOT / relative_path
    duplicates = guard.find_duplicate_keys(path)
    assert duplicates == [], (
        f"{relative_path} repeats {[d.key for d in duplicates]}. "
        "json.load keeps the last occurrence, so the earlier entry is dead text "
        "that still reads as authoritative - delete the stale one."
    )


def test_jp_anthropic_claude_sonnet_4_6_matches_across_price_maps():
    """NOL-90: the backup's winning entry had drifted from the canonical map.

    The duplicate's second (winning) copy was an older shape missing the 1-hour
    cache-write tier, so the backup map silently priced jp. 1hr cache writes at
    the 5-minute rate while the root map had it right.
    """
    model = "jp.anthropic.claude-sonnet-4-6"
    with open(REPO_ROOT / "model_prices_and_context_window.json") as f:
        root = json.load(f)
    with open(REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json") as f:
        backup = json.load(f)

    assert backup[model] == root[model], (
        f"{model} differs between the price map and its backup copy"
    )
    assert backup[model]["cache_creation_input_token_cost_above_1hr"] == 6.6e-06


def test_grok_imagine_entries_match_across_price_maps():
    """NOL-107: the canonical map shipped none of the four grok-imagine entries
    while the backup carried all of them, the same canonical/backup drift class
    NOL-90 pinned. The runtime reads the backup in our deployments, so the drift
    left live COGS resting on entries the canonical map disowned. Pin that the
    per-second video rates and per-image rates exist and are identical in both
    maps so the two cannot drift back apart."""
    with open(REPO_ROOT / "model_prices_and_context_window.json") as f:
        root = json.load(f)
    with open(REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json") as f:
        backup = json.load(f)

    expected_rates = {
        "xai/grok-imagine-video": ("output_cost_per_video_per_second", 0.05),
        "xai/grok-imagine-video-1.5": ("output_cost_per_video_per_second", 0.08),
        "xai/grok-imagine-image": ("input_cost_per_image", 0.02),
        "xai/grok-imagine-image-quality": ("input_cost_per_image", 0.05),
    }
    for model, (cost_key, rate) in expected_rates.items():
        assert model in root, f"{model} missing from canonical price map"
        assert model in backup, f"{model} missing from backup price map"
        assert root[model] == backup[model], f"{model} differs between the price map and its backup copy"
        assert root[model][cost_key] == rate, f"{model} {cost_key} is not {rate}"


def test_nol535_zero_cogs_entries_match_across_price_maps():
    """NOL-535: three live routes logged real generations at $0 COGS because
    their price-map keys did not exist - black_forest_labs/flux-3-video, the
    fal seedance reference-to-video variant (its t2v/i2v siblings were priced,
    r2v was not), and fal-ai/clarity-upscaler (the upscale pass behind every
    2k/4k image tier). Pin that all three entries exist and are byte-identical
    in both maps (the NOL-90 invariant; our deployments read the backup), that
    the flux-3-video tiers carry BFL's published per-second rates, that the r2v
    rate matches its siblings' basis, and that clarity carries fal's published
    $0.03/megapixel as a per-pixel rate."""
    with open(REPO_ROOT / "model_prices_and_context_window.json") as f:
        root = json.load(f)
    with open(REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json") as f:
        backup = json.load(f)

    models = (
        "black_forest_labs/flux-3-video",
        "fal_ai/bytedance/seedance-2.0/reference-to-video",
        "fal_ai/fal-ai/clarity-upscaler",
    )
    for model in models:
        assert model in root, f"{model} missing from canonical price map"
        assert model in backup, f"{model} missing from backup price map"
        assert root[model] == backup[model], f"{model} differs between the price map and its backup copy"

    flux3 = root["black_forest_labs/flux-3-video"]
    assert flux3["output_cost_per_second_hd"] == 0.17
    assert flux3["output_cost_per_second_fhd"] == 0.29
    assert flux3["output_cost_per_second_v2v_hd"] == 0.43
    assert flux3["output_cost_per_second_v2v_fhd"] == 0.54
    assert flux3["output_cost_per_second_v2v"] == flux3["output_cost_per_second_v2v_hd"]
    assert flux3["output_cost_per_second"] == flux3["output_cost_per_second_hd"], (
        "the base rate must be the hd tier - hd is the deployment default, so an "
        "untiered request must price as hd rather than $0"
    )
    assert "output_cost_per_video_per_second" not in flux3, (
        "output_cost_per_video_per_second is checked before the tiered keys and "
        "would flatten every tier to one rate"
    )

    r2v = root["fal_ai/bytedance/seedance-2.0/reference-to-video"]
    for sibling in (
        "fal_ai/bytedance/seedance-2.0/text-to-video",
        "fal_ai/bytedance/seedance-2.0/image-to-video",
    ):
        assert r2v["output_cost_per_video_per_second"] == root[sibling]["output_cost_per_video_per_second"], (
            f"r2v must share its siblings' per-second basis ({sibling})"
        )

    clarity = root["fal_ai/fal-ai/clarity-upscaler"]
    assert clarity["output_cost_per_pixel"] == 3e-08, "fal bills clarity at $0.03 per megapixel"
    assert "output_cost_per_image" not in clarity, (
        "a flat per-image rate would shadow nothing but would misprice any "
        "dimension-less fallback as nonzero guesswork"
    )


def test_fal_vendor_app_image_entries_match_across_price_maps():
    """Reve 2.1, Seedream 5 Pro, Ideogram v4 and Qwen Image 3 are dispatched by the
    fal image provider off their vendor-namespaced ids, so their price-map keys are
    fal_ai/<vendor>/... rather than fal_ai/fal-ai/.... Pin that every route (text-
    to-image and edit) exists in both maps, is byte-identical between them (the
    NOL-90 invariant; our deployments read the backup), and carries fal's published
    default-tier per-image rate, so a request cannot generate at $0 COGS."""
    with open(REPO_ROOT / "model_prices_and_context_window.json") as f:
        root = json.load(f)
    with open(REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json") as f:
        backup = json.load(f)

    expected_rates = {
        "fal_ai/reve/2.1/text-to-image": 0.25,
        "fal_ai/reve/2.1/edit": 0.25,
        "fal_ai/bytedance/seedream/v5/pro/text-to-image": 0.0675,
        "fal_ai/bytedance/seedream/v5/pro/edit": 0.0675,
        "fal_ai/ideogram/v4": 0.015,
        "fal_ai/ideogram/v4/image-to-image": 0.015,
        "fal_ai/alibaba/qwen-image-3/text-to-image": 0.04,
        "fal_ai/alibaba/qwen-image-3/edit": 0.04,
    }
    for model, rate in expected_rates.items():
        assert model in root, f"{model} missing from canonical price map"
        assert model in backup, f"{model} missing from backup price map"
        assert root[model] == backup[model], f"{model} differs between the price map and its backup copy"
        assert root[model]["output_cost_per_image"] == rate, f"{model} output_cost_per_image is not {rate}"
        assert root[model]["mode"] == "image_generation"
        assert root[model]["litellm_provider"] == "fal_ai"


def test_fleet_brain_cache_read_uses_the_key_the_cost_calculator_reads():
    """NOL-376: the fleet brain's cache-read rate has to sit under a consumed key.

    `input_cost_per_token_cache_hit` is declared on ModelInfoBase but no cost
    calculator ever reads it - `_calculate_input_cost` bills cached prompt tokens
    at `cache_read_input_token_cost`, which defaults to 0.0 when absent. So an
    entry carrying only the cache_hit spelling prices cache hits at $0 instead of
    $0.018/M and undercounts spend, which matters here because prod runs
    LITELLM_LOCAL_MODEL_COST_MAP=True and bills off the packaged backup map.
    Both spellings are kept at the same rate, as the 23 other dual-key entries do.
    """
    model = "openrouter/deepseek/deepseek-v4-flash-0731"
    with open(REPO_ROOT / "model_prices_and_context_window.json") as f:
        root = json.load(f)
    with open(REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json") as f:
        backup = json.load(f)

    for name, price_map in (("canonical", root), ("backup", backup)):
        assert model in price_map, f"{model} missing from the {name} price map"
    assert root[model] == backup[model], f"{model} differs between the price map and its backup copy"

    entry = root[model]
    assert entry["cache_read_input_token_cost"] == 1.8e-08, (
        "cache reads must be priced under cache_read_input_token_cost - it is the "
        "only cache-read key the cost calculator consumes"
    )
    assert entry["input_cost_per_token_cache_hit"] == entry["cache_read_input_token_cost"]
    assert entry["input_cost_per_token"] == 9e-08
    assert entry["output_cost_per_token"] == 1.8e-07


def test_guard_detects_duplicate_top_level_key(tmp_path):
    """The exact shape NOL-90 fixed: one model key written twice."""
    path = tmp_path / "dup.json"
    path.write_text(
        '{\n'
        '    "a-model": {"input_cost_per_token": 1e-06},\n'
        '    "jp.anthropic.claude-sonnet-4-6": {"input_cost_per_token": 3.3e-06},\n'
        '    "jp.anthropic.claude-sonnet-4-6": {"input_cost_per_token": 9.9e-06}\n'
        '}\n'
    )

    # json.load is blind to this - that is the whole problem.
    assert json.loads(path.read_text())["jp.anthropic.claude-sonnet-4-6"] == {
        "input_cost_per_token": 9.9e-06
    }

    duplicates = guard.find_duplicate_keys(path)
    assert [d.key for d in duplicates] == ["jp.anthropic.claude-sonnet-4-6"]
    assert duplicates[0].count == 2
    assert duplicates[0].lines == (3, 4)
    assert guard.check(path) is False
    assert guard.main([str(path)]) == 1


def test_guard_detects_duplicate_nested_key(tmp_path):
    """Duplicates below the top level count too (e.g. a repeated pricing field)."""
    path = tmp_path / "nested.json"
    path.write_text(
        '{\n'
        '    "a-model": {\n'
        '        "mode": "chat",\n'
        '        "mode": "video_generation"\n'
        '    }\n'
        '}\n'
    )

    duplicates = guard.find_duplicate_keys(path)
    assert [d.key for d in duplicates] == ["mode"]
    assert guard.main([str(path)]) == 1


def test_guard_passes_on_clean_file(tmp_path):
    path = tmp_path / "clean.json"
    path.write_text(
        '{\n'
        '    "a-model": {"mode": "chat"},\n'
        '    "b-model": {"mode": "video_generation"}\n'
        '}\n'
    )

    assert guard.find_duplicate_keys(path) == []
    assert guard.check(path) is True
    assert guard.main([str(path)]) == 0


def test_guard_reports_every_offending_file(tmp_path):
    """A clean file must not mask a dirty one when several are passed."""
    clean = tmp_path / "clean.json"
    clean.write_text('{"a": 1}\n')
    dirty = tmp_path / "dirty.json"
    dirty.write_text('{"a": 1, "a": 2}\n')

    assert guard.main([str(clean), str(dirty)]) == 1
    assert guard.main([str(dirty), str(clean)]) == 1


def test_guard_fails_on_invalid_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json")
    assert guard.main([str(path)]) == 1


def test_guard_fails_on_missing_file(tmp_path):
    assert guard.main([str(tmp_path / "nope.json")]) == 1


def test_guard_default_targets_are_the_tracked_price_maps():
    """Bare `python scripts/check_model_prices_duplicate_keys.py` must cover both."""
    assert set(guard.DEFAULT_TARGETS) == set(PRICE_MAPS)
    assert guard.main([]) == 0


def test_gemini_3_8_flash_entries_match_across_price_maps():
    """gemini-3.8-flash (stable 2026-09-02) is priced in all three forms the 3.6 sibling uses (gemini/,
    vertex_ai/ and the bare vertex_ai-language-models key). Pin that each exists in both maps, is byte-identical
    between them (the NOL-90 invariant; our deployments read the backup), and carries Google's paid-tier rates
    through 2026-12-31 with batch at half price and thinking billed at the output rate."""
    with open(REPO_ROOT / "model_prices_and_context_window.json") as f:
        root = json.load(f)
    with open(REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json") as f:
        backup = json.load(f)

    for model in (
        "gemini/gemini-3.8-flash",
        "vertex_ai/gemini-3.8-flash",
        "gemini-3.8-flash",
    ):
        assert model in root, f"{model} missing from canonical price map"
        assert model in backup, f"{model} missing from backup price map"
        assert root[model] == backup[model], f"{model} differs between the price map and its backup copy"
        entry = root[model]
        assert entry["input_cost_per_token"] == 7.5e-07
        assert entry["output_cost_per_token"] == 3.75e-06
        assert entry["output_cost_per_reasoning_token"] == entry["output_cost_per_token"]
        assert entry["cache_read_input_token_cost"] == 7.5e-08
        assert entry["input_cost_per_token_batches"] == entry["input_cost_per_token"] / 2
        assert entry["output_cost_per_token_batches"] == entry["output_cost_per_token"] / 2
        assert entry["max_input_tokens"] == 1048576
        assert entry["max_output_tokens"] == 65536
        assert entry["mode"] == "chat"
        assert entry["supports_reasoning"] is True
        assert entry["supports_minimal_reasoning_effort"] is False
        assert entry["supported_output_modalities"] == ["text"]
        assert "2027-01-01" in entry["metadata"]["notes"], "the 2027 rate step-up must stay documented on the entry"
