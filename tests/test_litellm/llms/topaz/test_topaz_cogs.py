"""
NOL-519: Topaz logged $0 COGS on every restore.

Unlike every other provider fixed under this ticket, Topaz cannot be priced by
a per-second rate. It bills CREDITS for the frames its engine processes, so cost
tracks source geometry, output geometry and the scale factor between them,
non-monotonically - measured on Topaz's own estimate endpoint, one classic
restore costs 0.25 credits per 5s from a 960x720 source and 9.67 from a
7680x4320 source AT THE SAME 720p OUTPUT TIER.

So the fix records the number Topaz reports rather than modelling the cost:
`estimates.cost` is a [lower, upper] credit range and Topaz bills the LOWER
bound. The create leg - the only leg that writes a spend row - polls for it
once the source upload lands, and hands the resulting USD to the logging path
as an explicit response_cost.

Verified live against api.topazlabs.com while writing this: POST /video/
returned `{"estimates": {"cost": [2, 3], "time": [299, 316]}}` for a 5s
1280x720 -> 1920x1080 prob-4 restore, and POST /video/express returned only
`{requestId, uploadId, uploadUrls}` with NO estimates - which is why the quote
is polled after upload rather than read off the create response.
"""

import httpx
import pytest

import litellm
from litellm.llms.topaz.cost_calculator import (
    DEFAULT_USD_PER_CREDIT,
    cost_calculator,
    cost_per_credit,
)
from litellm.llms.topaz.videos import transformation as topaz_transformation
from litellm.llms.topaz.videos.transformation import TopazVideoConfig

MODEL = "topaz/prob-4"


class TestTopazCreditRate:
    def test_rate_comes_from_the_cost_map(self):
        assert cost_per_credit(MODEL) == pytest.approx(0.12)
        assert cost_per_credit("prob-4") == pytest.approx(0.12)

    def test_unmapped_engine_falls_back_to_the_published_starter_rate(self):
        """An unmapped engine must still price, not silently record $0."""
        assert cost_per_credit("topaz/not-a-real-engine") == pytest.approx(DEFAULT_USD_PER_CREDIT)

    def test_starter_rate_is_the_conservative_plan(self):
        """
        Starter ($0.12) is deliberately the worst of Developer ($0.10) and Scale
        ($0.08), so a recorded cost is an upper bound on whichever plan we are
        on. A ledger that errs low is the failure this ticket exists to fix.
        """
        assert DEFAULT_USD_PER_CREDIT == 0.12

    def test_rate_is_visible_through_get_model_info(self):
        """
        Billing and /v1/model/info consumers read the standard model metadata,
        so the credit rate must survive the ModelInfo copy path and not stay
        private to this calculator's direct read of litellm.model_cost.
        """
        assert litellm.get_model_info(model=MODEL).get("output_cost_per_credit") == pytest.approx(0.12)

    def test_every_configured_topaz_engine_is_priced(self):
        """The 12 engines routed in litellm-config.yaml must all resolve a rate."""
        configured = [
            "prob-4", "rhea-1", "iris-3", "nyx-3", "thd-3", "ahq-12",
            "ghq-5", "dtd-4", "slf-2", "slp-2.5", "wonder-1", "hyp-2",
        ]
        for code in configured:
            entry = litellm.model_cost.get(f"topaz/{code}")
            assert entry is not None, f"topaz/{code} missing from the cost map"
            assert entry.get("output_cost_per_credit") == pytest.approx(0.12), code


class TestTopazCostCalculator:
    @pytest.mark.parametrize(
        "credits,expected",
        [
            (1, 0.12),
            (2, 0.24),   # the live quote for a 5s 720p -> 1080p prob-4 restore
            (6, 0.72),
            (10, 1.20),
            (0.25, 0.03),   # measured: 5s from a 960x720 source, 720p out
            (9.67, 1.1604),  # measured: the same 720p tier from 7680x4320
        ],
    )
    def test_credits_price_at_the_starter_rate(self, credits, expected):
        assert cost_calculator(model=MODEL, topaz_credits=credits) == pytest.approx(expected)

    @pytest.mark.parametrize("bad", [None, "2", True, -1, 0])
    def test_unusable_credit_counts_record_nothing_rather_than_guessing(self, bad):
        assert cost_calculator(model=MODEL, topaz_credits=bad) == 0.0


def _create_response(url: str = "https://api.topazlabs.com/video/express") -> httpx.Response:
    return httpx.Response(
        200,
        json={"requestId": "req-1", "uploadId": "u1", "uploadUrls": ["https://upload.test/put"]},
        request=httpx.Request("POST", url, headers={"X-API-Key": "test-key"}),
    )


class TestTopazCreditProbe:
    def test_status_url_is_derived_from_the_create_url(self):
        """Preserves a TOPAZ_API_BASE override instead of assuming the default host."""
        url = TopazVideoConfig._create_status_url(
            _create_response("https://topaz.internal.test/video/express"), "req-1"
        )
        assert url == "https://topaz.internal.test/video/req-1/status"

    def test_probe_reuses_the_create_request_api_key(self):
        assert TopazVideoConfig._probe_headers(_create_response()) == {"X-API-Key": "test-key"}

    def test_probe_reads_the_lower_bound_of_the_cost_range(self):
        """Topaz bills the LOWER bound; the upper is vendor uncertainty."""
        config = TopazVideoConfig()
        resp = httpx.Response(200, json={"status": "processing", "estimates": {"cost": [6, 9]}})
        assert config._credits_from_status(resp) == 6

    @pytest.mark.parametrize(
        "payload",
        [
            {"status": "processing"},                      # estimates not ready yet
            {"status": "processing", "estimates": {}},     # no cost key
            {"status": "processing", "estimates": {"cost": []}},
        ],
    )
    def test_missing_estimate_yields_no_credits(self, payload):
        config = TopazVideoConfig()
        assert config._credits_from_status(httpx.Response(200, json=payload)) is None

    @pytest.mark.parametrize("credits,expected", [(0.25, 0.25), (9.67, 9.67), (2, 2.0)])
    def test_fractional_quotes_are_read_at_full_precision(self, credits, expected):
        """
        Topaz quotes fractions for small jobs: 0.25 truncated to int would record
        $0 for exactly the cheap end of the range, and 9.67 would bill as 9.
        """
        config = TopazVideoConfig()
        resp = httpx.Response(200, json={"status": "processing", "estimates": {"cost": [credits, 12]}})
        assert config._credits_from_status(resp) == pytest.approx(expected)

    def test_non_200_and_unparseable_status_yield_no_credits(self):
        config = TopazVideoConfig()
        assert config._credits_from_status(httpx.Response(503, text="upstream down")) is None
        assert config._credits_from_status(httpx.Response(200, text="not json")) is None

    @pytest.mark.parametrize("body", ["null", "[]", '"queued"', "3"])
    def test_non_object_status_json_reads_as_a_failed_probe(self, body):
        """
        A 200 carrying valid non-object JSON must not raise: this runs after the
        upload succeeded, so it would turn accepted footage into a create error.
        """
        config = TopazVideoConfig()
        assert config._credits_from_status(httpx.Response(200, text=body)) is None

    def test_probe_gives_up_quietly_when_the_status_endpoint_is_unreachable(self, monkeypatch):
        """A bookkeeping probe must never fail a job the customer already paid for."""
        config = TopazVideoConfig()
        attempts = []

        class _Boom:
            def get(self, *a, **k):
                attempts.append(k.get("timeout"))
                raise httpx.ConnectError("topaz unreachable")

        monkeypatch.setattr(topaz_transformation.time, "sleep", lambda _: None)
        monkeypatch.setattr(config, "_http_client", lambda: _Boom())
        assert config._probe_billed_credits(_create_response(), "req-1") is None
        assert len(attempts) == topaz_transformation._CREDIT_PROBE_ATTEMPTS

    def test_transport_blip_is_retried_rather_than_ending_the_probe(self, monkeypatch):
        """
        The create leg is the only chance to record spend, so a transient
        ConnectError must spend the remaining attempts instead of returning None.
        """
        config = TopazVideoConfig()
        timeouts = []

        class _Flaky:
            def get(self, *a, **k):
                timeouts.append(k.get("timeout"))
                if len(timeouts) == 1:
                    raise httpx.ConnectError("transient blip")
                return httpx.Response(200, json={"estimates": {"cost": [2, 3]}})

        monkeypatch.setattr(topaz_transformation.time, "sleep", lambda _: None)
        monkeypatch.setattr(config, "_http_client", lambda: _Flaky())
        assert config._probe_billed_credits(_create_response(), "req-1") == pytest.approx(2)
        assert len(timeouts) == 2

    def test_every_probe_request_carries_its_own_timeout(self, monkeypatch):
        """
        The probe rides the caller's client, whose default timeout can be 60s.
        Without an explicit per-request timeout the attempt count would bound
        only the sleeps, not the network wait.
        """
        config = TopazVideoConfig()
        timeouts = []

        class _Silent:
            def get(self, *a, **k):
                timeouts.append(k.get("timeout"))
                return httpx.Response(200, json={"status": "processing"})

        monkeypatch.setattr(topaz_transformation.time, "sleep", lambda _: None)
        monkeypatch.setattr(config, "_http_client", lambda: _Silent())
        assert config._probe_billed_credits(_create_response(), "req-1") is None
        assert timeouts == [topaz_transformation._CREDIT_PROBE_TIMEOUT_SECS] * len(timeouts)
        assert timeouts and timeouts[0] is not None


class TestTopazCreatedVideoObject:
    def test_credits_become_usage_and_an_explicit_response_cost(self):
        video = TopazVideoConfig._created_video_object(
            model=MODEL, request_id="req-1", custom_llm_provider=None, seconds=5, topaz_credits=6
        )
        assert video.usage["topaz_credits"] == 6
        assert video.usage["duration_seconds"] == 5.0
        assert video._hidden_params["response_cost"] == pytest.approx(0.72)

    def test_without_credits_no_cost_is_asserted(self):
        """Pins the pre-fix shape: duration only, and no invented cost."""
        video = TopazVideoConfig._created_video_object(
            model=MODEL, request_id="req-1", custom_llm_provider=None, seconds=5
        )
        assert "topaz_credits" not in video.usage
        assert not video._hidden_params.get("response_cost")

    def test_duration_is_not_used_as_a_price_basis(self):
        """
        Two restores of identical duration must be able to cost different
        amounts - that is the whole reason a per-second rate is wrong here.
        """
        cheap = TopazVideoConfig._created_video_object(
            model=MODEL, request_id="a", custom_llm_provider=None, seconds=5, topaz_credits=1
        )
        dear = TopazVideoConfig._created_video_object(
            model=MODEL, request_id="b", custom_llm_provider=None, seconds=5, topaz_credits=39
        )
        assert cheap.usage["duration_seconds"] == dear.usage["duration_seconds"]
        assert cheap._hidden_params["response_cost"] == pytest.approx(0.12)
        assert dear._hidden_params["response_cost"] == pytest.approx(4.68)


class TestTopazCreditProbeWindow:
    """
    The probe window is sized from a live measurement, not a guess.

    Against api.topazlabs.com a job walks accepted -> initializing ->
    preprocessing once the upload lands, and `estimates` first appears at the
    preprocessing transition, ~2.6s in. A 3-attempt window sat on that boundary
    and missed it on a real prod restore, recording $0 for a job Topaz quoted at
    1 credit. These pin the shape of the fix so it cannot silently narrow again.
    """

    def test_window_outlasts_the_measured_estimate_delay(self):
        from litellm.llms.topaz.videos.transformation import (
            _CREDIT_PROBE_ATTEMPTS,
            _CREDIT_PROBE_DELAY_SECS,
            _CREDIT_PROBE_TIMEOUT_SECS,
        )

        measured_delay_secs = 2.6
        # Worst case each attempt costs its timeout, then sleeps before the next.
        window = _CREDIT_PROBE_ATTEMPTS * _CREDIT_PROBE_TIMEOUT_SECS + (
            _CREDIT_PROBE_ATTEMPTS - 1
        ) * _CREDIT_PROBE_DELAY_SECS
        assert window > measured_delay_secs * 1.5, (
            f"probe window {window}s leaves no margin over the measured {measured_delay_secs}s "
            "delay before Topaz publishes estimates"
        )

    def test_window_stays_bounded_enough_for_a_create_request(self):
        """A restore runs for minutes, but this sits on the customer's request."""
        from litellm.llms.topaz.videos.transformation import (
            _CREDIT_PROBE_ATTEMPTS,
            _CREDIT_PROBE_DELAY_SECS,
            _CREDIT_PROBE_TIMEOUT_SECS,
        )

        ceiling = _CREDIT_PROBE_ATTEMPTS * _CREDIT_PROBE_TIMEOUT_SECS + (
            _CREDIT_PROBE_ATTEMPTS - 1
        ) * _CREDIT_PROBE_DELAY_SECS
        assert ceiling <= 15.0, f"probe could hold the create leg for {ceiling}s"

    def test_probe_keeps_polling_through_the_pre_estimate_statuses(self, monkeypatch):
        """
        accepted and initializing carry no estimates; the quote lands at
        preprocessing. The probe must ride through the first two rather than
        treating an estimate-less 200 as a final answer.
        """
        config = TopazVideoConfig()
        pages = [
            {"status": "accepted"},
            {"status": "initializing"},
            {"status": "preprocessing", "estimates": {"cost": [1, 2]}},
        ]
        seen = []

        class _Staged:
            def get(self, *a, **k):
                payload = pages[min(len(seen), len(pages) - 1)]
                seen.append(payload["status"])
                return httpx.Response(200, json=payload)

        monkeypatch.setattr(topaz_transformation.time, "sleep", lambda _: None)
        monkeypatch.setattr(config, "_http_client", lambda: _Staged())

        assert config._probe_billed_credits(_create_response(), "req-1") == pytest.approx(1.0)
        assert seen == ["accepted", "initializing", "preprocessing"]
