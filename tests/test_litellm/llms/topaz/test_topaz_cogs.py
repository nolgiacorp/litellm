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
from litellm.llms.topaz.video_geometry import SourceGeometry
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
            "prob-4",
            "rhea-1",
            "iris-3",
            "nyx-3",
            "thd-3",
            "ahq-12",
            "ghq-5",
            "dtd-4",
            "slf-2",
            "slp-2.5",
            "wonder-1",
            "hyp-2",
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
            (2, 0.24),  # the live quote for a 5s 720p -> 1080p prob-4 restore
            (6, 0.72),
            (10, 1.20),
            (0.25, 0.03),  # measured: 5s from a 960x720 source, 720p out
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


def _mp4(width: int = 640, height: int = 360, timescale: int = 24000, duration: int = 312000, samples: int = 312):
    """A structurally valid ISO-BMFF clip; see test_video_geometry.py for the box layout."""
    from tests.test_litellm.llms.topaz.test_video_geometry import _mp4 as build

    return build(
        coded=(width, height), display=(width, height), timescale=timescale, duration=duration, samples=samples
    )


class _FakeClient:
    """
    Records every call so the tests can assert HOW MANY were made, not just what
    came back. The attempt count is the point: the reverted approach polled.
    """

    def __init__(self, post_result=None):
        self.post_calls = []
        self.put_calls = []
        self._post_result = post_result

    def put(self, url, content=None, headers=None):
        self.put_calls.append(url)
        return httpx.Response(200, request=httpx.Request("PUT", url))

    def post(self, url, json=None, headers=None, timeout=None):
        self.post_calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        if isinstance(self._post_result, Exception):
            raise self._post_result
        if self._post_result is not None:
            return self._post_result
        return httpx.Response(
            200,
            json={"requestId": "est-1", "estimates": {"cost": [1, 2], "time": [323, 336]}},
            request=httpx.Request("POST", url),
        )


def _pending(config, source, container="mp4", declared=None, output=(1280, 720), frame_rate=None):
    from litellm.llms.topaz.videos.transformation import _PendingUpload

    return _PendingUpload(
        source=source,
        container=container,
        seconds=13,
        model_code="prob-4",
        output_width=output[0],
        output_height=output[1],
        output_frame_rate=frame_rate,
        declared_geometry=declared,
    )


class TestTopazEstimateAcquisition:
    """
    NOL-519. The quote comes from Topaz's FREE `POST /video/` endpoint, priced
    off geometry read from the bytes this leg already downloaded.

    The approach this replaced polled the job's own status object, where
    `estimates` appears only at the `preprocessing` transition - measured at 2.6s
    on one live restore and still pending at 68s on the next. It therefore
    captured the quote only sometimes, and an intermittently populated COGS
    ledger is worse than an empty one: it looks healthy while understating
    unpredictably, and it permanently mutes the NOL-535 guard, which fires only
    for a model that has NEVER recorded a cost.
    """

    def test_estimate_url_is_derived_from_the_create_url(self):
        """Preserves a TOPAZ_API_BASE override instead of assuming the default host."""
        url = TopazVideoConfig._estimate_url(_create_response("https://topaz.internal.test/video/express"))
        assert url == "https://topaz.internal.test/video/"

    def test_estimate_url_falls_back_when_the_create_url_is_unexpected(self):
        url = TopazVideoConfig._estimate_url(_create_response("https://elsewhere.test/other"))
        assert url == "https://api.topazlabs.com/video/"

    def test_estimate_reuses_the_create_request_api_key(self):
        headers = TopazVideoConfig._estimate_headers(_create_response())
        assert headers["X-API-Key"] == "test-key"
        assert headers["Content-Type"] == "application/json"

    def test_body_matches_topaz_schema_for_parsed_geometry(self):
        """
        The vendor contract, pinned field by field. Verified live against
        api.topazlabs.com: this exact body for a 640x360 24fps 13s source at a
        1280x720 output returned {"estimates": {"cost": [1, 2], ...}} on five
        consecutive calls, in about 0.15s each.
        """
        client = _FakeClient()
        config = TopazVideoConfig(sync_client=client)
        data = _mp4()

        credits = config._estimate_billed_credits(_create_response(), _pending(config, source=None), data)

        assert credits == pytest.approx(1.0)
        body = client.post_calls[0]["json"]
        assert body["source"] == {
            "container": "mp4",
            "size": len(data),
            "duration": pytest.approx(13.0),
            "frameCount": 312,
            "frameRate": pytest.approx(24.0),
            "resolution": {"width": 640, "height": 360},
        }
        assert body["filters"] == [{"model": "prob-4"}]
        assert body["output"]["resolution"] == {"width": 1280, "height": 720}
        assert body["output"]["frameRate"] == pytest.approx(24.0)

    def test_exactly_one_request_is_made(self):
        """
        The regression guard against the reverted design. A quote that has to be
        polled is a quote whose arrival depends on Topaz's queue depth; this
        endpoint is deterministic over its inputs, so a second attempt could
        only ever return the same answer.
        """
        client = _FakeClient()
        config = TopazVideoConfig(sync_client=client)

        config._estimate_billed_credits(_create_response(), _pending(config, source=None), _mp4())

        assert len(client.post_calls) == 1

    def test_no_request_is_made_without_geometry(self):
        """An unparseable container with nothing declared cannot be quoted, so nothing is sent."""
        client = _FakeClient()
        config = TopazVideoConfig(sync_client=client)

        credits = config._estimate_billed_credits(
            _create_response(), _pending(config, source=None, container="mkv"), b"\x1a\x45\xdf\xa3not-isobmff"
        )

        assert credits is None
        assert client.post_calls == []

    def test_output_frame_rate_defaults_to_the_source_rate(self):
        """An upscale that was not asked to interpolate processes the frames it was given."""
        client = _FakeClient()
        config = TopazVideoConfig(sync_client=client)

        config._estimate_billed_credits(_create_response(), _pending(config, source=None, frame_rate=60.0), _mp4())
        assert client.post_calls[0]["json"]["output"]["frameRate"] == pytest.approx(60.0)

    def test_request_carries_its_own_timeout(self):
        """This sits inline on a customer's create request and rides a client whose default can be 60s."""
        client = _FakeClient()
        config = TopazVideoConfig(sync_client=client)

        config._estimate_billed_credits(_create_response(), _pending(config, source=None), _mp4())

        assert client.post_calls[0]["timeout"] == pytest.approx(6.0)


class TestTopazGeometryPrecedence:
    def test_measured_geometry_beats_declared_geometry(self):
        """
        A caller-supplied number that LOWERS the quote would lower our own
        recorded COGS. The bytes being uploaded are the authority.
        """
        client = _FakeClient()
        config = TopazVideoConfig(sync_client=client)
        declared = SourceGeometry(width=64, height=36, duration_seconds=1.0, frame_rate=1.0)

        config._estimate_billed_credits(_create_response(), _pending(config, None, declared=declared), _mp4())

        assert client.post_calls[0]["json"]["source"]["resolution"] == {"width": 640, "height": 360}

    def test_declared_geometry_covers_unparseable_containers(self):
        """mkv is EBML, not ISO-BMFF, so the declaration is the only source of truth for it."""
        client = _FakeClient()
        config = TopazVideoConfig(sync_client=client)
        declared = SourceGeometry(width=1920, height=1080, duration_seconds=5.0, frame_rate=30.0)

        credits = config._estimate_billed_credits(
            _create_response(), _pending(config, None, container="mkv", declared=declared), b"\x1a\x45\xdf\xa3"
        )

        assert credits == pytest.approx(1.0)
        assert client.post_calls[0]["json"]["source"]["frameCount"] == 150

    def test_partial_declared_geometry_is_refused(self):
        """Completing a declaration with defaults produces a plausible quote that is quietly wrong."""
        assert TopazVideoConfig._declared_geometry({"source_width": 1920, "source_height": 1080}) is None
        assert TopazVideoConfig._declared_geometry({}) is None

    def test_declared_duration_falls_back_to_seconds(self):
        geometry = TopazVideoConfig._declared_geometry(
            {"source_width": 1920, "source_height": 1080, "source_frame_rate": 30, "seconds": 5}
        )
        assert geometry == SourceGeometry(width=1920, height=1080, duration_seconds=5.0, frame_rate=30.0)

    def test_nonsense_declared_values_are_refused(self):
        assert (
            TopazVideoConfig._declared_geometry(
                {"source_width": 0, "source_height": 1080, "source_frame_rate": 30, "seconds": 5}
            )
            is None
        )


class TestTopazEstimateFailuresAreSilent:
    """
    Topaz has already accepted the footage by the time the quote is taken, so no
    bookkeeping failure may surface as a create error. Missing credits records
    no cost, which is the pre-existing behaviour and what NOL-535 detects.
    """

    @pytest.mark.parametrize(
        "response",
        [
            httpx.Response(503, text="upstream down", request=httpx.Request("POST", "https://t.test/video/")),
            httpx.Response(200, text="not json", request=httpx.Request("POST", "https://t.test/video/")),
            httpx.Response(200, json=[1, 2], request=httpx.Request("POST", "https://t.test/video/")),
            httpx.Response(200, json={"estimates": {}}, request=httpx.Request("POST", "https://t.test/video/")),
            httpx.Response(
                200, json={"estimates": {"cost": []}}, request=httpx.Request("POST", "https://t.test/video/")
            ),
        ],
    )
    def test_unusable_responses_yield_no_credits(self, response):
        config = TopazVideoConfig(sync_client=_FakeClient(post_result=response))
        assert config._estimate_billed_credits(_create_response(), _pending(config, None), _mp4()) is None

    @pytest.mark.parametrize(
        "raised",
        [
            httpx.ConnectError("no route"),
            httpx.ReadTimeout("slow"),
            litellm.Timeout(message="timed out", model="topaz/prob-4", llm_provider="topaz"),
        ],
    )
    def test_transport_and_timeout_failures_do_not_escape(self, raised):
        """
        The handler converts a read timeout into litellm.Timeout, which is NOT
        an httpx.HTTPError, so catching only httpx errors would let it through.
        """
        config = TopazVideoConfig(sync_client=_FakeClient(post_result=raised))
        assert config._estimate_billed_credits(_create_response(), _pending(config, None), _mp4()) is None

    def test_lower_bound_of_the_cost_range_is_what_topaz_bills(self):
        response = httpx.Response(
            200,
            json={"estimates": {"cost": [3, 9]}},
            request=httpx.Request("POST", "https://t.test/video/"),
        )
        config = TopazVideoConfig(sync_client=_FakeClient(post_result=response))
        assert config._estimate_billed_credits(_create_response(), _pending(config, None), _mp4()) == pytest.approx(3)

    def test_fractional_quotes_keep_full_precision(self):
        """Topaz quotes 0.25 credits for small jobs; truncating would record $0 for the cheap end."""
        response = httpx.Response(
            200,
            json={"estimates": {"cost": [0.25, 1]}},
            request=httpx.Request("POST", "https://t.test/video/"),
        )
        config = TopazVideoConfig(sync_client=_FakeClient(post_result=response))
        credits = config._estimate_billed_credits(_create_response(), _pending(config, None), _mp4())
        assert credits == pytest.approx(0.25)
        assert cost_calculator(model=MODEL, topaz_credits=credits) == pytest.approx(0.03)


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
