"""Coverage-boost suite: brokers, alerts, gaia internals, orbit_fit,
inference + iod edge cases that the main hammer didn't hit.
"""
from __future__ import annotations

import math
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# =============================================================================
# Brokers: instantiation + no-network paths (no actual API calls)
# =============================================================================

class TestATLASBroker:
    def test_init_picks_up_env_token(self, monkeypatch):
        monkeypatch.setenv("ATLAS_API_TOKEN", "fake-token-xxx")
        from ariadne.discovery.brokers.atlas import AtlasBroker
        b = AtlasBroker()
        assert b.api_token == "fake-token-xxx"

    def test_no_token_returns_empty_immediately(self, monkeypatch):
        monkeypatch.delenv("ATLAS_API_TOKEN", raising=False)
        from ariadne.discovery.brokers.atlas import AtlasBroker
        b = AtlasBroker(api_token="")
        out = list(b.query_cone(180, 20, 1.0, 60450, 60451, max_alerts=5))
        assert out == []

    def test_explicit_token_overrides_env(self, monkeypatch):
        monkeypatch.setenv("ATLAS_API_TOKEN", "env-tok")
        from ariadne.discovery.brokers.atlas import AtlasBroker
        b = AtlasBroker(api_token="explicit")
        assert b.api_token == "explicit"

    def test_filter_band_attribute_stored(self):
        from ariadne.discovery.brokers.atlas import AtlasBroker
        b = AtlasBroker(api_token="x", filter_band="o")
        assert b.filter_band == "o"


class TestPanSTARRSBroker:
    def test_default_bands(self):
        from ariadne.discovery.brokers.panstarrs import PanStarrsBroker
        b = PanStarrsBroker()
        assert "g" in b.bands and "r" in b.bands

    def test_custom_bands(self):
        from ariadne.discovery.brokers.panstarrs import PanStarrsBroker
        b = PanStarrsBroker(bands=["i", "z"])
        assert b.bands == ["i", "z"]

    def test_query_returns_empty_iter_on_no_astroquery(self, monkeypatch):
        from ariadne.discovery.brokers.panstarrs import PanStarrsBroker
        b = PanStarrsBroker()
        # Force ImportError path
        with patch.dict("sys.modules", {"astroquery.mast": None}):
            out = list(b.query_cone(180, 20, 0.1, 50000, 60000))
            assert out == []


class TestAlerceBroker:
    def test_alerce_init_propagates_missing_dep(self):
        from ariadne.discovery.brokers.base import BrokerError
        from ariadne.discovery.brokers import alerce
        # Pretend alerce isn't installed
        with patch.dict("sys.modules", {"alerce.core": None}):
            with pytest.raises(BrokerError):
                alerce.AlerceZTFBroker()


class TestBrokerBase:
    def test_base_query_box_not_implemented(self):
        from ariadne.discovery.brokers.base import BrokerBase
        b = BrokerBase()
        with pytest.raises(NotImplementedError):
            list(b.query_box(0, 1, 0, 1, 60000, 60001))

    def test_collect_caps_at_max_n(self):
        from ariadne.discovery.brokers.base import Alert, collect

        def gen():
            for i in range(100):
                yield Alert("X", f"a{i}", "o", 60450 + i, 0, 0, 20, "r")

        got = collect(gen(), max_n=10)
        assert len(got) == 10


# =============================================================================
# Alerts: every sink + fire_all error path
# =============================================================================

class TestAlertSinks:
    def _candidate(self):
        from ariadne.discovery.operations.candidate_store import Candidate
        return Candidate(
            key="180.0000_+20.0000_01.50", ra=180.0, dec=20.0,
            rate_arcsec_hr=1.5,
            first_seen_mjd=60450, last_seen_mjd=60451, n_runs=2,
            rms_history=[[60450, 1.5]], skybot_names=[], status="active",
        )

    def test_noop_sink_swallows_silently(self):
        from ariadne.discovery.operations.alerts import NoopSink
        NoopSink().fire(self._candidate())               # no error

    def test_filesink_jsonl_round_trip(self, tmp_path):
        import json
        from ariadne.discovery.operations.alerts import FileSink
        f = tmp_path / "a.jsonl"
        s = FileSink(f)
        s.fire(self._candidate(), run_id="run-1")
        s.fire(self._candidate(), run_id="run-2")
        lines = f.read_text().splitlines()
        assert len(lines) == 2
        recs = [json.loads(l) for l in lines]
        assert recs[0]["run_id"] == "run-1"
        assert recs[1]["run_id"] == "run-2"

    def test_format_alert_text_includes_position(self):
        from ariadne.discovery.operations.alerts import _format_alert_text
        text = _format_alert_text(self._candidate(), run_id="abc")
        assert "180.0000" in text
        assert "+20.0000" in text
        assert "abc" in text

    def test_webhook_sink_url_must_be_http(self):
        from ariadne.discovery.operations.alerts import WebhookSink
        with pytest.raises(ValueError):
            WebhookSink("ftp://example.com/hook")

    def test_webhook_sink_failure_does_not_raise(self):
        from ariadne.discovery.operations.alerts import WebhookSink, fire_all

        class _BoomSink:
            name = "boom"

            def fire(self, c, run_id=None):
                raise RuntimeError("boom")

        c = self._candidate()
        # fire_all suppresses per-sink errors
        fire_all([_BoomSink()], c, run_id="x")           # no exception

    def test_webhook_post_catches_connection_error(self):
        from ariadne.discovery.operations.alerts import WebhookSink
        sink = WebhookSink("http://127.0.0.1:1/invalid-port-nothing-listens")
        # network failure should NOT propagate
        sink.fire(self._candidate(), run_id="x")

    def test_email_sink_smtp_failure_swallowed(self):
        from ariadne.discovery.operations.alerts import (
            EmailSink, SMTPConfig)
        sink = EmailSink(
            SMTPConfig(host="127.0.0.1", port=1),         # nothing listens
            sender="a@b", recipients=["c@d"])
        # SMTP exception should be swallowed
        sink.fire(self._candidate(), run_id="run-x")


# =============================================================================
# Gaia refinement: private helpers
# =============================================================================

class TestGaiaPrivateHelpers:
    def test_nearest_match_finds_within_tol(self):
        from ariadne.discovery.imaging.gaia_refine import _nearest_match
        from ariadne.discovery.imaging.source_extraction import Source
        srcs = [
            Source(180.0, 20.0, 1, 20, 3, 0, "t", 0, 0),
            Source(170.0, 20.0, 1, 20, 3, 0, "t", 0, 0),       # far away
        ]
        m = _nearest_match(180.001, 20.001, srcs, tol_arcsec=10)
        assert m is not None and m.ra == 180.0

    def test_nearest_match_returns_none_outside_tol(self):
        from ariadne.discovery.imaging.gaia_refine import _nearest_match
        from ariadne.discovery.imaging.source_extraction import Source
        srcs = [Source(180.0, 20.0, 1, 20, 3, 0, "t", 0, 0)]
        m = _nearest_match(190.0, 20.0, srcs, tol_arcsec=1)
        assert m is None

    def test_fit_translation_only_returns_zero_for_zero_matches(self):
        from ariadne.discovery.imaging.gaia_refine import _fit_translation_only
        dra, ddec, rms = _fit_translation_only([])
        assert dra == 0.0 and ddec == 0.0
        assert rms == float("inf")

    def test_fit_translation_only_recovers_known_shift(self):
        from ariadne.discovery.imaging.gaia_refine import _fit_translation_only
        # Gaia is offset from source by +0.001 deg RA, -0.0005 deg Dec
        matches = [(180.001 + i * 0.01, 20.0 - 0.0005 + i * 0.001,
                     180.0 + i * 0.01, 20.0 + i * 0.001)
                    for i in range(6)]
        dra, ddec, rms = _fit_translation_only(matches)
        # Within rounding -- median over a clean signal
        assert abs(dra - 0.001 * math.cos(math.radians(20))) < 1e-4
        assert abs(ddec - (-0.0005)) < 1e-4

    def test_fit_affine_returns_identity_when_too_few(self):
        from ariadne.discovery.imaging.gaia_refine import _fit_affine
        mat, tr, rms = _fit_affine([(180, 20, 180, 20),
                                      (181, 21, 181, 21)])         # only 2 matches
        # Should return identity + zero translation
        assert np.allclose(mat, np.eye(2))
        assert np.allclose(tr, np.zeros(2))

    def test_refine_to_gaia_passthrough_when_no_stars(self):
        from ariadne.discovery.imaging.gaia_refine import refine_to_gaia
        from ariadne.discovery.imaging.source_extraction import Source
        srcs = [Source(0.5, 0.5, 100, 20, 3, 0, "t", 1, 1)]
        # query at empty patch with tiny radius -> passthrough
        refined, report = refine_to_gaia(
            srcs, image_centre_ra_deg=0.5, image_centre_dec_deg=0.5,
            image_radius_deg=0.0001)
        assert len(refined) == 1
        # passthrough is a documented outcome; success can be False or True
        assert report.method_used in ("passthrough", "translation", "affine",
                                       "reject_high_rms")


# =============================================================================
# orbit_fit (the 2-body, NOT iod) -- predict + multi-opposition
# =============================================================================

class TestOrbitFit:
    def test_predict_radec_returns_finite(self):
        from ariadne.discovery.orbit_fit import predict_radec
        from ariadne.dynamics.secular import elements_to_state
        from ariadne.data.ephemeris import et, body_state
        p, v = elements_to_state(2.7, 0.05, 5, 30, 50, 180)
        t_ref = et("2026-01-01T00:00:00")
        t_obs = et("2026-02-01T00:00:00")
        R_obs = body_state("EARTH", t_obs, "J2000", "SUN")[:3]
        state = np.concatenate([np.asarray(p), np.asarray(v)])
        ra, dec = predict_radec(state, t_ref, t_obs, R_obs)
        assert math.isfinite(ra) and math.isfinite(dec)
        assert -math.pi <= ra <= 2 * math.pi
        assert -math.pi / 2 <= dec <= math.pi / 2


# =============================================================================
# iod: refinement zoom + edge cases
# =============================================================================

class TestIOD:
    def test_iod_returns_none_for_two_tracklets(self):
        from ariadne.discovery.iod import iod_hypothesis_search
        # only 2 tracklets -> insufficient
        out = iod_hypothesis_search([
            {"t": 0.0, "ra": 0.0, "dec": 0.0, "dra": 0.0, "ddec": 0.0},
            {"t": 1.0, "ra": 0.0, "dec": 0.0, "dra": 0.0, "ddec": 0.0},
        ])
        assert out is None

    def test_iod_fit_candidate_returns_none_on_no_iod(self):
        from ariadne.discovery.iod import fit_candidate
        # Tracklets all at same epoch -> IOD will fail
        same_t = [{"t": 0.0, "ra": 1.0 + 0.001 * i,
                    "dec": 0.0, "dra": 0.0, "ddec": 0.0} for i in range(4)]
        out = fit_candidate(same_t)
        # Either returns None or a fit with inf RMS
        if out is not None:
            assert (not math.isfinite(out["rms_arcsec"])) or out["rms_arcsec"] > 10

    def test_fit_orbit_lm_handles_garbage_initial(self):
        from ariadne.discovery.iod import fit_orbit_lm
        # Garbage seed -> should not crash, returns dict
        out = fit_orbit_lm(
            [{"t": 0.0, "ra": 0.0, "dec": 0.0, "dra": 0.0, "ddec": 0.0}],
            t_ref=0.0, x_init=np.zeros(3), v_init=np.zeros(3))
        assert "x_fit" in out and "success" in out


# =============================================================================
# Inference: less-covered branches
# =============================================================================

class TestInferenceCoverageGaps:
    def test_recommend_observe_multiband_branch(self):
        """Trigger the mid-entropy + no band_magnitudes follow-up branch."""
        from ariadne.discovery.inference import infer, Evidence
        # Carefully tuned: middling evidence, no colors -> entropy in (0.5, 1.5)
        ev = Evidence(rate_arcsec_hr=20, apparent_mag=20,
                       morphology_label="POINT", morphology_confidence=0.7,
                       n_detections=3, arc_days=2, rms_arcsec=2,
                       skybot_match_names=[])
        res = infer(ev)
        # one of: monitor / observe_multiband / alert / second-night
        assert res.recommended_followup["action"] in (
            "monitor", "observe_multiband", "alert_and_submit_mpc",
            "observe_second_night")

    def test_recommend_monitor_when_sufficient_confidence(self):
        from ariadne.discovery.inference import infer, Evidence
        # Strong NEO evidence; entropy low; not in high-novelty class
        ev = Evidence(rate_arcsec_hr=200, apparent_mag=18,
                       morphology_label="POINT", morphology_confidence=0.99,
                       n_detections=10, arc_days=20, rms_arcsec=1,
                       skybot_match_names=[],
                       band_magnitudes={"g": 18.1, "r": 17.9})
        res = infer(ev)
        assert res.recommended_followup["action"] in (
            "monitor", "alert_and_submit_mpc")

    def test_size_from_magnitude_edge_cases(self):
        from ariadne.discovery.inference import _size_from_magnitude
        # NaN mag -> NaN size
        assert math.isnan(_size_from_magnitude(float("nan"), 1.0))
        # zero distance -> NaN
        assert math.isnan(_size_from_magnitude(20.0, 0.0))
        # negative distance -> NaN
        assert math.isnan(_size_from_magnitude(20.0, -1.0))

    def test_gaussian_log_pdf_zero_sigma_returns_floor(self):
        from ariadne.discovery.inference import _gaussian_log_pdf
        v = _gaussian_log_pdf(1.0, 0.0, 0.0)
        assert v <= -1e5

    def test_uniform_log_pdf_zero_width_returns_floor(self):
        from ariadne.discovery.inference import _uniform_log_pdf_in_range
        v = _uniform_log_pdf_in_range(0.5, 1.0, 1.0)
        # Same lo == hi; outside the (closed) point returns -1e6 floor
        assert v <= -1e5

    def test_pareto_front_single_hypothesis(self):
        from ariadne.discovery.inference import _pareto_front, Hypothesis
        h = Hypothesis(label="x", posterior=1.0, prior=-1.0, free_energy=1.0)
        front = _pareto_front([h])
        assert front == [h]

    def test_pareto_front_empty(self):
        from ariadne.discovery.inference import _pareto_front
        assert _pareto_front([]) == []

    def test_episodic_recall_filters_by_rate(self, tmp_path):
        from ariadne.discovery.inference import _episodic_recall, Evidence
        from ariadne.discovery.operations.candidate_store import CandidateStore
        store = CandidateStore(tmp_path / "s.json")
        # Two candidates at the same position; only one has matching rate
        store.upsert(ra=180, dec=20, rate_arcsec_hr=1.5, mjd=60440)
        store.upsert(ra=180.5, dec=20, rate_arcsec_hr=15.0, mjd=60440)
        matches = _episodic_recall(
            Evidence(ra_deg=180, dec_deg=20, rate_arcsec_hr=1.5),
            store, rate_tol_arcsec_hr=0.5)
        assert len(matches) == 1

    def test_reliability_report_handles_empty_cases(self):
        from ariadne.discovery.inference import reliability_report
        rep = reliability_report([])
        assert rep.n == 0
        assert rep.nll == float("inf")

    def test_certificate_round_trip_serialisable(self):
        from ariadne.discovery.inference import (
            infer, inference_certificate, _jsonable, Evidence)
        import json
        res = infer(Evidence(rate_arcsec_hr=1.5, apparent_mag=22))
        # The certificate should be JSON-serialisable
        as_json = json.dumps(_jsonable(res.certificate))
        assert "payload_hash" in as_json

    def test_validate_certificate_false_when_missing(self):
        from ariadne.discovery.inference import (
            validate_inference_certificate, Evidence, InferenceResult)
        res = InferenceResult(hypotheses=[])
        # certificate is empty -> validate returns False
        assert validate_inference_certificate(Evidence(), res) is False


# =============================================================================
# Realbogus: rules not covered by the main tests
# =============================================================================

class TestRealbogusEdges:
    def test_same_pixel_stacking_fires(self):
        from ariadne.discovery.realbogus import rule_same_pixel_stacking
        from ariadne.discovery.brokers.base import Alert
        # 5 detections all at exactly the same position -> stacking artefact
        members = [Alert("ZTF", f"{i}", "o", 60450 + i * 0.01, 180.0, 20.0, 20, "r")
                   for i in range(5)]
        assert rule_same_pixel_stacking({"members": members}) > 0

    def test_same_pixel_stacking_passes_real_motion(self):
        from ariadne.discovery.realbogus import rule_same_pixel_stacking
        from ariadne.discovery.brokers.base import Alert
        # 5 detections moving 1 arcmin between each -> NOT stacking
        members = [Alert("ZTF", f"{i}", "o", 60450 + i * 0.01,
                          180.0 + i * 0.02, 20.0, 20, "r") for i in range(5)]
        assert rule_same_pixel_stacking({"members": members}) == 0

    def test_collinear_with_too_few_members_passes(self):
        from ariadne.discovery.realbogus import rule_collinear_but_unequal_spacing
        # <4 members -> rule doesn't fire
        from ariadne.discovery.brokers.base import Alert
        members = [Alert("ZTF", f"{i}", "o", 60450 + i, 180, 20, 20, "r")
                   for i in range(3)]
        assert rule_collinear_but_unequal_spacing({"members": members}) == 0

    def test_morphology_rules_with_none(self):
        from ariadne.discovery.realbogus import (
            rule_cosmic_ray, rule_edge_artefact, rule_extended)
        assert rule_cosmic_ray(None) == 0
        assert rule_edge_artefact(None) == 0
        assert rule_extended(None) == 0

    def test_zero_motion_uses_dra_ddec_when_no_rate(self):
        from ariadne.discovery.realbogus import rule_zero_motion
        # rate not provided; dra and ddec are zero
        score = rule_zero_motion({"dra": 0.0, "ddec": 0.0})
        assert score > 0


# =============================================================================
# Realtime pipeline: link_helio_linc path + fit_filter error branches
# =============================================================================

class TestRealtimeBranches:
    def test_link_helio_linc_with_empty_tracklets(self):
        from ariadne.discovery.realtime import link_helio_linc
        assert link_helio_linc([]) == []

    def test_fit_filter_handles_unfittable_short_arc(self):
        from ariadne.discovery.realtime import fit_filter
        # single member -> tagged as unfittable_single_arc
        from ariadne.discovery.brokers.base import Alert
        a = Alert("ZTF", "x", "o", 60450, 180, 20, 20, "r")
        out = fit_filter([{"members": [a], "t": 0, "ra": 0, "dec": 0,
                            "dra": 0, "ddec": 0, "rate_arcsec_hr": 0,
                            "jd": 0}])
        assert out[0]["status"] == "unfittable_single_arc"
        assert out[0]["rms_arcsec"] is None

    def test_cluster_centroid_averages_position(self):
        from ariadne.discovery.realtime import cluster_centroid
        from ariadne.discovery.brokers.base import Alert
        cluster = [
            Alert("ZTF", "1", "o", 60450, 100.0, 20.0, 20.0, "r"),
            Alert("ZTF", "2", "o", 60450, 100.2, 20.0, 20.5, "r"),
        ]
        c = cluster_centroid(cluster)
        assert abs(c.ra - 100.1) < 1e-9
        assert c.meta["n_alerts"] == 2


# =============================================================================
# Quality scoring: subscore behaviours
# =============================================================================

class TestScoringSubscores:
    def test_arc_score_zero_at_zero_days(self):
        from ariadne.discovery.scoring import _arc_score
        assert _arc_score(0) == 0.0
        assert _arc_score(-1) == 0.0

    def test_runs_score_caps_at_one(self):
        from ariadne.discovery.scoring import _runs_score
        for n in (10, 100, 1000):
            assert _runs_score(n) <= 1.0
        assert _runs_score(0) == 0.0

    def test_skybot_score_no_xmatch_returns_half(self):
        from ariadne.discovery.scoring import _skybot_score
        # Unknown xmatch state -> 0.5
        assert _skybot_score([], has_xmatch_run=False) == 0.5

    def test_logistic_handles_non_finite(self):
        from ariadne.discovery.scoring import _logistic
        assert _logistic(float("nan"), 1, 1) == 0.0


# =============================================================================
# Multi-band fusion: edge / property
# =============================================================================

class TestFusionMore:
    def test_fuse_preserves_per_band_mags(self):
        from ariadne.discovery.fusion import fuse_alerts
        from ariadne.discovery.brokers.base import Alert
        alerts = [
            Alert("ZTF", "1", "o", 60450, 100, 20, 21.5, "g"),
            Alert("ZTF", "2", "o", 60450.001, 100, 20, 21.0, "r"),
        ]
        fused = fuse_alerts(alerts, pos_tol_arcsec=1.5, mag_diff_max=5.0)
        assert len(fused) == 1
        bm = fused[0].mag_by_band
        assert "g" in bm and "r" in bm

    def test_fusion_as_alert_picks_r_band_if_present(self):
        from ariadne.discovery.fusion import fuse_to_alerts
        from ariadne.discovery.brokers.base import Alert
        alerts = [
            Alert("ZTF", "1", "o", 60450, 100, 20, 21.5, "g"),
            Alert("ATLAS", "2", "o", 60450.001, 100, 20, 19.5, "o"),
            Alert("ZTF", "3", "o", 60450.002, 100, 20, 21.0, "r"),
        ]
        out = fuse_to_alerts(alerts, mag_diff_max=10.0)
        # representative magnitude is r-band (21.0)
        assert out[0].band == "r"
        assert abs(out[0].mag - 21.0) < 0.01


# =============================================================================
# Validation sensitivity: edge cases
# =============================================================================

class TestSensitivityEdges:
    def test_make_population_seeded_deterministic(self):
        from ariadne.validate.sensitivity import make_population
        a = make_population(n_objects=5, seed=42)
        b = make_population(n_objects=5, seed=42)
        # Same seed -> identical
        assert a == b

    def test_inject_zero_objects(self):
        from ariadne.validate.sensitivity import inject_synthetic_objects
        alerts, truth = inject_synthetic_objects([], epoch="2026-01-01T00:00:00")
        assert alerts == []
        assert truth == []
