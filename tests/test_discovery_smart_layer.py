"""Tests for the next-realm discovery layer: MPC submission, follow-up
prediction, candidate scoring, multi-cone nightly, HelioLinC wiring, dashboard.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ----------------------------- MPC SUBMISSION ---------------------------------

def test_mpc_format_record_exactly_80_chars():
    from ariadne.discovery.mpc_submit import format_record
    rec = format_record(
        mjd=60450.5, ra_deg=187.13456, dec_deg=-12.4321,
        mag=21.3, band="r",
        designation="~ABCDE ", observatory_code="I41",
        discovery_asterisk=True,
    )
    assert len(rec) == 80, f"got length {len(rec)}: {rec!r}"
    assert rec[12] == "*", f"discovery asterisk should be at col 13: {rec!r}"
    assert rec[14] == "C", "CCD note should be at col 15"
    assert rec[-3:] == "I41", "observatory code at cols 78-80"


def test_mpc_format_record_roundtrip():
    """Format -> parse round-trip preserves position/date within format precision."""
    from ariadne.discovery.mpc_submit import format_record, parse_record
    rec = format_record(
        mjd=60500.12345, ra_deg=42.1234, dec_deg=15.5678,
        mag=20.5, band="g",
        designation="~ABCDE ", observatory_code="500",
    )
    decoded = parse_record(rec)
    assert abs(decoded["mjd"] - 60500.12345) < 1e-5
    assert abs(decoded["ra_deg"] - 42.1234) < 1e-3       # 1 arcsec ~ 0.0003 deg
    assert abs(decoded["dec_deg"] - 15.5678) < 1e-3
    assert decoded["observatory_code"] == "500"
    assert decoded["band"] == "g"
    assert abs(decoded["mag"] - 20.5) < 0.01


def test_mpc_negative_declination_roundtrip():
    from ariadne.discovery.mpc_submit import format_record, parse_record
    rec = format_record(
        mjd=60100.0, ra_deg=350.0, dec_deg=-67.123,
        designation="~12345 ", observatory_code="W84",
    )
    decoded = parse_record(rec)
    assert abs(decoded["dec_deg"] - (-67.123)) < 1e-3
    assert abs(decoded["ra_deg"] - 350.0) < 1e-3


def test_mpc_format_never_emits_60_seconds():
    """Floating-point boundary: RA=260.0 deg sits at exactly 17h20m00s.
    Naive formatter prints '17 19 60.000' (invalid). Confirm rollover.
    """
    from ariadne.discovery.mpc_submit import _format_ra, _format_dec
    # RA = 260.0 deg -> 17h 20m 00s.000 -> must NOT be "17 19 60.000"
    ra_s = _format_ra(260.0)
    assert "60.000" not in ra_s, f"RA second rollover failed: {ra_s!r}"
    # Dec = +30.0 deg -> +30 00 00.00 -> must NOT be "+29 59 60.00"
    dec_s = _format_dec(30.0)
    assert "60.00" not in dec_s, f"Dec second rollover failed: {dec_s!r}"
    # spot-check parseability
    from ariadne.discovery.mpc_submit import format_record, parse_record
    rec = format_record(mjd=60460.0, ra_deg=260.0, dec_deg=30.0,
                        designation="~TST00 ", observatory_code="500")
    d = parse_record(rec)
    assert abs(d["ra_deg"] - 260.0) < 1e-3
    assert abs(d["dec_deg"] - 30.0) < 1e-3


def test_mpc_emit_submission_with_header():
    from ariadne.discovery.mpc_submit import MPCHeader, emit_submission, _pack_temp_designation
    from ariadne.discovery.operations.candidate_store import Candidate
    from ariadne.discovery.brokers.base import Alert
    header = MPCHeader(
        observatory_code="500", contact="J Doe, doe@example.com",
        observers="J Doe", measurers="J Doe", telescope="virtual",
        ack_keyword="ARI_TEST", ack_email="doe@example.com",
    )
    cand = Candidate(key="180.0000_+20.0000_01.50", ra=180.0, dec=20.0,
                     rate_arcsec_hr=1.5, first_seen_mjd=60450.0,
                     last_seen_mjd=60450.0, n_runs=1)
    alerts = [
        Alert(survey="ZTF", alert_id="a1", obj_id="x1", mjd=60450.0, ra=180.0,
              dec=20.0, mag=21.2, band="r", meta={}),
        Alert(survey="ZTF", alert_id="a2", obj_id="x1", mjd=60450.05, ra=180.001,
              dec=20.0001, mag=21.3, band="r", meta={}),
    ]
    text = emit_submission(header, [(cand, alerts)])
    lines = text.splitlines()
    assert any(line.startswith("COD 500") for line in lines)
    body_lines = [line for line in lines if len(line) == 80]
    assert len(body_lines) == 2, f"expected 2 80-col records, got {len(body_lines)}"
    # discovery asterisk on first record only
    assert body_lines[0][12] == "*"
    assert body_lines[1][12] == " "


# ----------------------------- FOLLOW-UP PREDICTOR ----------------------------

def _make_candidate_with_keplerian_state(a_au=80.0, e=0.05, mjd_ref=60450.0):
    """Build a Candidate whose orbit_state corresponds to a real heliocentric state
    of a Sedna-like distant object (predictable + finite uncertainty)."""
    from ariadne.discovery.operations.candidate_store import Candidate
    from ariadne.dynamics.secular import elements_to_state
    from ariadne.data.ephemeris import et
    p, v = elements_to_state(a_au, e, 12.0, 30.0, 50.0, 180.0)
    et_ref = et("2026-01-01T00:00:00")
    return Candidate(
        key="000.0000_+00.0000_00.00", ra=0.0, dec=0.0, rate_arcsec_hr=0.5,
        first_seen_mjd=mjd_ref, last_seen_mjd=mjd_ref,
        orbit_state=list(p) + list(v),
        meta={"t_ref_et": et_ref},
    )


def test_followup_predict_ephemeris_returns_real_radec():
    from ariadne.discovery.followup import predict_ephemeris
    c = _make_candidate_with_keplerian_state()
    eph = predict_ephemeris(c, mjd=60460.0)
    assert eph is not None
    assert 0 <= eph.ra_deg < 360
    assert -90 <= eph.dec_deg <= 90
    assert 1.0 < eph.sun_distance_au < 200
    assert 0.1 < eph.earth_distance_au < 200


def test_followup_uncertainty_scales_with_input_sigma():
    """Bigger input-state sigma -> bigger output ephemeris sigma. This is the
    monotonic property of the Monte-Carlo independent of orbital geometry.
    (Sigma vs lookahead time is NOT monotonic on the sky -- projection geometry,
    range, and phase rotate over an orbit and modulate the projected scatter.)
    """
    from ariadne.discovery.followup import predict_with_uncertainty
    c = _make_candidate_with_keplerian_state(a_au=5.0, e=0.10)
    tight = predict_with_uncertainty(c, mjd=60460.0, n_samples=40, seed=1,
                                     pos_sigma_km=1e3, vel_sigma_km_s=0.001)
    loose = predict_with_uncertainty(c, mjd=60460.0, n_samples=40, seed=1,
                                     pos_sigma_km=1e6, vel_sigma_km_s=1.0)
    assert tight is not None and loose is not None
    assert loose.sigma_arcsec > tight.sigma_arcsec * 5.0, \
        f"tighter inputs should give smaller sigma: " \
        f"tight={tight.sigma_arcsec:.2f}, loose={loose.sigma_arcsec:.2f}"


def test_followup_no_orbit_state_returns_none():
    from ariadne.discovery.followup import predict_ephemeris
    from ariadne.discovery.operations.candidate_store import Candidate
    c = Candidate(key="k", ra=0, dec=0, rate_arcsec_hr=0, first_seen_mjd=0, last_seen_mjd=0)
    assert predict_ephemeris(c, mjd=60450.0) is None


def test_next_night_targets_returns_sorted_dicts():
    from ariadne.discovery.followup import next_night_targets
    cands = [_make_candidate_with_keplerian_state(),
             _make_candidate_with_keplerian_state(a_au=50.0)]
    cands[1].key = "001.0000_+00.0000_00.00"
    targets = next_night_targets(cands, mjd_next=60460.0,
                                  max_sigma_arcsec=1e9, n_samples=10)
    assert len(targets) == 2
    sigmas = [t["sigma_arcsec"] for t in targets]
    assert sigmas == sorted(sigmas), "should be sorted by sigma ascending"
    for t in targets:
        for required in ("key", "ra_deg", "dec_deg", "sigma_arcsec",
                         "search_radius_arcsec", "range_au", "v_sky_arcsec_hr"):
            assert required in t


# ----------------------------- QUALITY SCORER ---------------------------------

def test_scoring_lower_rms_scores_higher():
    from ariadne.discovery.scoring import score_candidate
    from ariadne.discovery.operations.candidate_store import Candidate
    good = Candidate(key="g", ra=0, dec=0, rate_arcsec_hr=1.0,
                     first_seen_mjd=60450, last_seen_mjd=60460, n_runs=3,
                     rms_history=[[60450, 1.0]])
    bad = Candidate(key="b", ra=0, dec=0, rate_arcsec_hr=1.0,
                    first_seen_mjd=60450, last_seen_mjd=60460, n_runs=3,
                    rms_history=[[60450, 12.0]])
    assert score_candidate(good).total > score_candidate(bad).total


def test_scoring_longer_arc_scores_higher():
    from ariadne.discovery.scoring import score_candidate
    from ariadne.discovery.operations.candidate_store import Candidate
    short = Candidate(key="s", ra=0, dec=0, rate_arcsec_hr=1.0,
                      first_seen_mjd=60450, last_seen_mjd=60451,
                      rms_history=[[60450, 2.0]])
    long = Candidate(key="l", ra=0, dec=0, rate_arcsec_hr=1.0,
                     first_seen_mjd=60450, last_seen_mjd=60490,
                     rms_history=[[60450, 2.0]])
    assert score_candidate(long).total > score_candidate(short).total


def test_scoring_skybot_match_lowers_score():
    from ariadne.discovery.scoring import score_candidate
    from ariadne.discovery.operations.candidate_store import Candidate
    base = dict(ra=0, dec=0, rate_arcsec_hr=1.0,
                first_seen_mjd=60450, last_seen_mjd=60460, n_runs=2,
                rms_history=[[60450, 2.0]])
    unknown = Candidate(key="u", skybot_names=[], **base)
    known = Candidate(key="k", skybot_names=["Ceres"], **base)
    assert score_candidate(unknown).total > score_candidate(known).total


def test_scoring_grade_thresholds():
    from ariadne.discovery.scoring import QualityScore
    assert QualityScore(0.85, 1, 1, 1, 1, 1).grade() == "A"
    assert QualityScore(0.65, 0, 0, 0, 0, 0).grade() == "B"
    assert QualityScore(0.45, 0, 0, 0, 0, 0).grade() == "C"
    assert QualityScore(0.25, 0, 0, 0, 0, 0).grade() == "D"


def test_rank_candidates_sorts_best_first():
    from ariadne.discovery.scoring import rank_candidates
    from ariadne.discovery.operations.candidate_store import Candidate
    a = Candidate(key="a", ra=0, dec=0, rate_arcsec_hr=1.0,
                  first_seen_mjd=60450, last_seen_mjd=60460, n_runs=3,
                  rms_history=[[60450, 1.0]])
    b = Candidate(key="b", ra=0, dec=0, rate_arcsec_hr=1.0,
                  first_seen_mjd=60450, last_seen_mjd=60460, n_runs=1,
                  rms_history=[[60450, 11.0]])
    ranked = rank_candidates([b, a])           # intentionally reversed order
    assert ranked[0][0].key == "a", "best (low-RMS, many-run) should come first"


# ----------------------------- MULTI-CONE NIGHTLY -----------------------------

def test_nightly_config_cones_field_present():
    from ariadne.discovery.operations.nightly import NightlyConfig
    cfg = NightlyConfig(store_path="x.json",
                        cones=[(180.0, 20.0, 5.0), (10.0, -10.0, 3.0)])
    assert len(cfg.cones) == 2
    assert cfg.cones[0] == (180.0, 20.0, 5.0)


def test_nightly_synthetic_runs_end_to_end(tmp_path):
    from ariadne.discovery.operations.nightly import NightlyConfig, run_nightly
    cfg = NightlyConfig(
        store_path=str(tmp_path / "store.json"),
        source="synthetic",
        ra=180.0, dec=20.0, radius_deg=2.0,
        rms_threshold_arcsec=1e9,                          # accept everything
        do_xmatch=False,                                   # skip SkyBoT
    )
    summary = run_nightly(cfg)
    assert summary["n_cones"] == 1
    assert summary["alerts"] > 0


def test_nightly_multi_cone_aggregates(tmp_path):
    from ariadne.discovery.operations.nightly import NightlyConfig, run_nightly
    cfg = NightlyConfig(
        store_path=str(tmp_path / "store.json"),
        source="synthetic",
        cones=[(180.0, 20.0, 1.0), (60.0, -10.0, 1.0)],
        rms_threshold_arcsec=1e9, do_xmatch=False,
    )
    summary = run_nightly(cfg)
    assert summary["n_cones"] == 2
    assert len(summary["per_cone"]) == 2


# ----------------------------- HELIOLINC WIRING -------------------------------

def test_helio_linc_run_pipeline_smoke():
    """HelioLinC linker path runs end-to-end through run_pipeline without crashing."""
    from ariadne.discovery import realtime
    from ariadne.discovery.brokers.base import synthesise_keplerian_alerts
    alerts = synthesise_keplerian_alerts(orbits=[
        {"a_au": 60, "e": 0.05, "i": 8, "Omega": 30, "omega": 50, "M": 180},
    ], n_interlopers=20)
    # don't care about whether anything passes -- just verify no crash + no xmatch path
    res = realtime.run_pipeline(alerts, do_xmatch=False, use_helio_linc=True,
                                  rms_threshold_arcsec=1e9)
    assert isinstance(res, list)


# ----------------------------- WEB DASHBOARD ----------------------------------

def test_dashboard_factory_creates_flask_app(tmp_path):
    flask = pytest.importorskip("flask")
    from ariadne.discovery.dashboard import create_app
    # write a tiny store
    store_path = tmp_path / "store.json"
    store_path.write_text(json.dumps({
        "version": 1, "saved_at_unix": 1.0, "n_candidates": 1,
        "candidates": [{
            "key": "001.0000_+02.0000_01.00", "ra": 1.0, "dec": 2.0,
            "rate_arcsec_hr": 1.0, "first_seen_mjd": 60450.0,
            "last_seen_mjd": 60452.0, "n_runs": 2,
            "rms_history": [[60450.0, 1.5], [60452.0, 1.6]],
            "orbit_state": None, "skybot_names": [], "status": "active",
            "meta": {},
        }]
    }))
    app = create_app(store_path)
    client = app.test_client()
    home = client.get("/")
    assert home.status_code == 200
    assert b"Ariadne" in home.data
    assert b"001.0000_+02.0000_01.00" in home.data

    detail = client.get("/candidate/001.0000_+02.0000_01.00")
    assert detail.status_code == 200
    assert b"grade" in detail.data

    api_store = client.get("/api/store")
    assert api_store.status_code == 200
    payload = json.loads(api_store.data)
    assert payload["n_candidates"] == 1

    api_score = client.get("/api/score")
    assert api_score.status_code == 200
    score_payload = json.loads(api_score.data)
    assert len(score_payload) == 1
    assert score_payload[0]["key"] == "001.0000_+02.0000_01.00"
    assert score_payload[0]["grade"] in ("A", "B", "C", "D")

    skymap = client.get("/skymap.svg")
    assert skymap.status_code == 200
    assert skymap.mimetype == "image/svg+xml"
    assert b"<svg" in skymap.data and b"</svg>" in skymap.data
    # candidate should be rendered as a circle somewhere
    assert b"<circle" in skymap.data


def test_dashboard_404_unknown_candidate(tmp_path):
    flask = pytest.importorskip("flask")
    from ariadne.discovery.dashboard import create_app
    store_path = tmp_path / "store.json"
    store_path.write_text(json.dumps({
        "version": 1, "saved_at_unix": 1.0, "n_candidates": 0, "candidates": []
    }))
    app = create_app(store_path)
    client = app.test_client()
    assert client.get("/candidate/nope").status_code == 404
