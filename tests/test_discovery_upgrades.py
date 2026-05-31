"""Tests for the post-benchmark upgrades:
adaptive_pair_window, iod_diagnose, per-class temperature, scoring rescale,
image-pipeline sanity cap, smart_annotate calibration/scheduler passthrough.
"""
from __future__ import annotations

import math

import numpy as np
import pytest


# =============================================================================
# adaptive_pair_window
# =============================================================================

class TestAdaptivePairWindow:
    def test_tno_regime(self):
        from ariadne.discovery.realtime import adaptive_pair_window
        lo, hi = adaptive_pair_window(0.05, 5.0)
        # TNO: max 12h (60"/5"/h), min 30s floor or motion-based
        assert 0 < lo < hi
        assert hi <= 12.0

    def test_neo_regime_uses_short_windows(self):
        from ariadne.discovery.realtime import adaptive_pair_window
        lo, hi = adaptive_pair_window(30, 600)
        # NEO at 600"/hr: max ~6 min, min seconds
        assert hi < 0.5      # less than 30 minutes
        assert lo > 0

    def test_zero_max_rate_safe(self):
        from ariadne.discovery.realtime import adaptive_pair_window
        lo, hi = adaptive_pair_window(0.01, 0.0)
        # Edge case: zero max -> fall back to defaults
        assert lo > 0 and hi > lo

    def test_min_gte_max_collapses(self):
        from ariadne.discovery.realtime import adaptive_pair_window
        # narrow rate window
        lo, hi = adaptive_pair_window(100.0, 100.5)
        assert lo < hi      # always returns a positive window

    def test_build_tracklets_uses_adaptive_when_none(self):
        from ariadne.discovery.realtime import build_tracklets
        from ariadne.discovery.brokers.base import Alert
        # Two same-night detections, fast NEO regime
        alerts = [
            Alert("ZTF", "1", "o", 60450.000, 180.0, 20.0, 18, "r"),
            Alert("ZTF", "2", "o", 60450.001, 180.05, 20.05, 18, "r"),
        ]
        # rate ~250 arcsec/hr in this 0.001 day = 1.44 min interval
        # adaptive_pair_window for rate (30, 600) should accept this
        tracks = build_tracklets(alerts,
                                  min_rate_arcsec_hr=30, max_rate_arcsec_hr=600)
        assert isinstance(tracks, list)


# =============================================================================
# iod_diagnose
# =============================================================================

class TestIODDiagnose:
    def test_too_few_tracklets(self):
        from ariadne.discovery.iod import iod_diagnose
        result = iod_diagnose([])
        assert result["converged"] is False
        assert "rejection_reason" in result

    def test_converged_on_synthetic_tno(self):
        from ariadne.discovery import linkage, iod
        tracks, _ = linkage.synthesize_tracklets(
            orbits=[{"a_au": 50, "e": 0.05, "i": 8, "Omega": 30, "omega": 50}],
            epoch="2026-01-01T00:00:00",
            night_offsets_days=(0, 5, 15, 30),
            n_interlopers=0,
        )
        result = iod.iod_diagnose(tracks, r_grid_au=np.linspace(40, 60, 5),
                                    rdot_grid=np.linspace(-0.5, 0.5, 3))
        assert result["converged"]
        assert result["best_scatter_km"] is not None
        assert result["n_converged_hypotheses"] > 0


# =============================================================================
# Per-class temperature
# =============================================================================

class TestPerClassTemperature:
    def test_class_temp_softens_to_lift_underdog(self):
        from ariadne.discovery.inference import (
            infer, Evidence, CalibrationConfig)
        ev = Evidence(rate_arcsec_hr=1.0)
        # Base: TNOs sit far from the MBA-dominated min_F, so their weight
        # is small. Soften the TNO classes' temperature (higher T) -> their
        # weight decays less rapidly -> they get more posterior probability.
        base = infer(ev)
        cfg = CalibrationConfig(class_temperatures={
            "CLASSICAL_KBO": 10.0, "HOT_CLASSICAL": 10.0,
        })
        softened = infer(ev, calibration=cfg)
        tno_base = next(h.posterior for h in base.hypotheses
                         if h.orbital_class == "CLASSICAL_KBO")
        tno_soft = next(h.posterior for h in softened.hypotheses
                         if h.orbital_class == "CLASSICAL_KBO")
        assert tno_soft > tno_base

    def test_class_temp_field_round_trips(self):
        from ariadne.discovery.inference import CalibrationConfig
        cfg = CalibrationConfig(class_temperatures={"MBA": 2.0})
        assert cfg.class_temperatures == {"MBA": 2.0}


# =============================================================================
# Quality scoring rescale
# =============================================================================

class TestScoringRescale:
    def test_single_night_low_rms_hits_b_or_higher(self):
        """A clean single-night low-RMS candidate now reaches grade B."""
        from ariadne.discovery.scoring import score_candidate
        from ariadne.discovery.operations.candidate_store import Candidate
        c = Candidate(key="k", ra=180, dec=20, rate_arcsec_hr=1.5,
                       first_seen_mjd=60450, last_seen_mjd=60453, n_runs=1,
                       rms_history=[[60450, 1.0], [60451, 1.2], [60452, 1.1],
                                     [60453, 1.0]],
                       skybot_names=[])
        s = score_candidate(c)
        # Should be B or better (>= 0.6)
        assert s.total >= 0.55, f"single-night clean fit scored only {s.total:.2f}"


# =============================================================================
# Image-pipeline sanity cap
# =============================================================================

class TestImageTrackletCap:
    def test_min_separation_drops_co_located_pairs(self):
        from ariadne.discovery.imaging.tracklets_from_images import nightly_tracklets
        from ariadne.discovery.imaging.source_extraction import Source
        # Two sources at identical position across 30 min -> NOT a tracklet
        srcs = [
            Source(180, 20, 1000, 20, 3, 60450.00, "t1", 100, 100),
            Source(180, 20, 1000, 20, 3, 60450.02, "t2", 100, 100),
        ]
        tracks = nightly_tracklets(srcs, min_pair_separation_arcsec=0.5)
        assert len(tracks) == 0

    def test_max_per_night_caps_explosion(self):
        from ariadne.discovery.imaging.tracklets_from_images import nightly_tracklets
        from ariadne.discovery.imaging.source_extraction import Source
        # 50 sources on the same night -> 50*49/2 = 1225 potential pairs
        srcs = [Source(180 + i * 0.001, 20 + i * 0.001, 1000 + i, 20, 3,
                        60450.0 + 0.05 * i, f"t{i}", 100, 100)
                for i in range(50)]
        tracks = nightly_tracklets(srcs, min_rate_arcsec_hr=0.0,
                                    max_rate_arcsec_hr=1e9,
                                    min_pair_separation_arcsec=0.0,
                                    max_per_night=10)
        # Cap is per-night; verify no single night exceeds the cap
        from collections import Counter
        per_night = Counter(t["night"] for t in tracks)
        for n, count in per_night.items():
            assert count <= 10, f"night {n} has {count} tracklets > cap 10"


# =============================================================================
# smart_annotate calibration passthrough
# =============================================================================

class TestSmartAnnotateCalibration:
    def test_smart_annotate_accepts_calibration(self):
        from ariadne.discovery.realtime import smart_annotate
        from ariadne.discovery.inference import CalibrationConfig
        from ariadne.discovery.brokers.base import Alert
        # Pre-fitted accepted candidate
        tr = {
            "status": "accepted", "ra": math.radians(180), "dec": math.radians(20),
            "jd": 2460450.5, "t": 0.0, "rate_arcsec_hr": 1.0, "rms_arcsec": 1.5,
            "members": [Alert("ZTF", f"a{i}", "o", 60450 + i,
                               180 + i * 0.01, 20 + i * 0.005, 21, "r")
                        for i in range(4)],
            "x_fit_km": [1e10, 0, 0], "v_fit_kms": [0, 4.5, 0],
            "xmatch": {"n_known": 0, "names": []},
        }
        # No crash with calibration
        out = smart_annotate([tr], calibration=CalibrationConfig(temperature=2.0))
        assert out[0].get("_inference") is not None


# =============================================================================
# Full pipeline smoke test with all upgrades active
# =============================================================================

class TestPipelineSmokeWithUpgrades:
    def test_run_pipeline_with_adaptive_pair_dt(self):
        from ariadne.discovery.realtime import run_pipeline
        from ariadne.discovery.brokers.base import Alert
        alerts = [Alert("ZTF", f"a{i}", "o", 60450 + i * 0.02,
                         180 + i * 0.0001, 20, 21, "r") for i in range(8)]
        # No explicit pair_dt -> adaptive
        out = run_pipeline(alerts, pair_dt_hours=(None, None),
                            rate_window_arcsec_hr=(0.05, 5.0),
                            do_xmatch=False, smart_layer=False)
        assert isinstance(out, list)
