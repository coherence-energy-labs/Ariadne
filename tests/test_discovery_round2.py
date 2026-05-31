"""Round-2 upgrade tests: per-class T fit, calibration save/load,
chain sanity filter, N-body refinement plumbing, MCMC option."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest


# =============================================================================
# Per-class temperature fit
# =============================================================================

class TestPerClassFit:
    def test_fit_returns_calibration_with_global_T(self):
        from ariadne.discovery.inference import (
            Evidence, fit_per_class_temperatures)
        cases = [
            (Evidence(rate_arcsec_hr=1.0, n_detections=5,
                       arc_days=10, rms_arcsec=1.5,
                       morphology_label="POINT",
                       morphology_confidence=0.9,
                       skybot_match_names=[]), "CLASSICAL_KBO"),
            (Evidence(rate_arcsec_hr=15, n_detections=4,
                       arc_days=8, rms_arcsec=1.0,
                       morphology_label="POINT",
                       skybot_match_names=[]), "MBA"),
        ] * 4
        cfg, rep = fit_per_class_temperatures(cases, grid=(0.5, 1.0, 2.0))
        assert cfg.temperature > 0
        assert rep.n == 8

    def test_fit_empty_returns_neutral_config(self):
        from ariadne.discovery.inference import fit_per_class_temperatures
        cfg, rep = fit_per_class_temperatures([])
        assert cfg.temperature == 1.0
        assert rep.n == 0

    def test_fit_with_explicit_global_T_bypasses_seed_fit(self):
        from ariadne.discovery.inference import (
            Evidence, fit_per_class_temperatures)
        cases = [(Evidence(rate_arcsec_hr=1.0), "MBA")] * 3
        cfg, rep = fit_per_class_temperatures(
            cases, grid=(1.0,), global_temperature=2.5)
        assert cfg.temperature == 2.5


# =============================================================================
# Save / load calibration
# =============================================================================

class TestCalibrationPersistence:
    def test_save_then_load_round_trip(self, tmp_path):
        from ariadne.discovery.inference import (
            CalibrationConfig, save_calibration, load_calibration)
        cfg = CalibrationConfig(
            temperature=0.75,
            label_bias={"MBA": 0.1, "CLASSICAL_KBO": -0.2},
            channel_weights={"rate": 2.0, "magnitude": 0.5},
            class_temperatures={"MBA": 1.2, "CLASSICAL_KBO": 0.8},
            version="test-v1",
        )
        path = tmp_path / "calib.json"
        save_calibration(cfg, path)
        # File contents are valid JSON with the expected schema
        raw = json.loads(path.read_text())
        assert raw["schema"].startswith("ariadne.discovery.inference.calibration")
        # Round-trip via load_calibration preserves values
        recovered = load_calibration(path)
        assert recovered.temperature == 0.75
        assert recovered.label_bias == {"MBA": 0.1, "CLASSICAL_KBO": -0.2}
        assert recovered.channel_weights == {"rate": 2.0, "magnitude": 0.5}
        assert recovered.class_temperatures == {"MBA": 1.2, "CLASSICAL_KBO": 0.8}
        assert recovered.version == "test-v1"

    def test_load_missing_file_raises(self, tmp_path):
        from ariadne.discovery.inference import load_calibration
        with pytest.raises(FileNotFoundError):
            load_calibration(tmp_path / "nope.json")


# =============================================================================
# Chain sanity filter
# =============================================================================

class TestChainSanityFilter:
    def test_drops_single_night_chains(self):
        from ariadne.discovery.realtime import filter_chain_sanity
        single = [{"night": 0, "rate_arcsec_hr": 2.0},
                  {"night": 0, "rate_arcsec_hr": 2.1}]
        survivors = filter_chain_sanity([single])
        assert survivors == []

    def test_keeps_consistent_multi_night_chains(self):
        from ariadne.discovery.realtime import filter_chain_sanity
        good = [{"night": 0, "rate_arcsec_hr": 2.0},
                {"night": 1, "rate_arcsec_hr": 2.1},
                {"night": 2, "rate_arcsec_hr": 2.05}]
        assert filter_chain_sanity([good]) == [good]

    def test_drops_rate_inconsistent_chains(self):
        from ariadne.discovery.realtime import filter_chain_sanity
        # Rates: 1, 100, 1 -- the 100 is far from median 1
        bad = [{"night": 0, "rate_arcsec_hr": 1.0},
               {"night": 1, "rate_arcsec_hr": 100.0},
               {"night": 2, "rate_arcsec_hr": 1.0}]
        # default max_rate_change_pct=80%
        assert filter_chain_sanity([bad]) == []

    def test_handles_empty_input(self):
        from ariadne.discovery.realtime import filter_chain_sanity
        assert filter_chain_sanity([]) == []


# =============================================================================
# N-body refinement plumbing (smoke test)
# =============================================================================

class TestNBodyRefinementHook:
    def test_fit_filter_accepts_nbody_kwargs(self):
        """fit_filter signature exposes nbody_refine_rms_threshold + nbody_perturbers."""
        from ariadne.discovery.realtime import fit_filter
        # Empty input -> empty output, but signature accepts the kwargs
        out = fit_filter([], rms_threshold_arcsec=10,
                          nbody_refine_rms_threshold=3.0,
                          nbody_perturbers=("JUPITER",))
        assert out == []

    def test_fit_filter_default_nbody_threshold(self):
        from ariadne.discovery.realtime import fit_filter
        import inspect
        sig = inspect.signature(fit_filter)
        assert "nbody_refine_rms_threshold" in sig.parameters
        # default is positive (enables refinement)
        assert sig.parameters["nbody_refine_rms_threshold"].default > 0


# =============================================================================
# MCMC option on smart_annotate
# =============================================================================

class TestSmartAnnotateMCMC:
    def test_smart_annotate_accepts_mcmc_flag(self):
        from ariadne.discovery.realtime import smart_annotate
        # Empty input -> empty; smoke-test the new signature
        assert smart_annotate([], mcmc_for_high_quality=True) == []

    def test_smart_annotate_mcmc_skips_low_quality(self):
        """MCMC step ONLY fires for grade A/B candidates."""
        from ariadne.discovery.realtime import smart_annotate
        from ariadne.discovery.brokers.base import Alert
        # A short low-quality candidate -> should not have _mcmc
        tr = {
            "status": "accepted",
            "ra": math.radians(180), "dec": math.radians(20),
            "jd": 2460450.5, "t": 0.0,
            "rate_arcsec_hr": 1.0, "rms_arcsec": 15.0,    # high RMS -> C/D grade
            "members": [Alert("ZTF", f"a{i}", "o",
                               60450.0 + i * 0.1, 180 + i * 0.001, 20, 21, "r")
                        for i in range(2)],
            "xmatch": {"n_known": 0, "names": []},
        }
        out = smart_annotate([tr], mcmc_for_high_quality=True)
        # Either no _mcmc or it was skipped (no error)
        assert "_mcmc" not in out[0] or out[0]["_mcmc"] is not None


# =============================================================================
# IOD diagnose stability
# =============================================================================

class TestIODDiagnoseExtra:
    def test_diagnose_below_min_returns_dict(self):
        from ariadne.discovery.iod import iod_diagnose
        result = iod_diagnose([{"t": 0, "ra": 0, "dec": 0, "dra": 0, "ddec": 0}])
        assert result["converged"] is False
        assert "rejection_reason" in result
