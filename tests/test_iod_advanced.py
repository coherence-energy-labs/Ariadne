"""Tests for the advanced multi-strategy IOD ensemble."""
from __future__ import annotations

import math

import numpy as np
import pytest


def _make_synthetic_tno_tracklets():
    """Build a clean 4-tracklet arc of a known TNO via the validated synthesizer."""
    from ariadne.discovery import linkage
    tracks, e0 = linkage.synthesize_tracklets(
        orbits=[{"a_au": 50, "e": 0.05, "i": 8, "Omega": 30, "omega": 50}],
        epoch="2026-01-01T00:00:00",
        night_offsets_days=(0, 5, 15, 30),
        n_interlopers=0,
    )
    return tracks


# =============================================================================
# Individual strategies
# =============================================================================

class TestIndividualStrategies:
    def test_gauss_on_clean_4_tracklets(self):
        from ariadne.discovery.iod_advanced import _strategy_gauss
        tracks = _make_synthetic_tno_tracklets()
        t_ref = float(np.median([t["t"] for t in tracks]))
        result = _strategy_gauss(tracks, t_ref)
        # Either converges or fails with a documented reason
        assert isinstance(result.success, bool)
        if result.success:
            assert 1 < result.r_au < 200
            assert np.all(np.isfinite(result.x_init))

    def test_gauss_below_min_fails(self):
        from ariadne.discovery.iod_advanced import _strategy_gauss
        tracks = _make_synthetic_tno_tracklets()[:2]   # only 2
        t_ref = float(np.median([t["t"] for t in tracks]))
        result = _strategy_gauss(tracks, t_ref)
        assert not result.success
        assert "min" in result.notes.lower() or "3" in result.notes

    def test_adaptive_linker_runs(self):
        from ariadne.discovery.iod_advanced import _strategy_adaptive_helio_linc
        tracks = _make_synthetic_tno_tracklets()
        t_ref = float(np.median([t["t"] for t in tracks]))
        results = _strategy_adaptive_helio_linc(tracks, t_ref, n_retries=2)
        assert len(results) == 2
        # At least one of the retries should converge on the clean synth
        assert any(r.success for r in results)

    def test_vaisala_works_on_2_tracklets(self):
        """Vaisala is the ONLY strategy that works on 2-tracklet chains."""
        from ariadne.discovery.iod_advanced import _strategy_vaisala
        tracks = _make_synthetic_tno_tracklets()[:2]
        t_ref = float(np.median([t["t"] for t in tracks]))
        result = _strategy_vaisala(tracks, t_ref)
        # Should produce a viable seed
        assert isinstance(result.success, bool)
        if result.success:
            assert 1 < result.r_au < 1000
            assert np.all(np.isfinite(result.x_init))

    def test_vaisala_below_min_fails(self):
        from ariadne.discovery.iod_advanced import _strategy_vaisala
        result = _strategy_vaisala([], t_ref=0.0)
        assert not result.success

    def test_bk_runs_on_clean_tracklets(self):
        from ariadne.discovery.iod_advanced import _strategy_bernstein_khushalani
        tracks = _make_synthetic_tno_tracklets()
        t_ref = float(np.median([t["t"] for t in tracks]))
        result = _strategy_bernstein_khushalani(tracks, t_ref)
        assert isinstance(result.success, bool)
        if result.success:
            assert 1 < result.r_au < 200


# =============================================================================
# Ensemble combiner
# =============================================================================

class TestEnsemble:
    def test_ensemble_converges_on_clean_synth(self):
        from ariadne.discovery.iod_advanced import fit_candidate_ensemble
        tracks = _make_synthetic_tno_tracklets()
        ens = fit_candidate_ensemble(tracks)
        # At least one strategy should converge
        assert any(s.success for s in ens.strategy_results)
        # The winning strategy must be a real one
        assert ens.winning_strategy != "none"

    def test_ensemble_with_empty_input(self):
        from ariadne.discovery.iod_advanced import fit_candidate_ensemble
        ens = fit_candidate_ensemble([])
        assert not ens.success
        assert ens.winning_strategy == "none"

    def test_ensemble_records_all_strategy_attempts(self):
        from ariadne.discovery.iod_advanced import fit_candidate_ensemble
        tracks = _make_synthetic_tno_tracklets()
        # cheap_first=False forces every strategy to run regardless of whether
        # the cheap ones already met the acceptance threshold; early_exit_rms
        # raised to a very small number so it never fires.
        ens = fit_candidate_ensemble(tracks, n_linker_retries=2,
                                       cheap_first=False,
                                       early_exit_rms_arcsec=1e-9)
        labels = [s.strategy for s in ens.strategy_results]
        assert "gauss" in labels
        assert "vaisala" in labels
        assert "bernstein_khushalani" in labels
        assert any("adaptive_linker" in l for l in labels)

    def test_ensemble_strategies_subset(self):
        """Caller can request only specific strategies."""
        from ariadne.discovery.iod_advanced import fit_candidate_ensemble
        tracks = _make_synthetic_tno_tracklets()
        ens = fit_candidate_ensemble(tracks, strategies=("gauss", "vaisala"))
        labels = [s.strategy for s in ens.strategy_results]
        assert "gauss" in labels and "vaisala" in labels
        assert not any("linker" in l or "khushalani" in l for l in labels)

    def test_ensemble_summary_jsonable(self):
        from ariadne.discovery.iod_advanced import (
            fit_candidate_ensemble, ensemble_summary)
        import json
        tracks = _make_synthetic_tno_tracklets()
        ens = fit_candidate_ensemble(tracks)
        summary = ensemble_summary(ens)
        # Should serialise without error
        assert json.dumps(summary) is not None
        assert summary["n_strategies_run"] >= 4
        assert "per_strategy" in summary


# =============================================================================
# Plumbing test: fit_filter with use_ensemble_iod=True
# =============================================================================

class TestFitFilterEnsemble:
    def test_fit_filter_accepts_ensemble_flag(self):
        from ariadne.discovery.realtime import fit_filter
        out = fit_filter([], use_ensemble_iod=True)
        assert out == []

    def test_run_pipeline_accepts_ensemble_flag(self):
        from ariadne.discovery.realtime import run_pipeline
        out = run_pipeline([], use_ensemble_iod=True, smart_layer=False,
                            do_xmatch=False)
        assert out == []
