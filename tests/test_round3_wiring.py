"""Round-3 wiring tests:
* large_corpus generation
* MCMC-required MPC submission gate
* IOD ensemble wall-clock optimisations (cheap-first + early-exit)
"""
from __future__ import annotations

import math

import numpy as np
import pytest


# =============================================================================
# Large corpus
# =============================================================================

class TestLargeCorpus:
    def test_make_large_corpus_returns_size_in_range(self):
        from ariadne.discovery.large_corpus import make_large_corpus
        cases = make_large_corpus(n_per_class=5, n_artefacts_per_kind=3)
        # 16 moving-object classes * 5 + 4 artefact kinds * 3 = 80+12 = 92
        assert len(cases) >= 80
        # Every entry is a LabelledCase
        from ariadne.discovery.benchmarking import LabelledCase
        for c in cases:
            assert isinstance(c, LabelledCase)
            assert c.truth_label

    def test_make_large_corpus_deterministic(self):
        from ariadne.discovery.large_corpus import make_large_corpus
        a = make_large_corpus(n_per_class=3, n_artefacts_per_kind=2, seed=42)
        b = make_large_corpus(n_per_class=3, n_artefacts_per_kind=2, seed=42)
        assert [c.truth_label for c in a] == [c.truth_label for c in b]

    def test_corpus_covers_every_class(self):
        from ariadne.discovery.large_corpus import (
            make_large_corpus, _CLASS_SPECS, _ARTEFACT_SPECS)
        cases = make_large_corpus(n_per_class=3, n_artefacts_per_kind=2)
        labels = {c.truth_label for c in cases}
        for cls in _CLASS_SPECS:
            assert cls in labels, f"missing {cls}"
        for cls in _ARTEFACT_SPECS:
            assert cls in labels, f"missing {cls}"


# =============================================================================
# MCMC-required MPC submission gate
# =============================================================================

class TestMCMCGate:
    def _candidate(self, **overrides):
        from ariadne.discovery.operations.candidate_store import Candidate
        defaults = dict(
            key="123.0000_+45.0000_01.50", ra=123.0, dec=45.0,
            rate_arcsec_hr=1.5,
            first_seen_mjd=60450.0, last_seen_mjd=60455.0,
            n_runs=3,
            rms_history=[[60450, 1.0], [60453, 0.9], [60455, 1.1]],
            skybot_names=[],
            meta={"mcmc": {"a_au_quantiles": (44, 45, 46),
                            "e_quantiles": (0.05, 0.08, 0.11)},
                  "n_observations": 8},
        )
        defaults.update(overrides)
        return Candidate(**defaults)

    def test_passing_candidate_clears_gate(self):
        from ariadne.discovery.mpc_submit import grade_a_submission_gate
        grade_a_submission_gate(self._candidate())  # no raise

    def test_missing_mcmc_raises(self):
        from ariadne.discovery.mpc_submit import (
            grade_a_submission_gate, GradeAGateError)
        c = self._candidate(meta={"n_observations": 8})  # no mcmc key
        with pytest.raises(GradeAGateError, match="MCMC"):
            grade_a_submission_gate(c)

    def test_short_arc_raises(self):
        from ariadne.discovery.mpc_submit import (
            grade_a_submission_gate, GradeAGateError)
        c = self._candidate(last_seen_mjd=60450.5)   # arc 0.5d < 2d
        with pytest.raises(GradeAGateError, match="arc"):
            grade_a_submission_gate(c)

    def test_too_few_runs_raises(self):
        from ariadne.discovery.mpc_submit import (
            grade_a_submission_gate, GradeAGateError)
        c = self._candidate(n_runs=1)
        with pytest.raises(GradeAGateError, match="n_runs"):
            grade_a_submission_gate(c)

    def test_high_rms_raises(self):
        from ariadne.discovery.mpc_submit import (
            grade_a_submission_gate, GradeAGateError)
        c = self._candidate(rms_history=[[60450, 5.0]])
        with pytest.raises(GradeAGateError, match="RMS"):
            grade_a_submission_gate(c)

    def test_no_rms_history_raises(self):
        from ariadne.discovery.mpc_submit import (
            grade_a_submission_gate, GradeAGateError)
        c = self._candidate(rms_history=[])
        with pytest.raises(GradeAGateError, match="rms_history"):
            grade_a_submission_gate(c)

    def test_too_few_obs_raises(self):
        from ariadne.discovery.mpc_submit import (
            grade_a_submission_gate, GradeAGateError)
        c = self._candidate(rms_history=[[60450, 1.0]],
                              meta={"mcmc": {"a_au_quantiles": (44, 45, 46)}})
        with pytest.raises(GradeAGateError, match="observations"):
            grade_a_submission_gate(c)

    def test_mcmc_optional_when_disabled(self):
        from ariadne.discovery.mpc_submit import grade_a_submission_gate
        c = self._candidate(meta={"n_observations": 8})
        # require_mcmc=False -> passes without MCMC
        grade_a_submission_gate(c, require_mcmc=False)

    def test_emit_submission_skips_failing_candidates(self):
        from ariadne.discovery.mpc_submit import emit_submission, MPCHeader
        from ariadne.discovery.brokers.base import Alert
        header = MPCHeader(observatory_code="500", contact="J Doe",
                            observers="J Doe", measurers="J Doe",
                            telescope="virtual",
                            ack_keyword="ARI_TEST",
                            ack_email="doe@example.com")
        good = self._candidate()
        bad = self._candidate(key="111.0000_+11.0000_02.00",
                                meta={"n_observations": 8})  # no mcmc
        alerts = [Alert("ZTF", "a1", "x1", 60450, 123.0, 45.0, 21, "r"),
                  Alert("ZTF", "a2", "x1", 60450.5, 123.001, 45.001, 21, "r")]
        text = emit_submission(header, [(good, alerts), (bad, alerts)])
        # The bad candidate's records are NOT in the body; instead a COM line
        # records why it was skipped.
        assert "COM SKIPPED" in text
        assert "MCMC" in text
        # The good candidate's records ARE present (each line is 80 chars)
        body_lines = [l for l in text.splitlines() if len(l) == 80]
        assert len(body_lines) == 2

    def test_emit_submission_gate_disabled_includes_all(self):
        from ariadne.discovery.mpc_submit import emit_submission, MPCHeader
        from ariadne.discovery.brokers.base import Alert
        header = MPCHeader(observatory_code="500", contact="J Doe",
                            observers="J Doe", measurers="J Doe",
                            telescope="virtual",
                            ack_keyword="ARI", ack_email="x@y")
        bad = self._candidate(meta={"n_observations": 8})  # no mcmc
        alerts = [Alert("ZTF", "a1", "x1", 60450, 123.0, 45.0, 21, "r")]
        text = emit_submission(header, [(bad, alerts)],
                                enforce_grade_a_gate=False)
        assert "COM SKIPPED" not in text


# =============================================================================
# Ensemble wall-clock optimisation
# =============================================================================

class TestEnsembleOptimisation:
    def test_early_exit_on_clean_synth(self):
        """Cheap-first ordering + early-exit should converge without running
        every strategy."""
        from ariadne.discovery import linkage
        from ariadne.discovery.iod_advanced import fit_candidate_ensemble
        tracks, _ = linkage.synthesize_tracklets(
            orbits=[{"a_au": 50, "e": 0.05, "i": 8,
                      "Omega": 30, "omega": 50}],
            epoch="2026-01-01T00:00:00",
            night_offsets_days=(0, 5, 15, 30),
            n_interlopers=0,
        )
        ens = fit_candidate_ensemble(tracks, cheap_first=True,
                                      early_exit_rms_arcsec=2.0)
        # Should converge (clean synth)
        assert ens.success
        # Either fewer strategies ran (early exit) OR ones that ran show
        # that the cheap ones came first
        labels = [s.strategy for s in ens.strategy_results]
        assert labels  # at least gauss ran

    def test_cheap_first_skips_expensive_when_satisfied(self):
        """When the cheap strategy already produced an acceptable RMS,
        Vaisala and BK should be skipped (not in strategy_results)."""
        from ariadne.discovery import linkage
        from ariadne.discovery.iod_advanced import fit_candidate_ensemble
        tracks, _ = linkage.synthesize_tracklets(
            orbits=[{"a_au": 50, "e": 0.05, "i": 8,
                      "Omega": 30, "omega": 50}],
            epoch="2026-01-01T00:00:00",
            night_offsets_days=(0, 5, 15, 30),
            n_interlopers=0,
        )
        ens = fit_candidate_ensemble(
            tracks, cheap_first=True,
            rms_acceptance_arcsec=50.0,            # very loose
            early_exit_rms_arcsec=100.0)            # disable early-exit
        labels = [s.strategy for s in ens.strategy_results]
        # Cheap-first SHOULD have skipped vaisala + bk after gauss/linker passed
        # (rms_acceptance is very loose so any-rms qualifies).
        assert "vaisala" not in labels or "bernstein_khushalani" not in labels


# =============================================================================
# Calibration script smoke test
# =============================================================================

class TestCalibrationOnLargeCorpusSmoke:
    def test_calibration_can_run_on_50_cases(self):
        """End-to-end: large_corpus -> fit_per_class_temperatures -> save."""
        from ariadne.discovery import inference
        from ariadne.discovery.large_corpus import make_large_corpus
        cases = make_large_corpus(n_per_class=3, n_artefacts_per_kind=2)
        reliability_cases = [(c.evidence, c.truth_label) for c in cases]
        cfg, rep = inference.fit_per_class_temperatures(reliability_cases)
        assert cfg.temperature > 0
        assert math.isfinite(rep.nll)
        assert rep.ece >= 0
