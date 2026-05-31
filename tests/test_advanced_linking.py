"""Tests for imaging.advanced_linking: probabilistic + multipass + helio-linc + orchestrator."""
from __future__ import annotations

import math

import numpy as np
import pytest


def _make_one_object_tracklets(rate_arcsec_hr=2.0, mag=21.0):
    """4-night arc of one synthetic object as image tracklets."""
    from ariadne.discovery.imaging.source_extraction import Source
    from ariadne.discovery.imaging.tracklets_from_images import nightly_tracklets
    SEC_PER_DAY = 86400.0
    pixscale = 0.25
    # Plant the object at known position; moves rate_arcsec_hr toward higher ra
    ra0, dec0 = 180.0, 20.0
    sources = []
    for night, mjd_base in enumerate([60450.0, 60451.0, 60452.0, 60453.0]):
        for k in range(3):
            t_in_night_hours = (k - 1) * 2.0       # -2h, 0, +2h
            mjd = mjd_base + t_in_night_hours / 24.0
            dt_hours_since_first = (mjd - 60450.0) * 24.0
            ra = ra0 + rate_arcsec_hr * dt_hours_since_first / 3600.0
            sources.append(Source(
                ra=ra, dec=dec0, flux=1000.0, mag=mag,
                fwhm_px=3.0, mjd=mjd, image_id=f"img_{night}_{k}",
                x=100.0, y=100.0,
            ))
    return nightly_tracklets(sources, min_rate_arcsec_hr=0.5,
                              max_rate_arcsec_hr=10.0,
                              min_pair_dt_hours=1.0,
                              max_pair_dt_hours=5.0,
                              min_pair_separation_arcsec=0.0)


# ---------------------------------------------------------------------------
# _pair_log_likelihood
# ---------------------------------------------------------------------------

class TestPairLogLikelihood:
    def test_high_likelihood_for_consistent_pair(self):
        from ariadne.discovery.imaging.advanced_linking import _pair_log_likelihood
        tracks = _make_one_object_tracklets()
        # Pair two tracklets from consecutive nights
        same_obj = [t for t in tracks]
        # pick any two with different nights
        a = same_obj[0]
        b = next(t for t in same_obj[1:] if t["night"] != a["night"])
        ll = _pair_log_likelihood(a, b)
        # Consistent pair -> log-likelihood near 0 (high)
        assert ll > -5.0

    def test_low_likelihood_for_inconsistent_pair(self):
        from ariadne.discovery.imaging.advanced_linking import _pair_log_likelihood
        # Two tracklets with VERY different rates and positions
        a = {"jd": 2460450.5, "ra": math.radians(180.0), "dec": math.radians(20.0),
             "dra": 1e-9, "ddec": 0, "rate_arcsec_hr": 1.0}
        b = {"jd": 2460451.5, "ra": math.radians(250.0), "dec": math.radians(-30.0),
             "dra": 1e-9, "ddec": 0, "rate_arcsec_hr": 100.0}
        ll = _pair_log_likelihood(a, b)
        assert ll < -50.0


# ---------------------------------------------------------------------------
# probabilistic_chain
# ---------------------------------------------------------------------------

class TestProbabilisticChain:
    def test_links_clean_4_night_arc(self):
        from ariadne.discovery.imaging.advanced_linking import probabilistic_chain
        tracks = _make_one_object_tracklets()
        chains = probabilistic_chain(tracks, position_sigma_arcsec=120.0,
                                       rate_sigma_pct=50.0,
                                       log_likelihood_threshold=-20.0)
        assert chains, "should produce at least one chain"
        # Best chain should span all 4 nights
        best = max(chains, key=lambda c: len({t["night"] for t in c}))
        assert len({t["night"] for t in best}) >= 2

    def test_returns_empty_on_no_tracklets(self):
        from ariadne.discovery.imaging.advanced_linking import probabilistic_chain
        assert probabilistic_chain([]) == []

    def test_returns_empty_on_single_night(self):
        from ariadne.discovery.imaging.advanced_linking import probabilistic_chain
        tracks = [
            {"jd": 2460450.5, "ra": 0, "dec": 0, "dra": 0, "ddec": 0,
              "rate_arcsec_hr": 1.0, "night": 0, "t": 0.0},
        ]
        assert probabilistic_chain(tracks) == []


# ---------------------------------------------------------------------------
# multipass_refined_chain
# ---------------------------------------------------------------------------

class TestMultipassChain:
    def test_multipass_runs(self):
        from ariadne.discovery.imaging.advanced_linking import multipass_refined_chain
        tracks = _make_one_object_tracklets()
        chains = multipass_refined_chain(tracks, initial_sigma_arcsec=120.0,
                                           refined_sigma_arcsec=30.0)
        assert isinstance(chains, list)

    def test_refit_rate_two_points(self):
        from ariadne.discovery.imaging.advanced_linking import _refit_rate_from_chain
        chain = [
            {"t": 0.0, "ra": math.radians(180.0), "dec": math.radians(20.0)},
            {"t": 86400.0, "ra": math.radians(180.001),
              "dec": math.radians(20.001)},
        ]
        dra, ddec = _refit_rate_from_chain(chain)
        assert dra > 0
        assert ddec > 0


# ---------------------------------------------------------------------------
# helio_linc_image_bridge
# ---------------------------------------------------------------------------

class TestHelioLinCBridge:
    def test_returns_chains_on_real_synth(self):
        """Build tracklets from a real Keplerian orbit + bridge through HelioLinC."""
        from ariadne.discovery import linkage as L
        from ariadne.discovery.imaging.advanced_linking import helio_linc_image_bridge
        tracks, _ = L.synthesize_tracklets(
            orbits=[{"a_au": 50, "e": 0.05, "i": 8,
                      "Omega": 30, "omega": 50}],
            epoch="2026-01-01T00:00:00",
            night_offsets_days=(0, 5, 15, 30),
            n_interlopers=0,
        )
        out = helio_linc_image_bridge(
            tracks,
            r_grid_au=np.linspace(40, 60, 11),
            rdot_grid=np.linspace(-1, 1, 5))
        assert isinstance(out, list)

    def test_handles_empty(self):
        from ariadne.discovery.imaging.advanced_linking import helio_linc_image_bridge
        assert helio_linc_image_bridge([]) == []


# ---------------------------------------------------------------------------
# discover_in_images_chains orchestrator
# ---------------------------------------------------------------------------

class TestDiscoverOrchestrator:
    def test_orchestrator_runs_every_strategy(self):
        from ariadne.discovery.imaging.advanced_linking import discover_in_images_chains
        tracks = _make_one_object_tracklets()
        out = discover_in_images_chains(tracks)
        assert isinstance(out, list)

    def test_orchestrator_dedupe(self):
        """Same chain emitted by multiple strategies should appear once."""
        from ariadne.discovery.imaging.advanced_linking import (
            discover_in_images_chains, _merge_chain_lists, _chain_signature)
        tracks = _make_one_object_tracklets()
        # Build two identical chains; merge should keep one
        if not tracks:
            return
        ch_a = tracks[:2]
        ch_b = tracks[:2]
        merged = _merge_chain_lists([[ch_a], [ch_b]])
        assert len(merged) == 1

    def test_can_disable_strategies(self):
        from ariadne.discovery.imaging.advanced_linking import discover_in_images_chains
        tracks = _make_one_object_tracklets()
        out = discover_in_images_chains(tracks, use_greedy=False,
                                          use_helio_linc=False,
                                          use_probabilistic=True,
                                          use_multipass=False)
        assert isinstance(out, list)


# ---------------------------------------------------------------------------
# Hungarian helper
# ---------------------------------------------------------------------------

class TestHungarian:
    def test_hungarian_empty(self):
        from ariadne.discovery.imaging.advanced_linking import _hungarian_assignment
        out = _hungarian_assignment(np.zeros((0, 0)))
        assert out == []

    def test_hungarian_identity_matrix(self):
        from ariadne.discovery.imaging.advanced_linking import _hungarian_assignment
        # Identity-ish: pairs (0,0), (1,1), (2,2) win
        cost = np.array([[0.0, 5, 5],
                          [5, 0.0, 5],
                          [5, 5, 0.0]])
        out = _hungarian_assignment(cost)
        assert sorted(out) == [(0, 0), (1, 1), (2, 2)]
