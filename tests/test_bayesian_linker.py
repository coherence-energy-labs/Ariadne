"""Tests for the hierarchical Bayesian chain linker."""
from __future__ import annotations

import math
import pytest


def _et(mjd: float) -> float:
    """MJD -> SPICE ET seconds."""
    return (mjd - 51544.5) * 86400.0


def test_orbital_prior_picks_correct_class():
    from ariadne.discovery.imaging.bayesian_linker import log_orbital_prior

    # NEO-like rate (~200 "/hr) -> dominant class 'neo'
    _, neo = log_orbital_prior(200.0)
    assert neo == "neo"
    # MBA-like rate (~30) -> 'mba_inner'
    _, mba = log_orbital_prior(30.0)
    assert mba == "mba_inner"
    # TNO-like rate (~2) -> 'tno_class'
    _, tno = log_orbital_prior(2.0)
    assert tno == "tno_class"
    # Slow scattered (~0.7) -> 'tno_scatter'
    _, scatter = log_orbital_prior(0.7)
    assert scatter == "tno_scatter"


def test_orbital_prior_returns_negative_log_probability():
    from ariadne.discovery.imaging.bayesian_linker import log_orbital_prior
    log_p, _ = log_orbital_prior(2.5)
    assert log_p < 0.0
    assert math.isfinite(log_p)


def test_iod_prior_higher_for_better_geometry():
    from ariadne.discovery.imaging.bayesian_linker import (
        log_iod_convergence_prior)
    # 5 epochs, 96 hours, 2"/hr, clean noise
    p_good = log_iod_convergence_prior(n_epochs=5, arc_hours=96.0,
                                          median_rate_arcsec_hr=2.0,
                                          sigma_arcsec=0.3)
    # 2 epochs, 4 hours, 2"/hr, noisy
    p_bad = log_iod_convergence_prior(n_epochs=2, arc_hours=4.0,
                                         median_rate_arcsec_hr=2.0,
                                         sigma_arcsec=1.0)
    assert p_good > p_bad
    # Both log-probabilities should be <= 0 (proper probs)
    assert p_good <= 0.0
    assert p_bad <= 0.0


def test_pair_likelihood_prefers_consistent_motion():
    from ariadne.discovery.imaging.bayesian_linker import log_pair_likelihood
    t_a = {"t": _et(60450.0), "ra": math.radians(180.0),
           "dec": math.radians(20.0),
           "rate_arcsec_hr": 5.0, "mag": 21.5}
    # b: 3 days later, moved 5"/hr * 72h = 360" along RA
    t_b_good = {"t": _et(60453.0),
                 "ra": math.radians(180.0 + 360 / 3600 / math.cos(math.radians(20))),
                 "dec": math.radians(20.0),
                 "rate_arcsec_hr": 5.0, "mag": 21.5}
    # b': 3 days later but 10x position offset and totally different rate
    t_b_bad = {"t": _et(60453.0),
                "ra": math.radians(180.0 + 0.5),       # 0.5 deg = 1800" off
                "dec": math.radians(20.0 + 0.3),
                "rate_arcsec_hr": 50.0, "mag": 21.5}
    log_good = log_pair_likelihood(t_a, t_b_good)
    log_bad = log_pair_likelihood(t_a, t_b_bad)
    assert log_good > log_bad


def test_chain_score_decomposes_components():
    from ariadne.discovery.imaging.bayesian_linker import score_chain_bayesian
    ch = [
        {"t": _et(60450 + k * 3.0),
         "jd": (60450 + k * 3.0) + 2400000.5,
         "ra": math.radians(180.0 + k * 0.01),
         "dec": math.radians(20.0),
         "rate_arcsec_hr": 3.0, "mag": 21.5}
        for k in range(3)
    ]
    sc = score_chain_bayesian(ch)
    assert sc.n_entries == 3
    assert sc.n_unique_epochs == 3
    assert sc.arc_hours == pytest.approx(6 * 24.0, abs=0.5)
    assert math.isfinite(sc.log_likelihood)
    # Sum of components should equal the total
    assert (sc.log_pair_total + sc.log_orbit_prior + sc.log_iod_prior
            == pytest.approx(sc.log_likelihood))


def test_score_chains_returns_sorted_descending():
    from ariadne.discovery.imaging.bayesian_linker import score_chains_bayesian

    def _ch(rate, n_epochs):
        return [
            {"t": _et(60450 + k * 3.0),
             "jd": (60450 + k * 3.0) + 2400000.5,
             "ra": math.radians(180.0 + k * 0.01),
             "dec": math.radians(20.0),
             "rate_arcsec_hr": rate, "mag": 21.5}
            for k in range(n_epochs)
        ]

    # Three chains of varying quality
    chains = [_ch(2.0, 1), _ch(2.0, 5), _ch(2.0, 3)]
    scored = score_chains_bayesian(chains)
    # Output should be sorted descending by log_likelihood
    for a, b in zip(scored[:-1], scored[1:]):
        assert a.log_likelihood >= b.log_likelihood


def test_filter_chains_by_likelihood_caps_output():
    from ariadne.discovery.imaging.bayesian_linker import (
        filter_chains_by_likelihood)

    def _ch(n_epochs, rate_arcsec_hr=3.0):
        # Spacing the positions so that actual_dra ~= rate * dt_hours
        # (otherwise the pair likelihood blows up negative).
        dt_days = 3.0
        dt_hours = dt_days * 24.0
        # Per-pair sky motion = rate * dt_hours arcsec along RA
        dra_per_pair_deg = (rate_arcsec_hr * dt_hours / 3600.0
                              / math.cos(math.radians(20.0)))
        return [
            {"t": _et(60450 + k * dt_days),
             "jd": (60450 + k * dt_days) + 2400000.5,
             "ra": math.radians(180.0 + k * dra_per_pair_deg),
             "dec": math.radians(20.0),
             "rate_arcsec_hr": rate_arcsec_hr, "mag": 21.5}
            for k in range(n_epochs)
        ]

    chains = [_ch(3), _ch(4), _ch(5)]
    # All chains should pass a permissive threshold; the cap matters.
    kept, scores = filter_chains_by_likelihood(
        chains, log_l_threshold=-100.0, max_chains=2)
    assert len(kept) == 2


def test_filter_chains_respects_threshold():
    from ariadne.discovery.imaging.bayesian_linker import (
        filter_chains_by_likelihood)
    # Very low quality single-entry chain
    chain = [[{"t": _et(60450.0), "jd": 60450.0 + 2400000.5,
                "ra": math.radians(180.0), "dec": math.radians(20.0),
                "rate_arcsec_hr": 2.0, "mag": 21.5}]]
    kept, _ = filter_chains_by_likelihood(chain, log_l_threshold=1.0)
    # log_likelihood is negative, threshold +1.0 -> nothing kept
    assert len(kept) == 0
