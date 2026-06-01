"""Tests for the Bayesian IOD module."""
from __future__ import annotations

import math
import numpy as np
import pytest


def _et(mjd):
    return (mjd - 51544.5) * 86400.0


def test_orbital_class_prior_tno_at_45_au():
    from ariadne.discovery.iod_bayesian import log_orbital_class_prior
    log_p, dom = log_orbital_class_prior(45.0)
    assert math.isfinite(log_p)
    assert dom == "tno_class"


def test_orbital_class_prior_mba_at_2p7_au():
    from ariadne.discovery.iod_bayesian import log_orbital_class_prior
    _, dom = log_orbital_class_prior(2.7)
    assert dom == "mba"


def test_orbital_class_prior_neo_at_1p2_au():
    from ariadne.discovery.iod_bayesian import log_orbital_class_prior
    _, dom = log_orbital_class_prior(1.2)
    assert dom == "neo"


def test_orbital_class_prior_tno_scatter_at_70():
    from ariadne.discovery.iod_bayesian import log_orbital_class_prior
    _, dom = log_orbital_class_prior(70.0)
    assert dom == "tno_scatter"


def test_seed_from_class_returns_n_samples():
    from ariadne.discovery.iod_bayesian import (
        _seed_from_class, ORBITAL_PRIORS)
    # Need a non-trivial chain with t values
    chain = [{"t": _et(60450.0), "ra": math.radians(180.0),
                "dec": math.radians(20.0)}]
    rng = np.random.default_rng(0)
    seeds = _seed_from_class(ORBITAL_PRIORS[2], chain, n_samples=5, rng=rng)
    # ORBITAL_PRIORS[2] is "hilda" -- check we get 5 seeds with reasonable r
    assert len(seeds) == 5
    from ariadne.data.constants import AU_KM
    for r0, v0, t_ref in seeds:
        r_au = float(np.linalg.norm(r0)) / AU_KM
        # hilda prior r_au_mean=4.0, sigma=0.3 -> within +/- 3 sigma
        assert 2.5 < r_au < 5.5


def test_seed_from_class_empty_chain_returns_empty():
    from ariadne.discovery.iod_bayesian import (
        _seed_from_class, ORBITAL_PRIORS)
    seeds = _seed_from_class(ORBITAL_PRIORS[0], [], n_samples=5)
    assert seeds == []


def test_bayesian_iod_handles_too_short_chain():
    from ariadne.discovery.iod_bayesian import bayesian_iod
    fit = bayesian_iod([])
    assert not fit.success
    assert fit.winning_strategy == "bayesian_none"

    fit = bayesian_iod([{"t": _et(60450.0), "ra": math.radians(180.0),
                            "dec": math.radians(20.0)}])
    assert not fit.success


def test_bayesian_iod_runs_on_synthetic_tno_chain():
    """Build a clean 3-night TNO chain and verify Bayesian IOD returns
    a viable fit (success or failure -- not raising, not returning
    NaN)."""
    from ariadne.discovery.iod_bayesian import bayesian_iod
    # Synthetic 6-observation chain at TNO-like positions/rates
    ra0 = math.radians(180.0)
    dec0 = math.radians(20.0)
    rate_rad_hr = math.radians(2.0 / 3600.0)   # 2 "/hr
    pa = math.radians(45.0)
    chain = []
    for k in range(6):
        mjd = 60450.0 + (k // 2) * 3 + (k % 2) * 0.083
        dt_hr = (mjd - 60453.0) * 24.0
        dra = rate_rad_hr * dt_hr * math.sin(pa) / math.cos(dec0)
        ddec = rate_rad_hr * dt_hr * math.cos(pa)
        chain.append({
            "t": _et(mjd), "jd": mjd + 2400000.5,
            "ra": ra0 + dra, "dec": dec0 + ddec,
            "dra": 1e-9, "ddec": 1e-9,
            "rate_arcsec_hr": 2.0, "mag": 21.5,
            "source_pair": (),
        })
    fit = bayesian_iod(chain, n_seeds_per_class=2,
                         rms_acceptance_arcsec=300.0)
    # Just verify the call structure works -- whether it succeeds depends
    # on whether ANY orbital class fits these synthetic positions.
    assert fit.t_ref > 0
    assert math.isfinite(fit.rms_arcsec) or fit.rms_arcsec == float("inf")
