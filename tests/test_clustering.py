"""Stage 34 tests: circular-clustering statistics + real catalog ingestion."""
import math

import numpy as np

from ariadne.discovery.clustering import (load_distant_tnos, filter_population,
                                          circular_stats, rayleigh_mc, clustering_report)


def test_circular_stats_perfectly_clustered():
    s = circular_stats([30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0])
    assert abs(s["R"] - 1.0) < 1e-12
    assert abs(s["mean_dir_deg"] - 30.0) < 1e-9
    assert s["p_analytic"] < 1e-2                          # R=1 -> strongly significant


def test_circular_stats_uniform_is_not_significant():
    ang = list(np.linspace(0, 360, 24, endpoint=False))   # perfectly spread
    s = circular_stats(ang)
    assert s["R"] < 1e-6
    assert s["p_analytic"] > 0.5


def test_analytic_matches_montecarlo():
    rng = np.random.default_rng(0)
    ang = list(20.0 + rng.normal(0, 25, 15))
    s = circular_stats(ang)
    p_mc = rayleigh_mc(ang, n_mc=50000, seed=1)
    assert abs(s["p_analytic"] - p_mc) < 0.02


def test_montecarlo_flags_clustered_not_uniform():
    rng = np.random.default_rng(4)
    assert rayleigh_mc(list(rng.uniform(0, 360, 19)), n_mc=40000, seed=5) > 0.2
    assert rayleigh_mc(list(10.0 + rng.normal(0, 8, 19)), n_mc=40000, seed=6) < 1e-3


def test_real_catalog_loads_and_filters():
    rows = load_distant_tnos()                              # cached JPL SBDB
    assert len(rows) > 100
    extreme = filter_population(rows, a_min=250.0, q_min=42.0)
    assert 5 <= len(extreme) <= 60
    assert all(r["a_au"] >= 250 and r["q_au"] >= 42 for r in extreme)
    rep = clustering_report(extreme, n_mc=20000)
    assert set(rep) >= {"n", "varpi", "omega", "Omega"}
    assert 0.0 <= rep["varpi"]["p_mc"] <= 1.0
