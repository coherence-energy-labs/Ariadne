"""Stage 40 tests: dynamical-structure mining (orbital poles, Neptune decoupling)."""
import numpy as np

from ariadne.discovery.clustering import load_distant_tnos
from ariadne.discovery.structure import (orbital_poles, pole_clustering_vs_control,
                                         nearest_low_order_resonance, neptune_decoupling)


def test_orbital_poles_are_unit_vectors():
    rows = load_distant_tnos()[:20]
    h = orbital_poles(rows)
    assert np.allclose(np.linalg.norm(h, axis=1), 1.0, atol=1e-9)


def test_pole_clustering_runs_and_bounded():
    rows = load_distant_tnos()
    r = pole_clustering_vs_control(rows, n_mc=5000)
    assert r["n_ctrl"] >= 20
    assert 0.0 <= r["p_vs_selection"] <= 1.0
    assert 0.0 <= r["ext_R"] <= 1.0 and 0.0 <= r["ctrl_R"] <= 1.0


def test_distant_object_has_no_low_order_resonance():
    # a very distant object: period ratio with Neptune is huge -> no low-order p:q
    p, q, off = nearest_low_order_resonance(500.0, max_order=12)
    assert p is None


def test_close_object_finds_resonance():
    # an object near the 2:1 Neptune resonance (a ~ 30.07 * 2^(2/3) ~ 47.8 AU)
    a_21 = 30.069 * 2 ** (2 / 3)
    p, q, off = nearest_low_order_resonance(a_21, max_order=12)
    assert (p, q) == (2, 1) and off < 0.01


def test_neptune_decoupling_of_extreme_population():
    rows = load_distant_tnos()
    d = neptune_decoupling(rows)
    assert d["n_ext"] >= 10
    assert d["frac_near"] < 0.2          # detached objects are not in low-order resonances
