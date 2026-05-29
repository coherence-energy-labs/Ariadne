"""Stage 35 tests: differentiable RK4 dynamics + gradient-based shooting."""
import math

import numpy as np
import pytest

from ariadne.optimize import autodiff as AD

GM_SUN = 1.32712440018e11
AU = 1.495978707e8

pytestmark = pytest.mark.skipif(not AD.HAVE_JAX, reason="JAX required")


def test_propagator_circular_orbit_returns():
    r0 = np.array([AU, 0.0, 0.0])
    v0 = np.array([0.0, math.sqrt(GM_SUN / AU), 0.0])
    P = 2 * math.pi * math.sqrt(AU ** 3 / GM_SUN)
    rf, vf = AD.propagate(r0, v0, P, n_steps=2000)
    assert np.linalg.norm(rf - r0) / AU < 1e-4
    assert np.linalg.norm(vf - v0) / np.linalg.norm(v0) < 1e-4


def test_gradient_matches_finite_difference():
    r1 = np.array([AU, 0.0, 0.0])
    r2 = 1.5 * AU * np.array([math.cos(1.2), math.sin(1.2), 0.0])
    tof = 200 * 86400.0
    v0 = np.array([0.0, math.sqrt(GM_SUN / AU) * 1.05, 0.0])
    g = AD.dv_gradient(v0, r1, r2, tof)
    fd = np.zeros(3)
    for k in range(3):
        e = np.zeros(3); e[k] = 1e-3

        def miss(v):
            rf, _ = AD.propagate(r1, v, tof)
            return float(np.sum((rf - r2) ** 2))
        fd[k] = (miss(v0 + e) - miss(v0 - e)) / 2e-3
    assert np.max(np.abs(g - fd) / (np.abs(fd) + 1e-3)) < 1e-5


def test_gauss_newton_shooting_hits_target():
    r1 = np.array([AU, 0.0, 0.0])
    th = math.radians(75.0)
    r2 = 1.524 * AU * np.array([math.cos(th), math.sin(th), 0.0])
    tof = 220 * 86400.0
    v0 = np.array([0.0, math.sqrt(GM_SUN / AU) * 1.05, 0.0])
    sol = AD.solve_lambert_shooting(r1, r2, tof, v0, iters=30, tol_km=1e-2)
    assert sol["miss_km"] < 1.0
    assert sol["iters"] <= 25
    # the solution actually reaches the target when re-propagated
    rf, _ = AD.propagate(r1, sol["v1"], tof)
    assert np.linalg.norm(rf - r2) / AU < 1e-5
