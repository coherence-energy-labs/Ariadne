"""CR3BP dynamics tests (Gate G1)."""
import numpy as np

from ariadne.data.constants import EARTH_MOON
from ariadne.dynamics.cr3bp import (
    jacobi_constant, eom, jacobian, propagate, propagate_stm,
)

MU = EARTH_MOON.mu


def test_jacobi_conserved_long_propagation():
    s0 = np.array([0.5 - MU, np.sqrt(3) / 2, 0.0, 0.02, -0.01, 0.005])
    c0 = jacobi_constant(s0, MU)
    sol = propagate(s0, (0.0, 50.0), MU, t_eval=np.linspace(0, 50, 1001))
    devs = [abs(jacobi_constant(sol.y[:, i], MU) - c0)
            for i in range(sol.y.shape[1])]
    assert max(devs) < 1e-10


def test_jacobian_matches_finite_difference():
    s = np.array([0.85, 0.02, 0.01, 0.05, 0.12, -0.03])
    A = jacobian(s, MU)
    h = 1e-7
    fd = np.zeros((6, 6))
    for i in range(6):
        sp = s.copy(); sp[i] += h
        sm = s.copy(); sm[i] -= h
        fd[:, i] = (eom(0, sp, MU) - eom(0, sm, MU)) / (2 * h)
    assert np.max(np.abs(A - fd)) < 1e-6


def test_stm_matches_flow_finite_difference():
    s0 = np.array([0.85, 0.02, 0.01, 0.05, 0.12, -0.03])
    T = 1.0
    _, stm = propagate_stm(s0, (0.0, T), MU)
    h = 1e-7
    fd = np.zeros((6, 6))
    for i in range(6):
        sp = s0.copy(); sp[i] += h
        sm = s0.copy(); sm[i] -= h
        yp = propagate(sp, (0.0, T), MU).y[:, -1]
        ym = propagate(sm, (0.0, T), MU).y[:, -1]
        fd[:, i] = (yp - ym) / (2 * h)
    assert np.max(np.abs(stm - fd)) < 1e-5


def test_time_reversibility():
    # Benign bounded (L4-region) orbit: round-trip integration error is a
    # meaningful smoke test only when the trajectory is not near a primary.
    s0 = np.array([0.5 - MU, np.sqrt(3) / 2, 0.0, 0.02, -0.01, 0.005])
    fwd = propagate(s0, (0.0, 6.0), MU).y[:, -1]
    back = propagate(fwd, (0.0, -6.0), MU).y[:, -1]
    assert np.max(np.abs(back - s0)) < 1e-8
