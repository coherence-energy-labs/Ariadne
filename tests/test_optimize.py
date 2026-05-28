"""Lambert + collocation optimizer tests."""
import numpy as np

from ariadne.data.ephemeris import body_gm, et
from ariadne.dynamics.ephemeris_nbody import propagate_test_particle
from ariadne.optimize.lambert import lambert
from ariadne.optimize.collocation import solve_hermite_simpson


def test_lambert_self_consistency():
    mu = body_gm("EARTH")
    e0 = et("2025-06-01T00:00:00")
    r1 = np.array([7000.0, 1000.0, 500.0])
    v1 = np.array([1.0, 7.5, 0.3])
    # period ~6116 s; keep tof < one period (0-rev Lambert)
    for tof in (2000.0, 4000.0):
        sol = propagate_test_particle(r1, v1, e0, (0.0, tof), perturbers=())
        r2, v2 = sol.y[:3, -1], sol.y[3:, -1]
        v1l, v2l = lambert(r1, r2, tof, mu, prograde=True)
        assert np.linalg.norm(v1 - v1l) < 1e-6        # km/s
        assert np.linalg.norm(v2 - v2l) < 1e-6


def test_collocation_min_energy_double_integrator():
    # xdot=v, vdot=u; min int u^2 dt; (0,0)->(1,0), T=1
    # analytic optimum: u(t) = 6 - 12 t, J* = 12
    f = lambda x, u: np.array([x[1], u[0]])
    L = lambda x, u: u[0] ** 2
    sol = solve_hermite_simpson(f, L, [0, 0], [1, 0], 1.0, 20, 1)
    assert sol["success"]
    assert abs(sol["J"] - 12.0) < 1e-3
    assert sol["max_defect"] < 1e-6
    assert abs(sol["U"][0, 0] - 6.0) < 1e-2
    assert abs(sol["U"][-1, 0] + 6.0) < 1e-2
    assert np.linalg.norm(sol["X"][-1] - np.array([1.0, 0.0])) < 1e-6
