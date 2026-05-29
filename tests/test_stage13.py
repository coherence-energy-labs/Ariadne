"""Stage 13 tests: Jovian generalization (G_jov) + low-thrust regime (G_lt)."""
import numpy as np

from ariadne.data.constants import GALILEAN, EARTH_MOON
from ariadne.dynamics.cr3bp import jacobi_constant
from ariadne.dynamics.low_thrust import propagate_low_thrust, energy_gain_rate_check
from ariadne.transfers.jovian import moon_libration, moon_tour_deltav

MU = EARTH_MOON.mu


def test_engine_generalizes_to_galilean_moons():
    for S in GALILEAN:
        m = moon_libration(S)
        assert m["orbit"].half_period_residual < 1e-9      # periodic Lyapunov orbit
        assert 1e3 < m["L1_km"] < 1e5                       # sensible L1 distance
        assert 0.1 < m["lyap_period_d"] < 30


def test_moon_tour_deltav_sensible():
    legs = moon_tour_deltav()
    assert len(legs) == 3
    for leg in legs:
        assert 1000 < leg["dv_ms"] < 6000


def test_low_thrust_zero_equals_cr3bp():
    s0 = np.array([0.5 - MU, np.sqrt(3) / 2, 0.0, 0.4, -0.2, 0.05])
    sol = propagate_low_thrust(s0, (0.0, 5.0), MU, 0.0,
                               t_eval=np.linspace(0, 5, 300))
    c = [jacobi_constant(sol.y[:, i], MU) for i in range(sol.y.shape[1])]
    assert max(abs(np.array(c) - c[0])) < 1e-9             # Jacobi conserved = CR3BP


def test_low_thrust_energy_rate_matches_theorem():
    s0 = np.array([0.5 - MU, np.sqrt(3) / 2, 0.0, 0.4, -0.2, 0.05])
    a, dt = 1e-3, 0.02
    sol = propagate_low_thrust(s0, (0.0, dt), MU, a, mode="tangential")
    rate = (jacobi_constant(sol.y[:, -1], MU) - jacobi_constant(s0, MU)) / dt
    pred = energy_gain_rate_check(s0, MU, a)               # -2 a |v|
    assert abs(rate - pred) / abs(pred) < 1e-2
