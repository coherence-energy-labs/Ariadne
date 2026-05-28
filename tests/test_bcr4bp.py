"""BCR4BP tests (Gate G8a)."""
import numpy as np

from ariadne.data.constants import EARTH_MOON
from ariadne.dynamics.cr3bp import eom
from ariadne.dynamics.bcr4bp import (
    eom_bcr4bp, solar_acceleration, sun_params, propagate_bcr4bp,
)

MU = EARTH_MOON.mu
SP = sun_params(EARTH_MOON)


def test_sun_params_reasonable():
    assert abs(SP["m_S"] - 328900.6) < 5.0
    assert abs(SP["a_S"] - 389.17) < 0.1
    assert abs(SP["omega_S"] + 0.9253) < 1e-3       # ~ -0.925
    assert abs(2 * np.pi / abs(SP["omega_S"]) - 6.79) < 0.05   # synodic ~29.5 d


def test_no_sun_limit_equals_cr3bp():
    rng = np.random.default_rng(1)
    for _ in range(100):
        s = rng.uniform(-1.5, 1.5, size=6)
        d = eom_bcr4bp(0.7, s, MU, 0.0, SP["a_S"], SP["omega_S"]) - eom(0.7, s, MU)
        assert np.max(np.abs(d)) < 1e-12


def test_solar_accel_zero_at_barycenter():
    a = solar_acceleration(np.zeros(3), 0.0, SP["m_S"], SP["a_S"], SP["omega_S"])
    assert np.linalg.norm(a) < 1e-10


def test_solar_accel_tidal_magnitude_at_moon():
    a = solar_acceleration(np.array([1 - MU, 0, 0]), 0.0,
                           SP["m_S"], SP["a_S"], SP["omega_S"])
    assert 1e-3 < np.linalg.norm(a) < 5e-2          # ~1.1e-2 nondim


def test_propagation_runs():
    s0 = np.array([0.5 - MU, np.sqrt(3) / 2, 0.0, 0.01, -0.005, 0.0])
    sol = propagate_bcr4bp(s0, (0.0, 2.0), MU, SP["m_S"], SP["a_S"], SP["omega_S"])
    assert sol.success and sol.y.shape[0] == 6
