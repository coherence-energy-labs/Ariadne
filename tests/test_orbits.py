"""Periodic-orbit and family tests (Gates G3, G4)."""
import numpy as np

from ariadne.data.constants import EARTH_MOON
from ariadne.dynamics.cr3bp import propagate
from ariadne.orbits.differential_correction import correct_lyapunov
from ariadne.orbits.families import lyapunov_family, find_halo_bifurcation
from ariadne.orbits.linear import collinear_linear_modes, linear_lyapunov_guess

MU = EARTH_MOON.mu


def _periodicity(orbit):
    sol = propagate(orbit.s0, (0.0, orbit.period), MU)
    return float(np.max(np.abs(sol.y[:, -1] - orbit.s0)))


def test_linear_modes_well_formed():
    for pt in ("L1", "L2", "L3"):
        m = collinear_linear_modes(MU, pt)
        assert m["omega"] > 0 and m["lambda"] > 0 and m["omega_v"] > 0
        assert m["kappa"] > 0


def test_small_amplitude_period_matches_linear():
    m = collinear_linear_modes(MU, "L1")
    s0, Tg = linear_lyapunov_guess(MU, "L1", 1e-4)
    orb = correct_lyapunov(MU, s0, Tg)
    assert abs(orb.period - m["linear_period"]) / m["linear_period"] < 1e-3


def test_lyapunov_orbit_is_periodic():
    s0, Tg = linear_lyapunov_guess(MU, "L1", 0.02)
    orb = correct_lyapunov(MU, s0, Tg)
    assert orb.half_period_residual < 1e-11
    assert _periodicity(orb) < 1e-9


def test_family_monotonic_jacobi_and_halo_bifurcation():
    fam = lyapunov_family(MU, "L1", amplitude0=1e-3, dx=2e-3, n=30)
    assert len(fam) >= 20
    jac = np.array([m.orbit.jacobi for m in fam])
    assert np.all(np.diff(jac) < 1e-9)            # energy increases -> C decreases
    bif = find_halo_bifurcation(fam)
    assert bif is not None
    # Earth-Moon L1 halo bifurcation is near C ~ 3.18-3.19
    assert 3.15 < bif["jacobi"] < 3.22
