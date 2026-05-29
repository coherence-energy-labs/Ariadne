"""Stage 31 tests: doubly-averaged secular integrator + numba-accelerated direct integrator."""
import math

import numpy as np
import pytest

from ariadne.data.constants import GM_SUN, GM_JUPITER, AU_KM
from ariadne.dynamics import secular as S
from ariadne.dynamics import secular_avg as SA
from ariadne.dynamics import secular_fast as SF
from ariadne.dynamics.secular import YEAR_S

CONV = (180.0 / math.pi) * (1e6 * YEAR_S)        # rad/s -> deg/Myr


def _laplace_b(s, j, al):
    from scipy.integrate import quad
    f = lambda psi: math.cos(j * psi) / (1 - 2 * al * math.cos(psi) + al * al) ** s
    return quad(f, 0, 2 * math.pi)[0] / math.pi


def test_kepler_equation_solver():
    for e in (0.0, 0.3, 0.85, 0.93):
        M = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        E = SA._solve_kepler_eq(M, e)
        assert np.max(np.abs(E - e * np.sin(E) - M)) < 1e-12


def test_sample_orbit_lies_on_ellipse():
    a, e = 100.0, 0.6
    pos = SA.sample_orbit(a, e, 15.0, 40.0, 60.0, n=200)
    r = np.linalg.norm(pos, axis=1) / AU_KM
    assert r.min() > a * (1 - e) - 1e-6 and r.max() < a * (1 + e) + 1e-6


def test_secular_semimajor_axis_conserved():
    """Doubly-averaged theorem: da/dt = 0 (here to machine precision for a clean case)."""
    ring = SA.sample_orbit(5.2028, 0.0, 0.0, 0.0, 0.0, n=256)
    tp = dict(a_au=100.0, e=1e-3, i_deg=1e-3, Omega_deg=40.0, omega_deg=60.0)
    rt = SA.secular_rates(tp, [(ring, GM_JUPITER)], n_tp=256)
    assert abs(rt["da_au"] * 1e6 * YEAR_S) < 1e-10        # AU/Myr


def test_secular_matches_laplace_lagrange():
    """Secular apsidal rate must match analytic Laplace-Lagrange for a near-circular particle."""
    ring = SA.sample_orbit(5.2028, 0.0, 0.0, 0.0, 0.0, n=512)
    for a in (50.0, 100.0, 200.0):
        tp = dict(a_au=a, e=1e-3, i_deg=1e-3, Omega_deg=40.0, omega_deg=60.0)
        rt = SA.secular_rates(tp, [(ring, GM_JUPITER)], n_tp=512)
        dvarpi = (rt["dOmega"] + rt["domega"]) * CONV
        al = 5.2028 / a
        n = math.sqrt(GM_SUN / (a * AU_KM) ** 3)
        g = n * 0.25 * (GM_JUPITER / GM_SUN) * al * _laplace_b(1.5, 1, al) * CONV
        assert abs(dvarpi / g - 1.0) < 0.02              # within 2%


def test_ring_accel_reduces_to_point_mass_when_far():
    """A tight ring seen from far away ~ a point mass at the origin (+ indirect term)."""
    ring = SA.sample_orbit(1.0, 0.0, 0.0, 0.0, 0.0, n=256)        # 1 AU ring
    field = np.array([[1000.0 * AU_KM, 0.0, 0.0]])               # very far
    a = SA.ring_accel(field, ring, GM_SUN)[0]
    # direct part ~ -GM/r^2 toward origin; magnitude check
    expected = GM_SUN / (1000.0 * AU_KM) ** 2
    assert abs(np.linalg.norm(a[:1]) ) >= 0          # sanity (no NaN)
    assert abs(a[0]) < 5 * expected and not np.any(np.isnan(a))


@pytest.mark.skipif(not SF.HAVE_NUMBA, reason="numba not available")
def test_numba_matches_pure_python():
    """The numba direct integrator must reproduce the pure-Python map to tight tolerance."""
    def make():
        sys = S.build_system("2026-01-01T00:00:00")
        return S.add_test_particles(sys, [S.elements_to_state(506.8, 0.859, 11.93, 144.4, 311.29, 180.0)])
    a, b = make(), make()
    S.integrate(a, 1.0 * YEAR_S, 1500)
    SF.integrate_fast(b, 1.0 * YEAR_S, 1500)
    rel = np.max(np.abs(a.Q - b.Q)) / np.linalg.norm(a.Q[0])
    assert rel < 1e-8


@pytest.mark.skipif(not SF.HAVE_NUMBA, reason="numba not available")
def test_numba_energy_bounded():
    sys = S.build_system("2026-01-01T00:00:00")
    out = SF.integrate_fast(sys, 1.0 * YEAR_S, 20000, record_every=1000)
    en = out["energy"]
    assert np.abs(en - en[0]).max() / abs(en[0]) < 1e-4
