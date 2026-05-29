"""Stage 30 tests: universal-variable Kepler solver + symplectic Wisdom-Holman map."""
import math

import numpy as np

from ariadne.data.constants import GM_SUN, AU_KM
from ariadne.dynamics import secular as S


def _period(a_au, mu=GM_SUN):
    a = a_au * AU_KM
    return 2 * math.pi * math.sqrt(a ** 3 / mu)


def test_kepler_full_period_returns_to_start():
    """A full orbital period must map the state back onto itself (machine precision)."""
    for e in (0.0, 0.5, 0.9, 0.93):
        r0, v0 = S.elements_to_state(40.0, e, 17.0, 80.0, 250.0, 0.0)
        rn, vn = S.kepler_step(r0, v0, GM_SUN, _period(40.0))
        assert np.linalg.norm(rn - r0) / np.linalg.norm(r0) < 1e-10
        assert np.linalg.norm(vn - v0) / np.linalg.norm(v0) < 1e-10


def test_kepler_half_period_reaches_apoapsis():
    a_au, e = 5.0, 0.6
    r0, v0 = S.elements_to_state(a_au, e, 0, 0, 0, 0.0)         # at perihelion
    rh, _ = S.kepler_step(r0, v0, GM_SUN, _period(a_au) / 2)
    assert abs(np.linalg.norm(rh) / AU_KM - a_au * (1 + e)) < 1e-6


def test_kepler_vectorised_matches_scalar():
    R = np.array([[AU_KM, 0, 0], [0, 2 * AU_KM, 0]], float)
    V = np.array([[0, 30.0, 0], [-20.0, 0, 1.0]], float)
    Rn, Vn = S.kepler_step(R, V, GM_SUN, 1.0e6)
    for i in range(2):
        r1, v1 = S.kepler_step(R[i], V[i], GM_SUN, 1.0e6)
        assert np.allclose(Rn[i], r1) and np.allclose(Vn[i], v1)


def test_stumpff_continuous_through_zero():
    c2m, c3m = S._stumpff(np.array([-1e-9]))
    c2p, c3p = S._stumpff(np.array([1e-9]))
    assert abs(c2m[0] - 0.5) < 1e-9 and abs(c2p[0] - 0.5) < 1e-9
    assert abs(c3m[0] - 1.0 / 6) < 1e-9 and abs(c3p[0] - 1.0 / 6) < 1e-9


def test_elements_state_roundtrip():
    a, e, i, Om, om, nu = 506.8, 0.859, 11.93, 144.40, 311.29, 42.0
    r, v = S.elements_to_state(a, e, i, Om, om, nu)
    el = S.state_to_elements(r, v)
    assert abs(el["a_au"] - a) < 1e-6
    assert abs(el["e"] - e) < 1e-9
    assert abs(el["i_deg"] - i) < 1e-9
    assert abs(((el["varpi_deg"] - (Om + om) + 180.0) % 360.0) - 180.0) < 1e-6


def test_clustering_metric_aligned_vs_random():
    aligned = [{"peri_hat": np.array([1.0, 0.05 * k, 0.0]) / np.linalg.norm([1.0, 0.05 * k, 0.0]),
                "varpi_deg": 10.0 + k} for k in range(6)]
    rng = np.random.default_rng(0)
    rand_hats = rng.normal(size=(40, 3))
    rand = [{"peri_hat": h / np.linalg.norm(h),
             "varpi_deg": math.degrees(math.atan2(h[1], h[0])) % 360} for h in rand_hats]
    assert S.perihelion_resultant(aligned) > 0.9
    assert S.perihelion_resultant(rand) < 0.4
    assert S.varpi_dispersion_deg(aligned) < S.varpi_dispersion_deg(rand)


def test_symplectic_energy_bounded_and_second_order():
    """Energy error must stay bounded (no secular drift) and fall ~dt^2 (2nd-order)."""
    def max_dE(dt_yr, span_yr=4000.0):
        sys = S.build_system("2026-01-01T00:00:00")
        dt = dt_yr * S.YEAR_S
        n = int(span_yr / dt_yr)
        out = S.integrate(sys, dt, n, record_every=max(1, n // 20))
        en = out["energy"]
        return np.abs(en - en[0]).max() / abs(en[0])

    d1 = max_dE(1.0)
    d_half = max_dE(0.5)
    assert d1 < 1e-4                                   # bounded, small
    assert d_half < d1                                 # smaller step -> smaller error
    assert d_half / d1 < 0.5                            # ~dt^2 scaling (theory 0.25)


def test_angular_momentum_conserved():
    sys = S.build_system("2026-01-01T00:00:00")
    L0 = S.angular_momentum(sys)
    S.integrate(sys, 1.0 * S.YEAR_S, 2000)
    assert abs(S.angular_momentum(sys) - L0) / L0 < 1e-9
