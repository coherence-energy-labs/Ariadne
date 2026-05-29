"""Stage 19 tests: Gateway-class NRHO construction + near-stability."""
import numpy as np
import pytest

from ariadne.data.constants import EARTH_MOON, R_MOON
from ariadne.dynamics.cr3bp import propagate
from ariadne.orbits.differential_correction import monodromy
from ariadne.orbits.nrho import nrho_family

MU = EARTH_MOON.mu
T_DAYS = EARTH_MOON.T_star / 86400.0
LSTAR = EARTH_MOON.L_star


@pytest.mark.slow
def test_nrho_matches_gateway_geometry():
    nrho, fam = nrho_family(MU, "L2", t_star_days=T_DAYS, l_star=LSTAR,
                            target_period_d=6.56, ds=4e-3)
    assert nrho is not None and len(fam) > 50
    sol = propagate(nrho.s0, (0.0, nrho.period), MU,
                    t_eval=np.linspace(0, nrho.period, 800))
    d = np.sqrt((sol.y[0] - (1 - MU)) ** 2 + sol.y[1] ** 2 + sol.y[2] ** 2) * LSTAR
    period_d = nrho.period * T_DAYS
    assert 6.3 <= period_d <= 6.8                          # Gateway ~6.56 d
    assert 2800 <= d.min() <= 3800                         # perilune over the pole
    assert 60000 <= d.max() <= 78000                       # apolune ~70,000 km
    assert nrho.residual < 1e-9                            # genuinely periodic


@pytest.mark.slow
def test_nrho_is_far_more_stable_than_lyapunov():
    from ariadne.orbits.families import lyapunov_orbit_at_jacobi
    nrho, _ = nrho_family(MU, "L2", t_star_days=T_DAYS, l_star=LSTAR,
                          target_period_d=6.56, ds=4e-3)
    floq_nrho = float(np.max(np.abs(np.linalg.eigvals(monodromy(MU, nrho)))))
    floq_lyap = float(np.max(np.abs(np.linalg.eigvals(
        monodromy(MU, lyapunov_orbit_at_jacobi(MU, "L1", 3.16))))))
    assert floq_nrho < 100.0                               # near-stable
    assert floq_nrho < 0.1 * floq_lyap                     # vastly more stable
