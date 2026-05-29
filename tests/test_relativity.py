"""Stage 33 tests: 1PN (Schwarzschild) general-relativistic correction."""
import math

import numpy as np
from scipy.integrate import solve_ivp

from ariadne.data.constants import GM_SUN, AU_KM
from ariadne.dynamics.relativity import (gr_1pn_accel, newtonian_plus_gr_accel,
                                         perihelion_advance_per_orbit, C2)


def test_gr_term_is_tiny_correction_at_planetary_scale():
    r = np.array([AU_KM, 0.0, 0.0]); v = np.array([0.0, 29.78, 0.0])
    aN = GM_SUN / AU_KM ** 2
    aGR = np.linalg.norm(gr_1pn_accel(r, v, GM_SUN))
    assert aGR / aN < 1e-7                       # GR is a ~1e-8 fractional correction at 1 AU


def test_analytic_advance_matches_mercury():
    adv = perihelion_advance_per_orbit(0.387098, 0.205630, GM_SUN)
    arcsec_per_century = adv * (100 * 365.25 * 86400 /
                                (2 * math.pi * math.sqrt((0.387098 * AU_KM) ** 3 / GM_SUN))) \
        * (180 / math.pi) * 3600
    assert abs(arcsec_per_century - 42.98) < 0.5


def test_numeric_precession_matches_analytic():
    """Integrate Mercury (Newtonian+1PN) and confirm the measured advance matches theory."""
    a_au, e, mu = 0.387098, 0.205630, GM_SUN
    a = a_au * AU_KM
    P = 2 * math.pi * math.sqrt(a ** 3 / mu)
    rp = a * (1 - e); vp = math.sqrt(mu * (1 + e) / rp)
    y0 = np.array([rp, 0, 0, 0, vp, 0])

    def rhs(t, y):
        return np.concatenate([y[3:], newtonian_plus_gr_accel(y[:3], y[3:], mu)])

    n_orb = 60
    sol = solve_ivp(rhs, (0, n_orb * P), y0, rtol=1e-12, atol=1e-9,
                    dense_output=True, max_step=P / 200)
    ts = np.linspace(0, n_orb * P, n_orb * 40)
    Y = sol.sol(ts); r = Y[:3].T; v = Y[3:].T
    evec = ((np.einsum("ij,ij->i", v, v)[:, None] - mu / np.linalg.norm(r, axis=1)[:, None]) * r
            - np.einsum("ij,ij->i", r, v)[:, None] * v) / mu
    ang = np.unwrap(np.arctan2(evec[:, 1], evec[:, 0]))
    slope = np.linalg.lstsq(np.vstack([ts, np.ones_like(ts)]).T, ang, rcond=None)[0][0]
    num = slope * P
    ana = perihelion_advance_per_orbit(a_au, e, mu)
    assert abs(num / ana - 1.0) < 0.02
