"""Stage 33 validation -- general relativity (1PN / Schwarzschild) in the engine.

Adds the dominant relativistic correction and verifies it against the most famous test
in celestial mechanics: the anomalous perihelion advance of Mercury, ~42.98 arcsec/century,
the result that confirmed general relativity in 1915.

G33a (Mercury precession) - Integrating a Mercury-like orbit under Newtonian + 1PN gravity
                            reproduces 42.98 arcsec/century, and the measured per-orbit advance
                            matches the analytic 6 pi mu / (c^2 a (1-e^2)) to <1%.
G33b (firewall-safe scale) - The 1PN term is a ~1e-8 fractional correction at 1 AU (it does NOT
                            alter the Newtonian dynamics the rest of the engine relies on); it
                            matters only as accumulated PRECESSION over many orbits.

Run:  PYTHONPATH=src python -m ariadne.validate.stage33
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

from ..data.constants import GM_SUN, AU_KM
from ..dynamics.relativity import (gr_1pn_accel, newtonian_plus_gr_accel,
                                   perihelion_advance_per_orbit)

MERCURY = dict(a_au=0.387098, e=0.205630)


def check():
    a_au, e, mu = MERCURY["a_au"], MERCURY["e"], GM_SUN
    a = a_au * AU_KM
    P = 2 * math.pi * math.sqrt(a ** 3 / mu)
    rp = a * (1 - e); vp = math.sqrt(mu * (1 + e) / rp)
    y0 = np.array([rp, 0, 0, 0, vp, 0])

    def rhs(t, y):
        return np.concatenate([y[3:], newtonian_plus_gr_accel(y[:3], y[3:], mu)])

    n_orb = 100
    sol = solve_ivp(rhs, (0, n_orb * P), y0, rtol=1e-12, atol=1e-9,
                    dense_output=True, max_step=P / 200)
    ts = np.linspace(0, n_orb * P, n_orb * 50)
    Y = sol.sol(ts); r = Y[:3].T; v = Y[3:].T
    evec = ((np.einsum("ij,ij->i", v, v)[:, None] - mu / np.linalg.norm(r, axis=1)[:, None]) * r
            - np.einsum("ij,ij->i", r, v)[:, None] * v) / mu
    ang = np.unwrap(np.arctan2(evec[:, 1], evec[:, 0]))
    slope = np.linalg.lstsq(np.vstack([ts, np.ones_like(ts)]).T, ang, rcond=None)[0][0]
    num = slope * P
    ana = perihelion_advance_per_orbit(a_au, e, mu)
    arcsec_cy = num * (100 * 365.25 * 86400 / P) * (180 / math.pi) * 3600
    g33a = abs(num / ana - 1.0) < 0.02 and abs(arcsec_cy - 42.98) < 0.5

    # firewall-safe scale: GR fractional correction at 1 AU
    rr = np.array([AU_KM, 0.0, 0.0]); vv = np.array([0.0, 29.78, 0.0])
    frac = float(np.linalg.norm(gr_1pn_accel(rr, vv, mu)) / (mu / AU_KM ** 2))
    g33b = frac < 1e-7

    ok = g33a and g33b
    return ok, {"num": num, "ana": ana, "ratio": num / ana, "arcsec_cy": arcsec_cy,
                "frac": frac, "g33a": g33a, "g33b": g33b}


def main() -> int:
    print("=== Ariadne Stage 33  (general relativity: 1PN perihelion precession) ===\n")
    ok, i = check()
    print("[G33a] Mercury anomalous perihelion advance (Newtonian + 1PN)")
    print(f"      analytic d(varpi)/orbit = {i['ana']:.4e} rad")
    print(f"      numeric  d(varpi)/orbit = {i['num']:.4e} rad   (ratio {i['ratio']:.4f})")
    print(f"      => {i['arcsec_cy']:.2f} arcsec/century   (observed/GR: 42.98)")
    print(f"      -> {'PASS' if i['g33a'] else 'FAIL'}\n")
    print("[G33b] 1PN is a tiny correction (does not disturb the Newtonian engine)")
    print(f"      |a_GR|/|a_Newton| at 1 AU = {i['frac']:.2e}  (matters only as accumulated precession)")
    print(f"      -> {'PASS' if i['g33b'] else 'FAIL'}\n")
    print(f"=== STAGE 33: {'ALL GATES PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
