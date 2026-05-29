"""First post-Newtonian (1PN / Schwarzschild) gravity -- MASTER_PLAN.md Stage 33.

The integrators so far are purely Newtonian. General relativity adds a small correction
that, crucially, makes orbits PRECESS -- the effect that famously explains Mercury's
anomalous perihelion advance (~43 arcsec/century) and that, over Myr baselines, competes
with the secular signal a distant planet would imprint on the eTNOs. Adding it makes the
engine GR-capable (NASA-grade fidelity) and is validated against the textbook number.

Dominant two-body (Schwarzschild) 1PN acceleration of a body at heliocentric r, v
about a central mass mu (km^3/s^2), with c the speed of light (km/s):

    a_GR = (mu / (c^2 r^3)) [ (4 mu/r - v^2) r  +  4 (r . v) v ]

It produces an analytic perihelion advance per orbit of  d(varpi) = 6 pi mu / (c^2 a (1-e^2)),
which is what Stage 33 reproduces to verify correctness.
"""
from __future__ import annotations

import math

import numpy as np

C_LIGHT_KMS = 299792.458
C2 = C_LIGHT_KMS * C_LIGHT_KMS


def gr_1pn_accel(r, v, mu):
    """1PN Schwarzschild acceleration (km/s^2). r,v in km, km/s; mu in km^3/s^2."""
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    rn = float(np.linalg.norm(r))
    v2 = float(v @ v)
    rdotv = float(r @ v)
    return (mu / (C2 * rn ** 3)) * ((4.0 * mu / rn - v2) * r + 4.0 * rdotv * v)


def perihelion_advance_per_orbit(a_au, e, mu):
    """Analytic 1PN perihelion advance per orbit (radians): 6 pi mu / (c^2 a (1-e^2))."""
    from ..data.constants import AU_KM
    a = a_au * AU_KM
    return 6.0 * math.pi * mu / (C2 * a * (1.0 - e * e))


def newtonian_plus_gr_accel(r, v, mu):
    """Total acceleration: Newtonian central + 1PN correction (km/s^2)."""
    r = np.asarray(r, float)
    rn = float(np.linalg.norm(r))
    return -mu * r / rn ** 3 + gr_1pn_accel(r, v, mu)
