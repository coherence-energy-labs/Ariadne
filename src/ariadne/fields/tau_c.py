"""The principled coherence field tau_c (MASTER_PLAN.md - Stage 27).

Implements the coherence field at its OWN definition from the S_One framework, rather than the
ad-hoc FLI proxy of Stage 25. From the framework's central identity (weak field):

    tau_c(x) / tau_inf = 1 + Phi(x)/c^2          (Form 8 / delta tau_c = Phi/c^2)
    g_coh(x) = -c^2 grad ln(tau_c)               (the coherence force/acceleration)

where Phi is the TOTAL Newtonian potential of every included mass. tau_c is therefore a DIAGNOSTIC
of the gravitational potential (the framework is explicit that it is "derived, not an independent
field"), so g_coh REDUCES to Newtonian gravity in the weak field:

    g_coh = -grad Phi / (1 + Phi/c^2) = g_Newton * (1 + |Phi|/c^2 + ...)

In the solar system |Phi|/c^2 ~ 1e-8, so g_coh = g_Newton to ~1 part in 1e8 -- the framework's own
"Newton recovery." This makes the coherence field rigorous (it IS the potential field), keeps the
credibility firewall (no new dynamics), and -- because Phi sums over ALL masses -- is exactly the
substrate for detecting an UNMODELED mass from a trajectory residual (Stage 27 hidden_mass.py).

Units: GM in km^3/s^2, positions in km, Phi in km^2/s^2, accelerations in km/s^2.
"""
from __future__ import annotations

import numpy as np

C_LIGHT_KMS = 299792.458          # speed of light, km/s
C2 = C_LIGHT_KMS ** 2


def potential(x, masses):
    """Total Newtonian potential Phi(x) = -sum GM_i / |x - r_i|  (km^2/s^2).

    `masses` is a list of (GM, position3) tuples.
    """
    x = np.asarray(x, float)
    phi = 0.0
    for gm, r in masses:
        d = np.linalg.norm(x - np.asarray(r, float))
        if d > 0:
            phi -= gm / d
    return float(phi)


def tau_c(x, masses):
    """Coherence field tau_c/tau_inf = 1 + Phi/c^2 (dimensionless; < 1 inside potential wells)."""
    return 1.0 + potential(x, masses) / C2


def newtonian_accel(x, masses):
    """Newtonian acceleration g_N = -grad Phi = sum GM_i (r_i - x)/|r_i - x|^3  (km/s^2)."""
    x = np.asarray(x, float)
    a = np.zeros(3)
    for gm, r in masses:
        d = np.asarray(r, float) - x
        dn = np.linalg.norm(d)
        if dn > 0:
            a += gm * d / dn ** 3
    return a


def coherence_accel(x, masses):
    """Coherence acceleration g_coh = -c^2 grad ln(tau_c) = g_N / tau_c  (analytic; km/s^2)."""
    return newtonian_accel(x, masses) / tau_c(x, masses)


def coherence_accel_fd(x, masses, h=1.0):
    """g_coh by finite-difference of -c^2 grad ln(tau_c) -- an implementation cross-check."""
    x = np.asarray(x, float)
    g = np.zeros(3)
    for i in range(3):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        g[i] = -C2 * (np.log(tau_c(xp, masses)) - np.log(tau_c(xm, masses))) / (2.0 * h)
    return g
