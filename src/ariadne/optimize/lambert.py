"""Lambert's problem — the two-body boundary value problem (MASTER_PLAN.md §3.13).

Given two position vectors and a time of flight, find the connecting conic's
terminal velocities. Universal-variable formulation (Bate-Mueller-White /
Vallado) with bisection on the universal anomaly. This is the workhorse for
patched-conic transfer first guesses.
"""
from __future__ import annotations

import math

import numpy as np


def _stumpff(psi: float):
    if psi > 1e-6:
        s = math.sqrt(psi)
        c2 = (1.0 - math.cos(s)) / psi
        c3 = (s - math.sin(s)) / s ** 3
    elif psi < -1e-6:
        s = math.sqrt(-psi)
        c2 = (1.0 - math.cosh(s)) / psi
        c3 = (math.sinh(s) - s) / s ** 3
    else:
        c2 = 0.5 - psi / 24.0
        c3 = 1.0 / 6.0 - psi / 120.0
    return c2, c3


def lambert(r1, r2, tof: float, mu: float, prograde: bool = True,
            tol: float = 1e-9, max_iter: int = 300):
    """Solve Lambert's problem. Returns (v1, v2) in the same units as r/tof/mu.

    r1, r2 : position vectors (length 3). tof: time of flight (> 0).
    prograde : choose the prograde (True) or retrograde (False) transfer plane.
    """
    r1 = np.asarray(r1, float)
    r2 = np.asarray(r2, float)
    r1n = np.linalg.norm(r1)
    r2n = np.linalg.norm(r2)

    cos_dnu = float(np.dot(r1, r2) / (r1n * r2n))
    cross_z = float(np.cross(r1, r2)[2])
    if prograde:
        dm = 1.0 if cross_z >= 0.0 else -1.0
    else:
        dm = -1.0 if cross_z >= 0.0 else 1.0
    A = dm * math.sqrt(r1n * r2n * (1.0 + cos_dnu))
    if abs(A) < 1e-12:
        raise ValueError("Lambert: degenerate transfer geometry (A ~ 0)")

    psi = 0.0
    psi_up, psi_low = 4.0 * math.pi ** 2, -4.0 * math.pi ** 2
    y = r1n + r2n
    for _ in range(max_iter):
        c2, c3 = _stumpff(psi)
        y = r1n + r2n + A * (psi * c3 - 1.0) / math.sqrt(c2)
        if A > 0.0 and y < 0.0:
            psi_low = psi
            psi = 0.5 * (psi_up + psi_low)
            continue
        chi = math.sqrt(y / c2)
        dt = (chi ** 3 * c3 + A * math.sqrt(y)) / math.sqrt(mu)
        if abs(dt - tof) < tol * max(1.0, tof):
            break
        if dt <= tof:
            psi_low = psi
        else:
            psi_up = psi
        psi = 0.5 * (psi_up + psi_low)

    f = 1.0 - y / r1n
    g = A * math.sqrt(y / mu)
    gdot = 1.0 - y / r2n
    v1 = (r2 - f * r1) / g
    v2 = (gdot * r2 - r1) / g
    return v1, v2
