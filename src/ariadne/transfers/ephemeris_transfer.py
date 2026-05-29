"""Full-ephemeris Earth->Moon transfer design (MASTER_PLAN.md Stage 8).

Designs a real trans-lunar transfer on the JPL DE440 ephemeris:
  1. seed the departure velocity with a two-body Lambert arc to the Moon's real
     position at arrival,
  2. differentially correct it by SHOOTING in the full ephemeris (Earth central +
     Sun perturbation, real SPICE positions) so the trajectory hits the Moon's
     actual position at the arrival epoch,
  3. patch the lunar capture (v_infinity -> low-lunar-orbit insertion), and
  4. minimize the total LEO->LLO Delta-v over time of flight.

The trans-lunar leg uses Earth + Sun gravity (Moon treated as the target body, so
the relative arrival speed is a clean v_infinity); the lunar capture is patched
analytically (as in optimize/budget.py). This is a genuine ephemeris transfer, not
a CR3BP idealization.
"""
from __future__ import annotations

import math

import numpy as np

from ..data.constants import GM_MOON, R_EARTH, R_MOON
from ..data.ephemeris import body_gm, body_state
from ..dynamics.ephemeris_nbody import propagate_test_particle
from ..optimize.lambert import lambert


def _rodrigues(v, axis, angle):
    """Rotate vector v about unit axis by angle (rad)."""
    axis = axis / np.linalg.norm(axis)
    return (v * math.cos(angle)
            + np.cross(axis, v) * math.sin(angle)
            + axis * np.dot(axis, v) * (1.0 - math.cos(angle)))


def _target_moon(et0, tof, v1, r1, target, perturbers=("SUN",),
                 tol=1e-1, max_iter=30):
    """Correct departure velocity v1 so the ephemeris trajectory from r1 reaches
    `target` (Moon position at et0+tof). Returns (v1_corrected, miss_km)."""
    v = np.asarray(v1, float).copy()
    h = 1e-3  # km/s finite-difference step
    for _ in range(max_iter):
        rf = propagate_test_particle(r1, v, et0, (0.0, tof),
                                     perturbers=perturbers).y[:3, -1]
        miss = rf - target
        if np.linalg.norm(miss) < tol:
            return v, float(np.linalg.norm(miss))
        J = np.zeros((3, 3))
        for i in range(3):
            vp = v.copy(); vp[i] += h
            rfp = propagate_test_particle(r1, vp, et0, (0.0, tof),
                                          perturbers=perturbers).y[:3, -1]
            J[:, i] = (rfp - rf) / h
        v = v + np.linalg.solve(J, -miss)
    return v, float(np.linalg.norm(miss))


def design_transfer(et0, tof_days, leo_alt=200.0, llo_alt=100.0,
                    lead_deg=110.0, prograde=True):
    """Design one ephemeris trans-lunar transfer. Returns a dict of Delta-v's (m/s)
    and the targeting miss (km), or None if it fails to converge."""
    gm_e = body_gm("EARTH")
    tof = tof_days * 86400.0
    r_leo = R_EARTH + leo_alt
    r_llo = R_MOON + llo_alt

    moon0 = body_state("MOON", et0, "J2000", "EARTH")
    moonf = body_state("MOON", et0 + tof, "J2000", "EARTH")
    n_hat = np.cross(moon0[:3], moon0[3:])
    n_hat /= np.linalg.norm(n_hat)                      # Moon orbit plane normal
    arrive_hat = moonf[:3] / np.linalg.norm(moonf[:3])
    # departure position: lead the arrival direction by lead_deg in the orbit plane
    r1 = r_leo * _rodrigues(arrive_hat, n_hat, -math.radians(lead_deg))
    r2 = moonf[:3]

    try:
        v1, _ = lambert(r1, r2, tof, gm_e, prograde=prograde)
    except Exception:
        return None
    v1c, miss = _target_moon(et0, tof, v1, r1, r2)
    if miss > 5.0:                                      # > 5 km: didn't converge
        return None

    # arrival velocity (Earth-centered) at the Moon, in full ephemeris
    arr = propagate_test_particle(r1, v1c, et0, (0.0, tof), perturbers=("SUN",)).y[:, -1]
    v_inf = float(np.linalg.norm(arr[3:] - moonf[3:]))   # clean hyperbolic excess

    # departure burn: minimal (optimal tangential) injection from a circular LEO
    # onto the trans-lunar orbit of energy C3. The parking orbit is oriented so the
    # burn is tangential, so dv = v_peri(transfer) - v_circ (standard TLI).
    c3 = float(v1c @ v1c) - 2.0 * gm_e / np.linalg.norm(r1)     # km^2/s^2
    v_peri = math.sqrt(c3 + 2.0 * gm_e / r_leo)
    v_circ_leo = math.sqrt(gm_e / r_leo)
    dv_tli = v_peri - v_circ_leo

    # lunar capture (patched): v_inf -> circular LLO
    v_circ_llo = math.sqrt(GM_MOON / r_llo)
    v_hyp = math.sqrt(v_inf ** 2 + 2.0 * GM_MOON / r_llo)
    dv_loi = v_hyp - v_circ_llo

    return {"tof_days": tof_days, "miss_km": miss, "v_inf_kms": v_inf,
            "dv_tli_ms": dv_tli * 1000.0, "dv_loi_ms": dv_loi * 1000.0,
            "total_ms": (dv_tli + dv_loi) * 1000.0,
            "r1": r1, "v1": v1c, "et0": et0}


def optimize_transfer(et0, tof_grid=None, leo_alt=200.0, llo_alt=100.0,
                      lead_deg=110.0):
    """Sweep TOF; return (best, all_records) minimizing total Delta-v."""
    if tof_grid is None:
        tof_grid = np.arange(3.0, 6.01, 0.5)
    recs = []
    for tof in tof_grid:
        d = design_transfer(et0, float(tof), leo_alt, llo_alt, lead_deg)
        if d is not None:
            recs.append(d)
    if not recs:
        return None, []
    best = min(recs, key=lambda r: r["total_ms"])
    return best, recs
