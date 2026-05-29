"""Dynamical-structure mining of the real distant-TNO catalog -- MASTER_PLAN.md Stage 40.

Beyond the perihelion (varpi) clustering (Stages 34/36/38, found marginal + selection-explained),
the Planet 9 case also invokes (a) ORBITAL-POLE clustering (the eTNO orbit planes aligning) and the
objects' decoupling from Neptune. This stage checks those additional signatures on the live catalog,
selection-aware, and reports honestly whether any reveal structure a perturber would imprint.

Standard astrometry only; no coherence, no new physics.
"""
from __future__ import annotations

import numpy as np

from .clustering import load_distant_tnos, filter_population

NEPTUNE_A = 30.069        # AU


def orbital_poles(rows):
    """Unit orbit-normal (pole) vectors h-hat from (i, Omega)."""
    i = np.radians([r["i_deg"] for r in rows])
    Om = np.radians([r["Omega_deg"] for r in rows])
    return np.stack([np.sin(i) * np.sin(Om), -np.sin(i) * np.cos(Om), np.cos(i)], axis=1)


def pole_clustering_vs_control(rows, a_min=250.0, q_min=42.0, n_mc=20000, seed=0):
    """Is the extreme population's orbital-pole concentration above the survey selection function?

    The control (scattered, Neptune-coupled) population traces selection. R = |mean pole|.
    Returns the test/control R and p = fraction of control draws with R >= the test R.
    """
    ext = filter_population(rows, a_min, q_min)
    ctrl = [r for r in rows if r["a_au"] >= 150.0 and q_min - 12.0 < r["q_au"] <= q_min]
    he, hc = orbital_poles(ext), orbital_poles(ctrl)
    Re = float(np.linalg.norm(he.mean(0)))
    Rc = float(np.linalg.norm(hc.mean(0)))
    rng = np.random.default_rng(seed)
    N = len(ext)
    Rs = np.array([np.linalg.norm(hc[rng.choice(len(hc), N)].mean(0)) for _ in range(n_mc)])
    return {"n_ext": len(ext), "n_ctrl": len(ctrl), "ext_R": Re, "ctrl_R": Rc,
            "p_vs_selection": float((Rs >= Re).mean())}


def nearest_low_order_resonance(a_au, max_order=12, a_ref=NEPTUNE_A):
    """Nearest LOW-ORDER p:q mean-motion resonance with Neptune; returns (p, q, fractional offset).

    Only p,q <= max_order count (high-order resonances are dynamically negligible). The period
    ratio is (a/a_ref)^1.5. For a very distant object the ratio is huge, so no low-order p:q exists
    (p would have to be enormous) -> returns (None, None, 1.0), i.e. decoupled from Neptune.
    """
    ratio = (a_au / a_ref) ** 1.5
    p, q, off = None, None, 1.0
    for qq in range(1, max_order + 1):
        pp = round(ratio * qq)
        if 1 <= pp <= max_order and abs(ratio - pp / qq) < off:
            p, q, off = pp, qq, abs(ratio - pp / qq)
    return p, q, off


def neptune_decoupling(rows, a_min=250.0, q_min=42.0, max_order=12, near=0.02):
    """Fraction of the extreme population near a LOW-ORDER Neptune resonance (should be ~0: detached)."""
    ext = filter_population(rows, a_min, q_min)
    near_count = 0
    for r in ext:
        p, q, off = nearest_low_order_resonance(r["a_au"], max_order)
        if p is not None and off < near:
            near_count += 1
    return {"n_ext": len(ext), "n_near_resonance": near_count,
            "frac_near": near_count / max(len(ext), 1)}
