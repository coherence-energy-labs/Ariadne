"""Solar-system-wide coherence atlas (MASTER_PLAN.md - Stage 26).

Scales the coherence-skeleton analysis to the WHOLE solar system: every major CR3BP system
(Sun-planet, the giant planets' major moons, the Pluto-Charon binary) gets its libration
structure and characteristic transport scale catalogued, and a spanning subset gets the
falsifiable coherence-skeleton test (does the Stage-25 result -- manifolds are MORE coherent
than the background -- hold across 7 orders of magnitude in mass ratio?).

The per-system test is REGION-MATCHED: random comparison states are drawn from the same (x,y)
bounding box the manifold tube occupies, so the manifold-vs-background comparison is fair at
every mass ratio (the libration zone scales with the Hill radius).
"""
from __future__ import annotations

import numpy as np

from ..orbits.families import lyapunov_family
from ..manifolds.manifold import manifold_seeds, manifold_trajectory
from .coherence_field import fli, accessible_speed, _prograde_velocity


def _tube_states(mu, orbit, t_max=3.0, n_target=40):
    """Sample states along the L1 unstable-manifold tube, away from the secondary."""
    seeds, _ = manifold_seeds(mu, orbit, n_seeds=24, stable=False, branch=1)
    states = []
    for s in seeds:
        _, Y = manifold_trajectory(mu, s, stable=False, t_max=t_max, n=40)
        for k in range(0, Y.shape[1], 6):
            st = Y[:, k]
            dsec = np.hypot(st[0] - (1 - mu), st[1])
            if 0.02 < dsec < 0.5 and abs(st[1]) < 0.45:
                states.append(st)
    return states[:n_target]


def coherence_skeleton_test(system, t_max=3.0, seed=0):
    """Region-matched FLI test: are L1 manifold-tube states more coherent than the background?

    Returns means, the one-sided Mann-Whitney p (manifold < random), and the verdict.
    """
    from scipy.stats import mannwhitneyu
    mu = system.mu
    fam = lyapunov_family(mu, "L1", amplitude0=1e-3, dx=2e-3, n=24)
    orbit = fam[len(fam) // 2].orbit               # moderate-amplitude orbit (well-developed tube)
    C = orbit.jacobi

    man = _tube_states(mu, orbit, t_max=t_max)
    if len(man) < 12:
        return {"system": system.name, "mu": mu, "ok": False, "reason": "too few tube states"}
    man = np.array(man)
    xs, ys = man[:, 0], man[:, 1]
    box = (xs.min(), xs.max(), ys.min(), ys.max())

    man_fli = np.array([fli(s, mu, t_max=t_max) for s in man])

    rng = np.random.default_rng(seed)
    rnd_fli, tries = [], 0
    while len(rnd_fli) < len(man) and tries < 50 * len(man):
        tries += 1
        x = rng.uniform(box[0], box[1])
        y = rng.uniform(box[2], box[3])
        sp = accessible_speed(x, y, mu, C)
        if sp is None:
            continue
        vx, vy = _prograde_velocity(x, y, sp)
        rnd_fli.append(fli([x, y, 0.0, vx, vy, 0.0], mu, t_max=t_max))
    rnd_fli = np.array(rnd_fli)
    if len(rnd_fli) < 12:
        return {"system": system.name, "mu": mu, "ok": False, "reason": "too few random states"}

    _, p_less = mannwhitneyu(man_fli, rnd_fli, alternative="less")
    _, p_greater = mannwhitneyu(man_fli, rnd_fli, alternative="greater")
    return {"system": system.name, "mu": mu, "jacobi": float(C),
            "man_mean": float(man_fli.mean()), "rnd_mean": float(rnd_fli.mean()),
            "p_less": float(p_less), "p_greater": float(p_greater),
            "skeleton": bool(p_less < 0.05),       # tube MORE coherent (the Stage-25 claim)
            "separatrix": bool(p_greater < 0.05),  # tube LESS coherent (textbook separatrix)
            "n": len(man_fli), "ok": True}


def system_catalog(system):
    """Libration + characteristic transport scale for one system (fast)."""
    from ..transfers.jovian import moon_libration
    m = moon_libration(system)
    return {"system": system.name, "primary": system.primary, "secondary": system.secondary,
            "mu": system.mu, "L_star_km": system.L_star, "V_star_kms": system.V_star,
            "L1_km": float(m["L1_km"]), "L2_km": float(m["L2_km"]),
            "lyap_period_d": float(m["lyap_period_d"]),
            "half_period_residual": float(m["orbit"].half_period_residual)}


def build_solar_atlas(systems, test_systems):
    """Catalog every system + run the coherence-skeleton test on the spanning subset."""
    catalog = [system_catalog(s) for s in systems]
    tests = [coherence_skeleton_test(s) for s in test_systems]
    return {"catalog": catalog, "skeleton_tests": tests}
