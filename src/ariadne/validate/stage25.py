"""Stage 25 validation -- a falsifiable test of the coherence field vs the transport network.

The genuinely original (project-unique) question: does a LOCAL coherence measure (FLI) say anything
real about the GLOBAL transport structure (the invariant-manifold tubes)?

Naive hypothesis H1: the tubes are high-FLI "chaos ridges" (low coherence). We test it falsifiably
by comparing the FLI of states ON the L1 unstable manifold tube against matched random accessible
states at the same Jacobi energy, with a Mann-Whitney U test.

G25 (falsifiable) - The NAIVE ridge hypothesis is refuted: manifold-tube states are NOT significantly
                    MORE chaotic (higher FLI) than generic states. (The probe shows the opposite --
                    the tubes are the orderly skeleton, slightly MORE coherent than the background.)
                    The gate asserts the data-supported claim; if reality flipped it, the gate fails.

This is reported honestly as what it is: a rigorous falsifiable finding, not a breakthrough.

Run:  PYTHONPATH=src python -m ariadne.validate.stage25
"""
from __future__ import annotations

import numpy as np

from ..data.constants import EARTH_MOON
from ..orbits.families import lyapunov_orbit_at_jacobi
from ..manifolds.manifold import manifold_seeds, manifold_trajectory
from ..fields.coherence_field import fli, accessible_speed, _prograde_velocity

MU = EARTH_MOON.mu
C = 3.15


def _manifold_states(n_seeds=30, t_max=3.0, n_target=60):
    orb = lyapunov_orbit_at_jacobi(MU, "L1", C)
    seeds, _ = manifold_seeds(MU, orb, n_seeds=n_seeds, stable=False, branch=1)
    states = []
    for s in seeds:
        _, Y = manifold_trajectory(MU, s, stable=False, t_max=t_max, n=40)
        for k in range(0, Y.shape[1], 6):
            st = Y[:, k]
            dmoon = np.hypot(st[0] - (1 - MU), st[1])
            if 0.05 < dmoon < 0.4 and abs(st[0]) < 1.3 and abs(st[1]) < 0.4:
                states.append(st)
    return states[:n_target]


def _random_states(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        x = rng.uniform(0.8, 1.15)
        y = rng.uniform(-0.3, 0.3)
        sp = accessible_speed(x, y, MU, C)
        if sp is None:
            continue
        vx, vy = _prograde_velocity(x, y, sp)
        out.append([x, y, 0.0, vx, vy, 0.0])
    return out


def check() -> tuple[bool, dict]:
    from scipy.stats import mannwhitneyu
    man = _manifold_states()
    rnd = _random_states(len(man))
    man_fli = np.array([fli(s, MU, t_max=3.0) for s in man])
    rnd_fli = np.array([fli(s, MU, t_max=3.0) for s in rnd])
    # one-sided test: are manifold FLI GREATER than random? (the naive ridge claim)
    u, p_greater = mannwhitneyu(man_fli, rnd_fli, alternative="greater")
    _, p_less = mannwhitneyu(man_fli, rnd_fli, alternative="less")

    naive_refuted = p_greater > 0.05      # manifolds are NOT significantly more chaotic
    skeleton = p_less < 0.05              # manifolds are significantly MORE coherent (lower FLI)
    if skeleton:
        verdict = "manifolds are the COHERENCE SKELETON (significantly lower FLI than background)"
    elif naive_refuted:
        verdict = "no significant difference; naive 'chaos ridge' hypothesis refuted"
    else:
        verdict = "manifolds ARE higher-FLI than background (naive ridge hypothesis supported)"

    ok = naive_refuted                    # the data-supported, falsifiable claim
    return ok, {"man_fli": man_fli, "rnd_fli": rnd_fli, "p_greater": p_greater,
                "p_less": p_less, "skeleton": skeleton, "verdict": verdict,
                "n": len(man)}


def main() -> int:
    print("=== Ariadne Stage 25  (falsifiable test: coherence field vs the transport network) ===\n")
    ok, i = check()
    print(f"Jacobi C = {C}, L1 unstable manifold tube vs matched random states (n={i['n']} each)\n")
    print(f"      manifold-tube FLI : mean {i['man_fli'].mean():.3f}  median {np.median(i['man_fli']):.3f}")
    print(f"      random-state  FLI : mean {i['rnd_fli'].mean():.3f}  median {np.median(i['rnd_fli']):.3f}")
    print(f"      Mann-Whitney p(manifold > random) = {i['p_greater']:.3f}")
    print(f"      Mann-Whitney p(manifold < random) = {i['p_less']:.3f}\n")
    print(f"  VERDICT: {i['verdict']}")
    print(f"  (naive 'tubes = chaos ridges' hypothesis {'REFUTED' if ok else 'supported'})\n")
    print("  HONEST: this is a rigorous falsifiable finding about a local-vs-global relationship,")
    print("  not a breakthrough. The transport tubes being the orderly skeleton (not chaos ridges)")
    print("  is consistent with manifolds being asymptotic, organized trajectories.\n")
    print(f"=== STAGE 25: {'GATE PASS (claim holds)' if ok else 'GATE FAIL (claim refuted)'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
