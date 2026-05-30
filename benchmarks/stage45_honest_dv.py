"""Honest Coherence-HJB Delta-v comparison: count both velocity AND position mismatches.

The proxy Dv in the earlier bench_stage45_lowthrust.py was the per-edge velocity mismatch
||v_j − v_propagated|| only. That undercounts the real OCP cost because each edge also has
a POSITION mismatch ||r_j − r_propagated||; to actually follow the greedy policy you have
to spend Dv_position ≈ ||Dr|| / Dt (rendezvous burn) on top of the velocity correction.

This benchmark computes total per-edge cost as:

    Dv_edge  =  ||v_j − v_prop||                      (velocity mismatch)
              +  ||r_j − r_prop|| / Dt                (position-correction rendezvous Dv)

Reference comparison: a TRUE Hohmann LEO→Moon transfer = ~3.9 km/s (depart LEO @ 7.78 km/s,
arrive at lunar circular v_inf, insert). The greedy policy with HONEST per-edge cost should
be in the same regime if the value function is finding sensible transfers, otherwise it's
not actually doing OCP-grade work.
"""
import warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

import ariadne
from ariadne.optimize.coherence_hjb import (graph_laplacian, solve_helmholtz,
                                            surprise_value)
from ariadne.validate.stage45 import (_propagate_rk4, _spatial_eom_vec,
                                      _cr3bp_phase_samples_6d)

em = ariadne.system("EARTH_MOON")
MU, L_STAR, V_STAR = em.mu, em.L_star, em.V_star
TSTAR_S = em.T_star

DT_NONDIM = 0.4     # edge propagation time (nondim units)
DT_SECONDS = DT_NONDIM * TSTAR_S

print("=" * 76)
print("Coherence-HJB HONEST Delta-v comparison")
print(f"Edge cost = ||Dv|| + ||Dr|| / Dt  (rendezvous formula)")
print(f"Dt per edge = {DT_NONDIM} nondim = {DT_SECONDS:.0f} s = {DT_SECONDS/86400.0:.2f} days")
print("=" * 76)

print("\n[1] Build graph (20k samples, dynamics-aware k-NN, position-gap filter)")
samples = _cr3bp_phase_samples_6d(20000, MU, seed=0)
N = len(samples)
prop = _propagate_rk4(samples, MU, DT_NONDIM, n_steps=40, eom_vec=_spatial_eom_vec)
tree = cKDTree(samples)
dists, idxs = tree.query(prop, k=20)
sigma = float(np.median(dists))
pos_gaps = np.linalg.norm(samples[idxs][:, :, :3] - prop[:, np.newaxis, :3], axis=2)
weights = np.exp(-(dists ** 2) / (sigma ** 2 + 1e-30)) * (pos_gaps < 0.18)
rows = np.repeat(np.arange(N), 20)
W_graph = sp.csr_matrix((weights.flatten(), (rows, idxs.flatten())), shape=(N, N))
W_graph = 0.5 * (W_graph + W_graph.T).tocsr(); W_graph.eliminate_zeros()
print(f"    {N} samples, {W_graph.nnz} edges")

print("\n[2] Helmholtz solve at lunar-vicinity goal")
goal_pos = np.array([1.0 - MU, 0.0, -0.01, 0.0, 0.0, 0.0])
goal_idx = int(np.argmin(np.linalg.norm(samples - goal_pos, axis=1)))
L = graph_laplacian(W_graph, normalized=True)
V, info, n_iter = solve_helmholtz(L, gamma=1.0, D_coef=1.0, source_idx=goal_idx)
W_field = surprise_value(V, V_max=float(V[goal_idx]))
print(f"    CG: {n_iter} iters")


def greedy_honest(start_idx, max_steps=500):
    """Walk -grad W from start_idx; return (path, dv_velocity_kms, dv_position_kms)."""
    path = [int(start_idx)]
    dv_vel, dv_pos = 0.0, 0.0
    for _ in range(max_steps):
        i = path[-1]
        nbrs = W_graph[i].nonzero()[1]
        if not len(nbrs): break
        deltas = W_field[nbrs] - W_field[i]
        best = int(nbrs[int(np.argmin(deltas))])
        if W_field[best] >= W_field[i] - 1e-6: break
        # honest per-edge cost
        v_mismatch_nondim = float(np.linalg.norm(samples[best, 3:] - prop[i, 3:]))
        r_mismatch_nondim = float(np.linalg.norm(samples[best, :3] - prop[i, :3]))
        dv_vel += v_mismatch_nondim * V_STAR
        # rendezvous: Dv to cover Dr in Dt is ||Dr|| / Dt
        dv_pos += (r_mismatch_nondim * L_STAR) / DT_SECONDS
        path.append(best)
    return path, dv_vel, dv_pos


print("\n[3] Greedy from random starts -- HONEST per-edge cost")
rng = np.random.default_rng(0)
starts = rng.choice(N, size=30, replace=False)
results = []
for s in starts:
    if s == goal_idx: continue
    path, dv_v, dv_p = greedy_honest(int(s))
    end_dist = float(np.linalg.norm(samples[path[-1]] - samples[goal_idx]))
    reached = end_dist < 0.05
    if reached:
        results.append({"start": int(s), "steps": len(path), "dv_vel": dv_v,
                        "dv_pos": dv_p, "dv_total": dv_v + dv_p})

if not results:
    print("    no starts reached goal")
else:
    n_reach = len(results)
    n_total = len(starts) - 1
    print(f"    reach rate: {n_reach}/{n_total}  ({100*n_reach/n_total:.0f}%)")
    print(f"\n    {'metric':<35s}  {'p25':>10s}  {'p50':>10s}  {'p75':>10s}")
    print("    " + "-" * 70)
    for label, key in [("dv from velocity mismatch (km/s)", "dv_vel"),
                       ("dv from position rendezvous (km/s)", "dv_pos"),
                       ("dv TOTAL (km/s)", "dv_total")]:
        vals = sorted(r[key] for r in results)
        p25, p50, p75 = vals[len(vals)//4], vals[len(vals)//2], vals[3*len(vals)//4]
        print(f"    {label:<35s}  {p25:>10.2f}  {p50:>10.2f}  {p75:>10.2f}")

    p50 = sorted(r["dv_total"] for r in results)[len(results)//2]
    print(f"\n[4] Reference comparison:")
    print(f"    Hohmann LEO -> Moon (impulsive, idealized):  3.9 km/s")
    print(f"    Apollo-class direct lunar transfer:          ~3.9 km/s (LEO escape) +")
    print(f"                                                  ~0.9 km/s (LOI = lunar orbit insertion)")
    print(f"    Edelbaum low-thrust LEO -> lunar regime:     ~7-10 km/s")
    print(f"    Coherence-HJB greedy (honest total):         {p50:.1f} km/s (p50)")
    if 3.0 <= p50 <= 15.0:
        print(f"    -> in the physically plausible regime; the value function is doing real work.")
    elif p50 < 3.0:
        print(f"    -> SUSPICIOUSLY LOW: below Hohmann's physical minimum; metric is still proxying.")
    else:
        print(f"    -> high; greedy policy is taking inefficient paths through phase space.")

print(f"\n[5] Honest scope of this benchmark:")
print(f"    - This sums per-edge velocity AND position mismatch costs along the greedy path.")
print(f"    - It's still a proxy: it doesn't compute the OCP-optimal continuous-thrust control")
print(f"      that would actually steer the spacecraft through the chosen sample sequence.")
print(f"    - A direct OCP solver (collocation, indirect shooting) would give the exact answer.")
print(f"    - What this validates: the value field's gradient points toward the goal, and")
print(f"      following it gives a finite, sensibly-ordered cost. NOT that Ariadne replaces")
print(f"      a full OCP solver.")
