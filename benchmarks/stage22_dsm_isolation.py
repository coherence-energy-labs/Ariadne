"""Stage 22 DSM contribution isolation: separate 'different epoch' from 'real DSM benefit'.

The Stage-22 commit message claimed a "16% Delta-v reduction" from the DSM-enabled
optimizer (5392 m/s vs 6449 m/s reference). But the reference Galileo VEEGA was found by
an EARLIER 5-dim DE; the DSM optimizer ran a NEW 9-dim DE that could find a different
epoch + TOF combination. So part of the reduction might just be "second DE run with
slightly different settings converged to a slightly different basin."

To isolate the real DSM contribution, this benchmark runs THREE DE optimizers under
matched conditions (same seed, same iter budget, same population size):

  A) NO DSM, epoch + 4 TOFs free  (5 dims)
  B) DSM allowed (1-DOF on long legs), epoch + 4 TOFs free  (9 dims)
  C) Same as B but DSMs forced OFF (frac in {0, 1})  (should match A; sanity check)

If B beats A consistently, DSMs help. If they're within ~1%, the "16% reduction" was a
seed/topology artefact of the original comparison, not a real DSM benefit.
"""
import warnings, time
warnings.filterwarnings("ignore")
import numpy as np
from scipy.optimize import differential_evolution
from ariadne.interplanetary.flyby import evaluate_chain, GALILEO_VEEGA
from ariadne.data.ephemeris import et

DAY = 86400.0
BODIES = GALILEO_VEEGA['bodies']
TOF_BOUNDS = [(120, 220), (250, 400), (650, 900), (900, 1300)]
DEP_WINDOW_DAYS = 730
e0 = et("2029-01-01T00:00:00")

print("=" * 76)
print("Stage 22 DSM contribution isolation: 3 matched-condition DE runs")
print(f"Galileo VEEGA  {BODIES}")
print(f"Same seed (0), same maxiter (300), same popsize (30), same DE settings")
print("=" * 76)


def run_de(label, bounds, obj_fn, n_seeds=5):
    """Run the DE with n_seeds different seeds; return median final objective."""
    results = []
    for seed in range(n_seeds):
        t0 = time.time()
        res = differential_evolution(obj_fn, bounds, seed=seed, maxiter=300,
                                     popsize=30, tol=1e-7, mutation=(0.5, 1.5),
                                     recombination=0.7, polish=True)
        results.append((res.fun, time.time() - t0))
    fs = sorted(r[0] for r in results)
    t_total = sum(r[1] for r in results)
    return {"label": label, "best": fs[0], "median": fs[len(fs)//2], "worst": fs[-1],
            "all": fs, "time_s": t_total}


# A) NO DSM (5 dims)
print("\n[A] NO DSM, free (epoch, 4 TOFs)  ->  5 dims, 5 seeds")
bounds_A = [(0.0, DEP_WINDOW_DAYS)] + TOF_BOUNDS


def obj_A(x):
    r = evaluate_chain(BODIES, e0 + x[0] * DAY, x[1:], flyby_alt_km=300.0)
    if r is None: return 1e12
    return r['dv_launch_ms'] + r['mismatch_dv_ms'] + 5000 * r['infeasible']


A = run_de("no_dsm", bounds_A, obj_A)
print(f"    {len(A['all'])} seeds   best={A['best']:7.0f}   median={A['median']:7.0f}   "
      f"worst={A['worst']:7.0f} m/s   ({A['time_s']:.0f}s total)")
print(f"    all seeds: {['%.0f' % f for f in A['all']]}")

# B) DSM allowed on all 4 legs (frac in [0, 1]; <0.02 or >0.98 = no-DSM zone)
print("\n[B] DSM ALLOWED on legs 0-3, free (epoch, 4 TOFs, 4 DSM fracs)  ->  9 dims, 5 seeds")
bounds_B = bounds_A + [(0.0, 1.0)] * 4


def obj_B(x):
    dsm_fracs = [float(x[5 + k]) for k in range(4)]
    r = evaluate_chain(BODIES, e0 + x[0] * DAY, x[1:5], flyby_alt_km=300.0,
                       dsm_fracs=dsm_fracs)
    if r is None: return 1e12
    return (r['dv_launch_ms'] + r['mismatch_dv_ms'] + r.get('dsm_dv_ms', 0.0)
            + 5000 * r['infeasible'])


B = run_de("with_dsm", bounds_B, obj_B)
print(f"    {len(B['all'])} seeds   best={B['best']:7.0f}   median={B['median']:7.0f}   "
      f"worst={B['worst']:7.0f} m/s   ({B['time_s']:.0f}s total)")
print(f"    all seeds: {['%.0f' % f for f in B['all']]}")

# Comparison
print("\n[3] COMPARISON")
print(f"    {'Setting':<40s}  {'best':>8s}  {'median':>8s}  {'worst':>8s}")
print(f"    {'-' * 70}")
print(f"    {'A: no DSM (5 dims)':<40s}  {A['best']:8.0f}  {A['median']:8.0f}  {A['worst']:8.0f}")
print(f"    {'B: DSM allowed (9 dims)':<40s}  {B['best']:8.0f}  {B['median']:8.0f}  {B['worst']:8.0f}")

dv_improvement_best = A['best'] - B['best']
dv_improvement_median = A['median'] - B['median']
print(f"\n    Improvement (A best - B best):     {dv_improvement_best:+.0f} m/s  "
      f"({100*dv_improvement_best/A['best']:+.1f}%)")
print(f"    Improvement (A median - B median): {dv_improvement_median:+.0f} m/s  "
      f"({100*dv_improvement_median/A['median']:+.1f}%)")

print(f"\n[4] HONEST VERDICT")
if abs(dv_improvement_median) < 0.02 * A['median']:
    print(f"    A and B are within 2% on median over 5 seeds.")
    print(f"    The DSM mechanism doesn't appreciably help on the Galileo VEEGA target.")
    print(f"    Earlier claimed '16% reduction' was likely seed/topology artefact.")
elif dv_improvement_median > 0.05 * A['median']:
    print(f"    B is consistently {100*dv_improvement_median/A['median']:.1f}% better than A.")
    print(f"    DSMs do give real benefit on this trajectory.")
else:
    print(f"    B is marginally better than A ({100*dv_improvement_median/A['median']:+.1f}%).")
    print(f"    DSM benefit is small but consistent.")
