"""Unified performance harness -- benchmark every integration backend (MASTER_PLAN.md Stage 32).

Measures, on the current machine:
  * single long trajectory  : pure-Python vs numba symplectic map (steps/s, speedup)
  * ensemble integration     : numba 1-core vs numba 24-core vs numba.cuda GPU (and the crossover)
and prints what the intelligent selector (`dynamics.integrators`) would choose at each size.

Run:  PYTHONPATH=src python -m ariadne.perf
"""
from __future__ import annotations

import os
import time

import numpy as np

from .dynamics import secular as S
from .dynamics import secular_fast as SF
from .dynamics import secular_gpu as SG
from .dynamics import integrators as IN
from .dynamics.secular import System, YEAR_S
from .fields.hidden_mass import CLUSTERED_ETNOS


def _ensemble(N, seed=0):
    sys = S.build_system("2026-01-01T00:00:00")
    g0 = np.hstack([sys.Q[:sys.n_massive], sys.V[:sys.n_massive]])
    rng = np.random.default_rng(seed)
    tp = []
    for k in range(N):
        o = CLUSTERED_ETNOS[k % len(CLUSTERED_ETNOS)]
        pos, vel = S.elements_to_state(o["a_au"] * (1 + rng.normal(0, 1e-3)),
                                       min(.97, max(.01, o["e"] + rng.normal(0, 1e-3))),
                                       o["i"], o["Omega"], o["omega"], 180.0 + rng.normal(0, .5))
        tp.append(np.hstack([pos, vel + sys._v_sun0]))
    return g0, sys.gm.copy(), sys.gm0, np.array(tp)


def bench_single_trajectory(steps=50000):
    """Pure-Python vs numba symplectic map on one trajectory (Sun + giants + 1 eTNO)."""
    def make():
        sys = S.build_system("2026-01-01T00:00:00")
        return S.add_test_particles(sys, [S.elements_to_state(506.8, 0.859, 11.93, 144.4, 311.29, 180.0)])
    b = make(); SF.integrate_fast(b, 1.0 * YEAR_S, 100)              # JIT warmup
    a = make(); t0 = time.time(); S.integrate(a, 1.0 * YEAR_S, steps); tp = time.time() - t0
    b = make(); t0 = time.time(); SF.integrate_fast(b, 1.0 * YEAR_S, steps); tf = time.time() - t0
    return {"steps": steps, "python_sps": steps / tp, "numba_sps": steps / tf,
            "speedup": tp / tf}


def bench_ensemble(sizes=(1000, 10000, 100000), n_steps=4000):
    """numba 1-core vs 24-core vs GPU for ensembles of N test particles."""
    SF.integrate_ensemble_parallel(*_ensemble(8)[:3], _ensemble(8)[3], 1.0 * YEAR_S, 50)  # warmup
    rows = []
    for N in sizes:
        g0, gm, gm0, tp0 = _ensemble(N)
        def one():
            Q = np.vstack([g0[:, :3], tp0[:, :3]]); V = np.vstack([g0[:, 3:], tp0[:, 3:]])
            sys = System(Q=Q, V=V, gm=gm.copy(), gm0=gm0, n_massive=g0.shape[0])
            SF.integrate_fast(sys, 1.0 * YEAR_S, n_steps)
        times = {}
        if N <= 100000:
            t0 = time.time(); one(); times["numba-1core"] = time.time() - t0
        t0 = time.time(); SF.integrate_ensemble_parallel(g0, gm, gm0, tp0.copy(), 1.0 * YEAR_S, n_steps)
        times["numba-24core"] = time.time() - t0
        if SG.HAVE_CUDA:
            t0 = time.time(); SG.integrate_ensemble_gpu(g0, gm, gm0, tp0.copy(), 1.0 * YEAR_S, n_steps)
            times["gpu"] = time.time() - t0
        winner = min(times, key=times.get)
        rec, _ = IN.recommend_backend(N, n_steps)
        rows.append({"N": N, "times": times, "winner": winner, "selector": rec})
    return rows


def main():
    print(f"=== Ariadne performance harness ===  ({os.cpu_count()} CPU cores, "
          f"CUDA={SG.HAVE_CUDA}, numba={SF.HAVE_NUMBA})\n")

    st = bench_single_trajectory()
    print("Single trajectory (symplectic Wisdom-Holman map):")
    print(f"  pure-Python : {st['python_sps']:>10.0f} steps/s")
    print(f"  numba       : {st['numba_sps']:>10.0f} steps/s   ({st['speedup']:.0f}x)\n")

    print(f"Ensemble integration (n_steps={4000}):")
    print(f"  {'N':>8}  {'numba-1core':>12}  {'numba-24core':>13}  {'gpu':>9}  {'winner':>13}  {'selector':>10}")
    for r in bench_ensemble():
        c1 = f"{r['times'].get('numba-1core'):.2f}s" if 'numba-1core' in r['times'] else "--"
        c2 = f"{r['times']['numba-24core']:.2f}s"
        cg = f"{r['times'].get('gpu'):.2f}s" if 'gpu' in r['times'] else "--"
        print(f"  {r['N']:>8}  {c1:>12}  {c2:>13}  {cg:>9}  {r['winner']:>13}  {r['selector']:>10}")
    print("\n  (single trajectory -> numba; the GPU only pays off for very large ensembles, and on a")
    print("   GeForce GPU float64 is throttled ~1/64, so the 24-core CPU is the workhorse here.)")


if __name__ == "__main__":
    main()
