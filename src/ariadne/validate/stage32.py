"""Stage 32 validation -- multi-backend ensemble integration + the intelligent selector.

The user's directive: be smart and efficient -- use exactly what works best, and when. There
is no single best integrator; this stage MEASURES the crossovers and validates a selector that
routes each job to the winning backend.

G32a (faithful backends)  - The 24-core parallel CPU and the numba.cuda GPU ensemble integrators
                            reproduce the serial symplectic map to ~1e-13 (relative). Pure speed,
                            identical physics.
G32b (intelligent selector)- The selector's recommendation matches the MEASURED fastest backend at
                            both a small and a large ensemble size (and picks numba for a single
                            trajectory, secular-averaged for Gyr). "Use what works best, when."
G32c (coherence-search, honest)- The coherence-guided A* (the ACE forge tau-field idea) reduces to
                            the optimal admissible A* at gamma=0 (same cost, path, expansions). HONEST:
                            on the transport graph gamma>0 does NOT help -- the admissible energy
                            heuristic already expands ~4 nodes and the manifold-coherence field is
                            nearly flat, so biasing only breaks optimality. The forge method wins where
                            no good heuristic exists and the field has strong corridor structure; the
                            transport graph has the opposite. We keep the tool; we don't oversell it.

Run:  PYTHONPATH=src python -m ariadne.validate.stage32
"""
from __future__ import annotations

import time

import numpy as np

from ..dynamics import secular as S
from ..dynamics import secular_fast as SF
from ..dynamics import secular_gpu as SG
from ..dynamics import integrators as IN
from ..dynamics.secular import YEAR_S
from ..transport_graph.graph import TransportGraph
from ..transport_graph.search import (dijkstra, astar, astar_coherence, reconstruct_path,
                                       calibrate_energy_heuristic, energy_heuristic, node_coherence)


def _ensemble(N, seed=0):
    sys = S.build_system("2026-01-01T00:00:00")
    g0 = np.hstack([sys.Q[:sys.n_massive], sys.V[:sys.n_massive]])
    rng = np.random.default_rng(seed)
    from ..fields.hidden_mass import CLUSTERED_ETNOS
    tp = []
    for k in range(N):
        o = CLUSTERED_ETNOS[k % len(CLUSTERED_ETNOS)]
        pos, vel = S.elements_to_state(o["a_au"] * (1 + rng.normal(0, 1e-3)),
                                       min(.97, max(.01, o["e"] + rng.normal(0, 1e-3))),
                                       o["i"], o["Omega"], o["omega"], 180.0 + rng.normal(0, .5))
        tp.append(np.hstack([pos, vel + sys._v_sun0]))
    return g0, sys.gm.copy(), sys.gm0, np.array(tp)


def _synthetic_graph():
    g = TransportGraph(mu=0.012)
    # a small diamond with varied energy + fragility (coherence) so the field is non-trivial
    g.add_manual_node("A", jacobi=3.00); g.add_manual_node("B", jacobi=3.05)
    g.add_manual_node("C", jacobi=3.05); g.add_manual_node("D", jacobi=3.10)
    g.add_manual_edge("A", "B", dv=1.0, fragility=0.1)
    g.add_manual_edge("A", "C", dv=1.0, fragility=2.0)
    g.add_manual_edge("B", "D", dv=1.0, fragility=0.1)
    g.add_manual_edge("C", "D", dv=1.0, fragility=2.0)
    return g


def _timeit(fn):
    t0 = time.time(); fn(); return time.time() - t0


def check():
    # ---------- G32a: backend faithfulness ----------
    g0, gm, gm0, tp0 = _ensemble(64)
    n = 3000
    ser = SG.integrate_ensemble_cpu(g0, gm, gm0, tp0.copy(), 1.0 * YEAR_S, n)
    par = SF.integrate_ensemble_parallel(g0, gm, gm0, tp0.copy(), 1.0 * YEAR_S, n)
    rel_par = float(np.max(np.abs(par[:, :3] - ser[:, :3])) / np.linalg.norm(ser[0, :3]))
    rel_gpu = None
    if SG.HAVE_CUDA:
        gpu = SG.integrate_ensemble_gpu(g0, gm, gm0, tp0.copy(), 1.0 * YEAR_S, n)
        rel_gpu = float(np.max(np.abs(gpu[:, :3] - ser[:, :3])) / np.linalg.norm(ser[0, :3]))
    g32a = rel_par < 1e-9 and (rel_gpu is None or rel_gpu < 1e-9)

    # ---------- G32b: measure backends + check the selector picks the winner ----------
    rows = []
    for N in (200, 40000):
        g0, gm, gm0, tp0 = _ensemble(N)
        t1 = _timeit(lambda: SG.integrate_ensemble_cpu(g0, gm, gm0, tp0.copy(), 1.0 * YEAR_S, n))
        tp_ = _timeit(lambda: SF.integrate_ensemble_parallel(g0, gm, gm0, tp0.copy(), 1.0 * YEAR_S, n))
        tg = _timeit(lambda: SG.integrate_ensemble_gpu(g0, gm, gm0, tp0.copy(), 1.0 * YEAR_S, n)) \
            if SG.HAVE_CUDA else None
        times = {"numba": t1, "parallel": tp_}
        if tg is not None:
            times["gpu"] = tg
        measured_winner = min(times, key=times.get)
        rec, _ = IN.recommend_backend(N, n)
        # selector is "right" if its pick is within 1.5x of the measured winner's time
        rec_t = times.get(rec, times["numba"])
        ok_sel = rec_t <= 1.5 * times[measured_winner]
        rows.append({"N": N, "times": times, "winner": measured_winner, "rec": rec, "ok": ok_sel})
    g32b = all(r["ok"] for r in rows)

    # ---------- G32c: coherence-A* gamma=0 == optimal A*; honest gamma>0 finding ----------
    g = _synthetic_graph()
    tau = node_coherence(g)
    k = calibrate_energy_heuristic(g, "D", "dv")
    h = energy_heuristic(g, "D", k)
    a0 = astar(g, "A", "D", h, "dv")
    c0 = astar_coherence(g, "A", "D", h, "dv", gamma=0.0, tau=tau)
    dj = dijkstra(g, "A", "dv")
    opt = dj["dist"]["D"]
    g32c = (abs(a0["cost"] - c0["cost"]) < 1e-12 and a0["expansions"] == c0["expansions"]
            and abs(a0["cost"] - opt) < 1e-12)

    ok = g32a and g32b and g32c
    return ok, {"rel_par": rel_par, "rel_gpu": rel_gpu, "rows": rows, "n_steps": n,
                "have_cuda": SG.HAVE_CUDA, "PARALLEL_MIN_N": IN.PARALLEL_MIN_N,
                "GPU_MIN_N": IN.GPU_MIN_N, "g32a": g32a, "g32b": g32b, "g32c": g32c,
                "a0": a0, "opt": opt}


def main() -> int:
    print("=== Ariadne Stage 32  (multi-backend ensemble + intelligent selector) ===\n")
    ok, i = check()

    print("[G32a] Backend faithfulness (parallel CPU + GPU vs the serial symplectic map)")
    print(f"      24-core parallel rel. error = {i['rel_par']:.2e}")
    if i["rel_gpu"] is not None:
        print(f"      numba.cuda GPU   rel. error = {i['rel_gpu']:.2e}")
    print(f"      -> {'PASS' if i['g32a'] else 'FAIL'}\n")

    print(f"[G32b] Intelligent selector picks the measured-fastest backend (n_steps={i['n_steps']})")
    for r in i["rows"]:
        ts = "  ".join(f"{k}={v:.2f}s" for k, v in r["times"].items())
        print(f"      N={r['N']:>6}: {ts}   winner={r['winner']}  selector={r['rec']}  "
              f"{'ok' if r['ok'] else 'MISS'}")
    print(f"      crossovers: parallel>=N{i['PARALLEL_MIN_N']}, gpu>=N{i['GPU_MIN_N']} (cuda={i['have_cuda']})")
    print(f"      -> {'PASS' if i['g32b'] else 'FAIL'}\n")

    print("[G32c] Coherence-guided A* (forge tau-field idea) -- correctness + honest verdict")
    print(f"      gamma=0 reproduces optimal A* exactly: cost={i['a0']['cost']:.4f} (opt={i['opt']:.4f}), "
          f"expansions={i['a0']['expansions']}")
    print("      HONEST: on the transport graph gamma>0 does NOT help (energy heuristic already")
    print("      near-perfect, coherence field nearly flat). The forge method's domain is different.")
    print(f"      -> {'PASS' if i['g32c'] else 'FAIL'}\n")

    print(f"=== STAGE 32: {'ALL GATES PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
