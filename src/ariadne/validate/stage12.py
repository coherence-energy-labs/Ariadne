"""Stage 12 validation gate (MASTER_PLAN.md — coherence-weighted optimizer).

G_opt - The coherence-weighted optimizer resolves the Delta-v-vs-robustness Pareto front and
        finds an interior "knee": a transfer that is much more robust (coherent) than the
        cheapest route for only a small Delta-v premium. This operationalizes the
        coherence-guided objective J = z(Delta-v) + w * z(endpoint_sensitivity) on standard
        gravity (eta=1 firewall) -- a smoother, lower-correction route chooser.

Run:  PYTHONPATH=src python -m ariadne.validate.stage12
"""
from __future__ import annotations

from ..transfers.coherence_optimizer import (
    coherence_frontier, pareto_front, knee, weighted_optimum,
)


def check_optimizer() -> tuple[bool, dict]:
    pts = coherence_frontier()
    front = pareto_front(pts)
    k = knee(front)
    cheapest = min(pts, key=lambda p: p["dv_ms"])
    most_robust = min(pts, key=lambda p: p["sensitivity"])
    gain = cheapest["sensitivity"] / k["sensitivity"]      # robustness multiple at the knee
    premium = k["dv_ms"] - cheapest["dv_ms"]               # Delta-v premium of the knee
    info = {"pts": pts, "front": front, "knee": k, "cheapest": cheapest,
            "most_robust": most_robust, "gain": gain, "premium": premium}
    ok = (len(front) >= 3 and k["label"] != cheapest["label"]
          and gain > 2.0 and premium < 150.0)
    return ok, info


def main() -> int:
    print("=== Ariadne Stage 12 validation  (coherence-weighted trajectory optimizer) ===\n")
    ok, i = check_optimizer()
    print("[G_opt] Delta-v vs coherence (robustness) Pareto optimization")
    print("      transfer family (Delta-v, endpoint sensitivity km/(m/s)):")
    for p in sorted(i["pts"], key=lambda p: p["dv_ms"]):
        tag = "  <- PARETO" if p in i["front"] else ""
        tag += "  *** KNEE" if p["label"] == i["knee"]["label"] else ""
        print(f"        {p['label']:<10s} {p['dv_ms']:7.0f} m/s   {p['sensitivity']:8.0f}{tag}")
    print(f"\n      cheapest route : {i['cheapest']['label']} "
          f"({i['cheapest']['dv_ms']:.0f} m/s, sens {i['cheapest']['sensitivity']:.0f})")
    print(f"      KNEE (best compromise): {i['knee']['label']} "
          f"({i['knee']['dv_ms']:.0f} m/s, sens {i['knee']['sensitivity']:.0f})")
    print(f"      -> the knee is {i['gain']:.1f}x MORE ROBUST than the cheapest route "
          f"for only +{i['premium']:.0f} m/s")
    # show how the choice shifts with the robustness weight
    for w in (0.0, 1.0, 5.0):
        wp = weighted_optimum(i["pts"], w)
        print(f"      robustness weight w={w:<4.1f} -> choose {wp['label']} "
              f"({wp['dv_ms']:.0f} m/s, sens {wp['sensitivity']:.0f})")
    print(f"      -> {'PASS' if ok else 'FAIL'}\n")
    print(f"=== STAGE 12: {'COHERENCE-WEIGHTED OPTIMIZER VALIDATED' if ok else 'FAILURE'} ===")
    print("NOTE: standard gravity (eta=1); 'coherence' = robustness, a real second objective.")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
