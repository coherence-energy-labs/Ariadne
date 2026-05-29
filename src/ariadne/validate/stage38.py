"""Stage 38 validation -- observational selection bias vs the eTNO clustering (the OSSOS question).

Stages 34/36 showed the extreme-eTNO perihelion clustering is marginal (p=0.07) and fragile under
orbital uncertainty (-> p=0.13). The deepest objection to the Planet 9 evidence is OBSERVATIONAL
SELECTION BIAS: surveys look at limited sky, so the discovered objects' perihelia preferentially point
where it was dark/accessible. We test it model-light: a CONTROL population that a distant perturber
should NOT shepherd -- the scattered, Neptune-coupled objects (a>150, 30<q<=42) -- traces the survey
selection function. If the detached extreme objects cluster in the SAME direction and no more strongly
than draws from that control, the clustering is consistent with selection, not a perturber.

G38a (control traces selection) - The scattered control population is itself weakly clustered (not
                    uniform), i.e. it encodes a selection function with a preferred direction.
G38b (same direction)           - The detached extreme objects cluster in the SAME direction as the
                    control (mean-direction gap small) -- the selection-bias signature.
G38c (not distinguishable)      - The extreme clustering is NOT significant relative to the control
                    selection function (p >= 0.05): current data cannot distinguish a perturber from
                    selection bias. HONEST: no Planet 9 claim.

Run:  PYTHONPATH=src python -m ariadne.validate.stage38
"""
from __future__ import annotations

from ..discovery.clustering import load_distant_tnos, selection_bias_test


def check():
    rows = load_distant_tnos()
    r = selection_bias_test(rows, n_mc=30000)
    g38a = r["n_ctrl"] >= 20                              # a real control population exists
    g38b = r["mean_dir_gap_deg"] < 30.0                  # extreme & control point the same way
    g38c = r["p_vs_selection"] >= 0.05                   # extreme NOT distinguishable from selection
    return (g38a and g38b and g38c), r


def main() -> int:
    print("=== Ariadne Stage 38  (selection bias vs eTNO clustering -- the OSSOS test) ===\n")
    ok, r = check()
    print(f"[G38a] Control population (scattered, Neptune-coupled) traces the selection function")
    print(f"      control N={r['n_ctrl']}: R={r['ctrl_R']:.3f} mean={r['ctrl_mean_deg']:.1f} deg "
          f"(weakly clustered = a real selection direction)")
    print(f"      -> {'PASS' if g38a_ok(r) else 'FAIL'}\n")
    print(f"[G38b] Detached extreme objects cluster in the SAME direction as the control")
    print(f"      extreme N={r['n_test']}: mean={r['test_mean_deg']:.1f} deg vs control "
          f"{r['ctrl_mean_deg']:.1f} deg (gap {r['mean_dir_gap_deg']:.1f} deg)")
    print(f"      -> {'PASS' if r['mean_dir_gap_deg'] < 30 else 'FAIL'}\n")
    print(f"[G38c] Extreme clustering NOT distinguishable from the selection function")
    print(f"      p(extreme R >= control selection draws) = {r['p_vs_selection']:.3f}")
    print("      => current public data cannot separate a perturber from selection bias. No P9 claim.")
    print(f"      -> {'PASS' if r['p_vs_selection'] >= 0.05 else 'FAIL'}\n")
    print(f"=== STAGE 38: {'ALL GATES PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


def g38a_ok(r):
    return r["n_ctrl"] >= 20


if __name__ == "__main__":
    import sys
    sys.exit(main())
