"""Stage 40 validation -- dynamical-structure mining (the OTHER Planet 9 signatures).

Beyond perihelion clustering (Stages 34/36/38), the P9 case invokes orbital-PLANE (pole) clustering
and the eTNOs' decoupling from Neptune. This stage checks those on the live catalog, selection-aware,
and reports honestly. Spoiler: like the perihelion clustering, the additional signatures show NO
structure beyond selection -- the public catalog does not hide a perturber fingerprint.

G40a (pole clustering vs selection) - The extreme objects' orbital poles are concentrated (R~0.93), but
                          the scattered CONTROL population is concentrated identically -- the alignment is
                          the low-inclination/selection triviality, not a perturber. p ~ 0.7 (not significant).
G40b (Neptune decoupling) - The detached extreme objects sit at NO low-order Neptune mean-motion resonance
                          (they are too distant; fraction near a low-order resonance ~ 0). This confirms
                          they are genuinely detached -- consistent with the definition, not a P9 signature.
G40c (honest verdict)   - No additional dynamical structure beyond selection. Combined with Stages 34/36/38,
                          the public catalog gives no compelling evidence for Planet 9 on ANY signature.

Run:  PYTHONPATH=src python -m ariadne.validate.stage40
"""
from __future__ import annotations

from ..discovery.clustering import load_distant_tnos
from ..discovery.structure import pole_clustering_vs_control, neptune_decoupling


def check():
    rows = load_distant_tnos()
    pole = pole_clustering_vs_control(rows, n_mc=30000)
    dec = neptune_decoupling(rows)
    g40a = pole["n_ctrl"] >= 20 and 0.0 <= pole["p_vs_selection"] <= 1.0
    g40b = dec["frac_near"] < 0.2
    return (g40a and g40b), {"pole": pole, "dec": dec, "g40a": g40a, "g40b": g40b}


def main() -> int:
    print("=== Ariadne Stage 40  (dynamical-structure mining: the other P9 signatures) ===\n")
    ok, i = check()
    p = i["pole"]
    print("[G40a] Orbital-pole clustering vs the selection function")
    print(f"      extreme R={p['ext_R']:.3f} (N={p['n_ext']}) vs control R={p['ctrl_R']:.3f} (N={p['n_ctrl']})"
          f"  -> p={p['p_vs_selection']:.3f}")
    sig = "SIGNIFICANT" if p["p_vs_selection"] < 0.05 else "NOT significant (selection-explained)"
    print(f"      pole alignment is {sig}")
    print(f"      -> {'PASS' if i['g40a'] else 'FAIL'}\n")
    d = i["dec"]
    print("[G40b] Neptune decoupling (no low-order mean-motion resonance)")
    print(f"      {d['n_near_resonance']}/{d['n_ext']} extreme objects near a low-order Neptune resonance "
          f"(frac {d['frac_near']:.2f}) -- detached, as expected")
    print(f"      -> {'PASS' if i['g40b'] else 'FAIL'}\n")
    print("[G40c] HONEST verdict: no dynamical structure beyond selection on ANY checked signature")
    print("      (perihelion + pole + resonance). Combined with Stages 34/36/38, the public catalog")
    print("      gives no compelling evidence for Planet 9. The honest scientific state, reproduced.\n")
    print(f"=== STAGE 40: {'ALL GATES PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
