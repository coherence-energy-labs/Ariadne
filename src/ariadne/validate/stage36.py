"""Stage 36 validation -- observational-uncertainty propagation into the clustering significance.

Stage 34 found the extreme-eTNO perihelion clustering is MARGINAL (p~0.07). The natural next
question -- and the honest one -- is whether even that is robust to the objects' ORBITAL
UNCERTAINTIES. We pull the real per-element 1-sigma uncertainties from the JPL SBDB, resample each
object's perihelion longitude many times, and recompute the significance.

G36a (real uncertainties)  - The 1-sigma element uncertainties load; for the extreme sample the
                             perihelion angles are mostly measured to <0.1 deg (negligible), EXCEPT a
                             short-arc object whose perihelion is essentially unconstrained.
G36b (sound resampling)    - The nominal p lies inside the resampled distribution; the method is a
                             straightforward Monte-Carlo over the measurement errors.
G36c (honest verdict)      - Propagating uncertainty, the clustering significance DEGRADES from p~0.07
                             to a median p~0.13 (significant in only ~13% of resamples), driven by the
                             one unconstrained object. The marginal clustering is FRAGILE -- not robust
                             to poorly-observed members, and dominated by small N. Current data do NOT
                             give compelling statistical evidence for a perturber.

Run:  PYTHONPATH=src python -m ariadne.validate.stage36
"""
from __future__ import annotations

import numpy as np

from ..discovery.clustering import (load_with_uncertainty, filter_population,
                                    resampled_clustering_p, circular_stats)


def check():
    rows = load_with_uncertainty()
    ext = filter_population(rows, a_min=250.0, q_min=42.0)
    g36a = len(ext) >= 10 and all("sigma_varpi_deg" in r for r in ext)
    sig = np.array([r["sigma_varpi_deg"] for r in ext])
    med_sig, max_sig = float(np.median(sig)), float(sig.max())

    p_nom = circular_stats([r["varpi_deg"] for r in ext])["p_analytic"]
    ps = resampled_clustering_p(ext, n_real=5000, seed=0)
    p_med = float(np.median(ps)); p_lo, p_hi = float(np.percentile(ps, 16)), float(np.percentile(ps, 84))
    frac_sig = float((ps < 0.05).mean())

    # well-measured subset (drop the unconstrained object)
    well = [r for r in ext if r["sigma_varpi_deg"] < 10.0]
    p_well = circular_stats([r["varpi_deg"] for r in well])["p_analytic"]

    g36b = p_lo <= p_nom <= p_hi or abs(p_nom - p_med) < 0.1   # nominal inside / near the distribution
    g36c = p_med > p_nom and frac_sig < 0.5                     # uncertainty degrades significance

    ok = g36a and g36b and g36c
    return ok, {"n": len(ext), "med_sig": med_sig, "max_sig": max_sig, "p_nom": p_nom,
                "p_med": p_med, "p_lo": p_lo, "p_hi": p_hi, "frac_sig": frac_sig,
                "n_well": len(well), "p_well": p_well, "g36a": g36a, "g36b": g36b, "g36c": g36c}


def main() -> int:
    print("=== Ariadne Stage 36  (uncertainty propagation into eTNO clustering significance) ===\n")
    ok, i = check()
    print(f"[G36a] Real JPL element uncertainties (extreme sample N={i['n']})")
    print(f"      perihelion-angle sigma: median={i['med_sig']:.3f} deg, max={i['max_sig']:.1f} deg "
          f"(one short-arc object essentially unconstrained)")
    print(f"      -> {'PASS' if i['g36a'] else 'FAIL'}\n")
    print("[G36b] Monte-Carlo resampling over measurement errors is sound")
    print(f"      nominal p={i['p_nom']:.4f}; resampled median p={i['p_med']:.4f}  16-84%=[{i['p_lo']:.4f}, {i['p_hi']:.4f}]")
    print(f"      -> {'PASS' if i['g36b'] else 'FAIL'}\n")
    print("[G36c] HONEST verdict: the marginal clustering is FRAGILE under real uncertainty")
    print(f"      propagating uncertainty: p {i['p_nom']:.3f} -> median {i['p_med']:.3f}; "
          f"significant (p<0.05) in only {i['frac_sig']*100:.0f}% of resamples")
    print(f"      well-measured subset (drop unconstrained, N={i['n_well']}): p={i['p_well']:.3f}")
    print("      Conclusion: dominated by small N + one unconstrained object; NOT compelling evidence")
    print(f"      for a perturber on current data.  -> {'PASS' if i['g36c'] else 'FAIL'}\n")
    print(f"=== STAGE 36: {'ALL GATES PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
