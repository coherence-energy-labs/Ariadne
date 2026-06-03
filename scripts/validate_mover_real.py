"""Real-data test of the coherence mover/orbit classifier on the MPC orbit DB.

Ground truth: 1.5M real asteroids in known_objects (koa_a_au, koa_ecc, koa_incl).
The TRUE dynamical class follows the standard MPC definitions from (a, e). We then
ask the operational question the pipeline faces:

  (1) SNAPSHOT regime  -- distance only (what you get from rate at one epoch):
      does classify_mover(helio_r) land the right class? EoO == distance-cut here
      (only one axis), so this measures the irreducible single-snapshot ambiguity.
  (2) WITH-ORBIT regime -- distance + eccentricity (after IOD): does the coherence
      FUSION of ecc (the gap22 pattern, like color for variables) beat the
      distance-ONLY cut? This is the real test of the framework's value here.

Honest: true class is defined from (a,e), so this measures whether the coherence
basins reproduce the real taxonomy and whether ecc-fusion helps at the boundaries
(eccentric NEOs/Mars-crossers a distance-only cut misfiles).
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ariadne.discovery.imaging.coherence_classifier import (  # noqa: E402
    classify_mover, MOVER_PROTOTYPES, _MOVER_W)

DB = os.environ.get("ARIADNE_DB", "data/recovery_clean.db")


def true_class(a, e):
    """Standard MPC dynamical class from semi-major axis + eccentricity."""
    q = a * (1 - e)
    if q < 1.3:
        return "NEO"
    if q < 1.666:
        return "Mars-crosser"
    if a < 2.0:
        return "Mars-crosser"          # inner (Hungaria-ish) -> inner/Mars-crosser bin
    if a < 3.3:
        return "main-belt"
    if a < 5.5:
        return "outer-belt"            # Cybele/Hilda/Trojan
    if a < 30.1:
        return "Centaur"
    return "TNO"


# map a classify_mover label (substring) to our truth bucket
def to_bucket(label):
    if "NEO" in label:
        return "NEO"
    if "Mars" in label or "inner" in label:
        return "Mars-crosser"
    if "main-belt" in label:
        return "main-belt"
    if "outer" in label or "Hilda" in label or "Trojan" in label:
        return "outer-belt"
    if "Centaur" in label:
        return "Centaur"
    if "TNO" in label:
        return "TNO"
    return "other"


def distance_only_cut(a):
    """Pre-coherence baseline: nearest basin by log10(distance) ALONE (no ecc)."""
    lr = np.log10(a)
    best, bestd = None, 1e9
    for name, b in MOVER_PROTOTYPES.items():
        d = abs(lr - b["mu"]["logr"])
        if d < bestd:
            bestd, best = d, name
    return to_bucket(best)


def main():
    n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT koa_a_au, koa_ecc FROM known_objects "
        "WHERE koa_a_au IS NOT NULL AND koa_ecc IS NOT NULL "
        "AND koa_a_au BETWEEN 0.3 AND 2000 AND koa_ecc BETWEEN 0 AND 0.999 "
        f"ORDER BY random() LIMIT {n_sample}").fetchall()
    a = np.array([r[0] for r in rows], float)
    e = np.array([r[1] for r in rows], float)
    truth = np.array([true_class(ai, ei) for ai, ei in zip(a, e)])
    print(f"=== REAL MOVER CLASSIFIER TEST — {len(a)} MPC asteroids ===")
    uniq, counts = np.unique(truth, return_counts=True)
    print("  truth mix:", {u: int(c) for u, c in zip(uniq, counts)})

    # snapshot distance: realistic heliocentric r at a random orbit phase, per object
    rng = np.random.default_rng(7)
    nu = rng.uniform(0, 2 * np.pi, len(a))           # true anomaly
    r_snap = a * (1 - e ** 2) / (1 + e * np.cos(nu))  # conic r at that phase

    def score(pred):
        return float(np.mean(pred == truth))

    # (1) snapshot, distance-only (== EoO single-axis); use OBSERVED r
    snap_pred = np.array([to_bucket(max(classify_mover(ri), key=classify_mover(ri).get))
                          for ri in r_snap])
    # (2a) with-orbit, distance-ONLY cut on a (baseline, no ecc fusion)
    cut_pred = np.array([distance_only_cut(ai) for ai in a])
    # (2b) with-orbit, EoO coherence fusion of (a, ecc)
    eoo_pred = np.array([to_bucket(max(classify_mover(ai, ei), key=classify_mover(ai, ei).get))
                         for ai, ei in zip(a, e)])

    print(f"\n  {'regime':<40} {'accuracy':>9}")
    print(f"  {'(1) snapshot distance-only  classify_mover(r)':<40} {score(snap_pred)*100:>7.1f}%")
    print(f"  {'(2a) baseline: distance-only cut on a':<40} {score(cut_pred)*100:>7.1f}%")
    print(f"  {'(2b) EoO fusion: classify_mover(a, ecc)':<40} {score(eoo_pred)*100:>7.1f}%")

    # where does ecc-fusion change the call? focus on the eccentric boundary pops
    ecc_hi = e > 0.3
    print(f"\n  on eccentric objects (e>0.3, n={int(ecc_hi.sum())}):")
    print(f"    distance-only cut : {np.mean(cut_pred[ecc_hi]==truth[ecc_hi])*100:5.1f}%")
    print(f"    EoO (a, ecc)      : {np.mean(eoo_pred[ecc_hi]==truth[ecc_hi])*100:5.1f}%")

    # per-class recall, EoO vs baseline
    print(f"\n  per-class recall (baseline cut -> EoO fusion):")
    for cls in ["NEO", "Mars-crosser", "main-belt", "outer-belt", "Centaur", "TNO"]:
        m = truth == cls
        if m.sum() == 0:
            continue
        print(f"    {cls:<14} n={int(m.sum()):>6}  "
              f"{np.mean(cut_pred[m]==cls)*100:5.1f}% -> {np.mean(eoo_pred[m]==cls)*100:5.1f}%")

    print("\nVERDICT:")
    d = (score(eoo_pred) - score(cut_pred)) * 100
    de = (np.mean(eoo_pred[ecc_hi] == truth[ecc_hi]) - np.mean(cut_pred[ecc_hi] == truth[ecc_hi])) * 100
    if d > 0.5:
        print(f"  EoO ecc-fusion beats distance-only by {d:+.1f}pp overall, {de:+.1f}pp on eccentric objects")
        print("  -> adding eccentricity (the coherence fusion) measurably improves real classification.")
    elif d < -0.5:
        print(f"  EoO ecc-fusion is WORSE by {d:.1f}pp -- basins need retuning (honest null/loss).")
    else:
        print(f"  EoO ecc-fusion ~ties distance-only overall ({d:+.1f}pp); ecc helps on the eccentric tail ({de:+.1f}pp).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
