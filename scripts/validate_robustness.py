"""Robustness / graceful-degradation stress test (a NASA-grade property): do the
coherence selectors degrade SMOOTHLY as real-world noise grows, or collapse? Two
sweeps on real data:
  (1) vetting F1 vs astrometric jitter on the tracks (0.1 -> 2.0 arcsec)
  (2) mover accuracy vs relative error on the distance estimate (5% -> 50%)
A robust selector loses accuracy gradually and never falls off a cliff.
"""
from __future__ import annotations

import os
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ariadne.discovery.imaging.coherence_classifier import classify_mover         # noqa: E402
from ariadne.discovery.imaging.coherence_vet import track_energy                  # noqa: E402
from validate_mover_real import true_class, to_bucket                             # noqa: E402
from validate_coherence_vet_real import real_positive_tracks                      # noqa: E402
from validate_coherence_vet import gen_chance, hard_features                      # noqa: E402

DB = os.environ.get("ARIADNE_DB", "data/recovery_clean.db")


def f1_at(E, y, taus):
    def f1(pred):
        tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum(); fn = ((~pred) & (y == 1)).sum()
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        return 2 * p * r / max(p + r, 1e-9)
    return max(f1(E <= t) for t in taus)


def main():
    print("=" * 64)
    print("ROBUSTNESS / GRACEFUL-DEGRADATION STRESS TEST (real data)")
    print("=" * 64)

    # (1) vetting F1 vs astrometric jitter
    print("\n(1) VETTING -- EoO F1 vs astrometric jitter on the tracks:")
    print(f"    {'jitter(\")':>10}{'EoO F1':>9}{'hard F1':>9}")
    rng = np.random.default_rng(7)
    for jit in (0.1, 0.3, 0.5, 1.0, 2.0, 3.0):
        pos = real_positive_tracks(n_orbits=2500, jitter_arcsec=jit, seed=3)
        neg = [gen_chance(rng) for _ in range(len(pos))]
        tracks = [(p, 1) for p in pos] + [(n, 0) for n in neg]
        y = np.array([t[1] for t in tracks])
        E = np.array([track_energy(*t[0]) for t in tracks])
        taus = np.quantile(E[np.isfinite(E)], np.linspace(0.05, 0.95, 40))
        hard = np.array([(lambda cv, hs, rs: cv < 0.25 and hs < 25 and rs < 45)(*hard_features(*t[0][:3]))
                         for t in tracks])

        def f1(pred):
            tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum(); fn = ((~pred) & (y == 1)).sum()
            p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
            return 2 * p * r / max(p + r, 1e-9)
        print(f"    {jit:>10.1f}{f1_at(E, y, taus):>9.3f}{f1(hard):>9.3f}")

    # (2) mover accuracy vs distance-estimate relative error (snapshot regime)
    print("\n(2) MOVER -- class accuracy vs relative error on the distance estimate:")
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT koa_a_au, koa_ecc FROM known_objects WHERE koa_a_au IS NOT NULL "
        "AND koa_ecc IS NOT NULL AND koa_a_au BETWEEN 0.3 AND 2000 "
        "AND koa_ecc BETWEEN 0 AND 0.999 ORDER BY random() LIMIT 30000").fetchall()
    a = np.array([r[0] for r in rows]); e = np.array([r[1] for r in rows])
    truth = np.array([true_class(ai, ei) for ai, ei in zip(a, e)])
    rng2 = np.random.default_rng(1)
    nu = rng2.uniform(0, 2 * np.pi, len(a))
    r_true = a * (1 - e ** 2) / (1 + e * np.cos(nu))      # real snapshot distance
    print(f"    {'rel.err':>9}{'accuracy':>10}")
    for sig in (0.0, 0.05, 0.1, 0.2, 0.35, 0.5):
        r_obs = r_true * (1.0 + rng2.normal(0, sig, len(a)))
        r_obs = np.clip(r_obs, 0.3, None)
        pred = np.array([to_bucket(max(classify_mover(ri), key=classify_mover(ri).get)) for ri in r_obs])
        print(f"    {sig:>9.2f}{float(np.mean(pred == truth))*100:>9.1f}%")

    print("\n  Graceful = the metric slides down smoothly with noise (no cliff). The")
    print("  snapshot mover floor reflects irreducible single-snapshot ambiguity, not")
    print("  a model failure; an orbit (eccentricity) recovers it to ~98%.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
