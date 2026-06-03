"""Calibrate the coherence posterior's CONFIDENCE on real data (a NASA-grade
requirement): fit the temperature on a train split of real orbits so that when the
mover classifier reports p%, it is right ~p% of the time. Report the expected
calibration error (ECE) before/after on a held-out test split, plus the curve.
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
from ariadne.discovery.imaging.coherence_classifier import MOVER_PROTOTYPES, _MOVER_W  # noqa: E402
from ariadne.discovery.imaging.coherence_calibrate import fit_temperature, reliability  # noqa: E402
from validate_mover_real import true_class, to_bucket                                   # noqa: E402

DB = os.environ.get("ARIADNE_DB", "data/recovery_clean.db")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 80000
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT koa_a_au, koa_ecc FROM known_objects WHERE koa_a_au IS NOT NULL "
        "AND koa_ecc IS NOT NULL AND koa_a_au BETWEEN 0.3 AND 2000 "
        f"AND koa_ecc BETWEEN 0 AND 0.999 ORDER BY random() LIMIT {n}").fetchall()
    X = [{"logr": math.log10(a), "ecc": float(e),
          "logq": math.log10(max(a * (1 - e), 1e-6))} for a, e in rows]
    Y = [to_bucket_name(true_class(a, e)) for a, e in rows]
    rng = np.random.default_rng(0); idx = rng.permutation(len(X)); h = len(X) // 2
    Xtr = [X[i] for i in idx[:h]]; Ytr = [Y[i] for i in idx[:h]]
    Xte = [X[i] for i in idx[h:]]; Yte = [Y[i] for i in idx[h:]]
    basins = {to_bucket(k): v for k, v in MOVER_PROTOTYPES.items()}

    T, ece0_tr, eceT_tr = fit_temperature(Xtr, Ytr, basins, _MOVER_W)
    _, ece_before = reliability(Xte, Yte, basins, _MOVER_W, temperature=1.0)
    curve, ece_after = reliability(Xte, Yte, basins, _MOVER_W, temperature=T)

    print(f"=== CONFIDENCE CALIBRATION (mover, {len(X)} real orbits, held-out) ===")
    print(f"  fitted temperature T = {T:.2f}")
    print(f"  ECE before (T=1): {ece_before:.4f}")
    print(f"  ECE after  (T={T:.2f}): {ece_after:.4f}   ({(ece_before-ece_after)/max(ece_before,1e-9)*100:+.0f}% )")
    print(f"\n  reliability (confidence -> empirical accuracy), calibrated:")
    for conf, acc, cnt in curve:
        bar = "#" * int(acc * 30)
        print(f"    p~{conf:4.2f}  acc={acc:4.2f}  n={cnt:>6}  {bar}")
    print(f"\n  A calibrated selector has acc ~= confidence in every bin. ECE is the")
    print(f"  mean gap; {ece_after:.3f} means confidences are trustworthy to ~{ece_after*100:.0f}%.")
    return 0


def to_bucket_name(cls):
    return cls   # truth buckets already match the calibrated-basin keys


if __name__ == "__main__":
    raise SystemExit(main())
