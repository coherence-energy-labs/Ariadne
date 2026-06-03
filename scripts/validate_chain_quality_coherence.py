"""Consolidation check: does chain_quality's NEW unified Equation-of-ONE score
(chain_coherence_score, backed by the validated track-energy engine) separate
real-orbit chains from chance at least as well as the OLD ad-hoc geometric-mean
chain_purity_score? Same real asteroid orbits propagated across nights vs chance.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ariadne.discovery.imaging.chain_quality import (  # noqa: E402
    chain_purity_score, chain_coherence_score)
from validate_coherence_vet_real import real_positive_tracks  # noqa: E402
from validate_coherence_vet import gen_chance                 # noqa: E402


def to_chain(ra, dec, mjd, mag=None):
    order = np.argsort(mjd); ra, dec, mjd = ra[order], dec[order], mjd[order]
    mag = (np.asarray(mag)[order] if mag is not None else [20.0] * len(ra))
    dec0 = float(np.median(dec))
    x = (ra - np.median(ra)) * math.cos(math.radians(dec0)) * 3600.0
    y = (dec - np.median(dec)) * 3600.0
    th = (mjd - mjd[0]) * 24.0
    chain = []
    for i in range(len(ra)):
        if i == 0 and len(ra) > 1:
            j = 1
        else:
            j = i - 1 if i > 0 else 0
        dt = th[i] - th[j] if i != j else (th[1] - th[0] if len(ra) > 1 else 1.0)
        rate = math.hypot(x[i] - x[j], y[i] - y[j]) / abs(dt) if dt else 0.0
        chain.append({"ra": math.radians(float(ra[i])), "dec": math.radians(float(dec[i])),
                      "t": float(mjd[i]) * 86400.0, "mag": float(mag[i]),
                      "rate_arcsec_hr": float(rate)})
    return chain


def auc(scores, labels):
    """P(score_real > score_chance) via rank statistic."""
    order = np.argsort(scores); ranks = np.empty(len(scores)); ranks[order] = np.arange(len(scores))
    pos = labels == 1; n_pos = pos.sum(); n_neg = (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


def main():
    pos = real_positive_tracks(n_orbits=3000)
    rng = np.random.default_rng(5)
    neg = [gen_chance(rng) for _ in range(len(pos))]
    chains = [to_chain(*p) for p in pos] + [to_chain(*n) for n in neg]
    labels = np.array([1] * len(pos) + [0] * len(neg))
    purity = np.array([chain_purity_score(c) for c in chains])
    coh = np.array([chain_coherence_score(c) for c in chains])
    print(f"=== chain_quality consolidation — {len(pos)} real-orbit chains + {len(neg)} chance ===")
    print(f"  {'score':<34}{'AUC (real vs chance)':>20}")
    print(f"  {'OLD ad-hoc geometric-mean purity':<34}{auc(purity, labels):>20.4f}")
    print(f"  {'NEW unified EoO coherence score':<34}{auc(coh, labels):>20.4f}")
    print("\n  (both are valid; the EoO score uses the SAME engine as vetting +"
          " classification, so the pipeline now has ONE coherence implementation.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
