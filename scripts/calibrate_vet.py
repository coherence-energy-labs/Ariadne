"""Unify vetting + chain-quality into the ONE coherence engine, and check it beats
BOTH prior scorers. Fit 'real'/'chance' basins over track_features via the shared
coherence engine (coherence_posterior, same as classification), discriminatively
from real asteroid-orbit tracks vs chance, then compare AUC (real vs chance) to:
  - old coherence_vet.track_energy (4-sector, linear-residual heavy)
  - chain_quality.chain_purity_score (ad-hoc geometric mean)
A win means the pipeline can use ONE coherence model everywhere with no regression.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ariadne.discovery.imaging.coherence_vet import track_features, track_energy   # noqa: E402
from ariadne.discovery.imaging.coherence_field import coherence_posterior          # noqa: E402
from ariadne.discovery.imaging.coherence_calibrate import fit_basins, fit_weights  # noqa: E402
from ariadne.discovery.imaging.chain_quality import chain_purity_score             # noqa: E402
from validate_coherence_vet_real import real_positive_tracks                       # noqa: E402
from validate_coherence_vet import gen_chance                                      # noqa: E402

AXES = ["rate_cv", "heading", "resid", "mag_scatter", "log_epochs", "log_arc", "log_n"]


def to_chain(ra, dec, mjd, mag):
    order = np.argsort(mjd); ra, dec, mjd, mag = ra[order], dec[order], mjd[order], np.asarray(mag)[order]
    return [{"ra": math.radians(float(ra[i])), "dec": math.radians(float(dec[i])),
             "t": float(mjd[i]) * 86400.0, "mag": float(mag[i]),
             "rate_arcsec_hr": 0.0} for i in range(len(ra))]


def auc(scores, labels):
    s = np.asarray(scores, float); s = np.where(np.isfinite(s), s, np.nanmin(s[np.isfinite(s)]) - 1)
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(len(s))
    pos = labels == 1; np_, nn = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - np_ * (np_ - 1) / 2) / (np_ * nn))


def main():
    pos = real_positive_tracks(n_orbits=4000)
    rng = np.random.default_rng(5)
    neg = [gen_chance(rng) for _ in range(len(pos))]
    tracks = [(p, 1) for p in pos] + [(n, 0) for n in neg]
    labels = np.array([t[1] for t in tracks])

    feats = [track_features(*t[0]) for t in tracks]
    keep = [i for i, f in enumerate(feats) if f]
    feats = [feats[i] for i in keep]; labels = labels[keep]
    lab = ["real" if labels[i] == 1 else "chance" for i in range(len(labels))]

    # split for honest fit/eval
    idx = rng.permutation(len(feats)); h = len(feats) // 2
    tr, te = idx[:h], idx[h:]
    Ftr = [feats[i] for i in tr]; ytr = [lab[i] for i in tr]
    Fte = [feats[i] for i in te]; yte = labels[te]

    # ONE engine: fit real/chance basins + self-tune weights, score = P(real)
    basins = fit_basins(Ftr, ytr, AXES, robust=True)
    weights, _ = fit_weights(Ftr, ytr, basins, AXES, objective="balanced")
    p_real = [coherence_posterior(feats[i], basins, weights).get("real", 0.0) for i in te]

    # baselines on the SAME test tracks
    old_e = [-track_energy(*tracks[keep[i]][0]) for i in te]          # higher = more real
    purity = [chain_purity_score(to_chain(*tracks[keep[i]][0])) for i in te]

    print(f"=== UNIFY vetting+chain-quality into ONE coherence engine "
          f"({int((yte==1).sum())} real / {int((yte==0).sum())} chance held-out) ===")
    print(f"  {'scorer':<42}{'AUC':>8}")
    print(f"  {'chain_purity_score (ad-hoc geom-mean)':<42}{auc(purity, yte):>8.4f}")
    print(f"  {'old track_energy (4-sector, residual)':<42}{auc(old_e, yte):>8.4f}")
    print(f"  {'UNIFIED engine P(real) [fit basins+weights]':<42}{auc(p_real, yte):>8.4f}")
    print(f"\n  learned weights: {{{', '.join(f'{k}:{weights[k]:.2f}' for k in AXES)}}}")

    uni = auc(p_real, yte); best_prior = max(auc(purity, yte), auc(old_e, yte))
    print("\nVERDICT:")
    if uni >= best_prior - 0.005:
        print(f"  UNIFIED engine matches/beats both priors ({uni:.4f} vs best prior {best_prior:.4f})")
        print("  -> ONE coherence model can replace track_energy AND chain_purity. Saving fit.")
        out = {"basins": basins, "weights": weights, "axes": AXES}
        Path("data").mkdir(exist_ok=True)
        with open("data/vet_basins_calibrated.json", "w") as f:
            json.dump(out, f, indent=2)
    else:
        print(f"  unified {uni:.4f} < best prior {best_prior:.4f}; a prior scorer stays (honest).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
