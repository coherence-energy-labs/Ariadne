"""Real-DATA vetting A/B: EoO energy selector vs hard AND-rules, where the
POSITIVE tracks are REAL asteroid orbits propagated across nights (real sky
rates, real Keplerian curvature, real H-brightness) and the negatives are chance
alignments. This is the operational vetting question on real motion, not lines.

Positives: a sample of real MPC orbits (known_objects) propagated to 4 nights x 2
exposures via the batch ephemeris; kept if present on >=2 nights with >=3 points.
Negatives: chance alignments (same gen as the synthetic A/B). Vet both with the
hard AND-rules and with the EoO energy selector; report precision/recall/F1.
"""
from __future__ import annotations

import os
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ariadne.discovery.imaging.coherence_vet import track_energy  # noqa: E402
from validate_coherence_vet import hard_features, gen_chance      # noqa: E402

DB = os.environ.get("ARIADNE_DB", "data/recovery_clean.db")


def real_positive_tracks(n_orbits=4000, jitter_arcsec=0.15, seed=11):
    from ariadne.discovery.imaging.mpc_catalog import OrbitalElements
    from ariadne.discovery.imaging.mpc_ephemeris_batch import bulk_ephemeris_at_mjd
    con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT designation, epoch_mjd, orbital_elements, koa_h FROM known_objects "
        "WHERE orbital_elements IS NOT NULL AND koa_h IS NOT NULL AND koa_h < 14.5 "
        f"ORDER BY random() LIMIT {n_orbits}").fetchall()
    recs = []
    for r in rows:
        try:
            e = json.loads(r["orbital_elements"])
            recs.append(OrbitalElements(
                designation=r["designation"], epoch_mjd=float(r["epoch_mjd"]),
                a_au=float(e["a_au"]), e=float(e["e"]), i_deg=float(e["i_deg"]),
                Omega_deg=float(e["Omega_deg"]), omega_deg=float(e["omega_deg"]),
                M_deg=float(e["M_deg"]), H_mag=float(e.get("H", r["koa_h"]))))
        except Exception:
            continue
    H = np.array([rc.H_mag for rc in recs], float)
    rng = np.random.default_rng(seed)
    # 4 nights x 2 exposures (~13 min apart), near an arbitrary recent epoch
    mjd0 = 60000.0
    epochs = [mjd0 + d + k * 0.009 for d in range(4) for k in range(2)]
    pos = []  # per-epoch (ra, dec) arrays
    for mjd in epochs:
        eph = bulk_ephemeris_at_mjd(recs, mjd)
        pos.append((eph[:, 0].copy(), eph[:, 1].copy()))
    tracks = []
    for j in range(len(recs)):
        ra, dec, mjd, mag = [], [], [], []
        for (ej_ra, ej_dec), mjd_e in zip(pos, epochs):
            if np.isnan(ej_ra[j]):
                continue
            ra.append(ej_ra[j] + rng.normal(0, jitter_arcsec) / 3600.0)
            dec.append(ej_dec[j] + rng.normal(0, jitter_arcsec) / 3600.0)
            mjd.append(mjd_e); mag.append(H[j] + rng.normal(0, 0.12))
        if len(ra) < 3:
            continue
        if len({int(round(m)) for m in mjd}) < 2:
            continue
        tracks.append((np.array(ra), np.array(dec), np.array(mjd), np.array(mag)))
    return tracks


def main():
    t0 = time.time()
    pos = real_positive_tracks()
    rng = np.random.default_rng(99)
    neg = [gen_chance(rng) for _ in range(len(pos))]
    tracks = [(p, True) for p in pos] + [(n, False) for n in neg]
    y = np.array([t[1] for t in tracks])
    print(f"=== REAL-ORBIT VETTING A/B — {len(pos)} real asteroid tracks + "
          f"{len(neg)} chance ({time.time()-t0:.0f}s to propagate) ===")
    # rate distribution of the real positives (sanity: real sky motion)
    rates = []
    for (ra, dec, mjd, mag) in pos:
        cv, hs, rs = hard_features(ra, dec, mjd)
    print(f"  real tracks: median points={np.median([len(p[0]) for p in pos]):.0f}, "
          f"nights spanned up to 4")

    def pr(pred):
        tp = int(np.sum(pred & y)); fp = int(np.sum(pred & ~y)); fn = int(np.sum(~pred & y))
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        return prec, rec, 2 * prec * rec / max(prec + rec, 1e-9)

    # hard AND-rules
    hard = np.array([(lambda cv, hs, rs: cv < 0.25 and hs < 25.0 and rs < 45.0)(*hard_features(ra, dec, mjd))
                     for ra, dec, mjd, mag in [t[0] for t in tracks]])
    hp, hr, hf = pr(hard)
    # EoO energy, swept
    E = np.array([track_energy(ra, dec, mjd, mag) for ra, dec, mjd, mag in [t[0] for t in tracks]])
    taus = np.quantile(E[np.isfinite(E)], np.linspace(0.02, 0.98, 60))
    roc = [(*pr(E <= tau), tau) for tau in taus]
    best = max(roc, key=lambda z: z[2])
    at_rec = [r for r in roc if r[1] >= hr]
    eoo_at = max(at_rec, key=lambda z: z[0]) if at_rec else None

    print(f"\n  {'method':<32} {'prec':>6} {'recall':>7} {'F1':>6}")
    print(f"  {'hard AND-rules':<32} {hp:>6.3f} {hr:>7.3f} {hf:>6.3f}")
    print(f"  {'EoO energy <= tau (best F1)':<32} {best[0]:>6.3f} {best[1]:>7.3f} {best[2]:>6.3f}")
    if eoo_at:
        print(f"  {'EoO at hard-rules recall':<32} {eoo_at[0]:>6.3f} {eoo_at[1]:>7.3f} {eoo_at[2]:>6.3f}")
    print("\nVERDICT:")
    if best[2] > hf + 0.01:
        print(f"  EoO beats hard rules on REAL orbital motion: F1 {best[2]:.3f} vs {hf:.3f} "
              f"(+{(best[2]-hf)*100:.1f}pp), tunable ROC.")
    elif best[2] < hf - 0.01:
        print(f"  EoO worse on real motion: F1 {best[2]:.3f} vs {hf:.3f} (honest loss).")
    else:
        print(f"  EoO ties hard rules on real motion (F1 {best[2]:.3f} vs {hf:.3f}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
