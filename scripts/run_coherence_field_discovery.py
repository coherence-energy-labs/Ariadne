"""Measure the DEEP coherence-field discovery method on the real 2015 DECam field.

The pairwise rate linker tied at 3/22 because it needs a >=3-point within-night
tracklet before it can link a night -- a faint object detected once or twice per
night is invisible to it. The coherence field (coherence_field_tracks) instead
lets EVERY detection source the field: every viable inter-night pair proposes an
exact rate, and a real object's detections collapse to one coherent reference
peak even with no within-night tracklet. This script asks the only question that
matters: does that recover MORE of the same 22 recoverable knowns?

Truth is built with the SAME n-body ephemeris labelling as run_discovery_benchmark
(same DB, same match radius, same recoverable=>=2-nights set), so the recovery
count is directly comparable to the benchmark's pairwise 3/22.
"""
from __future__ import annotations

import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from run_discovery_benchmark import extract_exposure, _source_key, OBS, DB  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/decam_discovery_field")
    ap.add_argument("--n-ccd", type=int, default=60)
    ap.add_argument("--sigma", type=float, default=5.0)
    ap.add_argument("--match-arcsec", type=float, default=2.5)
    ap.add_argument("--stationary-veto-arcsec", type=float, default=0.8)
    ap.add_argument("--max-detections-per-exposure", type=int, default=20000)
    ap.add_argument("--truth-limit", type=int, default=300000)
    ap.add_argument("--max-rate", type=float, default=60.0)
    ap.add_argument("--cluster-tol", type=float, default=2.5)
    ap.add_argument("--min-nights", type=int, default=2)
    ap.add_argument("--min-points", type=int, default=3)
    args = ap.parse_args()

    from ariadne.discovery.imaging.source_extraction import Source
    from ariadne.discovery.imaging.tracklets_from_images import suppress_stationary_sources
    from ariadne.discovery.imaging.rate_constrained_linker import (
        coherence_field_tracks, build_within_night_tracks, link_rate_constrained)
    from ariadne.discovery.imaging.detection_db import open_db
    from ariadne.discovery.imaging.injection_recovery import pick_orbits_in_field
    from ariadne.discovery.imaging.mpc_ephemeris_nbody import bulk_ephemeris_at_mjd_nbody
    from ariadne.discovery.imaging.mpc_catalog import observatory_geo_km

    files = sorted(Path(args.data_dir).glob("c4d_*_ooi_*_v*.fits.fz"))
    if not files:
        print(f"no exposures in {args.data_dir}"); return 1
    cache_dir = Path(args.data_dir) / "_cache"; cache_dir.mkdir(exist_ok=True)
    print(f"{len(files)} exposures", flush=True)

    all_sources = []
    by_exposure = defaultdict(list)
    t0 = time.time()
    for f in files:
        try:
            ra, dec, mag, flux, mjd = extract_exposure(f, args.sigma, args.n_ccd, cache_dir)
        except Exception as exc:
            print(f"  SKIP {f.name}: {type(exc).__name__}: {exc}", flush=True); continue
        if args.max_detections_per_exposure and len(ra) > args.max_detections_per_exposure:
            print(f"  SKIP {f.name}: anomalous {len(ra)} detections", flush=True); continue
        for i in range(len(ra)):
            s = Source(ra=float(ra[i]), dec=float(dec[i]), flux=float(flux[i]),
                       mag=float(mag[i]), fwhm_px=4.0, mjd=mjd, image_id=f.stem, x=0.0, y=0.0)
            all_sources.append(s); by_exposure[f.stem].append(s)
    nights = sorted({int(round(s.mjd)) for s in all_sources})
    print(f"  {len(all_sources)} detections over {len(nights)} nights "
          f"({time.time()-t0:.1f}s)", flush=True)

    # --- truth: n-body ephemeris labels (same logic as the benchmark) ---
    db = open_db(DB)
    source_known = {}
    known_seen = defaultdict(set)
    tt = time.time()
    for image_id, ss in sorted(by_exposure.items()):
        ra = np.array([s.ra for s in ss]); dec = np.array([s.dec for s in ss])
        mjd = float(np.median([s.mjd for s in ss])); night = int(round(mjd))
        inf = pick_orbits_in_field(db, mjd, (ra.min(), ra.max()), (dec.min(), dec.max()),
                                   max_mag=22.0, limit_candidates=args.truth_limit)
        recs = [x[0] for x in inf]
        eph = bulk_ephemeris_at_mjd_nbody(recs, mjd, observer_geo_km=observatory_geo_km(OBS, mjd))
        for i in range(len(recs)):
            if np.isnan(eph[i, 0]):
                continue
            cd = math.cos(math.radians(eph[i, 1]))
            sep = np.hypot((ra - eph[i, 0]) * cd, dec - eph[i, 1]) * 3600
            if sep.size:
                j = int(np.argmin(sep))
                if float(sep[j]) <= args.match_arcsec:
                    desig = str(recs[i].designation)
                    source_known[_source_key(ss[j])] = desig
                    known_seen[desig].add(night)
    recoverable = {k for k, ns in known_seen.items() if len(ns) >= 2}
    print(f"  truth built ({time.time()-tt:.1f}s); knowns on >=2 nights "
          f"(recoverable): {len(recoverable)}", flush=True)

    # both linkers run on the SAME non-stationary detection set -> honest A/B.
    moving = suppress_stationary_sources(all_sources, args.stationary_veto_arcsec)
    print(f"  stationary veto -> {len(moving)} moving detections\n", flush=True)

    def _recover_from_known_sets(label_sets):
        """label_sets: list of {known -> set(nights)} per track/chain. A track
        recovers K if >=2 of its members on >=2 distinct nights carry K."""
        rec = set()
        for lab in label_sets:
            rec.update(k for k, ns in lab.items() if len(ns) >= 2)
        return rec & recoverable

    def coherence_run(srcs, min_nights, min_points):
        ct = time.time()
        tracks = coherence_field_tracks([s.ra for s in srcs], [s.dec for s in srcs],
                                        [s.mjd for s in srcs], max_rate_arcsec_hr=args.max_rate,
                                        cluster_tol_arcsec=args.cluster_tol,
                                        min_nights=min_nights, min_points=min_points)
        label_sets = []; unknown = 0
        for t in tracks:
            lab = defaultdict(set); known = False
            for idx in t["idx"]:
                k = source_known.get(_source_key(srcs[idx]))
                if k is not None:
                    lab[k].add(int(round(srcs[idx].mjd))); known = True
            label_sets.append(lab)
            if not known:
                unknown += 1
        return _recover_from_known_sets(label_sets), len(tracks), unknown, time.time() - ct

    def pairwise_run(srcs):
        ct = time.time()
        by_night = defaultdict(list)
        for s in srcs:
            by_night[int(round(s.mjd))].append(s)
        tracks = []
        for ss in by_night.values():
            tracks += build_within_night_tracks(ss, min_rate_arcsec_hr=2,
                                                max_rate_arcsec_hr=args.max_rate, min_points=3)
        chains = link_rate_constrained(tracks, min_nights=2)
        multinight = [c for c in chains if len({t.night for t in c}) >= 2]
        label_sets = []; unknown = 0
        for c in multinight:
            lab = defaultdict(set); known = False
            for trk in c:
                for s in trk.sources:
                    k = source_known.get(_source_key(s))
                    if k is not None:
                        lab[k].add(int(round(s.mjd))); known = True
            label_sets.append(lab)
            if not known:
                unknown += 1
        return _recover_from_known_sets(label_sets), len(multinight), unknown, time.time() - ct

    # scrambled control: per-night dec offset destroys real cross-night coherence,
    # preserves density -> chance floor for each method.
    rng = np.random.default_rng(0)
    offs = {n: rng.uniform(-0.1, 0.1) for n in nights}
    scr = [Source(ra=s.ra, dec=s.dec + offs[int(round(s.mjd))], flux=s.flux, mag=s.mag,
                  fwhm_px=4.0, mjd=s.mjd, image_id=s.image_id, x=0.0, y=0.0) for s in moving]

    N = len(recoverable)
    print(f"=== HEAD-TO-HEAD on identical {len(moving)} detections, {N} recoverable knowns ===")
    print(f"  {'method':36s} {'recall':>10s} {'unknown':>9s} {'chance':>8s} {'time':>7s}")
    pr, pn, pu, pt = pairwise_run(moving)
    _, _, psu, _ = pairwise_run(scr)
    print(f"  {'pairwise (>=3-pt within-night)':36s} {len(pr):>4d}/{N:<5d} {pu:>9d} {psu:>8d} {pt:>6.0f}s", flush=True)
    for mnn, mpp in ((2, 3), (3, 3), (3, 4)):
        cr, cn, cu, ctime = coherence_run(moving, mnn, mpp)
        _, _, csu, _ = coherence_run(scr, mnn, mpp)
        tag = f"coherence-field (>={mnn}nt,>={mpp}pt)"
        print(f"  {tag:36s} {len(cr):>4d}/{N:<5d} {cu:>9d} {csu:>8d} {ctime:>6.0f}s", flush=True)
    print("\n  recall = known asteroids recovered; chance = unknown tracks under the\n"
          "  scrambled control (lower is better). A genuine win = higher recall AND\n"
          "  a chance floor that stays low.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
