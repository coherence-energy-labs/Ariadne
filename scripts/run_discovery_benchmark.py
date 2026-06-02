"""Truthed END-TO-END discovery benchmark on a good-cadence near-ecliptic field.

This is the instrument that's been missing: run the FULL discovery chain on
real multi-night pixels where we know the truth, and measure it.

  per exposure : extract sources (fixed auto-PSF detection)
  per night    : pool sources -> within-night tracklets (nightly_tracklets)
  cross-match  : label tracklets that match a KNOWN asteroid (validated N-body
                 ephemeris) -> the truth set is knowns visible+detectable on >=2 nights
  link         : discover_in_images_chains across nights (HelioLinC + multipass
                 + n-body grow) on ALL tracklets
  measure      :
    - KNOWN RECOVERY: of recoverable knowns (>=2 nights), how many are recovered
      as a multi-night chain? (does the discovery machinery actually work?)
    - UNKNOWN CANDIDATES: chains not matching any known, vetted (>=2 nights,
      rate/mag consistency) -> potential discoveries
    - FALSE FLOOR: scrambled control (offset one night) -> chance chains
    - IOD (optional): fit an orbit on the best candidates

Extraction is cached per exposure (.npz) so re-runs are fast. PERF: the PSF
FWHM is measured ONCE per exposure (median of a few CCDs) and reused for all
CCDs -- per-CCD auto-measurement is more accurate but ~2x slower.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

OBS = "807"
DB = "C:/Users/Josh/AppData/Local/Temp/recovery_clean.db"


def extract_exposure(path, sigma, n_ccd, cache_dir):
    """Extract sources from one exposure (all CCDs). PSF measured once/exposure.
    Cached to .npz. Returns (ra, dec, mag, flux, mjd)."""
    cache = Path(cache_dir) / (Path(path).stem + f".s{sigma:.0f}.npz")
    if cache.exists():
        c = np.load(cache)
        return c["ra"], c["dec"], c["mag"], c["flux"], float(c["mjd"])
    from ariadne.discovery.imaging.decam_instcal import load_decam_instcal
    from ariadne.discovery.imaging.source_extraction import detect_sources_in_image
    from ariadne.discovery.imaging.trailed_rate import measure_image_fwhm
    inst = load_decam_instcal(str(path), read_dqm=False)
    mjd = inst.mjd
    ccds = [c for c in inst.ccds if c.wcs is not None and c.magzero > 0][:n_ccd]
    # measure PSF FWHM once (median over up to 4 central CCDs) -> reuse
    fwhms = []
    for c in ccds[:4]:
        m = measure_image_fwhm(np.asarray(c.science, float), fwhm_guess=4.0)
        if m:
            fwhms.append(m)
    fwhm = float(np.median(fwhms)) if fwhms else 4.0
    ra, dec, mag, flux = [], [], [], []
    for c in ccds:
        try:
            srcs = detect_sources_in_image(
                np.asarray(c.science, float), c.wcs, mjd=mjd, image_id=Path(path).stem,
                fwhm_px=fwhm, threshold_sigma=sigma, zeropoint_mag=c.magzero,
                auto_fwhm=False)
        except Exception:
            continue
        for s in srcs:
            ra.append(s.ra); dec.append(s.dec); mag.append(s.mag); flux.append(s.flux)
    ra = np.array(ra); dec = np.array(dec); mag = np.array(mag); flux = np.array(flux)
    np.savez(cache, ra=ra, dec=dec, mag=mag, flux=flux, mjd=mjd, fwhm=fwhm)
    return ra, dec, mag, flux, mjd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/decam_discovery_field")
    ap.add_argument("--n-ccd", type=int, default=60)
    ap.add_argument("--sigma", type=float, default=5.0)
    ap.add_argument("--min-rate", type=float, default=2.0, help="arcsec/hr")
    ap.add_argument("--max-rate", type=float, default=120.0)
    ap.add_argument("--match-arcsec", type=float, default=2.5)
    ap.add_argument("--iod", action="store_true", help="run IOD on candidates (slow)")
    args = ap.parse_args()

    from ariadne.discovery.imaging.source_extraction import Source
    from ariadne.discovery.imaging.tracklets_from_images import nightly_tracklets
    from ariadne.discovery.imaging.advanced_linking import discover_in_images_chains
    from ariadne.discovery.imaging.detection_db import open_db
    from ariadne.discovery.imaging.injection_recovery import pick_orbits_in_field
    from ariadne.discovery.imaging.mpc_ephemeris_nbody import bulk_ephemeris_at_mjd_nbody
    from ariadne.discovery.imaging.mpc_catalog import observatory_geo_km

    files = sorted(Path(args.data_dir).glob("c4d_*_ooi_*_v*.fits.fz"))
    if not files:
        print(f"no exposures in {args.data_dir}"); return 1
    cache_dir = Path(args.data_dir) / "_cache"; cache_dir.mkdir(exist_ok=True)
    print(f"{len(files)} exposures", flush=True)

    # extract all exposures -> Source list, grouped by night (round mjd)
    all_sources = []
    by_night_src = defaultdict(list)
    t0 = time.time()
    for f in files:
        ra, dec, mag, flux, mjd = extract_exposure(f, args.sigma, args.n_ccd, cache_dir)
        night = int(round(mjd))
        for i in range(len(ra)):
            s = Source(ra=float(ra[i]), dec=float(dec[i]), flux=float(flux[i]),
                       mag=float(mag[i]), fwhm_px=4.0, mjd=mjd,
                       image_id=f.stem, x=0.0, y=0.0)
            all_sources.append(s); by_night_src[night].append(s)
        print(f"  {f.name}: {len(ra)} det, night {night} ({time.time()-t0:.0f}s)", flush=True)
    nights = sorted(by_night_src)
    print(f"  {len(all_sources)} detections over {len(nights)} nights: {nights}", flush=True)

    # within-night tracklets
    tracks = nightly_tracklets(all_sources, min_rate_arcsec_hr=args.min_rate,
                                max_rate_arcsec_hr=args.max_rate,
                                min_pair_dt_hours=0.03, max_pair_dt_hours=3.0,
                                min_pair_separation_arcsec=0.5, max_per_night=4000)
    print(f"  {len(tracks)} within-night tracklets", flush=True)

    # KNOWN cross-match: label each tracklet by the known it matches (if any),
    # and build the recoverable-known truth set (detectable on >=2 nights).
    db = open_db(DB)
    known_seen = defaultdict(set)     # known_id -> set of nights it's detectable on
    for night in nights:
        ss = by_night_src[night]
        ra = np.array([s.ra for s in ss]); dec = np.array([s.dec for s in ss])
        mjd = float(np.median([s.mjd for s in ss]))
        inf = pick_orbits_in_field(db, mjd, (ra.min(), ra.max()), (dec.min(), dec.max()),
                                     max_mag=22.0, limit_candidates=1600000)
        recs = [x[0] for x in inf]
        eph = bulk_ephemeris_at_mjd_nbody(recs, mjd, observer_geo_km=observatory_geo_km(OBS, mjd))
        for i in range(len(recs)):
            if np.isnan(eph[i, 0]):
                continue
            cd = math.cos(math.radians(eph[i, 1]))
            sep = np.hypot((ra - eph[i, 0]) * cd, dec - eph[i, 1]) * 3600
            if sep.size and sep.min() <= args.match_arcsec:
                known_seen[id(recs[i])].add(night)   # detected this night
        # stash predictions for tracklet labelling
        by_night_src[night] = (ss, recs, eph)
    recoverable = {k for k, ns in known_seen.items() if len(ns) >= 2}
    print(f"  knowns detected on >=2 nights (recoverable truth set): {len(recoverable)}", flush=True)

    # label tracklets by known (midpoint within match radius of a known's eph)
    def label_tracklet(t):
        night = t["night"]; ss, recs, eph = by_night_src[night]
        tra = math.degrees(t["ra"]); tdec = math.degrees(t["dec"])
        cd = math.cos(math.radians(tdec))
        best = None; bestsep = 1e9
        for i in range(len(recs)):
            if np.isnan(eph[i, 0]):
                continue
            sep = math.hypot((eph[i, 0] - tra) * cd, eph[i, 1] - tdec) * 3600
            if sep < bestsep:
                bestsep = sep; best = id(recs[i])
        return best if bestsep <= args.match_arcsec else None
    for t in tracks:
        t["known_id"] = label_tracklet(t)

    # cross-night linking (full strategy suite incl. HelioLinC + n-body grow)
    chains = discover_in_images_chains(tracks, use_helio_linc=True, use_nbody_grow=True)
    multinight = [c for c in chains if len({t["night"] for t in c}) >= 2]
    print(f"  {len(chains)} chains, {len(multinight)} span >=2 nights", flush=True)

    # KNOWN RECOVERY: a chain recovers known K if >=2 of its tracklets on >=2
    # nights are labelled K.
    recovered = set()
    unknown_chains = []
    for c in multinight:
        labels = defaultdict(set)
        for t in c:
            if t.get("known_id") is not None:
                labels[t["known_id"]].add(t["night"])
        hit = [k for k, ns in labels.items() if len(ns) >= 2]
        if hit:
            recovered.update(hit)
        elif all(t.get("known_id") is None for t in c):
            unknown_chains.append(c)
    rec_known = len(recovered & recoverable)
    print(f"\n=== END-TO-END DISCOVERY BENCHMARK (real {len(nights)}-night field) ===")
    print(f"  KNOWN multi-night recovery: {rec_known}/{len(recoverable)} "
          f"= {rec_known/max(len(recoverable),1)*100:.0f}%")
    print(f"  UNKNOWN multi-night candidates (no known match): {len(unknown_chains)}")

    # vet unknown candidates: rate consistency across nights + magnitude
    def vet(c):
        rates = [t["rate_arcsec_hr"] for t in c]
        if np.std(rates) > 0.5 * (np.mean(rates) + 1e-6):
            return False
        return True
    vetted = [c for c in unknown_chains if vet(c)]
    print(f"  vetted unknown candidates (rate-consistent): {len(vetted)}")

    # FALSE FLOOR: scramble one night's source positions, re-link
    import copy
    scr_tracks = []
    off_night = nights[len(nights) // 2]
    for t in tracks:
        t2 = dict(t)
        if t2["night"] == off_night:
            t2["ra"] = t2["ra"] + math.radians(0.08)
            t2["dec"] = t2["dec"] + math.radians(0.05)
        scr_tracks.append(t2)
    scr_chains = discover_in_images_chains(scr_tracks, use_helio_linc=True, use_nbody_grow=True)
    scr_mn = [c for c in scr_chains if len({t["night"] for t in c}) >= 2
              and all(t.get("known_id") is None for t in c)]
    print(f"  scrambled-control false floor (unknown multi-night chains): {len(scr_mn)}")
    print(f"  => {len(vetted)} candidates vs {len(scr_mn)} chance floor")

    for c in sorted(vetted, key=lambda c: -len(c))[:8]:
        ns = sorted({t['night'] for t in c})
        ra0 = math.degrees(c[0]['ra']); dec0 = math.degrees(c[0]['dec'])
        rate = np.mean([t['rate_arcsec_hr'] for t in c])
        print(f"    cand: ({ra0:.4f},{dec0:.4f}) nights={ns} rate={rate:.0f}\"/hr "
              f"tracklets={len(c)}")

    if args.iod and vetted:
        print(f"\n  IOD on top {min(5,len(vetted))} candidates:")
        from ariadne.discovery.iod_robust import robust_iod
        for c in sorted(vetted, key=lambda c: -len(c))[:5]:
            try:
                ens = robust_iod(c, n_draws=2, rms_acceptance_arcsec=30.0,
                                  use_monte_carlo=True, use_rate_class=True)
                print(f"    chain({len(c)}): success={ens.success} "
                      f"RMS={getattr(ens,'rms_arcsec',-1):.2f}\"")
            except Exception as e:
                print(f"    chain IOD error: {str(e)[:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
