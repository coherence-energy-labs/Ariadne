"""Discovery run on NSC detections: rate-link -> vet -> scrambled floor -> SkyBoT.

Input: meas.npz from fetch_nsc_field.py (ra, dec, mjd, mag, fwhm, exposure).
Chain: build >=3-point within-night rate tracks -> rate-constrained cross-night
linking -> vet (rate/heading/linearity) -> scrambled-control false floor ->
SkyBoT (authoritative known-SSO check at the field epoch, which propagates
current orbits correctly, sidestepping far-epoch catalog fuzziness).

A small cone retains only SLOW movers across many nights (fast main-belt
objects leave in ~1 day), i.e. the distant-object/TNO regime where genuinely
new objects still hide. A vetted multi-night chain with NO SkyBoT match, above
the chance floor, is a discovery candidate.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meas", default="data/nsc_field/meas.npz")
    ap.add_argument("--min-rate", type=float, default=0.3)
    ap.add_argument("--max-rate", type=float, default=150.0)
    ap.add_argument("--min-points", type=int, default=3)
    ap.add_argument("--min-nights", type=int, default=3)
    ap.add_argument("--pos-tol", type=float, default=2.5)
    ap.add_argument("--vet-max-rate-cv", type=float, default=0.25)
    ap.add_argument("--vet-max-resid-arcsec", type=float, default=2.5)
    ap.add_argument("--skybot-radius", type=float, default=20.0, help="arcsec")
    ap.add_argument("--no-skybot", action="store_true")
    args = ap.parse_args()

    from ariadne.discovery.imaging.source_extraction import Source
    from ariadne.discovery.imaging.rate_constrained_linker import (
        build_within_night_tracks, link_rate_constrained)

    d = np.load(args.meas, allow_pickle=True)
    ra, dec, mjd, mag = d["ra"], d["dec"], d["mjd"], d["mag"]
    exp = d["exposure"]
    print(f"{len(ra)} NSC detections; mjd {mjd.min():.2f}-{mjd.max():.2f} "
          f"({(mjd.max()-mjd.min()):.1f} d), mag {mag.min():.1f}-{mag.max():.1f}", flush=True)
    by_night = defaultdict(list)
    for i in range(len(ra)):
        s = Source(ra=float(ra[i]), dec=float(dec[i]), flux=float(10 ** (-0.4 * mag[i])),
                   mag=float(mag[i]), fwhm_px=float(d["fwhm"][i]), mjd=float(mjd[i]),
                   image_id=str(exp[i]), x=0.0, y=0.0)
        by_night[math.floor(s.mjd - 0.5)].append(s)
    nights = sorted(by_night)
    print(f"  {len(nights)} nights; per-night detections: "
          f"{ {n: len(by_night[n]) for n in nights} }", flush=True)

    # within-night >=3-point rate tracks
    t0 = time.time()
    tracks = []
    for n in nights:
        tracks += build_within_night_tracks(
            by_night[n], min_rate_arcsec_hr=args.min_rate, max_rate_arcsec_hr=args.max_rate,
            pos_tol_arcsec=args.pos_tol, min_points=args.min_points)
    print(f"  {len(tracks)} within-night rate tracks ({time.time()-t0:.1f}s)", flush=True)

    chains = link_rate_constrained(tracks, min_nights=args.min_nights)
    print(f"  {len(chains)} chains spanning >={args.min_nights} nights", flush=True)

    def vet(ch):
        rates = np.array([t.rate_arcsec_hr for t in ch])
        if rates.mean() <= 0 or rates.std() / rates.mean() > args.vet_max_rate_cv:
            return False
        # linearity of the cross-night arc on the tangent plane
        ra0 = ch[0].ra0; dec0 = ch[0].dec0
        cd = math.cos(math.radians(dec0))
        xs = np.array([(t.ra_mid - ra0) * cd * 3600 for t in ch])
        ys = np.array([(t.dec_mid - dec0) * 3600 for t in ch])
        tt = np.array([t.jd_mid for t in ch]); tt = tt - tt.mean()
        if np.var(tt) == 0:
            return False
        rx = xs - np.polyval(np.polyfit(tt, xs, 1), tt)
        ry = ys - np.polyval(np.polyfit(tt, ys, 1), tt)
        return float(np.sqrt(np.mean(rx ** 2 + ry ** 2))) <= args.vet_max_resid_arcsec

    vetted = [c for c in chains if vet(c)]
    print(f"  {len(vetted)} vetted (rate-consistent, linear arc)", flush=True)

    # scrambled-control false floor: offset one night, re-link+vet
    import copy
    off = nights[len(nights) // 2]
    scr = []
    for t in tracks:
        t2 = copy.copy(t)
        if t2.night == off:
            t2.ra_mid += 0.07; t2.dec_mid += 0.05
        scr.append(t2)
    scr_chains = link_rate_constrained(scr, min_nights=args.min_nights)
    scr_vetted = [c for c in scr_chains if vet(c)]
    print(f"  scrambled-control false floor: {len(scr_vetted)} vetted chains", flush=True)

    # report candidates
    cand = sorted(vetted, key=lambda c: -len(c))
    print(f"\n=== CANDIDATES (vetted multi-night chains) vs floor {len(scr_vetted)} ===")
    for c in cand[:15]:
        ns = sorted({t.night for t in c})
        print(f"  ({c[0].ra_mid:.4f},{c[0].dec_mid:.4f}) nights={len(ns)} span={ns[-1]-ns[0]}d "
              f"rate={np.mean([t.rate_arcsec_hr for t in c]):.1f}\"/hr mag~{np.median([t.mag for t in c]):.1f}")

    # SkyBoT: is each candidate a KNOWN solar-system object at its epoch?
    if args.no_skybot or not cand:
        return 0
    print(f"\n  SkyBoT known-SSO check ({len(cand)} candidates) ...", flush=True)
    from astroquery.imcce import Skybot
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    import astropy.units as u
    known = unknown = 0; unknown_list = []
    for c in cand:
        cc = c[len(c) // 2]
        try:
            tb = Skybot.cone_search(SkyCoord(cc.ra_mid, cc.dec_mid, unit="deg"),
                                    args.skybot_radius * u.arcsec,
                                    Time(cc.jd_mid, format="jd"))
            n = 0 if tb is None else len(tb)
        except Exception:
            n = 0   # SkyBoT returns an error when no object is found
        if n > 0:
            known += 1
        else:
            unknown += 1
            unknown_list.append(c)
    print(f"  SkyBoT-known (recoveries): {known}   SkyBoT-UNKNOWN (candidates): {unknown}")
    print(f"  => {unknown} unknown vs {len(scr_vetted)} chance-floor chains")
    for c in unknown_list[:15]:
        ns = sorted({t.night for t in c})
        print(f"    UNKNOWN: ({c[0].ra_mid:.5f},{c[0].dec_mid:.5f}) nights={len(ns)} "
              f"span={ns[-1]-ns[0]}d rate={np.mean([t.rate_arcsec_hr for t in c]):.2f}\"/hr "
              f"mag~{np.median([t.mag for t in c]):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
