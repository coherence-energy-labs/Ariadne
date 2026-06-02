"""Scout the NOIRLab archive for a GOOD-CADENCE, NEAR-ECLIPTIC DECam field.

A real discovery needs: many exposures of the SAME field WITHIN a night (so a
moving object reappears -> a tracklet) over MULTIPLE nights (so a track
confirms and a chance alignment does not), near the ECLIPTIC (where the
asteroids are). This queries the NOIRLab Astro Data Archive REST API (metadata
only -- NO downloads), sweeping pointings along the ecliptic, groups exposures
into CTIO observing-nights per sub-field, and ranks candidates by
within-night cadence x multi-night coverage. Public (released) data only.

Output: ranked candidate fields with (ra, dec, ecliptic lat, band, #good
nights, exposures/night, date span, proposal) -- you pick one to fetch.
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict

import requests

API = "https://astroarchive.noirlab.edu/api/adv_search/find/"
EPS = math.radians(23.4393)        # obliquity of the ecliptic
TODAY = "2026-06-02"


def ecl0_to_eq(lon_deg):
    """Ecliptic (lon, lat=0) -> equatorial (ra, dec) degrees."""
    l = math.radians(lon_deg)
    ra = math.degrees(math.atan2(math.sin(l) * math.cos(EPS), math.cos(l))) % 360.0
    dec = math.degrees(math.asin(math.sin(l) * math.sin(EPS)))
    return ra, dec


def ecl_lat(ra, dec):
    a = math.radians(ra); d = math.radians(dec)
    return math.degrees(math.asin(math.sin(d) * math.cos(EPS)
                                  - math.cos(d) * math.sin(EPS) * math.sin(a)))


def to_mjd(iso):
    from astropy.time import Time
    return float(Time(iso, format="isot", scale="utc").mjd)


def query_box(ra, dec, half, limit, band=None):
    search = [["instrument", "decam"], ["proc_type", "instcal"],
              ["prod_type", "image"],
              ["ra_center", ra - half / math.cos(math.radians(dec)), ra + half / math.cos(math.radians(dec))],
              ["dec_center", dec - half, dec + half]]
    body = {"outfields": ["archive_filename", "ra_center", "dec_center",
                          "dateobs_center", "exposure", "ifilter", "proposal",
                          "release_date"],
            "search": search}
    r = requests.post(API + f"?limit={limit}", json=body, timeout=90)
    if not r.ok:
        return []
    j = r.json()
    return [x for x in j if isinstance(x, dict) and "ra_center" in x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lon-step", type=int, default=30)
    ap.add_argument("--half-deg", type=float, default=0.9)
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--min-per-night", type=int, default=4)
    ap.add_argument("--cell-deg", type=float, default=0.3)
    args = ap.parse_args()

    candidates = []
    for lon in range(0, 360, args.lon_step):
        ra, dec = ecl0_to_eq(lon)
        try:
            rows = query_box(ra, dec, args.half_deg, args.limit)
        except Exception as e:
            print(f"  lon={lon:3d} (ra={ra:.1f},dec={dec:.1f}) query failed: {str(e)[:60]}")
            continue
        # public only
        rows = [x for x in rows if str(x.get("release_date", "9999")) <= TODAY]
        print(f"  lon={lon:3d} ra={ra:6.1f} dec={dec:6.1f} ecllat={ecl_lat(ra,dec):+5.1f}  "
              f"{len(rows)} public instcal exposures", flush=True)
        if not rows:
            continue
        # cluster by sub-pointing cell x band, group exposures into nights
        cells = defaultdict(list)
        for x in rows:
            try:
                mjd = to_mjd(x["dateobs_center"])
            except Exception:
                continue
            band = str(x.get("ifilter", "?")).split()[0]
            key = (round(float(x["ra_center"]) / args.cell_deg),
                   round(float(x["dec_center"]) / args.cell_deg), band)
            cells[key].append((mjd, float(x["ra_center"]), float(x["dec_center"]),
                               band, str(x.get("proposal", "?"))))
        for (cx, cy, band), evs in cells.items():
            nights = defaultdict(list)
            for (mjd, r_, d_, b_, prop) in evs:
                nights[math.floor(mjd - 0.667)].append(mjd)   # CTIO observing night
            good_nights = {n: len(v) for n, v in nights.items() if len(v) >= args.min_per_night}
            if len(good_nights) >= 2:
                mjds = [m for v in nights.values() for m in v]
                cra = sum(e[1] for e in evs) / len(evs)
                cdec = sum(e[2] for e in evs) / len(evs)
                candidates.append({
                    "ra": cra, "dec": cdec, "ecllat": ecl_lat(cra, cdec),
                    "band": band, "n_good_nights": len(good_nights),
                    "max_per_night": max(good_nights.values()),
                    "total_good": sum(good_nights.values()),
                    "span_days": (max(mjds) - min(mjds)),
                    "proposal": evs[0][4], "n_total": len(evs),
                })

    candidates.sort(key=lambda c: (c["n_good_nights"], c["total_good"],
                                   -abs(c["ecllat"])), reverse=True)
    print(f"\n=== TOP DISCOVERY-FIELD CANDIDATES (>=2 nights with "
          f">={args.min_per_night} exposures, near ecliptic, public) ===")
    if not candidates:
        print("  none found -- widen --half-deg / lower --min-per-night / finer --lon-step")
    hdr = (f"  {'ra':>7} {'dec':>7} {'ecllat':>7} {'band':>5} {'nights':>6} "
           f"{'max/nt':>6} {'tot':>4} {'span_d':>7}  proposal")
    print(hdr)
    for c in candidates[:15]:
        print(f"  {c['ra']:7.2f} {c['dec']:7.2f} {c['ecllat']:+7.1f} {c['band']:>5} "
              f"{c['n_good_nights']:6d} {c['max_per_night']:6d} {c['total_good']:4d} "
              f"{c['span_days']:7.1f}  {c['proposal']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
