"""Fetch the chosen near-ecliptic, good-cadence DECam field for discovery.

Queries the NOIRLab Astro Data Archive (proposal 2013B-0536, ecliptic field
~ra 208, dec -11.7, VR band) for public instcal images, groups into observing
nights, and downloads up to --per-night exposures for the --nights chosen,
via the anonymous REST retrieve endpoint. Skips files already present.

Default target: the 2015-Apr contiguous 4-night block (within-night revisits
for tracklets, multi-night for confirmation).
"""
from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict
from pathlib import Path

import requests

FIND = "https://astroarchive.noirlab.edu/api/adv_search/find/"
RETRIEVE = "https://astroarchive.noirlab.edu/api/retrieve/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ra", type=float, default=208.0)
    ap.add_argument("--dec", type=float, default=-11.7)
    ap.add_argument("--half", type=float, default=0.7)
    ap.add_argument("--proposal", default="2013B-0536")
    ap.add_argument("--nights", default="57129,57130,57132,57133",
                    help="comma CTIO night ids (floor(mjd-0.667)); default 2015-Apr block")
    ap.add_argument("--per-night", type=int, default=5)
    ap.add_argument("--max-files", type=int, default=0, help="0 = no cap")
    ap.add_argument("--out", default="data/decam_discovery_field")
    args = ap.parse_args()

    from astropy.time import Time
    want_nights = {int(n) for n in args.nights.split(",") if n.strip()}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    body = {"outfields": ["md5sum", "archive_filename", "ra_center", "dec_center",
                          "dateobs_center", "exposure", "ifilter", "proposal",
                          "release_date"],
            "search": [["instrument", "decam"], ["proc_type", "instcal"],
                       ["prod_type", "image"],
                       ["ra_center", args.ra - args.half / math.cos(math.radians(args.dec)),
                        args.ra + args.half / math.cos(math.radians(args.dec))],
                       ["dec_center", args.dec - args.half, args.dec + args.half]]}
    rows = [x for x in requests.post(FIND + "?limit=4000", json=body, timeout=120).json()
            if isinstance(x, dict) and "ra_center" in x]
    rows = [x for x in rows if str(x.get("release_date", "9999")) <= "2026-06-02"
            and str(x.get("proposal")) == args.proposal]

    # DEDUPE multiple pipeline reductions of the SAME exposure: the archive
    # returns e.g. c4d_150418_082343_..._v4 AND ..._vx (identical date_time).
    # Keep one per unique exposure timestamp (prefer the highest version token),
    # else we'd fetch the same exposure twice (no motion -> useless tracklet).
    def expkey(fn):
        parts = Path(str(fn)).name.split("_")    # c4d, YYMMDD, HHMMSS, ooi, band, ver
        return (parts[1], parts[2]) if len(parts) >= 3 else (Path(str(fn)).name,)

    def ver_rank(fn):
        v = Path(str(fn)).name.split("_")[-1].split(".")[0]   # 'v4' / 'vx'
        # prefer standard numeric instcal versions (v1,v2,...) over odd tags
        # like 'vx'; among numeric, prefer the highest.
        if len(v) > 1 and v[0] == "v" and v[1:].isdigit():
            return (1, int(v[1:]))
        return (0, 0)
    best = {}
    for x in rows:
        k = expkey(x["archive_filename"])
        if k not in best or ver_rank(x["archive_filename"]) > ver_rank(best[k]["archive_filename"]):
            best[k] = x
    rows = list(best.values())

    by_night = defaultdict(list)
    for x in rows:
        mjd = float(Time(x["dateobs_center"], format="isot", scale="utc").mjd)
        by_night[math.floor(mjd - 0.667)].append((mjd, x))
    picks = []
    for n in sorted(want_nights):
        evs = sorted(by_night.get(n, []), key=lambda e: e[0])[:args.per_night]
        picks.extend(e[1] for e in evs)
        print(f"  night {n}: {len(evs)} picked of {len(by_night.get(n, []))} available")
    if args.max_files:
        picks = picks[:args.max_files]
    print(f"  total to fetch: {len(picks)} files -> {out}", flush=True)

    ok = 0
    for i, x in enumerate(picks):
        md5 = x.get("md5sum")
        name = Path(str(x["archive_filename"])).name
        dst = out / name
        if dst.exists() and dst.stat().st_size > 1_000_000:
            print(f"  [{i+1}/{len(picks)}] exists {name}", flush=True); ok += 1; continue
        if not md5:
            print(f"  [{i+1}/{len(picks)}] NO md5 for {name}", flush=True); continue
        try:
            with requests.get(RETRIEVE + md5 + "/", stream=True, timeout=600) as r:
                if not r.ok:
                    print(f"  [{i+1}/{len(picks)}] HTTP {r.status_code} {name}", flush=True); continue
                tmp = dst.with_suffix(dst.suffix + ".part")
                sz = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk); sz += len(chunk)
                os.replace(tmp, dst)
            print(f"  [{i+1}/{len(picks)}] {name}  {sz/1e6:.0f} MB", flush=True); ok += 1
        except Exception as e:
            print(f"  [{i+1}/{len(picks)}] FAIL {name}: {str(e)[:80]}", flush=True)
    print(f"\n  fetched {ok}/{len(picks)} files to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
