"""Fetch NSC DR2 detections for a target cone+window via the Data Lab ASYNC query.

The nsc_dr2.meas table is billions of rows; synchronous q3c cone queries time
out, so we submit an ASYNC job, poll, and retrieve. Saves point-like detections
(ra, dec, mjd, mag, fwhm, exposure) for the discovery linker.
"""
from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ra", type=float, default=247.0)
    ap.add_argument("--dec", type=float, default=-25.0)
    ap.add_argument("--radius", type=float, default=0.2)
    ap.add_argument("--mjd0", type=float, default=58552.0)
    ap.add_argument("--mjd1", type=float, default=58566.0)
    ap.add_argument("--band", default="r")
    ap.add_argument("--max-mag", type=float, default=23.5)
    ap.add_argument("--top", type=int, default=500000)
    ap.add_argument("--out", default="data/nsc_field/meas.npz")
    ap.add_argument("--max-wait", type=int, default=1500)
    args = ap.parse_args()

    from dl import queryClient as qc
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    sql = (f"SELECT TOP {args.top} ra,dec,mjd,mag_auto,fwhm,exposure "
           f"FROM nsc_dr2.meas "
           f"WHERE q3c_radial_query(ra,dec,{args.ra},{args.dec},{args.radius}) "
           f"AND mjd BETWEEN {args.mjd0} AND {args.mjd1} "
           f"AND class_star>0.3 AND mag_auto BETWEEN 1 AND {args.max_mag} "
           f"AND filter='{args.band}'")
    print(f"submitting async NSC query (cone {args.radius} deg @ {args.ra},{args.dec}, "
          f"mjd {args.mjd0}-{args.mjd1}, {args.band}<{args.max_mag}) ...", flush=True)
    jobid = qc.query(sql=sql, async_=True)
    print(f"  jobid={jobid}", flush=True)
    t0 = time.time()
    while True:
        st = qc.status(jobid)
        el = time.time() - t0
        if st in ("COMPLETED", "ERROR", "ABORTED"):
            print(f"  status={st} after {el:.0f}s", flush=True)
            break
        if el > args.max_wait:
            print(f"  giving up after {el:.0f}s (status={st})", flush=True)
            return 1
        time.sleep(15)
    if st != "COMPLETED":
        try:
            print(qc.error(jobid)[:400])
        except Exception:
            pass
        return 1
    csv = qc.results(jobid)
    import csv as _csv
    rows = list(_csv.DictReader(io.StringIO(csv)))
    if not rows:
        print("  0 rows returned"); return 1
    ra = np.array([float(r["ra"]) for r in rows])
    dec = np.array([float(r["dec"]) for r in rows])
    mjd = np.array([float(r["mjd"]) for r in rows])
    mag = np.array([float(r["mag_auto"]) for r in rows])
    fwhm = np.array([float(r["fwhm"]) for r in rows])
    exp = np.array([str(r["exposure"]) for r in rows])
    np.savez(out, ra=ra, dec=dec, mjd=mjd, mag=mag, fwhm=fwhm, exposure=exp)
    import math
    from collections import Counter
    nights = Counter(math.floor(m - 0.5) for m in mjd)
    print(f"  saved {len(ra)} detections -> {out}", flush=True)
    print(f"  nights (>=1 det): {len(nights)}; per-night counts: "
          f"{dict(sorted(nights.items()))}", flush=True)
    print(f"  mag range {mag.min():.1f}-{mag.max():.1f}; "
          f"distinct exposures: {len(set(exp))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
