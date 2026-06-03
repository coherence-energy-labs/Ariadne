"""NASA-grade validation of the light-curve analyzer on REAL data.

Pull known, classified variable stars (VSX) with published periods, fetch their
REAL ZTF r-band light curves (IRSA), run analyze_light_curve BLIND, and measure:
  * period recovery: |P - f*P0|/P0 < tol for f in {0.5,1,2} (the standard
    photometric aliases; eclipsing binaries fold at half the orbital period);
  * type accuracy: does the analyzer's top class match the VSX family?

This turns the light-curve analyzer from "synthetically tested" to
"validated against real labeled data" -- the actual NASA-grade bar.
"""
from __future__ import annotations

import argparse
import io
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

CACHE = Path("data/_lc_cache"); CACHE.mkdir(parents=True, exist_ok=True)
_SESSION = requests.Session()

ZTF = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"
# VSX type -> (query constraints, our-family label)
SAMPLES = {
    "RRAB": {"Period": ">0.4 & <0.9", "family": "RR Lyrae"},
    "EA":   {"Period": ">0.5 & <8",   "family": "eclipsing binary"},
    "EW":   {"Period": ">0.25 & <0.7", "family": "eclipsing binary"},
    "DCEP": {"Period": ">3 & <15",    "family": "Cepheid"},
}


def vsx(vtype, cons, n):
    from astroquery.vizier import Vizier
    v = Vizier(columns=["RAJ2000", "DEJ2000", "Name", "Type", "Period", "max"],
               row_limit=n * 6)
    res = v.query_constraints(catalog="B/vsx/vsx", Type=vtype,
                              Period=cons, **{"max": "<15.5"})
    if not res:
        return []
    out = []
    for r in res[0]:
        try:
            dec = float(r["DEJ2000"]); P = float(r["Period"])
            if dec > -25 and P > 0:
                out.append((float(r["RAJ2000"]), dec, str(r["Name"]), P))
        except Exception:
            continue
        if len(out) >= n:
            break
    return out


def ztf_lc(ra, dec):
    """Fetch the densest ZTF r-band light curve + the g-r color near (ra,dec).
    Returns (t, mag, magerr, g_minus_r) or None. Disk-cached."""
    key = CACHE / f"ztfc_{ra:.5f}_{dec:+.5f}.npz"
    if key.exists():
        c = np.load(key)
        if len(c["t"]) >= 20:
            return c["t"], c["m"], c["e"], float(c["gr"])
        return None
    u = f"{ZTF}?POS=CIRCLE+{ra}+{dec}+0.0007&FORMAT=CSV"     # all bands -> color
    try:
        txt = _SESSION.get(u, timeout=60).text
    except Exception:
        return None
    import csv
    by = defaultdict(lambda: defaultdict(list))   # filtercode -> oid -> [(mjd,mag,err)]
    for x in csv.DictReader(io.StringIO(txt)):
        try:
            if int(x.get("catflags", "0")) != 0:
                continue
            by[x["filtercode"]][x["oid"]].append(
                (float(x["mjd"]), float(x["mag"]), float(x["magerr"])))
        except Exception:
            continue
    r_oids = by.get("zr", {})
    best = max(r_oids.values(), key=len) if r_oids else []
    a = np.array(best) if best else np.empty((0, 3))
    # g-r color: median g (densest g oid) - median r (best r oid)
    gr = np.nan
    if len(a) and by.get("zg"):
        g_best = max(by["zg"].values(), key=len)
        gr = float(np.median([p[1] for p in g_best]) - np.median(a[:, 1]))
    np.savez(key, t=a[:, 0] if len(a) else np.array([]),
             m=a[:, 1] if len(a) else np.array([]),
             e=a[:, 2] if len(a) else np.array([]), gr=gr)
    return (a[:, 0], a[:, 1], a[:, 2], gr) if len(a) >= 20 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=12)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    from ariadne.discovery.imaging.light_curve import analyze_light_curve
    from ariadne.discovery.imaging.coherence_classifier import classify_variable, most_coherent
    import time
    t0 = time.time()
    # 1) gather all targets (a few VSX queries)
    targets = []   # (vtype, family, ra, dec, name, P0)
    for vtype, spec in SAMPLES.items():
        for (ra, dec, name, P0) in vsx(vtype, spec["Period"], args.per_type):
            targets.append((vtype, spec["family"], ra, dec, name, P0))
    print(f"  {len(targets)} known variables; fetching ZTF light curves "
          f"({args.workers} parallel) ...", flush=True)
    # 2) fetch ALL light curves in parallel (network-bound -> threads)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        lcs = list(ex.map(lambda T: ztf_lc(T[2], T[3]), targets))
    print(f"  fetched in {time.time()-t0:.0f}s; analyzing ...", flush=True)
    # 3) analyze + score
    per_type = defaultdict(lambda: [0, 0]); type_per = defaultdict(lambda: [0, 0])
    period_hits = period_tot = type_hits = type_tot = 0
    import numpy as _np
    for (vtype, family, ra, dec, name, P0), lc in zip(targets, lcs):
        if lc is None:
            continue
        t, m, e, gr = lc
        r = analyze_light_curve(t, m, e, min_period=0.1, max_period=25.0)
        period_tot += 1; type_tot += 1; per_type[vtype][1] += 1; type_per[vtype][1] += 1
        ok = False
        if r.best_period:
            rel = min(abs(r.best_period - f * P0) / P0 for f in (0.5, 1.0, 2.0))
            ok = rel < 0.05
            period_hits += ok; per_type[vtype][0] += ok
        # coherence-field classification with color (the degeneracy-breaking axis)
        post = classify_variable(r.best_period or 0.5, r.harmonic_r21, r.amplitude_mag,
                                 g_r=(gr if _np.isfinite(gr) else None),
                                 eclipse=(r.shape == "eclipse"))
        fam = most_coherent(post) or "?"
        tok = family.split()[0].lower() in fam.lower()
        type_hits += tok; type_per[vtype][0] += tok
        print(f"  {vtype:5} {name[:16]:<16} P0={P0:6.3f} rec={r.best_period if r.best_period else float('nan'):6.3f} "
              f"g-r={gr:+.2f} {'P-OK' if ok else 'miss':5} -> {fam[:26]:<26} {'TYPE-OK' if tok else ''}", flush=True)
    print(f"\n=== LIGHT-CURVE VALIDATION (real VSX+ZTF, {time.time()-t0:.0f}s) ===")
    for k in SAMPLES:
        h, n = per_type[k]
        if n:
            print(f"  {k:5}: period {h}/{n}={h/n*100:.0f}%  type {type_per[k][0]}/{n}={type_per[k][0]/n*100:.0f}%")
    print(f"  OVERALL period recovery: {period_hits}/{period_tot} = "
          f"{period_hits/max(period_tot,1)*100:.0f}%")
    print(f"  OVERALL type accuracy:   {type_hits}/{type_tot} = "
          f"{type_hits/max(type_tot,1)*100:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
