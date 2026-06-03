"""Close gap #1: INDEPENDENT mover labels. Fetch a per-class sample from JPL's
Small-Body Database (their official `orbit_class`, computed with JPL's criteria --
resonance-aware Hilda/Trojan, precise NEO sub-types -- not my (a,e) cuts), cache it
locally, and validate classify_mover against it. A match means the coherence
classifier reproduces JPL's authoritative classification, not just its own labels.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ariadne.discovery.imaging.coherence_classifier import classify_mover   # noqa: E402
from validate_mover_real import to_bucket                                   # noqa: E402

API = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
CACHE = Path("data/jpl_orbit_class.json")
# JPL orbit-class code -> our dynamical bucket
JPL_TO_BUCKET = {
    "IEO": "NEO", "ATE": "NEO", "APO": "NEO", "AMO": "NEO",
    "MCA": "Mars-crosser",
    "IMB": "main-belt", "MBA": "main-belt", "OMB": "main-belt",
    "HYA": "outer-belt", "TJN": "outer-belt",
    "CEN": "Centaur", "TNO": "TNO"}


def fetch(per_class=2500):
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    out = {}
    for code in JPL_TO_BUCKET:
        url = f"{API}?fields=a,e&sb-class={code}&limit={per_class}"
        try:
            r = json.load(urllib.request.urlopen(url, timeout=60))
            rows = [(float(a), float(e)) for a, e in r.get("data", [])
                    if a not in (None, "") and e not in (None, "")]
            out[code] = rows
            print(f"  {code}: {len(rows)}", flush=True)
        except Exception as ex:
            print(f"  {code}: ERR {type(ex).__name__} {ex}", flush=True); out[code] = []
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(out))
    return out


def main():
    print("Fetching JPL SBDB per orbit class (independent labels) ...", flush=True)
    data = fetch()
    a, e, jpl = [], [], []
    for code, rows in data.items():
        for ai, ei in rows:
            if 0.3 < ai < 2000 and 0 <= ei < 0.999:
                a.append(ai); e.append(ei); jpl.append(JPL_TO_BUCKET[code])
    a = np.array(a); e = np.array(e); jpl = np.array(jpl)
    print(f"\n=== INDEPENDENT VALIDATION vs JPL orbit_class ({len(a)} asteroids) ===")
    pred = np.array([to_bucket(max(classify_mover(ai, ei), key=classify_mover(ai, ei).get))
                     for ai, ei in zip(a, e)])
    overall = float(np.mean(pred == jpl))
    print(f"  classify_mover vs JPL overall agreement: {overall*100:.1f}%")
    print(f"\n  per JPL class:")
    for b in ["NEO", "Mars-crosser", "main-belt", "outer-belt", "Centaur", "TNO"]:
        m = jpl == b
        if m.sum():
            print(f"    {b:<14} n={int(m.sum()):>5}  agree {np.mean(pred[m]==b)*100:5.1f}%")
    print(f"\n  (JPL classes are derived with JPL's own criteria, so this is an")
    print(f"   independent check that the coherence classifier reproduces the")
    print(f"   authoritative dynamical taxonomy -- not its own (a,e) labels.)")
    print(f"  cached -> {CACHE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
