"""Stage 17 validation gate (MASTER_PLAN.md - deliverables: white paper + open release).

G_deliver - The capstone deliverables exist and are self-consistent:
  (a) the white paper (docs/WHITE_PAPER.md) is present and contains its key sections and the
      headline validated numbers (149 m GMAT, 3,907 m/s WSB, ~42x search efficiency);
  (b) the open release bundle exports cleanly: HDF5 atlas + INDEX.md + reference_routes.csv,
      the atlas round-trips, and the reference table carries an honestly GMAT-validated route
      AND keeps Earth->Moon transfers separate from libration-network routes.

Run:  PYTHONPATH=src python -m ariadne.validate.stage17
"""
from __future__ import annotations

import os

from ..atlas.release import export_release, reference_routes
from ..atlas.store import read_atlas

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
WHITE_PAPER = os.path.join(REPO_ROOT, "docs", "WHITE_PAPER.md")
RELEASE_DIR = os.path.join("results", "release")


def check() -> tuple[bool, dict]:
    # (a) white paper present + has key sections + headline numbers
    paper_ok, missing = False, []
    if os.path.exists(WHITE_PAPER):
        text = open(WHITE_PAPER, encoding="utf-8").read()
        required = ["## Abstract", "## 4. Validation gates", "## 6. Honest limitations",
                    "## 7. Reproducibility", "149 m", "3,907", "42"]
        missing = [r for r in required if r not in text]
        paper_ok = not missing

    # (b) release bundle exports + round-trips + honest reference table
    info = export_release(RELEASE_DIR)
    back = read_atlas(info["atlas_h5"])
    refs = reference_routes()
    gmat_validated = [r for r in refs if "GMAT-validated" in r["validation"]]
    classes = {r["route_class"] for r in refs}
    release_ok = (os.path.exists(info["index_md"])
                  and os.path.exists(info["reference_csv"])
                  and len(back["systems"]) == info["n_systems"]
                  and len(gmat_validated) >= 1
                  and {"Earth->Moon transfer", "libration-network route"} <= classes)

    ok = paper_ok and release_ok
    return ok, {"paper_ok": paper_ok, "missing": missing, "release_ok": release_ok,
                "info": info, "n_refs": len(refs), "n_gmat": len(gmat_validated),
                "classes": classes}


def main() -> int:
    print("=== Ariadne Stage 17 validation  (deliverables: white paper + open release) ===\n")
    ok, info = check()

    print("[G_deliver a] White paper (docs/WHITE_PAPER.md)")
    print(f"      present + key sections + headline numbers: {'PASS' if info['paper_ok'] else 'FAIL'}")
    if info["missing"]:
        print(f"      missing: {info['missing']}")
    print()

    i = info["info"]
    print("[G_deliver b] Open release bundle")
    print(f"      atlas:  {i['atlas_h5']}  ({i['n_systems']} systems)")
    print(f"      index:  {i['index_md']}")
    print(f"      routes: {i['reference_csv']}  ({info['n_refs']} reference routes, "
          f"{info['n_gmat']} GMAT-validated)")
    print(f"      route classes: {sorted(info['classes'])}")
    print(f"      -> {'PASS' if info['release_ok'] else 'FAIL'}\n")

    print(f"=== STAGE 17: {'ALL GATES PASS' if ok else 'FAILURE'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
