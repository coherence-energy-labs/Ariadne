"""Open atlas release + reference routes (MASTER_PLAN.md §12, Stage 17 deliverables).

Bundles the project's results into a self-describing, shareable release directory:
  - ariadne_atlas.h5    : the HDF5 atlas (systems + Earth-Moon graph + ranked routes),
  - INDEX.md            : a human-readable index (systems table, route catalog, the
                          curated reference routes with their validation rung),
  - reference_routes.csv: the reference routes as data.

`reference_routes()` is the curated, HONEST table: each route is tagged with the highest
fidelity rung it actually reached. We never label a route "GMAT-validated" unless GMAT
actually checked it (only the direct trans-lunar transfer, Stage 9, did). Earth->Moon
transfers and libration-network reconfigurations are kept in separate classes so their
very different Delta-v scales are never conflated.
"""
from __future__ import annotations

import csv
import os

from .build import build_atlas
from .store import write_atlas


def reference_routes() -> list[dict]:
    """Curated reference routes with honest per-route validation rung (from Stages 6-15)."""
    return [
        {"name": "Direct trans-lunar transfer", "route_class": "Earth->Moon transfer",
         "dv_ms": 3953.0, "tof_days": 5.0,
         "validation": "GMAT-validated: identical state agrees to 149 m / 0.89 mm/s (Stage 9)",
         "source": "Stage 8/9"},
        {"name": "Sun-assisted low-energy (WSB) transfer", "route_class": "Earth->Moon transfer",
         "dv_ms": 3907.0, "tof_days": 48.8,
         "validation": "Full DE440 ephemeris, converged; below Coimbra 3925 m/s (Stage 10)",
         "source": "Stage 10"},
        {"name": "Ballistic-capture end-to-end (LEO->LLO)", "route_class": "Earth->Moon transfer",
         "dv_ms": 3756.0, "tof_days": 6.0,
         "validation": "CR3BP manifold capture + vis-viva budget; brackets Coimbra (Stage 6)",
         "source": "Stage 6"},
        {"name": "L1->L2 3-hop reconfiguration (transport graph)", "route_class": "libration-network route",
         "dv_ms": 16.9, "tof_days": float("nan"),
         "validation": "Energy-exact CR3BP verification (Jacobi resid 1.3e-15) + solar "
                       "survivability 0.10 L*; discovered ~2x cheaper than direct patch (Stage 14/15)",
         "source": "Stage 14/15"},
    ]


def _md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _index_markdown(atlas: dict, refs: list[dict]) -> str:
    prov = atlas.get("provenance", {})
    lines = ["# Ariadne Atlas — open release", "",
             f"Built {prov.get('created_utc', '?')} (version {prov.get('version', '?')}).  ",
             prov.get("note", ""), "",
             "## Systems", "",
             "Engine output across the mass-ratio spectrum (periodic L1 Lyapunov orbits).", ""]
    rows = []
    for name, s in sorted(atlas["systems"].items(), key=lambda kv: kv[1]["params"]["mu"]):
        p, lib = s["params"], s["libration"]
        rows.append([name, f"{p['mu']:.3e}", f"{lib['L1_km']:.3f}",
                     f"{lib['lyap_period_d']:.3f}", f"{lib['half_period_residual']:.1e}"])
    lines.append(_md_table(["System", "mu", "L1 (km)", "Lyapunov period (d)", "periodicity resid"], rows))

    em = atlas["systems"].get("Earth-Moon", {})
    if em.get("routes"):
        lines += ["", "## Earth-Moon transport-graph route catalog (ranked)", "",
                  "Automatically mined (Yen k-shortest) and CR3BP-verified. These are "
                  "libration-network reconfigurations (L1<->L2), not Earth-departure transfers.", ""]
        rrows = [[i + 1, f"{r['dv_ms']:.1f}", r["hops"], " -> ".join(r["path"])]
                 for i, r in enumerate(em["routes"])]
        lines.append(_md_table(["#", "Delta-v (m/s)", "hops", "route"], rrows))

    lines += ["", "## Reference routes (curated, with validation rung)", "",
              "Honest fidelity tags. Earth->Moon transfers and libration-network routes are "
              "different classes -- their Delta-v scales are NOT comparable.", ""]
    refrows = [[r["name"], r["route_class"], f"{r['dv_ms']:.1f}",
                ("%.1f" % r["tof_days"]) if r["tof_days"] == r["tof_days"] else "n/a",
                r["validation"]] for r in refs]
    lines.append(_md_table(["Route", "Class", "Delta-v (m/s)", "TOF (d)", "Validation"], refrows))
    lines += ["", "---", "",
              "Standard gravity throughout (credibility firewall). Nothing here is new physics; "
              "the contribution is an open, validated, automated low-energy routing engine + atlas.",
              ""]
    return "\n".join(lines)


def export_release(outdir: str, atlas: dict | None = None) -> dict:
    """Write the release bundle (atlas .h5 + INDEX.md + reference_routes.csv). Returns paths."""
    os.makedirs(outdir, exist_ok=True)
    if atlas is None:
        atlas = build_atlas()
    refs = reference_routes()

    h5_path = os.path.join(outdir, "ariadne_atlas.h5")
    write_atlas(h5_path, atlas)

    index_path = os.path.join(outdir, "INDEX.md")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(_index_markdown(atlas, refs))

    csv_path = os.path.join(outdir, "reference_routes.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "route_class", "dv_ms", "tof_days",
                                          "validation", "source"])
        w.writeheader()
        w.writerows(refs)

    return {"atlas_h5": h5_path, "index_md": index_path, "reference_csv": csv_path,
            "n_systems": len(atlas["systems"]), "n_reference_routes": len(refs)}


def main() -> int:
    outdir = os.path.join("results", "release")
    print(f"Building Ariadne open release in {outdir} ...")
    info = export_release(outdir)
    print(f"  atlas:           {info['atlas_h5']}  ({info['n_systems']} systems)")
    print(f"  index:           {info['index_md']}")
    print(f"  reference routes:{info['reference_csv']}  ({info['n_reference_routes']} routes)")
    print("Done.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
