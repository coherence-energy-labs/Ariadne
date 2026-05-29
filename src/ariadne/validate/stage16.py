"""Stage 16 validation gates (MASTER_PLAN.md - generalization + HDF5 atlas).

G_gen   - The engine generalizes across a wide mass-ratio spectrum with ONLY constant
          changes: every system in ATLAS_SYSTEMS (mu ~ 1.6e-8 .. 7e-3 -- Mars-Phobos,
          Saturn moons, Sun-Mars, the DART binary asteroid) yields a sensible L1 distance
          and a periodic L1 Lyapunov orbit.
G_atlas - The atlas persists to HDF5 and round-trips EXACTLY: build -> write -> read
          reproduces the systems, the Earth-Moon transport graph (nodes + edges), the
          ranked route catalog, and full provenance.

Run:  PYTHONPATH=src python -m ariadne.validate.stage16
"""
from __future__ import annotations

import os

from ..data.constants import ATLAS_SYSTEMS
from ..atlas.build import build_atlas
from ..atlas.store import write_atlas, read_atlas

OUT = os.path.join("results", "atlas", "ariadne_atlas.h5")


def check() -> tuple[bool, dict]:
    atlas = build_atlas(n_seeds=120, route_K=8)

    # G_gen: every generalization system has sensible, periodic libration structure
    gen_rows = []
    g_gen = True
    expected = {s.name for s in ATLAS_SYSTEMS}
    for name, sysd in atlas["systems"].items():
        if name not in expected:
            continue
        lib = sysd["libration"]
        ok = (lib["L1_km"] > 0.0 and lib["lyap_period_d"] > 0.0
              and lib["half_period_residual"] < 1e-9)
        g_gen = g_gen and ok
        gen_rows.append((name, sysd["params"]["mu"], lib["L1_km"],
                         lib["lyap_period_d"], lib["half_period_residual"], ok))
    g_gen = g_gen and len(gen_rows) == len(ATLAS_SYSTEMS)

    # G_atlas: write -> read round-trip is exact
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    write_atlas(OUT, atlas)
    back = read_atlas(OUT)

    em0, em1 = atlas["systems"]["Earth-Moon"], back["systems"]["Earth-Moon"]
    checks = {
        "systems": set(atlas["systems"]) == set(back["systems"]),
        "graph_nodes": len(em0["graph"]["nodes"]) == len(em1["graph"]["nodes"]),
        "graph_edges": len(em0["graph"]["edges"]) == len(em1["graph"]["edges"]),
        "routes": ([r["path"] for r in em0["routes"]] == [r["path"] for r in em1["routes"]]),
        "route_dv": (max((abs(a["dv_ms"] - b["dv_ms"])
                          for a, b in zip(em0["routes"], em1["routes"])), default=0.0) < 1e-6),
        "provenance": ("version" in back["provenance"] and "created_utc" in back["provenance"]
                       and back["provenance"].get("config", {}).get("n_seeds") == 120),
        "libration_roundtrip": all(
            abs(atlas["systems"][n]["libration"]["L1_km"] - back["systems"][n]["libration"]["L1_km"]) < 1e-6
            for n in atlas["systems"]),
    }
    g_atlas = all(checks.values())

    ok = g_gen and g_atlas
    return ok, {"gen_rows": gen_rows, "g_gen": g_gen, "g_atlas": g_atlas,
                "checks": checks, "atlas": atlas, "path": OUT}


def main() -> int:
    print("=== Ariadne Stage 16 validation  (generalization + HDF5 atlas) ===\n")
    ok, info = check()

    print("[G_gen] Engine generalizes across the mass-ratio spectrum")
    for name, mu, l1, per, resid, row_ok in sorted(info["gen_rows"], key=lambda r: r[1]):
        print(f"      {name:20s} mu={mu:.3e}  L1={l1:12.3f} km  "
              f"period={per:8.3f} d  (periodic to {resid:.1e})  {'OK' if row_ok else 'FAIL'}")
    print(f"      -> {'PASS' if info['g_gen'] else 'FAIL'}\n")

    a = info["atlas"]
    em = a["systems"]["Earth-Moon"]
    print("[G_atlas] HDF5 atlas build + round-trip")
    print(f"      atlas: {len(a['systems'])} systems, "
          f"Earth-Moon graph {len(em['graph']['nodes'])} nodes / {len(em['graph']['edges'])} edges, "
          f"{len(em['routes'])} ranked routes")
    print(f"      written to {info['path']}  ({a['provenance']['created_utc']})")
    for k, v in info["checks"].items():
        print(f"        {k:22s}: {'ok' if v else 'MISMATCH'}")
    print(f"      -> {'PASS' if info['g_atlas'] else 'FAIL'}\n")

    print(f"=== STAGE 16: {'ALL GATES PASS' if ok else 'FAILURE'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
