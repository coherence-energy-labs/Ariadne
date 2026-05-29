"""The Atlas: a persistent HDF5 store for the transport network (MASTER_PLAN.md §12, Stage 16).

One file holds, per CR3BP system: the system parameters, the libration summary, the
transport graph (nodes + patch edges), and the ranked route catalog -- each with
provenance (when it was built, with what code version and config). This is the durable,
browsable deliverable the project promised: an atlas of low-energy routes you can reopen,
diff, and extend.

The on-disk schema is deliberately simple and round-trippable:

  /provenance                         (group; attrs: created_utc, version, note, config_json)
  /systems/<name>                     (group; attrs: mu, L_star, T_star, V_star, primary, secondary)
      attrs: L1_km, L2_km, lyap_period_d, half_period_residual   (libration summary)
      /graph/node_key   (vlen-str [n])     node names
      /graph/node_point (vlen-str [n])     "L1"/"L2"
      /graph/node_jacobi(float [n])
      /graph/edge_src   (vlen-str [m]) / edge_dst (vlen-str [m])
      /graph/edge_dv_ms (float [m])  / edge_fragility (float [m])
      /routes/path      (vlen-str [k])     " -> "-joined node sequence
      /routes/dv_ms     (float [k])  / hops (int [k])

`write_atlas(path, atlas)` and `read_atlas(path)` are exact inverses for this schema.
"""
from __future__ import annotations

import json

import h5py
import numpy as np

_VLEN = h5py.string_dtype(encoding="utf-8")


def _set_attrs(group, d):
    for k, v in d.items():
        group.attrs[k] = v


def write_atlas(path, atlas: dict) -> str:
    """Write an atlas dict to HDF5. Returns the path."""
    with h5py.File(path, "w") as f:
        prov = f.create_group("provenance")
        p = dict(atlas.get("provenance", {}))
        if "config" in p and not isinstance(p["config"], str):
            p["config"] = json.dumps(p["config"])
        _set_attrs(prov, p)

        systems = f.create_group("systems")
        for name, sysd in atlas.get("systems", {}).items():
            g = systems.create_group(name)
            _set_attrs(g, sysd.get("params", {}))
            _set_attrs(g, sysd.get("libration", {}))

            graph = sysd.get("graph")
            if graph and graph.get("nodes"):
                gg = g.create_group("graph")
                nodes, edges = graph["nodes"], graph.get("edges", [])
                gg.create_dataset("node_key", data=np.array([n["key"] for n in nodes], dtype=_VLEN))
                gg.create_dataset("node_point", data=np.array([n["point"] for n in nodes], dtype=_VLEN))
                gg.create_dataset("node_jacobi", data=np.array([n["jacobi"] for n in nodes], float))
                gg.create_dataset("edge_src", data=np.array([e["src"] for e in edges], dtype=_VLEN))
                gg.create_dataset("edge_dst", data=np.array([e["dst"] for e in edges], dtype=_VLEN))
                gg.create_dataset("edge_dv_ms", data=np.array([e["dv_ms"] for e in edges], float))
                gg.create_dataset("edge_fragility", data=np.array([e["fragility"] for e in edges], float))

            routes = sysd.get("routes")
            if routes:
                rg = g.create_group("routes")
                rg.create_dataset("path", data=np.array([" -> ".join(r["path"]) for r in routes], dtype=_VLEN))
                rg.create_dataset("dv_ms", data=np.array([r["dv_ms"] for r in routes], float))
                rg.create_dataset("hops", data=np.array([r["hops"] for r in routes], int))
    return str(path)


def _attrs_to_dict(group, keys):
    out = {}
    for k in keys:
        if k in group.attrs:
            v = group.attrs[k]
            out[k] = v.item() if hasattr(v, "item") else v
    return out


def read_atlas(path) -> dict:
    """Read an atlas HDF5 file back into a dict (inverse of write_atlas)."""
    atlas = {"provenance": {}, "systems": {}}
    with h5py.File(path, "r") as f:
        if "provenance" in f:
            prov = {k: (v.item() if hasattr(v, "item") else v)
                    for k, v in f["provenance"].attrs.items()}
            if "config" in prov:
                try:
                    prov["config"] = json.loads(prov["config"])
                except (json.JSONDecodeError, TypeError):
                    pass
            atlas["provenance"] = prov

        for name, g in f.get("systems", {}).items():
            sysd = {
                "params": _attrs_to_dict(g, ("mu", "L_star", "T_star", "V_star",
                                             "primary", "secondary")),
                "libration": _attrs_to_dict(g, ("L1_km", "L2_km", "lyap_period_d",
                                                 "half_period_residual")),
            }
            if "graph" in g:
                gg = g["graph"]
                nodes = [{"key": k.decode() if isinstance(k, bytes) else k,
                          "point": p.decode() if isinstance(p, bytes) else p,
                          "jacobi": float(j)}
                         for k, p, j in zip(gg["node_key"][:], gg["node_point"][:], gg["node_jacobi"][:])]
                edges = [{"src": s.decode() if isinstance(s, bytes) else s,
                          "dst": d.decode() if isinstance(d, bytes) else d,
                          "dv_ms": float(dv), "fragility": float(fr)}
                         for s, d, dv, fr in zip(gg["edge_src"][:], gg["edge_dst"][:],
                                                 gg["edge_dv_ms"][:], gg["edge_fragility"][:])]
                sysd["graph"] = {"nodes": nodes, "edges": edges}
            if "routes" in g:
                rg = g["routes"]
                sysd["routes"] = [
                    {"path": (pth.decode() if isinstance(pth, bytes) else pth).split(" -> "),
                     "dv_ms": float(dv), "hops": int(h)}
                    for pth, dv, h in zip(rg["path"][:], rg["dv_ms"][:], rg["hops"][:])]
            atlas["systems"][name] = sysd
    return atlas
