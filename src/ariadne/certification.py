"""Proof-carrying route certificates for Ariadne trajectories.

A route is not treated as mission-grade because an optimizer found it. It is
promoted rung by rung, and each rung leaves machine-checkable evidence:

1. CR3BP patch proof: position continuity, Jacobi bookkeeping, burn equality.
2. BCR4BP survivability: bounded solar-perturbation divergence.
3. Robustness envelope: deterministic Monte-Carlo injection sensitivity.
4. Optional DE440 retargeting: position-continuous multiple shooting in real ephemeris.
5. Optional GMAT status: recorded explicitly as pass/fail/not_available/not_run.

The resulting certificate is a canonical JSON object with a SHA-256 hash over
the payload. It is intentionally conservative: missing required evidence fails
closed, and optional evidence is never silently upgraded to a pass.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .data.ephemeris import et
from .discovery.verify import ephemeris_survivability, verify_route
from .dynamics.cr3bp import propagate, pseudo_potential
from .io.gmat_export import locate_gmat
from .transfers.ephemeris_retarget import retarget_heteroclinic
from .transport_graph.graph import Edge, Node, TransportGraph


@dataclass(frozen=True)
class CertificateThresholds:
    """Explicit pass/fail limits for route promotion."""

    cr3bp_pos_gap_max: float = 1e-9
    cr3bp_jacobi_resid_max: float = 1e-6
    cr3bp_dv_resid_max: float = 1e-9
    bcr4bp_divergence_max_km: float = 100_000.0
    monte_carlo_sensitivity_max_km_per_ms: float = 250_000.0
    de440_position_resid_max_km: float = 1.0
    de440_total_dv_max_ms: float = 5_000.0


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, float):
        if not np.isfinite(x):
            return None
        return x
    if isinstance(x, (str, int, bool)) or x is None:
        return x
    return str(x)


def canonical_json(payload: dict) -> str:
    """Stable JSON form used for replay hashes."""

    return json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)


def payload_hash(payload: dict) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def stamp_certificate(payload: dict) -> dict:
    """Attach certificate_hash over all fields except the hash itself."""

    out = copy.deepcopy(_jsonable(payload))
    out.pop("certificate_hash", None)
    out["certificate_hash"] = payload_hash(out)
    return out


def validate_certificate(certificate: dict) -> dict:
    """Validate hash integrity and rung verdict consistency."""

    expected = certificate.get("certificate_hash")
    body = copy.deepcopy(certificate)
    body.pop("certificate_hash", None)
    hash_ok = expected == payload_hash(body)
    required = certificate.get("required_rungs", [])
    rungs = certificate.get("rungs", {})
    missing = [r for r in required if rungs.get(r, {}).get("status") != "pass"]
    status_ok = certificate.get("status") == ("certified" if not missing else "rejected")
    return {"ok": bool(hash_ok and status_ok and not missing),
            "hash_ok": bool(hash_ok), "status_ok": bool(status_ok),
            "missing_or_failed_required": missing}


def _edge(graph, u, v):
    return next((e for e in graph.edges.get(u, []) if e.dst == v), None)


def _route_fingerprint(graph: TransportGraph, path: list[str], system) -> str:
    edges = []
    for u, v in zip(path[:-1], path[1:]):
        e = _edge(graph, u, v)
        edges.append({"src": u, "dst": v, "dv": None if e is None else e.dv,
                      "meta": None if e is None else e.meta})
    return payload_hash({"system": system.name, "mu": graph.mu, "v_star": graph.v_star,
                         "path": path, "edges": edges})


def _monte_carlo_envelope(graph: TransportGraph, path: list[str], system, *,
                          n_samples: int = 24, sigma_ms: float = 1.0,
                          horizon: float = 0.25, seed: int = 46) -> dict:
    """Deterministic CR3BP injection sensitivity in km per m/s.

    Each patch post-state is perturbed by velocity vectors of norm `sigma_ms`,
    propagated over a short horizon, and compared with the nominal trajectory.
    """

    rng = np.random.default_rng(seed)
    rows = []
    worst = 0.0
    eps_nd = (sigma_ms / 1000.0) / system.V_star
    t_eval = np.linspace(0.0, horizon, 80)
    for u, v in zip(path[:-1], path[1:]):
        e = _edge(graph, u, v)
        if e is None or "post" not in e.meta:
            rows.append({"leg": [u, v], "status": "missing_patch_state"})
            worst = float("inf")
            continue
        s0 = np.array(e.meta["post"], float)
        nominal = propagate(s0, (0.0, horizon), graph.mu, t_eval=t_eval).y[:3, -1]
        max_drift = 0.0
        for _ in range(n_samples):
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            sp = s0.copy()
            sp[3:] += eps_nd * d
            end = propagate(sp, (0.0, horizon), graph.mu, t_eval=t_eval).y[:3, -1]
            max_drift = max(max_drift, float(np.linalg.norm(end - nominal) * system.L_star))
        sens = max_drift / sigma_ms
        rows.append({"leg": [u, v], "sigma_ms": sigma_ms,
                     "max_drift_km": max_drift,
                     "sensitivity_km_per_ms": sens,
                     "n_samples": n_samples})
        worst = max(worst, sens)
    return {"status": "pass", "worst_sensitivity_km_per_ms": worst,
            "horizon_nondim": horizon,
            "horizon_days": horizon * system.T_star / 86400.0,
            "per_leg": rows, "seed": seed}


def _rung(status: str, **evidence) -> dict:
    out = {"status": status}
    out.update(evidence)
    return out


def certify_transport_route(graph: TransportGraph, path: list[str], system, *,
                            epoch_utc: str = "2025-06-01T00:00:00",
                            thresholds: CertificateThresholds | None = None,
                            require_de440: bool = False,
                            de440_result: dict | None = None,
                            include_gmat_status: bool = True,
                            monte_carlo_samples: int = 24) -> dict:
    """Build a proof-carrying certificate for a transport-graph route.

    `de440_result` may be supplied by a higher-fidelity retargeter. If
    `require_de440=True`, the certificate fails unless that evidence passes the
    configured residual and total-Delta-v thresholds.
    """

    if thresholds is None:
        thresholds = CertificateThresholds()

    rungs = {}
    route_id = _route_fingerprint(graph, path, system)

    cr = verify_route(graph, path, tol_pos=thresholds.cr3bp_pos_gap_max,
                      tol_jac=thresholds.cr3bp_jacobi_resid_max)
    cr_ok = bool(cr["ok"] and all(l.get("dv_resid", 0.0) <= thresholds.cr3bp_dv_resid_max
                                  for l in cr["legs"] if "dv_resid" in l))
    rungs["cr3bp_patch_proof"] = _rung("pass" if cr_ok else "fail", **cr)

    try:
        bc = ephemeris_survivability(graph, path, system)
        bc_ok = bc["worst_divergence_km"] <= thresholds.bcr4bp_divergence_max_km
        rungs["bcr4bp_survivability"] = _rung("pass" if bc_ok else "fail", **bc)
    except Exception as exc:
        rungs["bcr4bp_survivability"] = _rung("fail", error=str(exc)[:240])

    try:
        mc = _monte_carlo_envelope(graph, path, system, n_samples=monte_carlo_samples)
        mc_ok = mc["worst_sensitivity_km_per_ms"] <= thresholds.monte_carlo_sensitivity_max_km_per_ms
        mc["status"] = "pass" if mc_ok else "fail"
        rungs["robustness_envelope"] = mc
    except Exception as exc:
        rungs["robustness_envelope"] = _rung("fail", error=str(exc)[:240])

    if de440_result is None:
        rungs["de440_retarget"] = _rung("not_run")
    else:
        de_ok = (de440_result.get("max_resid_km", float("inf")) <= thresholds.de440_position_resid_max_km
                 and de440_result.get("total_dv_ms", float("inf")) <= thresholds.de440_total_dv_max_ms)
        rungs["de440_retarget"] = _rung("pass" if de_ok else "fail", **de440_result)

    if include_gmat_status:
        gmat = locate_gmat()
        rungs["gmat_replay"] = _rung("not_available" if gmat is None else "not_run",
                                     executable=gmat)

    required = ["cr3bp_patch_proof", "bcr4bp_survivability", "robustness_envelope"]
    if require_de440:
        required.append("de440_retarget")
    certified = all(rungs.get(r, {}).get("status") == "pass" for r in required)

    payload = {
        "schema": "ariadne.route_certificate.v1",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "route_id": route_id,
        "status": "certified" if certified else "rejected",
        "required_rungs": required,
        "model_ladder": ["CR3BP", "BCR4BP", "DE440", "GMAT"],
        "system": {"name": system.name, "mu": system.mu, "L_star": system.L_star,
                   "T_star": system.T_star, "V_star": system.V_star,
                   "primary": system.primary, "secondary": system.secondary},
        "epoch_utc": epoch_utc,
        "path": path,
        "thresholds": asdict(thresholds),
        "rungs": rungs,
        "replay": {
            "commands": [
                "PYTHONPATH=src python -m ariadne.validate.stage46",
                "PYTHONPATH=src python -m pytest tests/test_certification.py",
            ],
            "route_payload_hash": route_id,
        },
    }
    return stamp_certificate(payload)


def graph_from_heteroclinic(conn: dict, system) -> tuple[TransportGraph, list[str]]:
    """Represent a same-energy heteroclinic as a one-edge transport graph."""

    mu = system.mu
    c = float(conn["jacobi"])
    xsec = float(conn["x_section"])
    y, vy = [float(v) for v in conn["connection_yv"]]
    om = pseudo_potential([xsec, y, 0.0, 0.0, 0.0, 0.0], mu)
    arg = 2.0 * om - c - vy * vy
    if arg < 0.0:
        raise ValueError("heteroclinic crossing is energetically infeasible")
    vx = float(np.sqrt(arg))
    src = f"{conn['source']}@{c:.3f}"
    dst = f"{conn['target']}@{c:.3f}"

    g = TransportGraph(mu, system.V_star)
    g.add_node(Node(key=src, point=conn["source"], jacobi=c, orbit=conn["orbit_source"]))
    g.add_node(Node(key=dst, point=conn["target"], jacobi=c, orbit=conn["orbit_target"]))
    state = [xsec, y, 0.0, vx, vy, 0.0]
    g.add_edge(Edge(src=src, dst=dst, dv=0.0, fragility=0.0,
                    meta={"pre": state, "post": state, "y": y, "vy": vy,
                          "source": "heteroclinic_exact_same_energy"}))
    return g, [src, dst]


def certify_heteroclinic_route(conn: dict, system, *,
                               epoch_utc: str = "2025-06-01T00:00:00",
                               thresholds: CertificateThresholds | None = None,
                               require_de440: bool = True,
                               n_patch: int = 10,
                               t_leg: float = 2.6,
                               monte_carlo_samples: int = 16) -> dict:
    """Certify a heteroclinic route and automatically promote it to DE440."""

    if thresholds is None:
        thresholds = CertificateThresholds()
    g, path = graph_from_heteroclinic(conn, system)
    de = None
    if require_de440:
        raw = retarget_heteroclinic(system.mu, system, conn, et(epoch_utc),
                                    n_patch=n_patch, t_leg=t_leg)
        if raw is not None:
            de = {"total_dv_ms": float(raw["total_dv_ms"]),
                  "max_resid_km": float(raw["max_resid_km"]),
                  "n_segments": int(raw["n_segments"]),
                  "epoch_utc": epoch_utc,
                  "method": "DE440 position-continuous multiple shooting"}
        else:
            de = {"total_dv_ms": float("inf"), "max_resid_km": float("inf"),
                  "n_segments": 0, "epoch_utc": epoch_utc,
                  "method": "DE440 position-continuous multiple shooting",
                  "error": "retargeter returned None"}
    return certify_transport_route(g, path, system, epoch_utc=epoch_utc,
                                   thresholds=thresholds,
                                   require_de440=require_de440,
                                   de440_result=de,
                                   monte_carlo_samples=monte_carlo_samples)
