"""Universal solar-system mission navigator.

The navigator is a top-level architecture layer: it asks multiple engines for
candidate routes, keeps their fidelity/assumptions explicit, ranks the Pareto
front, and writes audit + visual artifacts.  It is meant for first-cut mission
architecture, not final flight design; every route says exactly what physics
was used.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from ..data.constants import (
    GM_JUPITER, GM_SATURN,
    GM_IO, GM_EUROPA, GM_GANYMEDE, GM_CALLISTO,
    GM_TITAN, GM_RHEA, GM_ENCELADUS, GM_IAPETUS,
    R_IO, R_EUROPA, R_GANYMEDE, R_CALLISTO,
)
from ..data.ephemeris import et, utc
from ..transfers.tisserand import (
    connecting_transfer,
    hohmann_leg_dv,
    max_turn_angle,
    vinf_at_moon,
)
from .flyby import evaluate_chain, optimize_chain
from .porkchop import (
    coherent_knee,
    lambert_transfer,
    launch_windows,
    optimize_window,
    porkchop,
    time_energy_pareto,
)


DAY = 86400.0


R_TITAN = 2574.73
R_RHEA = 763.8
R_ENCELADUS = 252.1
R_IAPETUS = 734.5


@dataclass(frozen=True)
class BodyTarget:
    """Normalized target definition."""
    name: str
    ephemeris_name: str
    kind: str
    primary: str | None = None
    system_barycenter: str | None = None
    orbit_radius_km: float | None = None
    body_radius_km: float | None = None
    gm_km3_s2: float | None = None


@dataclass(frozen=True)
class NavigatorConstraints:
    """Search controls for universal navigation."""
    origin: str = "EARTH"
    target: str = "MARS"
    epoch_start: str = "2028-01-01T00:00:00"
    departure_window_days: float = 365.25 * 4.0
    tof_range_days: tuple[float, float] = (120.0, 2500.0)
    n_dep: int = 60
    n_tof: int = 45
    optimize_direct: bool = True
    include_direct: bool = True
    include_gravity_assist: bool = True
    include_moon_tour: bool = True
    optimize_flybys: bool = False
    flyby_maxiter: int = 35
    flyby_alt_km: float = 300.0
    max_total_dv_ms: float | None = None
    max_tof_days: float | None = None
    max_direct_pareto_routes: int = 10


@dataclass(frozen=True)
class NavigatorWeights:
    """Scalar ranking weights. Lower score is better."""
    dv: float = 1.0
    tof: float = 0.002
    c3: float = 0.04
    arrival_vinf: float = 0.25
    risk: float = 1.0
    fidelity_bonus: float = 0.25


@dataclass(frozen=True)
class RouteEvent:
    """One route event for route cards and plotted labels."""
    body: str
    epoch_utc: str
    role: str
    coordinates_km: tuple[float, float, float] | None = None
    vinf_kms: float | None = None
    notes: str = ""


@dataclass(frozen=True)
class MissionRoute:
    """One candidate route."""
    route_id: str
    name: str
    engine: str
    sequence: tuple[str, ...]
    target: str
    fidelity: str
    total_dv_ms: float
    tof_days: float
    c3_km2_s2: float | None = None
    arrival_vinf_kms: float | None = None
    risk: float = 0.5
    feasible: bool = True
    pareto_optimal: bool = False
    assumptions: tuple[str, ...] = ()
    validations: tuple[str, ...] = ()
    events: tuple[RouteEvent, ...] = ()
    components: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)
    certificate_hash: str = ""


@dataclass(frozen=True)
class NavigatorReport:
    """Full route-search proof bundle."""
    schema: str
    created_utc: str
    constraints: NavigatorConstraints
    weights: NavigatorWeights
    origin: BodyTarget
    target: BodyTarget
    routes: tuple[MissionRoute, ...]
    pareto_front: tuple[str, ...]
    fastest_id: str | None
    cheapest_id: str | None
    balanced_id: str | None
    certificate_hash: str
    artifacts: dict = field(default_factory=dict)

    @property
    def fastest(self) -> MissionRoute | None:
        return _find_route(self.routes, self.fastest_id)

    @property
    def cheapest(self) -> MissionRoute | None:
        return _find_route(self.routes, self.cheapest_id)

    @property
    def balanced(self) -> MissionRoute | None:
        return _find_route(self.routes, self.balanced_id)


def _find_route(routes, rid):
    for route in routes:
        if route.route_id == rid:
            return route
    return None


def _jsonable(value):
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def stable_hash(value) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _target_registry() -> dict[str, BodyTarget]:
    targets = {
        "EARTH": BodyTarget("EARTH", "EARTH", "planet", system_barycenter="EARTH"),
        "VENUS": BodyTarget("VENUS", "VENUS", "planet", system_barycenter="VENUS"),
        "MARS": BodyTarget("MARS", "MARS BARYCENTER", "planet", system_barycenter="MARS BARYCENTER"),
        "JUPITER": BodyTarget("JUPITER", "JUPITER BARYCENTER", "planet", system_barycenter="JUPITER BARYCENTER"),
        "SATURN": BodyTarget("SATURN", "SATURN BARYCENTER", "planet", system_barycenter="SATURN BARYCENTER"),
        "TITAN": BodyTarget("TITAN", "SATURN BARYCENTER", "moon", "SATURN", "SATURN BARYCENTER",
                            1221870.0, R_TITAN, GM_TITAN),
        "RHEA": BodyTarget("RHEA", "SATURN BARYCENTER", "moon", "SATURN", "SATURN BARYCENTER",
                           527108.0, R_RHEA, GM_RHEA),
        "ENCELADUS": BodyTarget("ENCELADUS", "SATURN BARYCENTER", "moon", "SATURN", "SATURN BARYCENTER",
                                237948.0, R_ENCELADUS, GM_ENCELADUS),
        "IAPETUS": BodyTarget("IAPETUS", "SATURN BARYCENTER", "moon", "SATURN", "SATURN BARYCENTER",
                              3560820.0, R_IAPETUS, GM_IAPETUS),
        "IO": BodyTarget("IO", "JUPITER BARYCENTER", "moon", "JUPITER", "JUPITER BARYCENTER",
                         421800.0, R_IO, GM_IO),
        "EUROPA": BodyTarget("EUROPA", "JUPITER BARYCENTER", "moon", "JUPITER", "JUPITER BARYCENTER",
                             671100.0, R_EUROPA, GM_EUROPA),
        "GANYMEDE": BodyTarget("GANYMEDE", "JUPITER BARYCENTER", "moon", "JUPITER", "JUPITER BARYCENTER",
                               1070400.0, R_GANYMEDE, GM_GANYMEDE),
        "CALLISTO": BodyTarget("CALLISTO", "JUPITER BARYCENTER", "moon", "JUPITER", "JUPITER BARYCENTER",
                               1882700.0, R_CALLISTO, GM_CALLISTO),
    }
    return targets


def resolve_target(name: str) -> BodyTarget:
    key = name.upper().replace(" ", "_")
    aliases = {
        "MARS_BARYCENTER": "MARS",
        "JUPITER_BARYCENTER": "JUPITER",
        "SATURN_BARYCENTER": "SATURN",
    }
    key = aliases.get(key, key)
    registry = _target_registry()
    if key not in registry:
        raise KeyError(f"unknown navigation target {name!r}; known={sorted(registry)}")
    return registry[key]


def _central_moon_catalog(primary: str):
    if primary == "JUPITER":
        return GM_JUPITER, [
            resolve_target("IO"), resolve_target("EUROPA"),
            resolve_target("GANYMEDE"), resolve_target("CALLISTO"),
        ]
    if primary == "SATURN":
        return GM_SATURN, [
            resolve_target("ENCELADUS"), resolve_target("RHEA"),
            resolve_target("TITAN"), resolve_target("IAPETUS"),
        ]
    return None, []


def moon_tour_screen(target: BodyTarget, *, flyby_alt_km: float = 300.0) -> dict | None:
    """Tisserand-style moon tour from a large capture moon to target moon."""
    if target.kind != "moon" or target.primary is None:
        return None
    gm_central, moons = _central_moon_catalog(target.primary)
    if not moons:
        return None
    moons = sorted(moons, key=lambda m: m.orbit_radius_km or 0.0)
    target_idx = [i for i, m in enumerate(moons) if m.name == target.name][0]
    anchor_idx = len(moons) - 1 if target_idx < len(moons) - 1 else 0
    step = 1 if target_idx > anchor_idx else -1
    path = moons[anchor_idx:target_idx + step:step]
    if len(path) < 2:
        return {
            "sequence": [target.name],
            "legs": [],
            "junctions": [],
            "ga_deterministic_dv_ms": 0.0,
            "hohmann_dv_ms": 0.0,
            "saving_ms": 0.0,
            "anchor": target.name,
        }
    legs = []
    hohmann = 0.0
    for a, b in zip(path[:-1], path[1:]):
        r1, r2 = float(a.orbit_radius_km), float(b.orbit_radius_km)
        rin, rout = min(r1, r2), max(r1, r2)
        aa, e = connecting_transfer(rin, rout)
        vi = vinf_at_moon(aa, e, r1, gm_central)
        vo = vinf_at_moon(aa, e, r2, gm_central)
        turn_a = max_turn_angle(vi, float(a.gm_km3_s2), float(a.body_radius_km) + flyby_alt_km)
        turn_b = max_turn_angle(vo, float(b.gm_km3_s2), float(b.body_radius_km) + flyby_alt_km)
        legs.append({
            "from": a.name, "to": b.name,
            "vinf_from_kms": vi, "vinf_to_kms": vo,
            "turn_from_deg": math.degrees(turn_a),
            "turn_to_deg": math.degrees(turn_b),
        })
        hohmann += hohmann_leg_dv(r1, r2, gm_central)
    mismatch = 0.0
    junctions = []
    for left, right in zip(legs[:-1], legs[1:]):
        vin = left["vinf_to_kms"]
        vout = right["vinf_from_kms"]
        dv = abs(vout - vin)
        mismatch += dv
        junctions.append({"moon": left["to"], "vinf_in": vin, "vinf_out": vout,
                          "dv_ms": dv * 1000.0})
    return {
        "sequence": [m.name for m in path],
        "anchor": path[0].name,
        "legs": legs,
        "junctions": junctions,
        "ga_deterministic_dv_ms": mismatch * 1000.0,
        "hohmann_dv_ms": hohmann * 1000.0,
        "saving_ms": (hohmann - mismatch) * 1000.0,
    }


def _route_from_direct(rec: dict, target: BodyTarget, *, name="direct") -> MissionRoute:
    events = (
        RouteEvent(rec.get("dep_body", "EARTH"), utc(rec["et_dep"]), "departure",
                   tuple(float(x) for x in rec.get("r1", [0, 0, 0]))),
        RouteEvent(target.system_barycenter or target.ephemeris_name, utc(rec["et_arr"]), "arrival",
                   tuple(float(x) for x in rec.get("r2", [0, 0, 0])), rec.get("arr_vinf_kms")),
    )
    origin_name = rec.get("dep_body", "EARTH")
    shell = {
        "name": name,
        "sequence": (origin_name, target.system_barycenter or target.name),
        "total_dv_ms": rec["total_ms"],
        "tof_days": rec["tof_days"],
        "c3": rec["c3"],
    }
    cert = stable_hash(shell)
    return MissionRoute(
        route_id=f"route_{cert[:14]}",
        name=name,
        engine="porkchop.optimize_window",
        sequence=(origin_name, target.system_barycenter or target.name),
        target=target.name,
        fidelity="patched_conic_ephemeris",
        total_dv_ms=float(rec["total_ms"]),
        tof_days=float(rec["tof_days"]),
        c3_km2_s2=float(rec["c3"]),
        arrival_vinf_kms=float(rec["arr_vinf_kms"]),
        risk=min(0.95, 0.18 + float(rec["arr_vinf_kms"]) / 30.0),
        feasible=True,
        assumptions=("heliocentric Lambert arc on real ephemerides",
                     "departure parking-orbit cost used when configured; otherwise departure v-infinity",
                     "capture cost included only for bodies with configured capture model"),
        validations=(f"launch C3 {rec['c3']:.3f} km^2/s^2",
                     f"arrival v_inf {rec['arr_vinf_kms']:.3f} km/s"),
        events=events,
        components={"dv_depart_ms": rec.get("dv_dep_ms"),
                    "dv_arrival_ms": rec.get("dv_arr_ms"),
                    "c3_km2_s2": rec.get("c3")},
        raw={k: v for k, v in rec.items() if k not in {"r1", "v1", "r2", "v2"}},
        certificate_hash=cert,
    )


def _sample_front(front: list[dict], limit: int) -> list[dict]:
    """Keep a representative spread across a sorted Pareto front."""
    if limit <= 0 or len(front) <= limit:
        return list(front)
    idxs = np.linspace(0, len(front) - 1, limit)
    keep = []
    seen = set()
    for x in idxs:
        i = int(round(float(x)))
        if i not in seen:
            seen.add(i)
            keep.append(front[i])
    return keep


def _with_moon_tour(route: MissionRoute, target: BodyTarget,
                    tour: dict | None) -> MissionRoute:
    if not tour or target.kind != "moon":
        return route
    total = route.total_dv_ms + float(tour["ga_deterministic_dv_ms"])
    seq = tuple([*route.sequence, *tour["sequence"]])
    shell = {"base": route.certificate_hash, "moon_tour": tour, "total": total}
    cert = stable_hash(shell)
    return MissionRoute(
        route_id=f"route_{cert[:14]}",
        name=f"{route.name}_plus_{target.primary.lower()}_moon_tour_to_{target.name.lower()}",
        engine=f"{route.engine}+tisserand_moon_tour",
        sequence=seq,
        target=target.name,
        fidelity="patched_conic_ephemeris+tisserand_screen",
        total_dv_ms=total,
        tof_days=route.tof_days,
        c3_km2_s2=route.c3_km2_s2,
        arrival_vinf_kms=route.arrival_vinf_kms,
        risk=min(0.98, route.risk + 0.12),
        feasible=route.feasible,
        assumptions=(*route.assumptions,
                     "moon tour is Tisserand energy screening, not phased moon ephemeris"),
        validations=(*route.validations,
                     f"moon-tour deterministic mismatch {tour['ga_deterministic_dv_ms']:.1f} m/s",
                     f"moon-tour Hohmann baseline {tour['hohmann_dv_ms']:.1f} m/s"),
        events=route.events + tuple(
            RouteEvent(m, route.events[-1].epoch_utc, "moon-tour-node",
                       notes="abstract Tisserand screening node; phased moon ephemeris not attached")
            for m in tour["sequence"]
        ),
        components={**route.components, "moon_tour": tour},
        raw={**route.raw, "moon_tour": tour},
        certificate_hash=cert,
    )


def _pareto_ids(routes: list[MissionRoute]) -> set[str]:
    out = set()
    for a in routes:
        dominated = False
        for b in routes:
            if a is b:
                continue
            if (
                b.total_dv_ms <= a.total_dv_ms
                and b.tof_days <= a.tof_days
                and (b.risk <= a.risk + 1e-12)
                and (b.total_dv_ms < a.total_dv_ms
                     or b.tof_days < a.tof_days
                     or b.risk < a.risk)
            ):
                dominated = True
                break
        if not dominated:
            out.add(a.route_id)
    return out


def _score(route: MissionRoute, weights: NavigatorWeights) -> float:
    fidelity_rank = {
        "patched_conic_ephemeris": 2,
        "patched_conic_ephemeris+tisserand_screen": 2,
        "optimized_flyby_chain": 3,
    }.get(route.fidelity, 1)
    return (
        weights.dv * route.total_dv_ms / 1000.0
        + weights.tof * route.tof_days
        + weights.c3 * (route.c3_km2_s2 or 0.0)
        + weights.arrival_vinf * (route.arrival_vinf_kms or 0.0)
        + weights.risk * route.risk
        - weights.fidelity_bonus * fidelity_rank
    )


def _flyby_templates(target: BodyTarget) -> list[tuple[str, list[str], tuple[tuple[float, float], ...]]]:
    dest = target.system_barycenter or target.ephemeris_name
    if dest == "JUPITER BARYCENTER":
        return [
            ("VEEGA_Jupiter", ["EARTH", "VENUS", "EARTH", "EARTH", dest],
             ((120, 260), (220, 520), (500, 900), (800, 1300))),
            ("EJ_direct_assist", ["EARTH", "EARTH", "JUPITER BARYCENTER"],
             ((300, 800), (700, 1400))),
        ]
    if dest == "SATURN BARYCENTER":
        return [
            ("EVEEJS", ["EARTH", "VENUS", "EARTH", "EARTH", "JUPITER BARYCENTER", dest],
             ((120, 260), (220, 520), (500, 900), (650, 1300), (900, 1900))),
            ("EJS", ["EARTH", "JUPITER BARYCENTER", dest],
             ((700, 1400), (900, 2200))),
        ]
    return []


def _flyby_routes(target: BodyTarget, constraints: NavigatorConstraints,
                  optimizer: Callable = optimize_chain) -> list[MissionRoute]:
    if not constraints.optimize_flybys:
        return []
    routes = []
    start = et(constraints.epoch_start)
    for name, bodies, bounds in _flyby_templates(target):
        rec = optimizer(
            bodies,
            start,
            constraints.departure_window_days,
            bounds,
            flyby_alt_km=constraints.flyby_alt_km,
            maxiter=constraints.flyby_maxiter,
            seed=7,
            dsm_legs=None,
            popsize=8,
        )
        if rec is None:
            continue
        seq = tuple(bodies)
        events = tuple(
            RouteEvent(body, utc(epoch), "flyby" if 0 < i < len(bodies) - 1 else
                       ("departure" if i == 0 else "arrival"),
                       vinf_kms=None)
            for i, (body, epoch) in enumerate(zip(bodies, rec["epochs"]))
        )
        shell = {"name": name, "sequence": seq, "rec": rec}
        cert = stable_hash(shell)
        routes.append(MissionRoute(
            route_id=f"route_{cert[:14]}",
            name=name,
            engine="flyby.optimize_chain",
            sequence=seq,
            target=target.name,
            fidelity="optimized_flyby_chain",
            total_dv_ms=float(rec["total_dv_ms"]),
            tof_days=float(rec["tof_total_days"]),
            c3_km2_s2=float(rec["c3"]),
            arrival_vinf_kms=float(rec["arr_vinf_kms"]),
            risk=min(0.98, 0.15 + 0.2 * float(rec["infeasible"]) + rec["mismatch_dv_ms"] / 8000.0),
            feasible=float(rec["infeasible"]) <= 1e-9,
            assumptions=("patched-conic Lambert legs", "flybys rotate v_inf; mismatch is powered-flyby cost"),
            validations=(f"{len(rec['flybys'])} flyby checks",
                         f"turn infeasibility {rec['infeasible']:.6g} rad"),
            events=events,
            components={"launch_ms": rec["dv_launch_ms"],
                        "mismatch_ms": rec["mismatch_dv_ms"],
                        "dsm_ms": rec.get("dsm_dv_ms", 0.0),
                        "flybys": rec["flybys"]},
            raw={k: v for k, v in rec.items() if k not in {"r1", "v1"}},
            certificate_hash=cert,
        ))
    return routes


def navigate_solar_system(
    constraints: NavigatorConstraints,
    weights: NavigatorWeights | None = None,
    *,
    porkchop_solver: Callable = porkchop,
    direct_optimizer: Callable = optimize_window,
    launch_window_solver: Callable = launch_windows,
    flyby_optimizer: Callable = optimize_chain,
) -> NavigatorReport:
    """Search and rank routes to a planet or major moon."""
    weights = weights or NavigatorWeights()
    origin = resolve_target(constraints.origin)
    target = resolve_target(constraints.target)
    if origin.name != "EARTH" and constraints.include_gravity_assist:
        constraints = NavigatorConstraints(
            origin=constraints.origin,
            target=constraints.target,
            epoch_start=constraints.epoch_start,
            departure_window_days=constraints.departure_window_days,
            tof_range_days=constraints.tof_range_days,
            n_dep=constraints.n_dep,
            n_tof=constraints.n_tof,
            optimize_direct=constraints.optimize_direct,
            include_direct=constraints.include_direct,
            include_gravity_assist=False,
            include_moon_tour=constraints.include_moon_tour,
            optimize_flybys=False,
            flyby_maxiter=constraints.flyby_maxiter,
            flyby_alt_km=constraints.flyby_alt_km,
            max_total_dv_ms=constraints.max_total_dv_ms,
            max_tof_days=constraints.max_tof_days,
            max_direct_pareto_routes=constraints.max_direct_pareto_routes,
        )
    arrival = target.system_barycenter or target.ephemeris_name
    start = et(constraints.epoch_start)
    routes: list[MissionRoute] = []
    pork = None

    if constraints.include_direct:
        pork = porkchop_solver(
            origin.ephemeris_name,
            arrival,
            start,
            constraints.departure_window_days,
            constraints.tof_range_days,
            n_dep=constraints.n_dep,
            n_tof=constraints.n_tof,
            capture=False,
        )
        if pork.get("grid_best"):
            best = dict(pork["grid_best"])
            best["dep_body"] = origin.ephemeris_name
            route = _route_from_direct(best, target, name="direct_grid_best")
            routes.append(_with_moon_tour(route, target, moon_tour_screen(target, flyby_alt_km=constraints.flyby_alt_km)))
        front = time_energy_pareto(pork)
        knee = coherent_knee(front)
        selected = _sample_front(front, constraints.max_direct_pareto_routes)
        if knee is not None:
            selected.append(knee)
        seen_pairs = set()
        for idx, p in enumerate(selected):
            key = (round(float(p["dep_et"]), 6), round(float(p["tof_days"]), 6))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            rec = lambert_transfer(
                origin.ephemeris_name,
                arrival,
                float(p["dep_et"]),
                float(p["tof_days"]),
                capture=False,
            )
            if rec is None:
                continue
            rec = dict(rec)
            rec["dep_body"] = origin.ephemeris_name
            tag = "knee" if knee is not None and key == (
                round(float(knee["dep_et"]), 6), round(float(knee["tof_days"]), 6)) else f"{idx:02d}"
            route = _route_from_direct(rec, target, name=f"direct_pareto_{tag}")
            routes.append(_with_moon_tour(route, target, moon_tour_screen(target, flyby_alt_km=constraints.flyby_alt_km)))
        if constraints.optimize_direct:
            opt = direct_optimizer(
                origin.ephemeris_name,
                arrival,
                start,
                constraints.departure_window_days,
                constraints.tof_range_days,
                capture=False,
                maxiter=45,
                seed=3,
            )
            if opt:
                opt = dict(opt)
                opt["dep_body"] = origin.ephemeris_name
                route = _route_from_direct(opt, target, name="direct_optimized")
                routes.append(_with_moon_tour(route, target, moon_tour_screen(target, flyby_alt_km=constraints.flyby_alt_km)))

    if constraints.include_gravity_assist:
        for route in _flyby_routes(target, constraints, optimizer=flyby_optimizer):
            routes.append(_with_moon_tour(route, target, moon_tour_screen(target, flyby_alt_km=constraints.flyby_alt_km)))

    filtered = []
    for route in routes:
        if constraints.max_total_dv_ms is not None and route.total_dv_ms > constraints.max_total_dv_ms:
            continue
        if constraints.max_tof_days is not None and route.tof_days > constraints.max_tof_days:
            continue
        filtered.append(route)
    routes = filtered
    pareto = _pareto_ids(routes)
    routes = [
        MissionRoute(
            route_id=r.route_id,
            name=r.name,
            engine=r.engine,
            sequence=r.sequence,
            target=r.target,
            fidelity=r.fidelity,
            total_dv_ms=r.total_dv_ms,
            tof_days=r.tof_days,
            c3_km2_s2=r.c3_km2_s2,
            arrival_vinf_kms=r.arrival_vinf_kms,
            risk=r.risk,
            feasible=r.feasible,
            pareto_optimal=r.route_id in pareto,
            assumptions=r.assumptions,
            validations=r.validations,
            events=r.events,
            components=r.components,
            raw=r.raw,
            certificate_hash=r.certificate_hash,
        )
        for r in routes
    ]
    routes.sort(key=lambda r: (_score(r, weights), r.total_dv_ms, r.tof_days))
    fastest = min(routes, key=lambda r: r.tof_days).route_id if routes else None
    cheapest = min(routes, key=lambda r: r.total_dv_ms).route_id if routes else None
    balanced = min(routes, key=lambda r: _score(r, weights)).route_id if routes else None
    shell = {
        "schema": "ariadne.solar_system_navigator.v1",
        "constraints": constraints,
        "weights": weights,
        "origin": origin,
        "target": target,
        "routes": routes,
        "pareto_front": sorted(pareto),
        "fastest_id": fastest,
        "cheapest_id": cheapest,
        "balanced_id": balanced,
    }
    return NavigatorReport(
        schema=shell["schema"],
        created_utc=datetime.now(timezone.utc).isoformat(),
        constraints=constraints,
        weights=weights,
        origin=origin,
        target=target,
        routes=tuple(routes),
        pareto_front=tuple(sorted(pareto)),
        fastest_id=fastest,
        cheapest_id=cheapest,
        balanced_id=balanced,
        certificate_hash=stable_hash(shell),
    )


def _plot_fonts():
    from PIL import ImageFont
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]

    def load(size, bold=False):
        names = []
        for c in candidates:
            if bold:
                names.extend([
                    c.replace(".ttf", "bd.ttf"),
                    c.replace(".ttf", "b.ttf"),
                ])
            names.append(c)
        for name in names:
            try:
                return ImageFont.truetype(name, size=size)
            except Exception:
                pass
        return ImageFont.load_default()

    return load(27, True), load(20), load(17), load(15)


_VIRIDIS_STOPS = (
    (0.000, (68, 1, 84)),
    (0.130, (71, 44, 122)),
    (0.250, (59, 82, 139)),
    (0.380, (44, 113, 142)),
    (0.500, (33, 144, 141)),
    (0.630, (39, 173, 129)),
    (0.750, (92, 200, 99)),
    (0.880, (170, 220, 50)),
    (1.000, (253, 231, 37)),
)


def _viridis(t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, float(t)))
    for (t0, c0), (t1, c1) in zip(_VIRIDIS_STOPS[:-1], _VIRIDIS_STOPS[1:]):
        if t <= t1:
            q = (t - t0) / max(t1 - t0, 1e-12)
            return tuple(int(round(c0[i] + q * (c1[i] - c0[i]))) for i in range(3))
    return _VIRIDIS_STOPS[-1][1]


def _heatmap_image(z: np.ndarray, width: int, height: int,
                   zmin: float, zmax: float):
    """Render a smooth, perceptual heatmap without depending on Matplotlib."""
    from PIL import Image, ImageFilter
    arr = np.asarray(z, float)
    finite = np.isfinite(arr)
    arr = np.where(finite, arr, zmax)
    norm = np.clip((arr - zmin) / (zmax - zmin + 1e-12), 0.0, 1.0)
    rgb = np.zeros((*norm.shape, 3), dtype=np.uint8)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            rgb[i, j] = _viridis(norm[i, j])
    rgb = rgb[::-1, :, :]  # y axis increases upward
    small = Image.fromarray(rgb, mode="RGB")
    resample = getattr(Image, "Resampling", Image).BICUBIC
    heat = small.resize((width, height), resample=resample)
    return heat.filter(ImageFilter.GaussianBlur(radius=0.35))


def _nice_ticks(vmin: float, vmax: float, n: int = 6) -> list[float]:
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmin == vmax:
        return [vmin]
    raw = abs(vmax - vmin) / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((1, 2, 2.5, 5, 10), key=lambda s: abs(s * mag - raw)) * mag
    start = math.ceil(vmin / step) * step
    ticks = []
    x = start
    while x <= vmax + 0.5 * step:
        if x >= vmin - 1e-9:
            ticks.append(float(x))
        x += step
    return ticks or [vmin, vmax]


def _fmt_tick(x: float) -> str:
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}".rstrip("0").rstrip(".")
    return f"{x:.2f}".rstrip("0").rstrip(".")


def _draw_axes(d, left, top, plot_w, plot_h, *,
               x_min, x_max, y_min, y_max, x_label, y_label,
               tick_font, label_font):
    axis = (20, 20, 20)
    grid = (216, 216, 216)
    for xval in _nice_ticks(x_min, x_max, 7):
        x = int(left + (xval - x_min) / (x_max - x_min + 1e-12) * plot_w)
        d.line([x, top, x, top + plot_h], fill=grid, width=1)
        d.line([x, top + plot_h, x, top + plot_h + 7], fill=axis, width=2)
        text = _fmt_tick(xval)
        bbox = d.textbbox((0, 0), text, font=tick_font)
        d.text((x - (bbox[2] - bbox[0]) / 2, top + plot_h + 12), text,
               fill=axis, font=tick_font)
    for yval in _nice_ticks(y_min, y_max, 6):
        y = int(top + (1.0 - (yval - y_min) / (y_max - y_min + 1e-12)) * plot_h)
        d.line([left, y, left + plot_w, y], fill=grid, width=1)
        d.line([left - 7, y, left, y], fill=axis, width=2)
        text = _fmt_tick(yval)
        bbox = d.textbbox((0, 0), text, font=tick_font)
        d.text((left - 12 - (bbox[2] - bbox[0]), y - (bbox[3] - bbox[1]) / 2),
               text, fill=axis, font=tick_font)
    d.rectangle([left, top, left + plot_w, top + plot_h], outline=axis, width=2)
    bbox = d.textbbox((0, 0), x_label, font=label_font)
    d.text((left + plot_w / 2 - (bbox[2] - bbox[0]) / 2, top + plot_h + 52),
           x_label, fill=axis, font=label_font)
    _draw_rotated_text(d, (18, int(top + plot_h / 2)), y_label,
                       font=label_font, fill=axis, angle=90, anchor="center")


def _draw_colorbar(d, x, y, w, h, zmin, zmax, label, *, tick_font, label_font):
    for row in range(h):
        t = 1.0 - row / max(h - 1, 1)
        d.line([x, y + row, x + w, y + row], fill=_viridis(t), width=1)
    d.rectangle([x, y, x + w, y + h], outline=(20, 20, 20), width=2)
    for v in _nice_ticks(zmin, zmax, 7):
        yy = int(y + (1.0 - (v - zmin) / (zmax - zmin + 1e-12)) * h)
        d.line([x + w, yy, x + w + 7, yy], fill=(20, 20, 20), width=2)
        d.text((x + w + 12, yy - 9), _fmt_tick(v), fill=(20, 20, 20), font=tick_font)
    _draw_rotated_text(d, (x + w + 82, int(y + h / 2)), label,
                       font=label_font, fill=(20, 20, 20), angle=90, anchor="center")


def _draw_rotated_text(d, xy, text, *, font, fill, angle=90, anchor="center"):
    from PIL import Image, ImageDraw
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0] + 8, bbox[3] - bbox[1] + 8
    layer = Image.new("RGBA", (tw, th), (255, 255, 255, 0))
    td = ImageDraw.Draw(layer)
    td.text((4, 4), text, fill=fill + (255,) if len(fill) == 3 else fill, font=font)
    rotated = layer.rotate(angle, expand=True)
    x, y = xy
    if anchor == "center":
        x -= rotated.size[0] // 2
        y -= rotated.size[1] // 2
    d._image.paste(rotated, (int(x), int(y)), rotated)


def _draw_star(d, x: int, y: int, r: int, *, fill, outline):
    pts = []
    for k in range(10):
        ang = -math.pi / 2 + k * math.pi / 5
        rr = r if k % 2 == 0 else r * 0.42
        pts.append((x + rr * math.cos(ang), y + rr * math.sin(ang)))
    d.polygon(pts, fill=fill, outline=outline)


def _legend_box(d, x, y, font):
    d.rounded_rectangle([x, y, x + 245, y + 88], radius=5,
                        fill=(255, 255, 255), outline=(70, 70, 70))
    entries = [
        ((210, 40, 40), "balanced route"),
        ((40, 145, 95), "Pareto route"),
        ((115, 115, 115), "dominated route"),
    ]
    for i, (color, label) in enumerate(entries):
        yy = y + 18 + i * 24
        d.ellipse([x + 14, yy - 6, x + 26, yy + 6], fill=color, outline=(0, 0, 0))
        d.text((x + 38, yy - 9), label, fill=(25, 25, 25), font=font)


def _patch_matplotlib_deepcopy_bug():
    """Work around a Python 3.14/Matplotlib Path deepcopy recursion in this env."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.path as mpath
    mpath.Path.__deepcopy__ = lambda self, memo: self


def _visual_porkchop(report: NavigatorReport):
    c = report.constraints
    start = et(c.epoch_start)
    target = report.target.system_barycenter or report.target.ephemeris_name
    return porkchop(
        report.origin.ephemeris_name, target, start, c.departure_window_days,
        c.tof_range_days,
        n_dep=max(72, min(140, c.n_dep * 5)),
        n_tof=max(56, min(110, c.n_tof * 5)),
        capture=False,
    )


def _plot_porkchop_matplotlib(report: NavigatorReport, path: Path):
    _patch_matplotlib_deepcopy_bug()
    import matplotlib.pyplot as plt

    c = report.constraints
    pk = _visual_porkchop(report)
    z = pk["total_ms"] / 1000.0
    finite = z[np.isfinite(z)]
    if not finite.size:
        raise ValueError("porkchop grid has no finite transfer costs")

    dep_days = (pk["dep_grid"] - pk["dep_grid"][0]) / DAY
    vmax = float(np.nanpercentile(finite, 88))
    vmin = float(np.nanmin(finite))
    if vmax <= vmin:
        vmax = float(np.nanmax(finite))
    levels = np.linspace(vmin, vmax, 34)

    from . import nasa_plate as NP
    NP.apply_style()

    fig, ax = plt.subplots(figsize=(12.5, 7.8))
    fig.subplots_adjust(left=0.07, right=0.93, top=0.88, bottom=0.10)
    cs = ax.contourf(dep_days, pk["tof_grid"], z, levels=levels,
                     cmap="cividis", extend="max")
    cbar = fig.colorbar(cs, ax=ax, pad=0.018)
    cbar.set_label("COST PROXY (km/s)  -  launch + arrival + target screen",
                    color=NP.TEXT_SECONDARY, fontsize=8.5)
    cbar.ax.tick_params(colors=NP.TEXT_SECONDARY)
    cbar.outline.set_edgecolor(NP.GRID_LINE)

    contour_levels = np.linspace(vmin, vmax, 9)[1:-1]
    if contour_levels.size:
        lines = ax.contour(dep_days, pk["tof_grid"], z, levels=contour_levels,
                            colors=NP.TEXT_SECONDARY, linewidths=0.4,
                            alpha=0.45)
        ax.clabel(lines, inline=True, fontsize=7, fmt="%.1f",
                   colors=NP.TEXT_SECONDARY)

    if pk.get("grid_best"):
        gb = pk["grid_best"]
        od = (float(gb["et_dep"]) - float(pk["dep_grid"][0])) / DAY
        ax.plot(od, gb["tof_days"], marker="*", ms=22, mfc=NP.ACCENT_GOLD,
                 mec=NP.DEEP_SPACE_BG, mew=1.6, linestyle="None",
                 label=f"grid minimum  {gb['total_ms']/1000:.2f} km/s",
                 zorder=5)
    for role, route, marker in (
        ("fastest", report.fastest, "D"),
        ("cheapest", report.cheapest, "P"),
        ("balanced", report.balanced, "*"),
    ):
        if route is None:
            continue
        dep_utc = route.events[0].epoch_utc if route.events else None
        if not dep_utc:
            continue
        od = (et(dep_utc) - float(pk["dep_grid"][0])) / DAY
        color = NP.ROLE_COLORS.get(role, NP.ACCENT_CYAN)
        ax.scatter([od], [route.tof_days], marker=marker,
                    s=180 if marker != "*" else 300, facecolor=color,
                    edgecolors=NP.DEEP_SPACE_BG, linewidths=1.2,
                    zorder=6, label=f"{role.upper()} route")

    NP.style_axes(ax, title="",
                    xlabel=f"DEPARTURE OFFSET FROM {c.epoch_start[:10]} (days)",
                    ylabel="TIME OF FLIGHT (days)")
    leg = ax.legend(loc="upper right", facecolor=NP.PANEL_BG,
                     edgecolor=NP.GRID_LINE, labelcolor=NP.TEXT_PRIMARY,
                     fontsize=8.5, framealpha=0.92)
    NP.mission_title(fig,
        title=f"Porkchop: {report.origin.name} -> {report.target.name}",
        subtitle="Contours expose launch-window structure; "
                  "markers show ranked mission routes (real ephemerides)")
    NP.mission_footer(fig,
        mission_id=f"{report.origin.name}_{report.target.name}",
        cert=report.certificate_hash,
        fidelity="patched-conic + real ephemerides")
    fig.savefig(path, dpi=180, bbox_inches="tight",
                 facecolor=NP.DEEP_SPACE_BG)
    plt.close(fig)
    NP.reset_style()


def _plot_porkchop_pillow(report: NavigatorReport, path: Path):
    c = report.constraints
    pk = _visual_porkchop(report)
    z = pk["total_ms"] / 1000.0
    from PIL import Image, ImageDraw
    W, H = 1440, 900
    left, right, top, bottom = 130, 230, 92, 118
    plot_w, plot_h = W - left - right, H - top - bottom
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    title_font, label_font, tick_font, small_font = _plot_fonts()
    finite = z[np.isfinite(z)]
    zmin = float(np.nanmin(finite)) if finite.size else 0.0
    zmax = float(np.nanmax(finite)) if finite.size else 1.0

    heat = _heatmap_image(z, plot_w, plot_h, zmin, zmax)
    img.paste(heat, (left, top))
    _draw_axes(
        d, left, top, plot_w, plot_h,
        x_min=0.0, x_max=c.departure_window_days,
        y_min=c.tof_range_days[0], y_max=c.tof_range_days[1],
        x_label="departure offset (days)",
        y_label="time of flight (days)",
        tick_font=tick_font,
        label_font=label_font,
    )
    d.text(
        (left, 28),
        f"Ariadne {report.origin.name}->{report.target.name} porkchop -- route families from real ephemerides",
        fill=(20, 20, 20),
        font=title_font,
    )
    _draw_colorbar(
        d, left + plot_w + 48, top, 46, plot_h, zmin, zmax,
        "cost proxy (km/s)", tick_font=tick_font, label_font=label_font)
    if pk.get("grid_best"):
        gb = pk["grid_best"]
        dep_frac = (gb["et_dep"] - pk["dep_grid"][0]) / max(pk["dep_grid"][-1] - pk["dep_grid"][0], 1.0)
        tof_frac = (gb["tof_days"] - pk["tof_grid"][0]) / max(pk["tof_grid"][-1] - pk["tof_grid"][0], 1.0)
        x = int(left + dep_frac * plot_w)
        y = int(top + (1.0 - tof_frac) * plot_h)
        _draw_star(d, x, y, 19, fill=(235, 35, 35), outline=(120, 0, 0))
        d.rounded_rectangle([x + 16, y - 22, x + 246, y + 12], radius=4,
                            fill=(255, 255, 255), outline=(40, 40, 40))
        d.text((x + 24, y - 17),
               f"grid min {gb['tof_days']:.0f} d, {gb['total_ms']/1000:.2f} km/s",
               fill=(20, 20, 20), font=small_font)
    img.save(path)


def _route_axes_bounds(routes: list[MissionRoute]):
    xs = [r.tof_days for r in routes]
    ys = [r.total_dv_ms / 1000.0 for r in routes]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    padx = max(1.0, 0.08 * (xmax - xmin + 1e-9))
    pady = max(0.1, 0.10 * (ymax - ymin + 1e-9))
    return xmin - padx, xmax + padx, ymin - pady, ymax + pady


def _plot_route_summary_matplotlib(report: NavigatorReport, path: Path):
    _patch_matplotlib_deepcopy_bug()
    import matplotlib.pyplot as plt

    routes = list(report.routes)
    fig, ax = plt.subplots(figsize=(10.8, 7.0))
    if routes:
        risks = np.array([r.risk for r in routes])
        xs = np.array([r.tof_days for r in routes])
        ys = np.array([r.total_dv_ms / 1000.0 for r in routes])
        sizes = np.array([120 if r.pareto_optimal else 62 for r in routes])
        sc = ax.scatter(xs, ys, c=risks, cmap="plasma_r", s=sizes,
                        edgecolors="k", linewidths=0.55, alpha=0.9, zorder=3)
        fig.colorbar(sc, ax=ax, pad=0.018, label="route risk score (lower is better)")

        pareto = sorted((r for r in routes if r.pareto_optimal), key=lambda r: r.tof_days)
        if len(pareto) >= 2:
            ax.plot([r.tof_days for r in pareto], [r.total_dv_ms / 1000.0 for r in pareto],
                    color="tab:blue", lw=1.8, alpha=0.75, label="Pareto frontier", zorder=2)

        picks = []
        seen = {}
        for role, route, marker, color in (
            ("fastest", report.fastest, "D", "tab:red"),
            ("cheapest", report.cheapest, "P", "tab:blue"),
            ("balanced", report.balanced, "*", "limegreen"),
        ):
            if route is None:
                continue
            if route.route_id in seen:
                picks[seen[route.route_id]][0].append(role)
            else:
                seen[route.route_id] = len(picks)
                picks.append(([role], route, marker, color))
        for roles, route, marker, color in picks:
            role_label = " / ".join(roles)
            ax.scatter([route.tof_days], [route.total_dv_ms / 1000.0],
                       marker=marker, s=260 if marker != "*" else 360,
                       facecolor=color, edgecolor="k", linewidth=1.0,
                       zorder=6, label=f"{role_label}: {route.name}")
            xmid = 0.5 * (ax.get_xlim()[0] + ax.get_xlim()[1])
            dx = -92 if route.tof_days > xmid else 7
            ax.annotate(role_label, (route.tof_days, route.total_dv_ms / 1000.0),
                        textcoords="offset points", xytext=(dx, 7), fontsize=8,
                        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "0.75", "alpha": 0.9})

        ax.set_xlim(*_route_axes_bounds(routes)[:2])
        ax.set_ylim(*_route_axes_bounds(routes)[2:])
    ax.set_xlabel("time of flight (days)")
    ax.set_ylabel("deterministic cost / proxy (km/s)")
    ax.set_title(
        f"Ariadne route trade space to {report.target.name}\n"
        "time, energy, risk, feasibility, and Pareto dominance in one review plot"
    )
    ax.grid(True, alpha=0.24)
    ax.legend(loc="best", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_route_summary_pillow(report: NavigatorReport, path: Path):
    from PIL import Image, ImageDraw
    routes = list(report.routes)
    W, H = 1320, 850
    left, right, top, bottom = 125, 90, 92, 120
    plot_w, plot_h = W - left - right, H - top - bottom
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    title_font, label_font, tick_font, small_font = _plot_fonts()
    d.text((left, 28), f"Ariadne route trade space to {report.target.name}",
           fill=(20, 20, 20), font=title_font)
    if routes:
        xs = [r.tof_days for r in routes]
        ys = [r.total_dv_ms / 1000.0 for r in routes]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        padx = max(1.0, 0.08 * (xmax - xmin + 1e-9))
        pady = max(0.1, 0.10 * (ymax - ymin + 1e-9))
        xmin, xmax = xmin - padx, xmax + padx
        ymin, ymax = ymin - pady, ymax + pady
        _draw_axes(
            d, left, top, plot_w, plot_h,
            x_min=xmin, x_max=xmax,
            y_min=ymin, y_max=ymax,
            x_label="time of flight (days)",
            y_label="deterministic cost / proxy (km/s)",
            tick_font=tick_font,
            label_font=label_font,
        )
        pareto_pts = sorted(
            [r for r in routes if r.pareto_optimal],
            key=lambda r: r.tof_days,
        )
        if len(pareto_pts) >= 2:
            line = []
            for r in pareto_pts:
                x = int(left + (r.tof_days - xmin) / (xmax - xmin + 1e-12) * plot_w)
                y = int(top + (1.0 - ((r.total_dv_ms / 1000.0 - ymin) / (ymax - ymin + 1e-12))) * plot_h)
                line.append((x, y))
            d.line(line, fill=(45, 90, 150), width=3)
        for r in routes:
            x = int(left + (r.tof_days - xmin) / (xmax - xmin + 1e-12) * plot_w)
            y = int(top + (1.0 - ((r.total_dv_ms / 1000.0 - ymin) / (ymax - ymin + 1e-12))) * plot_h)
            color = (210, 40, 40) if r.route_id == report.balanced_id else (
                (40, 145, 95) if r.pareto_optimal else (115, 115, 115))
            radius = 10 if r.route_id == report.balanced_id else 7
            d.ellipse([x - radius, y - radius, x + radius, y + radius],
                      fill=color, outline=(0, 0, 0), width=2)
        label_routes = []
        for rid in (report.fastest_id, report.cheapest_id, report.balanced_id):
            rr = _find_route(routes, rid)
            if rr is not None and rr not in label_routes:
                label_routes.append(rr)
        for rr in pareto_pts:
            if "knee" in rr.name and rr not in label_routes:
                label_routes.append(rr)
        for r in label_routes[:5]:
            x = int(left + (r.tof_days - xmin) / (xmax - xmin + 1e-12) * plot_w)
            y = int(top + (1.0 - ((r.total_dv_ms / 1000.0 - ymin) / (ymax - ymin + 1e-12))) * plot_h)
            if r.route_id == report.fastest_id:
                label = "fastest"
            elif r.route_id == report.cheapest_id and r.route_id == report.balanced_id:
                label = "cheapest / balanced"
            elif r.route_id == report.cheapest_id:
                label = "cheapest"
            elif r.route_id == report.balanced_id:
                label = "balanced"
            elif "knee" in r.name:
                label = "Pareto knee"
            else:
                label = r.name[:22]
            bbox = d.textbbox((0, 0), label, font=small_font)
            text_w = bbox[2] - bbox[0]
            tx = x + 12
            if tx + text_w > left + plot_w - 8:
                tx = x - text_w - 14
            ty = max(top + 4, min(top + plot_h - 22, y - 9))
            d.rounded_rectangle([tx - 4, ty - 3, tx + text_w + 4, ty + 18],
                                radius=3, fill=(255, 255, 255), outline=(215, 215, 215))
            d.text((tx, ty), label, fill=(20, 20, 20), font=small_font)
        _legend_box(d, left + plot_w - 260, top + 20, small_font)
    else:
        _draw_axes(
            d, left, top, plot_w, plot_h,
            x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0,
            x_label="time of flight (days)", y_label="cost (km/s)",
            tick_font=tick_font, label_font=label_font)
    img.save(path)


def _plot_mission_plate(report: NavigatorReport, path: Path):
    _patch_matplotlib_deepcopy_bug()
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from . import nasa_plate as NP

    NP.apply_style()

    pk = _visual_porkchop(report)
    z = pk["total_ms"] / 1000.0
    finite = z[np.isfinite(z)]
    if not finite.size:
        raise ValueError("porkchop grid has no finite transfer costs")
    dep_days = (pk["dep_grid"] - pk["dep_grid"][0]) / DAY
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanpercentile(finite, 88))
    if vmax <= vmin:
        vmax = float(np.nanmax(finite))
    levels = np.linspace(vmin, vmax, 32)
    routes = list(report.routes)

    fig = plt.figure(figsize=(16.0, 9.5), constrained_layout=False)
    gs = GridSpec(2, 3, figure=fig, width_ratios=(1.35, 1.0, 0.86),
                  height_ratios=(1.0, 0.42),
                  left=0.05, right=0.99, top=0.91, bottom=0.06,
                  hspace=0.30, wspace=0.22)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[:, 2])
    ax3 = fig.add_subplot(gs[1, 0:2])

    cf = ax0.contourf(dep_days, pk["tof_grid"], z, levels=levels,
                       cmap="cividis", extend="max")
    ax0.contour(dep_days, pk["tof_grid"], z,
                 levels=np.linspace(vmin, vmax, 8)[1:-1],
                 colors=NP.GRID_LINE, linewidths=0.35, alpha=0.55)
    if pk.get("grid_best"):
        gb = pk["grid_best"]
        od = (float(gb["et_dep"]) - float(pk["dep_grid"][0])) / DAY
        ax0.plot(od, gb["tof_days"], marker="*", ms=20,
                 mfc=NP.ACCENT_GOLD, mec=NP.DEEP_SPACE_BG, mew=1.4,
                 linestyle="None", label="grid minimum", zorder=5)
        NP.halo_text(ax0, od, gb["tof_days"] + 18,
                      f"{float(z.min()):.2f} km/s",
                      color=NP.ACCENT_GOLD, fontsize=8.0,
                      weight="bold", ha="center")
    NP.style_axes(ax0, title="Launch-window energy topology",
                   xlabel="DEPARTURE OFFSET (days)",
                   ylabel="TIME OF FLIGHT (days)")
    cb = fig.colorbar(cf, ax=ax0, pad=0.01, label="cost proxy (km/s)")
    cb.ax.tick_params(colors=NP.TEXT_SECONDARY)
    cb.set_label("COST PROXY (km/s)", color=NP.TEXT_SECONDARY,
                  fontsize=8.5)
    cb.outline.set_edgecolor(NP.GRID_LINE)

    if routes:
        xs = [r.tof_days for r in routes]
        ys = [r.total_dv_ms / 1000.0 for r in routes]
        risks = [r.risk for r in routes]
        ax1.scatter(xs, ys, c=risks, cmap="plasma_r",
                     s=[125 if r.pareto_optimal else 58 for r in routes],
                     edgecolors=NP.DEEP_SPACE_BG, linewidths=0.55,
                     alpha=0.92, zorder=3)
        pareto = sorted((r for r in routes if r.pareto_optimal),
                          key=lambda r: r.tof_days)
        if len(pareto) >= 2:
            ax1.plot([r.tof_days for r in pareto],
                      [r.total_dv_ms / 1000.0 for r in pareto],
                      color=NP.ACCENT_CYAN, lw=1.4, alpha=0.5,
                      linestyle="-", zorder=2, label="Pareto front")
        picks = []
        seen = {}
        for role, route, marker in (
            ("fastest", report.fastest, "D"),
            ("cheapest", report.cheapest, "P"),
            ("balanced", report.balanced, "*"),
        ):
            if route is not None:
                if route.route_id in seen:
                    picks[seen[route.route_id]][0].append(role)
                else:
                    seen[route.route_id] = len(picks)
                    picks.append(([role], route, marker))
        for roles, route, marker in picks:
                role_label = " / ".join(roles)
                color = NP.ROLE_COLORS.get(roles[0], NP.ACCENT_CYAN)
                ax1.scatter([route.tof_days], [route.total_dv_ms / 1000.0],
                              marker=marker, s=300, facecolor=color,
                              edgecolor=NP.DEEP_SPACE_BG, linewidths=1.2,
                              zorder=5)
                xmid = 0.5 * (ax1.get_xlim()[0] + ax1.get_xlim()[1])
                dx = -78 if route.tof_days > xmid else 8
                NP.halo_text(ax1, route.tof_days, route.total_dv_ms / 1000.0,
                              role_label, fontsize=8.0, weight="bold",
                              color=color,
                              ha="left" if dx > 0 else "right",
                              va="bottom")
        xmin, xmax, ymin, ymax = _route_axes_bounds(routes)
        ax1.set_xlim(xmin, xmax); ax1.set_ylim(ymin, ymax)
        if len(pareto) >= 2:
            ax1.legend(loc="upper right", facecolor=NP.PANEL_BG,
                        edgecolor=NP.GRID_LINE, labelcolor=NP.TEXT_PRIMARY,
                        fontsize=8)
    NP.style_axes(ax1, title="Mission route trade space",
                   xlabel="TIME OF FLIGHT (days)",
                   ylabel="COST PROXY (km/s)")

    ax2.axis("off")
    ax2.set_title("SELECTED ROUTE CERTIFICATES", loc="left", pad=10,
                   color=NP.TEXT_PRIMARY, weight="bold", fontsize=10)
    def short(text: str, n: int = 33) -> str:
        return text if len(text) <= n else text[:n - 3] + "..."

    role_y = 0.95
    role_color = {"fastest": NP.ROLE_COLORS["fastest"],
                  "cheapest": NP.ROLE_COLORS["cheapest"],
                  "balanced": NP.ROLE_COLORS["balanced"]}
    for role, route in (("fastest", report.fastest),
                          ("cheapest", report.cheapest),
                          ("balanced", report.balanced)):
        if route is None:
            continue
        seq = " -> ".join(route.sequence)
        body = (
            f"{short(route.name, 36)}\n"
            f"{short(seq, 42)}\n"
            f"cost {route.total_dv_ms/1000:.2f} km/s  tof {route.tof_days:.0f} d\n"
            f"risk {route.risk:.2f}  C3 {(route.c3_km2_s2 or 0.0):.2f}\n"
            f"cert {route.certificate_hash[:18]}"
        )
        NP.card_box(ax2, 0.02, role_y, body, width=0.93,
                      header=role, fontsize=8.0,
                      color=NP.TEXT_PRIMARY,
                      bg=NP.PANEL_BG,
                      border=role_color.get(role, NP.GRID_LINE))
        role_y -= 0.31

    ax3.axis("off")
    route = report.balanced or report.cheapest or report.fastest
    NP.card_box(ax3, 0.005, 0.92,
                  text=(
                      f"target:           {report.target.name}\n"
                      f"routes searched:  {len(routes)}\n"
                      f"pareto routes:    {len(report.pareto_front)}\n"
                      f"selected route:   {route.name if route else '(none)'}\n"
                      f"engine:           {route.engine if route else '(n/a)'}\n"
                      f"fidelity:         {route.fidelity if route else '(n/a)'}\n"
                      f"validation:       "
                      + ("; ".join(route.validations[:2]) if route else '(n/a)')
                  ),
                  header="MISSION SUMMARY", width=0.99, fontsize=9.0,
                  color=NP.TEXT_PRIMARY,
                  border=NP.ACCENT_CYAN)

    NP.mission_title(fig,
        title=f"Solar-system navigator: {report.origin.name} -> {report.target.name}",
        subtitle=f"Multi-engine porkchop + Pareto trade space, "
                  f"{len(routes)} routes evaluated")
    NP.mission_footer(fig,
        mission_id=f"{report.origin.name}_{report.target.name}",
        cert=report.certificate_hash,
        fidelity=route.fidelity if route else "n/a",
        extras=[("routes", str(len(routes)))])

    fig.savefig(path, dpi=180, bbox_inches="tight",
                 facecolor=NP.DEEP_SPACE_BG)
    plt.close(fig)
    NP.reset_style()


def _plot_porkchop(report: NavigatorReport, path: Path):
    try:
        _plot_porkchop_matplotlib(report, path)
    except Exception:
        _plot_porkchop_pillow(report, path)


def _plot_route_summary(report: NavigatorReport, path: Path):
    try:
        _plot_route_summary_matplotlib(report, path)
    except Exception:
        _plot_route_summary_pillow(report, path)


def write_navigator_report(report: NavigatorReport, outdir: str | Path,
                           *, make_plots: bool = True) -> dict:
    """Write JSON route card and optional PNG visual artifacts."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    if make_plots:
        try:
            p = out / "porkchop_heatmap.png"
            _plot_porkchop(report, p)
            artifacts["porkchop_heatmap"] = str(p)
        except Exception as exc:
            artifacts["porkchop_heatmap_error"] = f"{type(exc).__name__}: {exc}"
        try:
            p = out / "route_trade_space.png"
            _plot_route_summary(report, p)
            artifacts["route_trade_space"] = str(p)
        except Exception as exc:
            artifacts["route_trade_space_error"] = f"{type(exc).__name__}: {exc}"
        try:
            p = out / "mission_plate.png"
            _plot_mission_plate(report, p)
            artifacts["mission_plate"] = str(p)
        except Exception as exc:
            artifacts["mission_plate_error"] = f"{type(exc).__name__}: {exc}"
    payload = {
        **_jsonable(report),
        "artifacts": artifacts,
    }
    report_path = out / "navigator_report.json"
    report_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    artifacts["report"] = str(report_path)
    cards_path = out / "route_cards.md"
    cards_path.write_text(_route_cards_markdown(report, artifacts), encoding="utf-8")
    artifacts["route_cards"] = str(cards_path)
    manifest_path = out / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(_figure_manifest(report, artifacts), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    artifacts["figure_manifest"] = str(manifest_path)
    return artifacts


def _figure_manifest(report: NavigatorReport, artifacts: dict) -> dict:
    """Machine-checkable semantic contract for navigator visuals."""
    route_roles = {
        "fastest": report.fastest_id,
        "cheapest": report.cheapest_id,
        "balanced": report.balanced_id,
    }
    return {
        "schema": "ariadne.navigator_figure_manifest.v1",
        "report_certificate_hash": report.certificate_hash,
        "origin": report.origin.name,
        "target": report.target.name,
        "route_count": len(report.routes),
        "pareto_count": len(report.pareto_front),
        "route_roles": route_roles,
        "required_figures": {
            "mission_plate": {
                "path": artifacts.get("mission_plate"),
                "must_show": ["origin", "target", "balanced route", "certificate", "route count"],
            },
            "porkchop_heatmap": {
                "path": artifacts.get("porkchop_heatmap"),
                "must_show": ["departure offset days", "time of flight days", "delta-v scale"],
            },
            "route_trade_space": {
                "path": artifacts.get("route_trade_space"),
                "must_show": ["time of flight", "delta-v", "pareto front", "balanced route"],
            },
        },
        "units": {
            "delta_v": "m/s",
            "time_of_flight": "days",
            "c3": "km^2/s^2",
            "arrival_vinf": "km/s",
            "coordinates": "km J2000 heliocentric unless otherwise noted",
        },
        "routes": [
            {
                "route_id": route.route_id,
                "name": route.name,
                "sequence": route.sequence,
                "fidelity": route.fidelity,
                "certificate_hash": route.certificate_hash,
                "event_count": len(route.events),
                "assumption_count": len(route.assumptions),
                "validation_count": len(route.validations),
            }
            for route in report.routes
        ],
    }


def _route_cards_markdown(report: NavigatorReport, artifacts: dict | None = None) -> str:
    """Human-review route cards for mission-design triage."""
    artifacts = artifacts or {}
    lines = [
        f"# Solar Navigator Route Cards: {report.origin.name} to {report.target.name}",
        "",
        f"- report certificate: `{report.certificate_hash}`",
        f"- routes: `{len(report.routes)}`",
        f"- fastest: `{report.fastest_id}`",
        f"- cheapest: `{report.cheapest_id}`",
        f"- balanced: `{report.balanced_id}`",
        "",
    ]
    if artifacts:
        lines.append("## Artifacts")
        lines.append("")
        for key, value in sorted(artifacts.items()):
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    lines += [
        "## Summary",
        "",
        "| Role | Route | Sequence | Cost m/s | TOF d | C3 | Arrival v-inf | Fidelity |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    roles = [("fastest", report.fastest), ("cheapest", report.cheapest), ("balanced", report.balanced)]
    for role, route in roles:
        if route is None:
            continue
        lines.append(
            f"| {role} | `{route.name}` | {' -> '.join(route.sequence)} | "
            f"{route.total_dv_ms:.2f} | {route.tof_days:.2f} | "
            f"{'' if route.c3_km2_s2 is None else f'{route.c3_km2_s2:.3f}'} | "
            f"{'' if route.arrival_vinf_kms is None else f'{route.arrival_vinf_kms:.3f}'} | "
            f"`{route.fidelity}` |"
        )
    lines.append("")
    lines.append("## Pareto Routes")
    lines.append("")
    lines.append("| Route | Sequence | Cost m/s | TOF d | Risk | Certificate |")
    lines.append("|---|---|---:|---:|---:|---|")
    for route in report.routes:
        if not route.pareto_optimal:
            continue
        lines.append(
            f"| `{route.name}` | {' -> '.join(route.sequence)} | "
            f"{route.total_dv_ms:.2f} | {route.tof_days:.2f} | {route.risk:.3f} | "
            f"`{route.certificate_hash[:16]}` |"
        )
    lines.append("")
    lines.append("## Route Details")
    for route in report.routes:
        lines += [
            "",
            f"### {route.name}",
            "",
            f"- id: `{route.route_id}`",
            f"- engine: `{route.engine}`",
            f"- sequence: `{' -> '.join(route.sequence)}`",
            f"- total cost: `{route.total_dv_ms:.3f} m/s`",
            f"- time of flight: `{route.tof_days:.3f} d`",
            f"- C3: `{route.c3_km2_s2}`",
            f"- arrival v-inf: `{route.arrival_vinf_kms}`",
            f"- fidelity: `{route.fidelity}`",
            f"- feasible: `{route.feasible}`",
            f"- Pareto: `{route.pareto_optimal}`",
            f"- certificate: `{route.certificate_hash}`",
            "",
            "Assumptions:",
        ]
        lines.extend(f"- {x}" for x in route.assumptions)
        lines.append("")
        lines.append("Validations:")
        lines.extend(f"- {x}" for x in route.validations)
    return "\n".join(lines) + "\n"
