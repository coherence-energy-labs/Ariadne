"""Cislunar mission architect.

This module coordinates Ariadne's separate Earth-Moon transfer engines into a
single round-trip architecture catalogue.  It does not pretend that patched,
CR3BP, and full-ephemeris results have the same fidelity; every candidate keeps
explicit assumptions, validation gates, and a deterministic certificate.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable

import numpy as np

from ..data.constants import GM_EARTH, GM_MOON, R_EARTH, R_MOON
from ..data.ephemeris import et
from ..optimize.budget import circular_speed, earth_moon_budget
from .coherence_optimizer import coherence_frontier, knee, pareto_front, weighted_optimum
from .ephemeris_transfer import optimize_transfer
from .low_energy_lunar import low_energy_lunar_transfer


FIDELITY_RANK = {
    "analytic_patch": 1,
    "cr3bp_patch": 2,
    "full_ephemeris": 3,
    "hybrid_verified": 4,
}


@dataclass(frozen=True)
class MissionConstraints:
    """Operational constraints for Earth-Moon-Moon-Earth architecture search."""
    epoch: str = "2025-06-01T00:00:00"
    leo_alt_km: float = 200.0
    llo_alt_km: float = 100.0
    lunar_stay_days: float = 7.0
    outbound_tof_days: tuple[float, ...] = (3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0)
    return_tof_days: tuple[float, ...] = (3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0)
    low_energy_jacobi: tuple[float, ...] = (3.05, 3.10, 3.15)
    include_direct: bool = True
    include_low_energy: bool = True
    include_coherence: bool = True
    include_free_return: bool = True
    max_total_dv_ms: float | None = None
    max_total_tof_days: float | None = None
    min_reentry_perigee_km: float = 6378.0 + 30.0
    max_reentry_perigee_km: float = 6378.0 + 120.0


@dataclass(frozen=True)
class ArchitectureWeights:
    """Scalar objective weights.  Lower score is better."""
    dv: float = 1.0
    tof: float = 0.12
    robustness: float = 0.35
    risk: float = 0.80
    fidelity_bonus: float = 0.20


@dataclass(frozen=True)
class MissionLeg:
    """One normalized mission leg candidate."""
    name: str
    direction: str
    engine: str
    fidelity: str
    dv_ms: float
    tof_days: float
    robustness: float = 0.5
    risk: float = 0.5
    assumptions: tuple[str, ...] = ()
    validations: tuple[str, ...] = ()
    components: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MissionArchitecture:
    """One complete Earth-Moon-Moon-Earth architecture."""
    architecture_id: str
    outbound: MissionLeg
    return_leg: MissionLeg
    lunar_stay_days: float
    total_dv_ms: float
    total_tof_days: float
    score: float
    pareto_optimal: bool
    free_return_capable: bool
    fidelity: str
    assumptions: tuple[str, ...]
    validations: tuple[str, ...]
    certificate_hash: str


@dataclass(frozen=True)
class MissionArchitectureReport:
    """Full proof bundle for a mission-architecture search."""
    schema: str
    created_utc: str
    constraints: MissionConstraints
    weights: ArchitectureWeights
    candidates: tuple[MissionArchitecture, ...]
    pareto_front: tuple[str, ...]
    recommended_id: str | None
    certificate_hash: str

    @property
    def recommended(self) -> MissionArchitecture | None:
        for candidate in self.candidates:
            if candidate.architecture_id == self.recommended_id:
                return candidate
        return None


def _jsonable(value):
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    return value


def stable_hash(value) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _return_capture_budget(llo_alt_km: float, *, ballistic: bool = False) -> dict:
    """Analytic LLO-to-Earth-return budget and corridor proxy.

    The departure burn is the reverse of lunar insertion: circular LLO to either
    a direct-return hyperbola or a near-parabolic ballistic return.  Earth
    arrival is atmospheric capture, not a propulsive Earth-orbit insertion.
    """
    b = earth_moon_budget(llo_alt=llo_alt_km)
    r_llo = R_MOON + llo_alt_km
    v_circ_llo = circular_speed(GM_MOON, r_llo)
    v_inf = 0.0 if ballistic else b["v_inf_direct"]
    v_peri = math.sqrt(v_inf * v_inf + 2.0 * GM_MOON / r_llo)
    dv_depart = max(0.0, v_peri - v_circ_llo)
    v_entry = math.sqrt(v_inf * v_inf + 2.0 * GM_EARTH / (R_EARTH + 80.0))
    return {
        "dv_depart_ms": dv_depart * 1000.0,
        "earth_entry_speed_kms": v_entry,
        "v_inf_kms": v_inf,
        "entry_perigee_km": R_EARTH + 80.0,
    }


def _leg_id(prefix: str, leg: MissionLeg) -> str:
    return f"{prefix}_{stable_hash(leg)[:12]}"


def _direct_outbound_legs(constraints: MissionConstraints,
                          direct_optimizer: Callable = optimize_transfer) -> list[MissionLeg]:
    e0 = et(constraints.epoch)
    best, records = direct_optimizer(
        e0,
        tof_grid=np.array(constraints.outbound_tof_days, dtype=float),
        leo_alt=constraints.leo_alt_km,
        llo_alt=constraints.llo_alt_km,
    )
    legs = []
    for rec in records:
        miss = float(rec.get("miss_km", 0.0))
        risk = _clip01(0.10 + miss / 50.0)
        legs.append(MissionLeg(
            name=f"direct_ephemeris_{rec['tof_days']:.2f}d",
            direction="earth_to_moon",
            engine="ephemeris_transfer.optimize_transfer",
            fidelity="full_ephemeris",
            dv_ms=float(rec["total_ms"]),
            tof_days=float(rec["tof_days"]),
            robustness=_clip01(1.0 - risk),
            risk=risk,
            assumptions=(
                "Earth-centered full-ephemeris targeting with Sun perturbation",
                "patched lunar orbit insertion at selected LLO altitude",
            ),
            validations=(
                "Lambert seed corrected by finite-difference shooting",
                f"targeting miss {miss:.3g} km",
            ),
            components={
                "tli_ms": float(rec.get("dv_tli_ms", 0.0)),
                "loi_ms": float(rec.get("dv_loi_ms", 0.0)),
                "v_inf_kms": float(rec.get("v_inf_kms", 0.0)),
                "miss_km": miss,
            },
            raw={k: v for k, v in rec.items() if k not in {"r1", "v1"}},
        ))
    return legs


def _low_energy_outbound_legs(constraints: MissionConstraints,
                              low_energy_solver: Callable = low_energy_lunar_transfer) -> list[MissionLeg]:
    best, records, baseline = low_energy_solver(
        jacobi_values=constraints.low_energy_jacobi,
        leo_alt=constraints.leo_alt_km,
        llo_alt=constraints.llo_alt_km,
    )
    legs = []
    for rec in records:
        match_hint = abs(float(rec.get("periapsis_alt_km", constraints.llo_alt_km)) - constraints.llo_alt_km)
        risk = _clip01(0.22 + match_hint / 10000.0)
        legs.append(MissionLeg(
            name=f"low_energy_{rec['point']}_C{rec['jacobi']:.3f}",
            direction="earth_to_moon",
            engine="low_energy_lunar_transfer",
            fidelity="cr3bp_patch",
            dv_ms=float(rec["total_ms"]),
            tof_days=float(rec["tof_days"]),
            robustness=_clip01(0.78 - risk * 0.35),
            risk=risk,
            assumptions=(
                "standard TLI paired with CR3BP manifold ballistic lunar capture",
                "Sun/BCR4BP phasing not fully optimized in this leg",
            ),
            validations=(
                "capture state generated from invariant-manifold propagation",
                "patched-conic direct budget retained as comparison baseline",
            ),
            components={
                "tli_ms": float(rec.get("tli_ms", 0.0)),
                "loi_ms": float(rec.get("loi_ms", 0.0)),
                "direct_baseline_ms": float(baseline.get("direct_total_ms", 0.0)),
                "coimbra_reference_ms": float(baseline.get("coimbra_ms", 0.0)),
                "periapsis_alt_km": float(rec.get("periapsis_alt_km", 0.0)),
            },
            raw=rec,
        ))
    return legs


def _coherence_outbound_legs(constraints: MissionConstraints,
                             frontier_solver: Callable = coherence_frontier) -> list[MissionLeg]:
    points = frontier_solver(
        epoch=constraints.epoch,
        tof_grid=constraints.outbound_tof_days,
        leo_alt=constraints.leo_alt_km,
        llo_alt=constraints.llo_alt_km,
        include_wsb=True,
    )
    if not points:
        return []
    front = pareto_front(points)
    chosen = {id(p) for p in front}
    k = knee(front)
    if k is not None:
        chosen.add(id(k))
    w = weighted_optimum(points, 1.0)
    if w is not None:
        chosen.add(id(w))
    sens = [float(p.get("sensitivity", 1.0)) for p in points]
    s_min, s_max = min(sens), max(sens)

    legs = []
    for p in points:
        if id(p) not in chosen:
            continue
        s = float(p.get("sensitivity", 1.0))
        sn = (s - s_min) / (s_max - s_min + 1e-30)
        legs.append(MissionLeg(
            name=f"coherence_{p['label'].replace(' ', '_')}",
            direction="earth_to_moon",
            engine="coherence_optimizer.coherence_frontier",
            fidelity="hybrid_verified" if str(p.get("label", "")).startswith("direct") else "cr3bp_patch",
            dv_ms=float(p["dv_ms"]),
            tof_days=float(p["tof_days"]),
            robustness=_clip01(1.0 - sn),
            risk=_clip01(0.18 + 0.55 * sn),
            assumptions=(
                "coherence score is endpoint-sensitivity over the propagated candidate",
                "selected from the delta-v/sensitivity Pareto frontier",
            ),
            validations=(
                "non-dominated by both delta-v and sensitivity",
                f"endpoint_sensitivity {s:.6g}",
            ),
            components={"sensitivity": s},
            raw=p,
        ))
    return legs


def _return_legs(constraints: MissionConstraints) -> list[MissionLeg]:
    legs = []
    for tof in constraints.return_tof_days:
        direct = _return_capture_budget(constraints.llo_alt_km, ballistic=False)
        corridor_ok = (
            constraints.min_reentry_perigee_km
            <= direct["entry_perigee_km"]
            <= constraints.max_reentry_perigee_km
        )
        legs.append(MissionLeg(
            name=f"direct_return_{tof:.2f}d",
            direction="moon_to_earth",
            engine="analytic_lunar_return_budget",
            fidelity="analytic_patch",
            dv_ms=direct["dv_depart_ms"],
            tof_days=float(tof),
            robustness=0.68 if corridor_ok else 0.35,
            risk=0.22 if corridor_ok else 0.70,
            assumptions=(
                "LLO departure uses patched reverse lunar-insertion energetics",
                "Earth arrival assumes atmospheric capture/reentry, not Earth orbit insertion",
            ),
            validations=(
                f"entry perigee proxy {direct['entry_perigee_km']:.1f} km",
                f"entry speed proxy {direct['earth_entry_speed_kms']:.3f} km/s",
            ),
            components=direct,
            raw={"tof_days": float(tof), "corridor_ok": corridor_ok},
        ))
        if constraints.include_free_return:
            ballistic = _return_capture_budget(constraints.llo_alt_km, ballistic=True)
            legs.append(MissionLeg(
                name=f"ballistic_safe_return_{tof + 1.5:.2f}d",
                direction="moon_to_earth",
                engine="analytic_ballistic_return_budget",
                fidelity="analytic_patch",
                dv_ms=ballistic["dv_depart_ms"],
                tof_days=float(tof + 1.5),
                robustness=0.82,
                risk=0.16,
                assumptions=(
                    "near-parabolic lunar escape branch used as safe-return proxy",
                    "requires later high-fidelity targeting to certified reentry corridor",
                ),
                validations=(
                    f"entry perigee proxy {ballistic['entry_perigee_km']:.1f} km",
                    "free-return capable by architecture intent",
                ),
                components=ballistic,
                raw={"tof_days": float(tof + 1.5), "corridor_ok": True},
            ))
    return legs


def _score(outbound: MissionLeg, ret: MissionLeg, constraints: MissionConstraints,
           weights: ArchitectureWeights) -> float:
    total_dv = outbound.dv_ms + ret.dv_ms
    total_tof = outbound.tof_days + ret.tof_days + constraints.lunar_stay_days
    robustness = 0.5 * (outbound.robustness + ret.robustness)
    risk = 0.5 * (outbound.risk + ret.risk)
    fidelity = min(FIDELITY_RANK.get(outbound.fidelity, 0), FIDELITY_RANK.get(ret.fidelity, 0))
    return (
        weights.dv * (total_dv / 1000.0)
        + weights.tof * total_tof
        + weights.robustness * (1.0 - robustness)
        + weights.risk * risk
        - weights.fidelity_bonus * fidelity
    )


def _architecture(outbound: MissionLeg, ret: MissionLeg, constraints: MissionConstraints,
                  weights: ArchitectureWeights) -> MissionArchitecture:
    total_dv = outbound.dv_ms + ret.dv_ms
    total_tof = outbound.tof_days + ret.tof_days + constraints.lunar_stay_days
    fidelity_rank = min(FIDELITY_RANK.get(outbound.fidelity, 0), FIDELITY_RANK.get(ret.fidelity, 0))
    fidelity = next((k for k, v in FIDELITY_RANK.items() if v == fidelity_rank), "mixed")
    free_return = "free-return" in " ".join(ret.validations).lower() or "safe_return" in ret.name
    assumptions = tuple(dict.fromkeys((*outbound.assumptions, *ret.assumptions)))
    validations = tuple(dict.fromkeys((*outbound.validations, *ret.validations)))
    shell = {
        "outbound": outbound,
        "return": ret,
        "lunar_stay_days": constraints.lunar_stay_days,
        "total_dv_ms": total_dv,
        "total_tof_days": total_tof,
        "score": _score(outbound, ret, constraints, weights),
    }
    cert = stable_hash(shell)
    return MissionArchitecture(
        architecture_id=f"cislunar_{cert[:14]}",
        outbound=outbound,
        return_leg=ret,
        lunar_stay_days=constraints.lunar_stay_days,
        total_dv_ms=total_dv,
        total_tof_days=total_tof,
        score=shell["score"],
        pareto_optimal=False,
        free_return_capable=free_return,
        fidelity=fidelity,
        assumptions=assumptions,
        validations=validations,
        certificate_hash=cert,
    )


def _pareto_ids(candidates: Iterable[MissionArchitecture]) -> set[str]:
    candidates = list(candidates)
    out = set()
    for a in candidates:
        dominated = False
        for b in candidates:
            if a is b:
                continue
            if (
                b.total_dv_ms <= a.total_dv_ms
                and b.total_tof_days <= a.total_tof_days
                and b.score <= a.score
                and (b.total_dv_ms < a.total_dv_ms
                     or b.total_tof_days < a.total_tof_days
                     or b.score < a.score)
            ):
                dominated = True
                break
        if not dominated:
            out.add(a.architecture_id)
    return out


def architect_cislunar_round_trip(
    constraints: MissionConstraints | None = None,
    weights: ArchitectureWeights | None = None,
    *,
    direct_optimizer: Callable = optimize_transfer,
    low_energy_solver: Callable = low_energy_lunar_transfer,
    frontier_solver: Callable = coherence_frontier,
) -> MissionArchitectureReport:
    """Build and rank Earth-Moon-Moon-Earth mission architectures.

    Optional solver injection keeps the function testable while defaulting to
    Ariadne's real engines in production.
    """
    constraints = constraints or MissionConstraints()
    weights = weights or ArchitectureWeights()
    outbound: list[MissionLeg] = []
    if constraints.include_direct:
        outbound.extend(_direct_outbound_legs(constraints, direct_optimizer))
    if constraints.include_low_energy:
        outbound.extend(_low_energy_outbound_legs(constraints, low_energy_solver))
    if constraints.include_coherence:
        outbound.extend(_coherence_outbound_legs(constraints, frontier_solver))
    returns = _return_legs(constraints)

    candidates = []
    for out_leg in outbound:
        for ret_leg in returns:
            arch = _architecture(out_leg, ret_leg, constraints, weights)
            if constraints.max_total_dv_ms is not None and arch.total_dv_ms > constraints.max_total_dv_ms:
                continue
            if constraints.max_total_tof_days is not None and arch.total_tof_days > constraints.max_total_tof_days:
                continue
            candidates.append(arch)

    pareto = _pareto_ids(candidates)
    candidates = [
        MissionArchitecture(
            architecture_id=c.architecture_id,
            outbound=c.outbound,
            return_leg=c.return_leg,
            lunar_stay_days=c.lunar_stay_days,
            total_dv_ms=c.total_dv_ms,
            total_tof_days=c.total_tof_days,
            score=c.score,
            pareto_optimal=c.architecture_id in pareto,
            free_return_capable=c.free_return_capable,
            fidelity=c.fidelity,
            assumptions=c.assumptions,
            validations=c.validations,
            certificate_hash=c.certificate_hash,
        )
        for c in candidates
    ]
    candidates.sort(key=lambda c: (c.score, c.total_dv_ms, c.total_tof_days))
    recommended = candidates[0].architecture_id if candidates else None
    report_shell = {
        "schema": "ariadne.cislunar_mission_architect.v1",
        "constraints": constraints,
        "weights": weights,
        "candidates": candidates,
        "pareto_front": sorted(pareto),
        "recommended_id": recommended,
    }
    return MissionArchitectureReport(
        schema=report_shell["schema"],
        created_utc=datetime.now(timezone.utc).isoformat(),
        constraints=constraints,
        weights=weights,
        candidates=tuple(candidates),
        pareto_front=tuple(sorted(pareto)),
        recommended_id=recommended,
        certificate_hash=stable_hash(report_shell),
    )


def write_architecture_report(report: MissionArchitectureReport, path) -> dict:
    """Write a deterministic JSON report for review/control boards."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _jsonable(report)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")
    return payload
