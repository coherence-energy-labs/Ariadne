#!/usr/bin/env python
"""Build an Earth-Moon-Moon-Earth mission architecture report."""
from __future__ import annotations

import argparse
import json

from ariadne.transfers.mission_architect import (
    ArchitectureWeights,
    MissionConstraints,
    architect_cislunar_round_trip,
    write_architecture_report,
)


def _grid(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in text.split(",") if x.strip())


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epoch", default="2025-06-01T00:00:00")
    p.add_argument("--out", default="data/benchmarks/cislunar_architecture/report.json")
    p.add_argument("--leo-alt-km", type=float, default=200.0)
    p.add_argument("--llo-alt-km", type=float, default=100.0)
    p.add_argument("--lunar-stay-days", type=float, default=7.0)
    p.add_argument("--outbound-tof-days", default="3.0,3.5,4.0,4.5,5.0,5.5,6.0")
    p.add_argument("--return-tof-days", default="3.0,3.5,4.0,4.5,5.0,5.5,6.0")
    p.add_argument("--low-energy-jacobi", default="3.05,3.10,3.15")
    p.add_argument("--no-direct", action="store_true")
    p.add_argument("--no-low-energy", action="store_true")
    p.add_argument("--no-coherence", action="store_true")
    p.add_argument("--no-free-return", action="store_true")
    p.add_argument("--max-total-dv-ms", type=float)
    p.add_argument("--max-total-tof-days", type=float)
    p.add_argument("--w-dv", type=float, default=1.0)
    p.add_argument("--w-tof", type=float, default=0.12)
    p.add_argument("--w-robustness", type=float, default=0.35)
    p.add_argument("--w-risk", type=float, default=0.80)
    p.add_argument("--w-fidelity-bonus", type=float, default=0.20)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    constraints = MissionConstraints(
        epoch=args.epoch,
        leo_alt_km=args.leo_alt_km,
        llo_alt_km=args.llo_alt_km,
        lunar_stay_days=args.lunar_stay_days,
        outbound_tof_days=_grid(args.outbound_tof_days),
        return_tof_days=_grid(args.return_tof_days),
        low_energy_jacobi=_grid(args.low_energy_jacobi),
        include_direct=not args.no_direct,
        include_low_energy=not args.no_low_energy,
        include_coherence=not args.no_coherence,
        include_free_return=not args.no_free_return,
        max_total_dv_ms=args.max_total_dv_ms,
        max_total_tof_days=args.max_total_tof_days,
    )
    weights = ArchitectureWeights(
        dv=args.w_dv,
        tof=args.w_tof,
        robustness=args.w_robustness,
        risk=args.w_risk,
        fidelity_bonus=args.w_fidelity_bonus,
    )
    report = architect_cislunar_round_trip(constraints, weights)
    write_architecture_report(report, args.out)
    top = report.recommended
    print(json.dumps({
        "report": args.out,
        "certificate_hash": report.certificate_hash,
        "n_candidates": len(report.candidates),
        "recommended_id": report.recommended_id,
        "recommended": None if top is None else {
            "outbound": top.outbound.name,
            "return": top.return_leg.name,
            "total_dv_ms": top.total_dv_ms,
            "total_tof_days": top.total_tof_days,
            "score": top.score,
            "fidelity": top.fidelity,
            "free_return_capable": top.free_return_capable,
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
