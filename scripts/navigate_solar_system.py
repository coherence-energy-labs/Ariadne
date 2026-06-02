#!/usr/bin/env python
"""Search solar-system mission routes and write JSON/PNG artifacts."""
from __future__ import annotations

import argparse
import json

from ariadne.interplanetary.navigator import (
    NavigatorConstraints,
    NavigatorWeights,
    navigate_solar_system,
    write_navigator_report,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--origin", default="EARTH")
    p.add_argument("--target", required=True)
    p.add_argument("--epoch-start", default="2028-01-01T00:00:00")
    p.add_argument("--departure-window-days", type=float, default=365.25 * 4.0)
    p.add_argument("--tof-min-days", type=float, default=120.0)
    p.add_argument("--tof-max-days", type=float, default=2500.0)
    p.add_argument("--n-dep", type=int, default=60)
    p.add_argument("--n-tof", type=int, default=45)
    p.add_argument("--no-direct", action="store_true")
    p.add_argument("--no-gravity-assist", action="store_true")
    p.add_argument("--no-moon-tour", action="store_true")
    p.add_argument("--optimize-flybys", action="store_true")
    p.add_argument("--flyby-maxiter", type=int, default=35)
    p.add_argument("--flyby-alt-km", type=float, default=300.0)
    p.add_argument("--max-total-dv-ms", type=float)
    p.add_argument("--max-tof-days", type=float)
    p.add_argument("--out-dir", default="data/benchmarks/solar_navigator")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--w-dv", type=float, default=1.0)
    p.add_argument("--w-tof", type=float, default=0.002)
    p.add_argument("--w-c3", type=float, default=0.04)
    p.add_argument("--w-arrival-vinf", type=float, default=0.25)
    p.add_argument("--w-risk", type=float, default=1.0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    constraints = NavigatorConstraints(
        origin=args.origin,
        target=args.target,
        epoch_start=args.epoch_start,
        departure_window_days=args.departure_window_days,
        tof_range_days=(args.tof_min_days, args.tof_max_days),
        n_dep=args.n_dep,
        n_tof=args.n_tof,
        include_direct=not args.no_direct,
        include_gravity_assist=not args.no_gravity_assist,
        include_moon_tour=not args.no_moon_tour,
        optimize_flybys=args.optimize_flybys,
        flyby_maxiter=args.flyby_maxiter,
        flyby_alt_km=args.flyby_alt_km,
        max_total_dv_ms=args.max_total_dv_ms,
        max_tof_days=args.max_tof_days,
    )
    weights = NavigatorWeights(
        dv=args.w_dv,
        tof=args.w_tof,
        c3=args.w_c3,
        arrival_vinf=args.w_arrival_vinf,
        risk=args.w_risk,
    )
    report = navigate_solar_system(constraints, weights)
    artifacts = write_navigator_report(report, args.out_dir, make_plots=not args.no_plots)
    top = report.balanced
    print(json.dumps({
        "report": artifacts.get("report"),
        "certificate_hash": report.certificate_hash,
        "n_routes": len(report.routes),
        "fastest_id": report.fastest_id,
        "cheapest_id": report.cheapest_id,
        "balanced_id": report.balanced_id,
        "balanced": None if top is None else {
            "name": top.name,
            "sequence": top.sequence,
            "total_dv_ms": top.total_dv_ms,
            "tof_days": top.tof_days,
            "c3_km2_s2": top.c3_km2_s2,
            "arrival_vinf_kms": top.arrival_vinf_kms,
            "fidelity": top.fidelity,
        },
        "artifacts": artifacts,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
