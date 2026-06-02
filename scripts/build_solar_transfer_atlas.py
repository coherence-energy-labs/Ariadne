#!/usr/bin/env python
"""Build whole-solar-system transfer corridor maps."""
from __future__ import annotations

import argparse
import json

from ariadne.interplanetary.solar_atlas import (
    build_solar_transfer_atlas,
    render_solar_transfer_atlas,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epoch-start", default="2028-01-01T00:00:00")
    p.add_argument("--departure-window-days", type=float, default=900.0)
    p.add_argument("--n-dep", type=int, default=14)
    p.add_argument("--n-tof", type=int, default=10)
    p.add_argument("--route-origin", default="EARTH")
    p.add_argument("--route-target", default="SATURN")
    p.add_argument("--out-dir", default="data/benchmarks/solar_transfer_atlas")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    atlas = build_solar_transfer_atlas(
        epoch_start=args.epoch_start,
        departure_window_days=args.departure_window_days,
        n_dep=args.n_dep,
        n_tof=args.n_tof,
        route_origin=args.route_origin,
        route_target=args.route_target,
    )
    artifacts = render_solar_transfer_atlas(atlas, args.out_dir)
    print(json.dumps({
        "schema": atlas.schema,
        "certificate_hash": atlas.certificate_hash,
        "corridors": len(atlas.corridors),
        "route_origin": atlas.route_origin,
        "route_target": atlas.route_target,
        "optimal_route": atlas.optimal_route,
        "optimal_route_score": atlas.optimal_route_score,
        "artifacts": artifacts,
    }, sort_keys=True))


if __name__ == "__main__":
    main()

