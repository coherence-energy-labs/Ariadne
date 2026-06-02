"""Promote routes from a navigator report through Ariadne's proof ladder."""
from __future__ import annotations

import argparse
from pathlib import Path

from ariadne.proof import (
    covariance_envelope_evidence,
    independent_crosscheck_evidence,
    load_routes_from_navigator_report,
    nbody_replay_evidence,
    promote_routes,
    write_promotion_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("navigator_report", help="path to navigator_report.json")
    parser.add_argument("--out-dir", default="data/benchmarks/route_promotion")
    parser.add_argument("--require-high-fidelity", action="store_true",
                        help="require n-body, covariance, and independent cross-check evidence")
    parser.add_argument("--generate-nbody-evidence", action="store_true",
                        help="generate Sun-centered n-body replay evidence for each route")
    parser.add_argument("--generate-covariance-evidence", action="store_true",
                        help="generate deterministic n-body covariance envelope evidence for each route")
    parser.add_argument("--generate-crosscheck-evidence", action="store_true",
                        help="generate independent Radau-vs-DOP853 n-body cross-check evidence")
    parser.add_argument("--require-nbody-replay", action="store_true",
                        help="require the n-body replay rung, without requiring all high-fidelity rungs")
    parser.add_argument("--require-covariance-envelope", action="store_true",
                        help="require the covariance envelope rung, without requiring all high-fidelity rungs")
    parser.add_argument("--require-independent-crosscheck", action="store_true",
                        help="require the independent cross-check rung, without requiring all high-fidelity rungs")
    parser.add_argument("--allow-missing-coordinates", action="store_true",
                        help="do not require finite ephemeris coordinates on route events")
    parser.add_argument("--fail-on-rejected", action="store_true")
    args = parser.parse_args()

    routes = load_routes_from_navigator_report(args.navigator_report)
    evidence_by_route = {}
    if args.generate_nbody_evidence or args.generate_covariance_evidence or args.generate_crosscheck_evidence:
        for route in routes:
            rid = str(route.get("route_id", ""))
            rows = []
            if args.generate_nbody_evidence:
                rows.append(nbody_replay_evidence(route))
            if args.generate_covariance_evidence:
                rows.append(covariance_envelope_evidence(route))
            if args.generate_crosscheck_evidence:
                rows.append(independent_crosscheck_evidence(route))
            evidence_by_route[rid] = tuple(rows)
    required_external = []
    if args.require_nbody_replay:
        required_external.append("nbody_replay")
    if args.require_covariance_envelope:
        required_external.append("covariance_envelope")
    if args.require_independent_crosscheck:
        required_external.append("independent_crosscheck")
    report = promote_routes(
        routes,
        external_evidence_by_route=evidence_by_route,
        require_high_fidelity=args.require_high_fidelity,
        required_external_rungs=tuple(required_external),
        require_ephemeris_coordinates=not args.allow_missing_coordinates,
    )
    outputs = write_promotion_report(report, Path(args.out_dir))
    print(f"status={report.status}")
    print(f"routes={report.route_count}")
    print(f"promoted={report.promoted_count}")
    print(f"rejected={report.rejected_count}")
    print(f"certificate_hash={report.certificate_hash}")
    print(f"json={outputs['json']}")
    print(f"markdown={outputs['markdown']}")
    return 2 if args.fail_on_rejected and report.rejected_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
