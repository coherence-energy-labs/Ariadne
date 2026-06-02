#!/usr/bin/env python
"""Run real-ephemeris solar navigator benchmarks."""
from __future__ import annotations

import argparse
import json
import sys

from ariadne.interplanetary.navigator_benchmark import run_navigator_benchmark


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="data/benchmarks/solar_navigator_benchmark")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--fail-on-error", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = run_navigator_benchmark(
        outdir=args.out_dir,
        make_plots=not args.no_plots,
    )
    payload = {
        "out_dir": args.out_dir,
        "passed": result.passed,
        "elapsed_s": result.elapsed_s,
        "certificate_hash": result.certificate_hash,
        "cases": result.cases,
        "validations": [
            {
                "target": v.target,
                "passed": v.passed,
                "failures": v.failures,
                "warnings": v.warnings,
                "route_count": v.route_count,
                "pareto_count": v.pareto_count,
            }
            for v in result.validations
        ],
    }
    print(json.dumps(payload, sort_keys=True))
    if args.fail_on_error and not result.passed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
