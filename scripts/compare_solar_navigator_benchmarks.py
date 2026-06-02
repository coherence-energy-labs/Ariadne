#!/usr/bin/env python
"""Compare two solar navigator benchmark summaries."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from ariadne.interplanetary.navigator_benchmark import compare_benchmark_summaries


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--fail-on-drift", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    diff = compare_benchmark_summaries(args.baseline, args.candidate)
    print(json.dumps(asdict(diff), sort_keys=True))
    if args.fail_on_drift and not diff.same_certificate:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
