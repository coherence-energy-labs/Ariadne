"""Compare two Ariadne replay manifests."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ariadne.discovery.operations.replay import compare_replay_manifests


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("baseline", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--fail-on-drift", action="store_true")
    args = p.parse_args()

    diff = compare_replay_manifests(args.baseline, args.candidate)
    payload = asdict(diff)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.fail_on_drift and not diff.same_output:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

