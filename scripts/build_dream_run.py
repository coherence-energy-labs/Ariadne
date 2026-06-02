"""Build Ariadne DreamLab experiment queue from a closure report."""
from __future__ import annotations

import argparse

from ariadne.proof import build_dream_run, write_dream_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure-report", default="data/benchmarks/closure/closure_report.json")
    parser.add_argument("--out-dir", default="data/benchmarks/dream_lab")
    args = parser.parse_args()
    run = build_dream_run(args.closure_report)
    outputs = write_dream_run(run, args.out_dir)
    print(f"status={run.status}")
    print(f"experiments={run.experiment_count}")
    print(f"certificate_hash={run.certificate_hash}")
    print(f"json={outputs['json']}")
    print(f"markdown={outputs['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
