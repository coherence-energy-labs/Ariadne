"""Build Ariadne's system-wide closure report."""
from __future__ import annotations

import argparse
from pathlib import Path

from ariadne.proof import build_default_ariadne_closure, write_closure_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".", help="Ariadne checkout root")
    parser.add_argument("--out-dir", default="data/benchmarks/closure", help="output directory")
    parser.add_argument("--fail-on-critical", action="store_true",
                        help="exit non-zero when critical closure gates fail")
    args = parser.parse_args()

    root = Path(args.project_root)
    report = build_default_ariadne_closure(root)
    outputs = write_closure_report(report, root / args.out_dir)
    print(f"status={report.status}")
    print(f"readiness_score={report.readiness_score:.6f}")
    print(f"critical_failures={report.critical_failures}")
    print(f"certificate_hash={report.certificate_hash}")
    print(f"json={outputs['json']}")
    print(f"markdown={outputs['markdown']}")
    return 2 if args.fail_on_critical and report.critical_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
