"""Build an external-corpus discovery inference benchmark.

Examples:

  python scripts/build_external_inference_benchmark.py --fetch-mpc --mpc-limit 500
  python scripts/build_external_inference_benchmark.py --ztf alerts.jsonl --rubin alerts.avro
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ariadne.discovery.benchmarking import run_inference_benchmark, write_benchmark_report
from ariadne.discovery.external_corpora import (
    ExternalCorpusManifest,
    fetch_mpcorb_cases,
    labelled_cases_from_rubin_file,
    labelled_cases_from_ztf_file,
    labelled_cases_from_mpcorb_lines,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch-mpc", action="store_true",
                        help="stream a live labelled sample from MPCORB.DAT.gz")
    parser.add_argument("--mpc-file", type=Path,
                        help="local MPCORB.DAT or .gz file")
    parser.add_argument("--mpc-limit", type=int, default=200)
    parser.add_argument("--ztf", type=Path, action="append", default=[],
                        help="ZTF alert/export file: JSON, JSONL, CSV, or Avro")
    parser.add_argument("--rubin", type=Path, action="append", default=[],
                        help="Rubin/LSST alert file: JSON, JSONL, CSV, or Avro")
    parser.add_argument("--out", type=Path, default=Path(".benchmarks/external_inference"))
    parser.add_argument("--fit-channels", action="store_true",
                        help="learn evidence-channel weights on the loaded cases")
    parser.add_argument("--fit-labels", action="store_true",
                        help="learn class-prior label biases on the loaded cases")
    parser.add_argument("--separate-calibration", action="store_true",
                        help="fit calibration on train/validation splits and score blind split")
    parser.add_argument("--adversarial", action="store_true",
                        help="include deterministic adversarial mutations")
    args = parser.parse_args()

    cases = []
    if args.fetch_mpc:
        cases.extend(fetch_mpcorb_cases(limit=args.mpc_limit))
    if args.mpc_file:
        if args.mpc_file.suffix.lower() == ".gz":
            import gzip
            with gzip.open(args.mpc_file, "rt", encoding="latin-1", errors="replace") as f:
                cases.extend(labelled_cases_from_mpcorb_lines(f, limit=args.mpc_limit,
                                                              source="mpc_mpcorb_file"))
        else:
            with args.mpc_file.open(encoding="latin-1", errors="replace") as f:
                cases.extend(labelled_cases_from_mpcorb_lines(f, limit=args.mpc_limit,
                                                              source="mpc_mpcorb_file"))
    for path in args.ztf:
        cases.extend(labelled_cases_from_ztf_file(path, require_truth=True))
    for path in args.rubin:
        cases.extend(labelled_cases_from_rubin_file(path, require_truth=True))

    if not cases:
        raise SystemExit("no external cases loaded; pass --fetch-mpc, --mpc-file, --ztf, or --rubin")

    result = run_inference_benchmark(
        cases, fit_channels=args.fit_channels, fit_labels=args.fit_labels,
        separate_calibration=args.separate_calibration,
        adversarial=args.adversarial)
    paths = write_benchmark_report(result, args.out)
    manifest = {
        "schema": "ariadne.discovery.external_benchmark_manifest.v1",
        "sources": ExternalCorpusManifest().__dict__,
        "n_cases": result.n,
        "accuracy": result.accuracy,
        "safe_accuracy": result.safe_accuracy,
        "macro_f1": result.macro_f1,
        "ece": result.reliability.ece,
        "certificate_hash": result.certificate_hash,
        "artifacts": paths,
    }
    (args.out / "external_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
