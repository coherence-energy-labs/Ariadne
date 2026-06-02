#!/usr/bin/env python
"""Acquire real labelled discovery corpora and emit audit artifacts.

This script is deliberately conservative: externally supplied labels are kept
separate from unlabeled operational alert streams, and every source is recorded
in a manifest so benchmark claims can be audited later.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ariadne.discovery.benchmarking import run_inference_benchmark, write_benchmark_report
from ariadne.discovery.brokers.base import BrokerError
from ariadne.discovery.external_corpora import (
    CorpusBuildRecord,
    ExternalCorpusManifest,
    alerts_from_rubin_file,
    alerts_from_ztf_file,
    fetch_mpcorb_cases,
    labelled_cases_from_rubin_file,
    labelled_cases_from_ztf_file,
    read_alerts_jsonl,
    read_labelled_cases_jsonl,
    write_alerts_jsonl,
    write_labelled_cases_jsonl,
)
from ariadne.discovery.operations.replay import (
    ProvenanceLedger,
    replay_alerts,
    stable_hash,
    write_replay_manifest,
)


def _jsonable(value):
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _load_existing(paths, loader):
    out = []
    for path in paths:
        out.extend(loader(path))
    return out


def _query_alerce(args, records: list[CorpusBuildRecord]):
    try:
        from ariadne.discovery.brokers.alerce import AlerceZTFBroker

        broker = AlerceZTFBroker(class_name=args.alerce_class)
        alerts = list(broker.query_cone(
            args.ra, args.dec, args.radius_deg,
            args.mjd_start, args.mjd_end,
            max_alerts=args.max_alerts,
        ))
        records.append(CorpusBuildRecord(
            source="alerce_ztf",
            path_or_url="https://api.alerce.online/",
            n_alerts=len(alerts),
        ))
        return alerts
    except (BrokerError, Exception) as exc:
        records.append(CorpusBuildRecord(
            source="alerce_ztf",
            path_or_url="https://api.alerce.online/",
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        ))
        return []


def build_corpus(args) -> dict:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records: list[CorpusBuildRecord] = []
    cases = []
    alerts = []

    if args.case_jsonl:
        loaded = _load_existing(args.case_jsonl, read_labelled_cases_jsonl)
        cases.extend(loaded)
        records.append(CorpusBuildRecord(
            source="case_jsonl",
            path_or_url=";".join(map(str, args.case_jsonl)),
            n_cases=len(loaded),
        ))

    if args.alert_jsonl:
        loaded = _load_existing(args.alert_jsonl, read_alerts_jsonl)
        alerts.extend(loaded)
        records.append(CorpusBuildRecord(
            source="alert_jsonl",
            path_or_url=";".join(map(str, args.alert_jsonl)),
            n_alerts=len(loaded),
        ))

    if args.fetch_mpc:
        mpc_cases = fetch_mpcorb_cases(limit=args.mpc_limit, timeout=args.timeout)
        cases.extend(mpc_cases)
        records.append(CorpusBuildRecord(
            source="mpc_mpcorb_live",
            path_or_url=ExternalCorpusManifest().mpcorb_url,
            n_cases=len(mpc_cases),
        ))

    for path in args.ztf_file:
        ztf_cases = labelled_cases_from_ztf_file(path, require_truth=not args.allow_unlabelled)
        ztf_alerts = alerts_from_ztf_file(path)
        cases.extend(ztf_cases)
        alerts.extend(ztf_alerts)
        records.append(CorpusBuildRecord(
            source="ztf_file",
            path_or_url=str(path),
            n_cases=len(ztf_cases),
            n_alerts=len(ztf_alerts),
        ))

    for path in args.rubin_file:
        rubin_cases = labelled_cases_from_rubin_file(path, require_truth=not args.allow_unlabelled)
        rubin_alerts = alerts_from_rubin_file(path)
        cases.extend(rubin_cases)
        alerts.extend(rubin_alerts)
        records.append(CorpusBuildRecord(
            source="rubin_file",
            path_or_url=str(path),
            n_cases=len(rubin_cases),
            n_alerts=len(rubin_alerts),
        ))

    if args.alerce:
        alerts.extend(_query_alerce(args, records))

    if not cases and not alerts:
        raise SystemExit("no cases or alerts were acquired")

    case_path = out / "labelled_cases.jsonl"
    alert_path = out / "alerts.jsonl"
    if cases:
        write_labelled_cases_jsonl(cases, case_path)
    if alerts:
        write_alerts_jsonl(alerts, alert_path)

    benchmark_paths = {}
    benchmark_summary = None
    if args.run_benchmark:
        if not cases:
            raise SystemExit("--run-benchmark requires at least one labelled case")
        result = run_inference_benchmark(
            cases,
            fit_channels=args.fit_channels,
            fit_labels=args.fit_labels,
            separate_calibration=args.separate_calibration,
            adversarial=args.adversarial,
        )
        benchmark_paths = write_benchmark_report(result, out / "benchmark")
        benchmark_summary = {
            "n": result.n,
            "accuracy": result.accuracy,
            "safe_accuracy": result.safe_accuracy,
            "macro_f1": result.macro_f1,
            "ece": result.reliability.ece,
            "certificate_hash": result.certificate_hash,
        }

    replay_summary = None
    if args.run_replay:
        if not alerts:
            raise SystemExit("--run-replay requires at least one alert")
        ledger_path = out / "provenance.jsonl"
        if ledger_path.exists():
            ledger_path.unlink()
        ledger = ProvenanceLedger(ledger_path)
        replay = replay_alerts(
            alerts,
            batch_days=args.batch_days,
            ledger=ledger,
            source="real_corpus_acquisition",
        )
        replay_manifest = write_replay_manifest(replay, out / "replay_manifest.json")
        replay_summary = {
            "n_alerts": replay.n_alerts,
            "n_batches": replay.n_batches,
            "n_outputs": replay.n_outputs,
            "output_hash": replay.output_hash,
            "manifest": str(out / "replay_manifest.json"),
            "provenance": str(out / "provenance.jsonl"),
            "status_counts": replay_manifest.get("status_counts", {}),
            "action_counts": replay_manifest.get("action_counts", {}),
        }

    manifest = {
        "schema": "ariadne.discovery.real_corpus_manifest.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_references": _jsonable(ExternalCorpusManifest()),
        "records": _jsonable(records),
        "n_cases": len(cases),
        "n_alerts": len(alerts),
        "case_path": str(case_path) if cases else "",
        "alert_path": str(alert_path) if alerts else "",
        "case_hash": stable_hash(cases),
        "alert_hash": stable_hash(alerts),
        "benchmark": benchmark_summary,
        "benchmark_artifacts": benchmark_paths,
        "replay": replay_summary,
    }
    manifest_path = out / "corpus_manifest.json"
    manifest_path.write_text(json.dumps(_jsonable(manifest), sort_keys=True, indent=2) + "\n",
                             encoding="utf-8")
    return manifest


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="data/benchmarks/real_corpus")
    p.add_argument("--case-jsonl", action="append", default=[])
    p.add_argument("--alert-jsonl", action="append", default=[])
    p.add_argument("--fetch-mpc", action="store_true")
    p.add_argument("--mpc-limit", type=int, default=500)
    p.add_argument("--timeout", type=float, default=45.0)
    p.add_argument("--ztf-file", action="append", default=[])
    p.add_argument("--rubin-file", action="append", default=[])
    p.add_argument("--allow-unlabelled", action="store_true")
    p.add_argument("--alerce", action="store_true")
    p.add_argument("--alerce-class", default="asteroid")
    p.add_argument("--ra", type=float, default=180.0)
    p.add_argument("--dec", type=float, default=0.0)
    p.add_argument("--radius-deg", type=float, default=1.0)
    p.add_argument("--mjd-start", type=float, default=60000.0)
    p.add_argument("--mjd-end", type=float, default=60400.0)
    p.add_argument("--max-alerts", type=int, default=1000)
    p.add_argument("--run-benchmark", action="store_true")
    p.add_argument("--fit-channels", action="store_true")
    p.add_argument("--fit-labels", action="store_true")
    p.add_argument("--separate-calibration", action="store_true")
    p.add_argument("--adversarial", action="store_true")
    p.add_argument("--run-replay", action="store_true")
    p.add_argument("--batch-days", type=float, default=1.0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    manifest = build_corpus(args)
    print(json.dumps({
        "manifest": str(Path(args.out_dir) / "corpus_manifest.json"),
        "n_cases": manifest["n_cases"],
        "n_alerts": manifest["n_alerts"],
        "benchmark": manifest.get("benchmark"),
        "replay": manifest.get("replay"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
