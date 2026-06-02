"""Operational alert replay and provenance ledger.

This module is the bridge between static benchmark cases and live survey
operations. It replays alerts chronologically, batches them by time window,
runs the realtime pipeline, and writes tamper-evident provenance records for
every batch and candidate output.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from ..brokers.base import Alert


def _jsonable(value):
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def stable_hash(value) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProvenanceRecord:
    """One replay/provenance event."""
    event: str
    ts_utc: str
    source: str
    n_alerts: int = 0
    n_outputs: int = 0
    input_hash: str = ""
    output_hash: str = ""
    parameters: dict = field(default_factory=dict)
    notes: str = ""


class ProvenanceLedger:
    """Append-only JSONL provenance ledger."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: ProvenanceRecord) -> ProvenanceRecord:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")
        return record

    def record(self, *, event: str, source: str, alerts=None, outputs=None,
               parameters: dict | None = None, notes: str = "") -> ProvenanceRecord:
        alerts = list(alerts or [])
        outputs = list(outputs or [])
        rec = ProvenanceRecord(
            event=event,
            ts_utc=datetime.now(timezone.utc).isoformat(),
            source=source,
            n_alerts=len(alerts),
            n_outputs=len(outputs),
            input_hash=stable_hash([asdict(a) if hasattr(a, "__dataclass_fields__") else a for a in alerts]),
            output_hash=stable_hash(outputs),
            parameters=parameters or {},
            notes=notes,
        )
        return self.append(rec)

    def read(self) -> list[ProvenanceRecord]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(ProvenanceRecord(**json.loads(line)))
        return out


@dataclass
class ReplayResult:
    """Summary of a chronological replay run."""
    n_alerts: int
    n_batches: int
    n_outputs: int
    output_hash: str
    ledger_path: str | None = None
    status_counts: dict = field(default_factory=dict)
    action_counts: dict = field(default_factory=dict)
    candidate_keys: list = field(default_factory=list)
    outputs: list = field(default_factory=list)


@dataclass(frozen=True)
class ReplayComparison:
    """Diff two replay results/manifests."""
    baseline_hash: str
    candidate_hash: str
    same_output: bool
    delta_outputs: int
    added_keys: list
    removed_keys: list
    status_delta: dict
    action_delta: dict


def output_key(output, idx: int) -> str:
    """Stable-ish candidate key for replay diffs."""
    if isinstance(output, dict):
        for key in ("key", "candidate_key", "desig", "status"):
            if output.get(key):
                return str(output[key])
        if "ra" in output and "dec" in output:
            return f"{output.get('ra')}_{output.get('dec')}_{output.get('rate_arcsec_hr', '')}"
    return f"output_{idx}_{stable_hash(output)[:12]}"


def summarize_outputs(outputs: list) -> dict:
    """Summarize statuses/actions/keys from pipeline outputs."""
    status_counts = {}
    action_counts = {}
    keys = []
    for idx, out in enumerate(outputs):
        keys.append(output_key(out, idx))
        if isinstance(out, dict):
            status = out.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            action = None
            if isinstance(out.get("_inference"), dict):
                action = out["_inference"].get("recommended_action")
            if action is None and isinstance(out.get("recommended_followup"), dict):
                action = out["recommended_followup"].get("action")
            if action:
                action_counts[action] = action_counts.get(action, 0) + 1
    return {
        "status_counts": status_counts,
        "action_counts": action_counts,
        "candidate_keys": sorted(keys),
    }


def write_replay_manifest(result: ReplayResult, path: str | Path) -> dict:
    """Write a compact replay manifest for drift comparison."""
    manifest = {
        "schema": "ariadne.discovery.replay_manifest.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_alerts": result.n_alerts,
        "n_batches": result.n_batches,
        "n_outputs": result.n_outputs,
        "output_hash": result.output_hash,
        "ledger_path": result.ledger_path,
        "status_counts": result.status_counts,
        "action_counts": result.action_counts,
        "candidate_keys": result.candidate_keys,
    }
    Path(path).write_text(json.dumps(_jsonable(manifest), sort_keys=True, indent=2) + "\n",
                          encoding="utf-8")
    return manifest


def compare_replay_manifests(baseline: dict | str | Path,
                             candidate: dict | str | Path) -> ReplayComparison:
    """Compare two replay manifests or manifest paths."""
    def load(x):
        if isinstance(x, dict):
            return x
        return json.loads(Path(x).read_text(encoding="utf-8"))

    a = load(baseline)
    b = load(candidate)
    a_keys = set(a.get("candidate_keys", []))
    b_keys = set(b.get("candidate_keys", []))
    status_keys = set(a.get("status_counts", {})) | set(b.get("status_counts", {}))
    action_keys = set(a.get("action_counts", {})) | set(b.get("action_counts", {}))
    return ReplayComparison(
        baseline_hash=a.get("output_hash", ""),
        candidate_hash=b.get("output_hash", ""),
        same_output=a.get("output_hash") == b.get("output_hash"),
        delta_outputs=int(b.get("n_outputs", 0)) - int(a.get("n_outputs", 0)),
        added_keys=sorted(b_keys - a_keys),
        removed_keys=sorted(a_keys - b_keys),
        status_delta={
            k: b.get("status_counts", {}).get(k, 0) - a.get("status_counts", {}).get(k, 0)
            for k in sorted(status_keys)
        },
        action_delta={
            k: b.get("action_counts", {}).get(k, 0) - a.get("action_counts", {}).get(k, 0)
            for k in sorted(action_keys)
        },
    )


def batch_alerts_by_time(alerts: Iterable[Alert], *, batch_days: float = 1.0) -> list[list[Alert]]:
    """Sort alerts chronologically and group into fixed-width MJD batches."""
    ordered = sorted(list(alerts), key=lambda a: a.mjd)
    if not ordered:
        return []
    batches: list[list[Alert]] = []
    current = [ordered[0]]
    batch_start = ordered[0].mjd
    for alert in ordered[1:]:
        if alert.mjd - batch_start <= batch_days:
            current.append(alert)
        else:
            batches.append(current)
            current = [alert]
            batch_start = alert.mjd
    batches.append(current)
    return batches


def replay_alerts(alerts: Iterable[Alert],
                  *,
                  pipeline: Callable[[Iterable[Alert]], list] | None = None,
                  batch_days: float = 1.0,
                  ledger: ProvenanceLedger | None = None,
                  source: str = "replay",
                  pipeline_kwargs: dict | None = None) -> ReplayResult:
    """Replay alerts in chronological batches through a pipeline function."""
    if pipeline is None:
        from .. import realtime

        def pipeline(batch):
            return realtime.run_pipeline(batch, **(pipeline_kwargs or {}))
    elif pipeline_kwargs:
        base = pipeline

        def pipeline(batch):
            return base(batch, **pipeline_kwargs)

    all_alerts = sorted(list(alerts), key=lambda a: a.mjd)
    batches = batch_alerts_by_time(all_alerts, batch_days=batch_days)
    outputs = []
    if ledger:
        ledger.record(event="replay_start", source=source, alerts=all_alerts,
                      parameters={"batch_days": batch_days, **(pipeline_kwargs or {})})
    for idx, batch in enumerate(batches):
        out = list(pipeline(batch))
        outputs.extend(out)
        if ledger:
            ledger.record(event="replay_batch", source=source, alerts=batch, outputs=out,
                          parameters={"batch_index": idx, "batch_days": batch_days})
    if ledger:
        ledger.record(event="replay_complete", source=source, alerts=all_alerts,
                      outputs=outputs, parameters={"n_batches": len(batches)})
    summary = summarize_outputs(outputs)
    return ReplayResult(
        n_alerts=len(all_alerts),
        n_batches=len(batches),
        n_outputs=len(outputs),
        output_hash=stable_hash(outputs),
        ledger_path=str(ledger.path) if ledger else None,
        status_counts=summary["status_counts"],
        action_counts=summary["action_counts"],
        candidate_keys=summary["candidate_keys"],
        outputs=outputs,
    )
