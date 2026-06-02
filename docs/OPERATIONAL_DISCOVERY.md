# Operational Discovery Hardening

This layer turns discovery experiments into replayable operations.

## Alert replay

Use `ariadne.discovery.operations.replay` to feed alerts chronologically in
fixed-width MJD batches. Every run can write a JSONL provenance ledger with
stable input/output hashes.

```python
from ariadne.discovery.operations.replay import ProvenanceLedger, replay_alerts

ledger = ProvenanceLedger(".benchmarks/replay/provenance.jsonl")
result = replay_alerts(alerts, batch_days=1.0, ledger=ledger,
                       pipeline_kwargs={"do_xmatch": False})
```

Write and compare replay manifests:

```python
from ariadne.discovery.operations.replay import (
    write_replay_manifest, compare_replay_manifests)

write_replay_manifest(result, ".benchmarks/replay/manifest.json")
diff = compare_replay_manifests("baseline.json", "candidate.json")
```

Or from the shell:

```powershell
$env:PYTHONPATH="src"
python scripts\compare_replay_manifests.py baseline.json candidate.json --fail-on-drift
```

## Real corpus acquisition

`scripts/acquire_real_labelled_corpus.py` is the operator path for building an
auditable real-data bundle. It can download live MPCORB labelled known-object
cases, ingest local ZTF/Rubin JSON/JSONL/CSV/Avro exports, collect ALeRCE ZTF
broker alerts, and then write portable JSONL plus a source manifest.

```powershell
$env:PYTHONPATH="src"
python scripts\acquire_real_labelled_corpus.py `
  --out-dir data\benchmarks\real_corpus_mpc_500 `
  --fetch-mpc --mpc-limit 500 `
  --run-benchmark --fit-channels --fit-labels
```

For alert replay:

```powershell
$env:PYTHONPATH="src"
python scripts\acquire_real_labelled_corpus.py `
  --out-dir data\benchmarks\real_corpus_alerce_probe `
  --alerce --ra 180 --dec 0 --radius-deg 2 `
  --mjd-start 60000 --mjd-end 60400 --max-alerts 100 `
  --run-replay
```

The acquisition manifest keeps labelled benchmark cases separate from
unlabelled operational alert streams. That separation is intentional: labels
prove inference accuracy; replay proves pipeline determinism and operational
drift behavior.

## Pipeline provenance

`realtime.run_pipeline_with_provenance()` wraps the existing live pipeline and
records `pipeline_start` / `pipeline_complete` events with hashes and runtime
parameters.

Nightly runs can write the same ledger by setting
`NightlyConfig(provenance_path="path/to/provenance.jsonl")`.

## Real/bogus triage

`realbogus.score_realbogus()` now returns:

* `severity`: `low`, `medium`, `high`, or `critical`
* `action`: `keep`, `review`, or `discard`
* `explanation`: compact fired-rule summary

Short arcs now route toward review instead of silent discard when evidence is
underdetermined.

## Scheduler separation metadata

`PredictiveScheduler.recommend()` accepts top alternative hypotheses and reports
which actions help separate them. The inference engine passes its top
hypotheses into the scheduler when a learned scheduler is attached.

## Dashboard ops API

The dashboard exposes `/api/ops`, a compact operator-facing state endpoint with
candidate counts, top-ranked leads, store path, alert path, score, grade, status,
and candidate metadata.
