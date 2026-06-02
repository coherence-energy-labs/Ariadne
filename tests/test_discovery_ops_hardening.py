from pathlib import Path


def _alert(i, mjd=60000.0, ra=180.0, dec=0.0):
    from ariadne.discovery.brokers.base import Alert
    return Alert(
        survey="TEST", alert_id=f"a{i}", obj_id=f"o{i}",
        mjd=mjd, ra=ra, dec=dec, mag=20.0, band="r",
        meta={"fixture": True},
    )


def test_replay_batches_chronologically_and_writes_provenance(tmp_path: Path):
    from ariadne.discovery.operations.replay import (
        ProvenanceLedger, replay_alerts, write_replay_manifest)

    alerts = [_alert(2, 60002.0), _alert(0, 60000.0), _alert(1, 60000.2)]
    ledger = ProvenanceLedger(tmp_path / "prov.jsonl")

    def pipeline(batch):
        return [{"n": len(list(batch))}]

    result = replay_alerts(alerts, pipeline=pipeline, batch_days=0.5,
                           ledger=ledger, source="unit")
    assert result.n_alerts == 3
    assert result.n_batches == 2
    assert result.n_outputs == 2
    records = ledger.read()
    assert [r.event for r in records] == [
        "replay_start", "replay_batch", "replay_batch", "replay_complete"]
    assert records[0].input_hash
    assert records[-1].output_hash == result.output_hash
    manifest = write_replay_manifest(result, tmp_path / "manifest.json")
    assert manifest["output_hash"] == result.output_hash
    assert manifest["n_batches"] == 2


def test_replay_manifest_comparison_detects_drift(tmp_path: Path):
    from ariadne.discovery.operations.replay import compare_replay_manifests

    a = {
        "output_hash": "aaa", "n_outputs": 1,
        "candidate_keys": ["x"],
        "status_counts": {"accepted": 1},
        "action_counts": {"monitor": 1},
    }
    b = {
        "output_hash": "bbb", "n_outputs": 2,
        "candidate_keys": ["x", "y"],
        "status_counts": {"accepted": 1, "rejected": 1},
        "action_counts": {"monitor": 0, "discard": 1},
    }
    diff = compare_replay_manifests(a, b)
    assert not diff.same_output
    assert diff.delta_outputs == 1
    assert diff.added_keys == ["y"]
    assert diff.status_delta["rejected"] == 1
    assert diff.action_delta["discard"] == 1


def test_realbogus_returns_severity_and_review_action():
    from ariadne.discovery.realbogus import score_realbogus

    verdict = score_realbogus({"members": [_alert(0), _alert(1, 60000.02)],
                               "arc_days": 0.02, "rate_arcsec_hr": 1.0})
    assert verdict.severity in {"medium", "high", "critical"}
    assert verdict.action in {"review", "discard"}
    assert verdict.explanation


def test_scheduler_reports_discriminating_actions():
    from ariadne.discovery.predictive import PredictiveScheduler

    sched = PredictiveScheduler()
    reco = sched.recommend(
        evidence_class="default",
        hypothesis_posterior=0.45,
        alternatives=[
            {"orbital_class": "JTROJAN"},
            {"orbital_class": "CENTAUR"},
        ],
    )
    assert isinstance(reco.separates, list)
    assert reco.action in {
        "observe_second_night", "observe_multi_band", "observe_deep_stack",
        "archive_search", "query_skybot", "monitor_only", "discard",
        "query_horizons", "alert_and_submit_mpc",
    }


def test_run_pipeline_with_provenance_records_hashes(tmp_path: Path):
    from ariadne.discovery.operations.replay import ProvenanceLedger
    from ariadne.discovery.realtime import run_pipeline_with_provenance

    ledger = ProvenanceLedger(tmp_path / "pipeline.jsonl")
    result = run_pipeline_with_provenance(
        [], ledger=ledger, source="unit", do_xmatch=False, smart_layer=False)
    assert result == []
    records = ledger.read()
    assert [r.event for r in records] == ["pipeline_start", "pipeline_complete"]
    assert records[-1].output_hash


def test_dashboard_ops_api(tmp_path: Path):
    from ariadne.discovery.dashboard import create_app

    store = tmp_path / "store.json"
    store.write_text('{"candidates":[]}', encoding="utf-8")
    app = create_app(store)
    client = app.test_client()
    resp = client.get("/api/ops")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["n_candidates"] == 0
    assert data["store_path"] == str(store)


def test_nightly_config_accepts_provenance_path():
    from ariadne.discovery.operations.nightly import NightlyConfig

    cfg = NightlyConfig(store_path="store.json", provenance_path="prov.jsonl")
    assert cfg.provenance_path == "prov.jsonl"
