"""Tests for the discovery operations layer (candidate_store, alerts, nightly)."""
import json
import tempfile
from pathlib import Path

import pytest

from ariadne.discovery.operations.candidate_store import (
    Candidate, CandidateStore, _canonical_key,
)
from ariadne.discovery.operations.alerts import (
    NoopSink, FileSink, AlertSink, fire_all,
)


# ---------- canonical key ----------
def test_canonical_key_stable_under_small_drift():
    # Same object at slightly different positions/rates should produce the same key
    k1 = _canonical_key(180.001, 20.0, 1.0)
    k2 = _canonical_key(180.005, 20.005, 1.05)
    assert k1 == k2


def test_canonical_key_differs_for_distinct_objects():
    k1 = _canonical_key(180.0, 20.0, 1.0)
    k2 = _canonical_key(180.5, 20.0, 1.0)
    k3 = _canonical_key(180.0, 21.0, 1.0)
    k4 = _canonical_key(180.0, 20.0, 5.0)
    assert len({k1, k2, k3, k4}) == 4


# ---------- candidate store ----------
def test_store_new_then_redetect_dedupes_correctly():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "store.json"
        s = CandidateStore(path)
        c1, is_new1 = s.upsert(ra=180.0, dec=20.0, rate_arcsec_hr=1.0,
                                mjd=60000.0, rms_arcsec=5.0)
        assert is_new1 and c1.n_runs == 1

        # Same object 3 days later: NOT new, n_runs bumped
        c2, is_new2 = s.upsert(ra=180.002, dec=20.0, rate_arcsec_hr=1.02,
                                mjd=60003.0, rms_arcsec=4.5)
        assert not is_new2
        assert c2.n_runs == 2
        assert c2.first_seen_mjd == 60000.0
        assert c2.last_seen_mjd == 60003.0
        assert len(c2.rms_history) == 2

        # Distinct object: NEW
        c3, is_new3 = s.upsert(ra=190.0, dec=20.0, rate_arcsec_hr=1.0,
                                mjd=60000.0, rms_arcsec=8.0)
        assert is_new3

        assert len(s) == 2


def test_store_round_trip_persists():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "store.json"
        s = CandidateStore(path)
        s.upsert(ra=180.0, dec=20.0, rate_arcsec_hr=1.0, mjd=60000.0, rms_arcsec=5.0)
        s.upsert(ra=190.0, dec=21.0, rate_arcsec_hr=2.0, mjd=60001.0, rms_arcsec=3.0)
        s.save()

        # Reload from disk
        s2 = CandidateStore(path)
        assert len(s2) == 2


def test_store_status_transitions():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "store.json"
        s = CandidateStore(path)
        c, _ = s.upsert(ra=180.0, dec=20.0, rate_arcsec_hr=1.0,
                         mjd=60000.0, rms_arcsec=5.0)
        assert c.status == "new"
        c, _ = s.upsert(ra=180.0, dec=20.0, rate_arcsec_hr=1.0,
                         mjd=60003.0, rms_arcsec=5.0)
        assert c.status == "active"
        n = s.mark_stale(max_age_days=1.0, current_mjd=60100.0)
        assert n == 1
        assert c.status == "stale"


def test_store_discovery_candidates_filters_skybot_matches():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "store.json"
        s = CandidateStore(path)
        s.upsert(ra=180.0, dec=20.0, rate_arcsec_hr=1.0, mjd=60000.0,
                 rms_arcsec=5.0, skybot_names=[])
        s.upsert(ra=190.0, dec=21.0, rate_arcsec_hr=2.0, mjd=60000.0,
                 rms_arcsec=3.0, skybot_names=["1 Ceres"])
        leads = s.discovery_candidates()
        assert len(leads) == 1
        assert not leads[0].skybot_names


# ---------- alert sinks ----------
def test_filesink_writes_jsonl():
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "alerts.jsonl"
        sink = FileSink(log)
        c = Candidate(key="180.0_+20.0_1.00", ra=180.0, dec=20.0,
                       rate_arcsec_hr=1.0, first_seen_mjd=60000.0,
                       last_seen_mjd=60000.0,
                       rms_history=[[60000.0, 5.0]])
        sink.fire(c, run_id="test-run-1")
        # second event
        sink.fire(c, run_id="test-run-2")
        # Read back
        lines = log.read_text().strip().splitlines()
        assert len(lines) == 2
        rec0 = json.loads(lines[0])
        assert rec0["run_id"] == "test-run-1"
        assert rec0["key"] == "180.0_+20.0_1.00"
        assert rec0["rms_arcsec"] == 5.0


def test_fire_all_continues_on_sink_failure():
    class BrokenSink(AlertSink):
        name = "broken"
        def fire(self, c, run_id=None):
            raise RuntimeError("intentional test failure")

    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "alerts.jsonl"
        sinks = [BrokenSink(), FileSink(log), NoopSink()]
        c = Candidate(key="k", ra=0, dec=0, rate_arcsec_hr=1.0,
                       first_seen_mjd=0, last_seen_mjd=0)
        # Should NOT raise, even though BrokenSink does
        fire_all(sinks, c, run_id="r")
        # FileSink should still have received the event
        assert log.read_text().strip()
