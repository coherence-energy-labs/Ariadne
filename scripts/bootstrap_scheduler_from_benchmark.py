"""Seed the PredictiveScheduler ledger from synthetic-injection recovery outcomes.

Pattern: run the recovery harness, classify each accepted candidate's
evidence class via predictive.classify_evidence, then record whether the
candidate was recovered (truth_id matched -> outcome='confirmed') or
not (no truth_id match -> outcome='refuted').

After running, the scheduler ledger at data/scheduler_ledger.json can be
loaded by any nightly run to give the live engine an empirically-trained
prior on what actions confirm what evidence classes.

This version calls the pipeline stages DIRECTLY (no run_pipeline wrapper)
so progress messages flush correctly under any process supervisor.
"""
from __future__ import annotations

import math
import sys
import time
from collections import Counter
from pathlib import Path


def main():
    from ariadne.discovery import realtime, predictive
    from ariadne.discovery.inference import Evidence
    from ariadne.validate.sensitivity import (
        make_population, inject_synthetic_objects)

    print("=" * 70, flush=True)
    print("BOOTSTRAPPING predictive scheduler from synthetic benchmark outcomes",
          flush=True)
    print("=" * 70, flush=True)

    sched_path = Path("data/scheduler_ledger.json")
    sched = predictive.PredictiveScheduler(ledger_path=sched_path)
    print(f"  starting ledger entries: {len(sched._history)}", flush=True)

    orbits = make_population(n_objects=5, seed=42)
    alerts, truth = inject_synthetic_objects(
        orbits, epoch="2026-04-01T00:00:00", n_nights=4, n_per_night=3,
        within_night_spread_hours=4.0, seed=42)
    print(f"  injected {len(orbits)} synthetic objects -> "
          f"{len(alerts)} alerts", flush=True)

    # Inline pipeline so every stage flushes immediately
    t0 = time.time()
    clusters = realtime.cluster_same_night(alerts, 1.0, 0.02)
    centroids = [realtime.cluster_centroid(c) for c in clusters]
    print(f"  [cluster] {len(centroids)} centroids", flush=True)
    tracks = realtime.build_tracklets(centroids,
                                        min_rate_arcsec_hr=0.05,
                                        max_rate_arcsec_hr=5.0)
    print(f"  [tracklets] {len(tracks)} tracklets", flush=True)
    chains = realtime.chain_tracklets(tracks)
    sane = realtime.filter_chain_sanity(chains)
    print(f"  [chains] {len(chains)} -> {len(sane)} after sanity", flush=True)

    chain_clusters = []
    for ch in sane:
        members = [m for sub in ch for m in sub.get("members", [])]
        centroid = dict(ch[len(ch) // 2])
        centroid["members"] = members
        centroid["chain"] = ch
        chain_clusters.append(centroid)

    print(f"  [iod] running ensemble on {len(chain_clusters)} chains...",
          flush=True)
    t1 = time.time()
    fitted = realtime.fit_filter(chain_clusters, rms_threshold_arcsec=1e9,
                                   use_ensemble_iod=True)
    print(f"  [iod] {time.time() - t1:.1f}s; "
          f"{sum(1 for f in fitted if f.get('status') == 'accepted')} accepted",
          flush=True)

    truth_id_recovered = set()
    n_recorded = 0
    for tr in fitted:
        if tr.get("status") != "accepted":
            continue
        members = tr.get("members", [])
        tids = Counter()
        for m in members:
            tid = (m.meta or {}).get("truth_id") if hasattr(m, "meta") else None
            if tid is not None:
                tids[tid] += 1
        if not tids:
            continue
        truth_id, n_match = tids.most_common(1)[0]
        recovered = (n_match >= 4)
        if recovered:
            truth_id_recovered.add(truth_id)

        ev = Evidence(
            mjd=float(tr["jd"] - 2400000.5) if "jd" in tr else None,
            ra_deg=math.degrees(tr["ra"]),
            dec_deg=math.degrees(tr["dec"]),
            rate_arcsec_hr=float(tr.get("rate_arcsec_hr", 0.0)),
            apparent_mag=None,
            n_detections=len(members),
            arc_days=(max(m.mjd for m in members) - min(m.mjd for m in members))
                       if members else 0.0,
            rms_arcsec=float(tr.get("rms_arcsec") or 0.0),
            orbit_state=(tr.get("x_fit_km", []) + tr.get("v_fit_kms", []))
                         if "x_fit_km" in tr else None,
            skybot_match_names=tr.get("xmatch", {}).get("names", []),
        )
        ev_class = predictive.classify_evidence(ev)
        outcome = "confirmed" if recovered else "refuted"
        sched.record_outcome(evidence_class=ev_class,
                              action="alert_and_submit_mpc",
                              outcome=outcome,
                              notes=f"benchmark truth_id={truth_id}")
        n_recorded += 1

    sched.save()
    elapsed = time.time() - t0
    print(f"  recorded {n_recorded} outcomes in {elapsed:.0f}s", flush=True)
    print(f"  recovered {len(truth_id_recovered)}/{len(orbits)} truth_ids",
          flush=True)
    print(f"  ledger written -> {sched_path}", flush=True)

    summary = sched.summary()
    print(f"  ledger summary: {summary['n_records']} records", flush=True)
    print(f"    by_evidence_class: {summary['by_evidence_class']}", flush=True)
    print(f"    outcomes: {summary['outcomes']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
