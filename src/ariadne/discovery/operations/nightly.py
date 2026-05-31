"""Nightly orchestrator -- one CLI invocation = one night's discovery run.

Workflow:
  1. Read config (sky window, MJD window, broker / archive, alert sinks, store path).
  2. Pull alerts/detections from the configured source (ALeRCE / archive).
  3. Run the discovery pipeline (cluster -> tracklet -> chain -> IOD+LM -> SkyBoT).
  4. For each surviving candidate, upsert into the candidate store. New keys ->
     fire alerts. Re-detections update history but don't re-fire.
  5. Mark stale candidates (not seen in N days).
  6. Save the candidate store atomically. Print a one-line run summary.

Idempotent: re-running the same config the same night won't double-fire alerts
(the store dedupes on canonical key). Designed for cron / systemd timer / Windows
Task Scheduler.
"""
from __future__ import annotations

import math
import time
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from .alerts import AlertSink, FileSink, fire_all
from .candidate_store import CandidateStore


@dataclass
class NightlyConfig:
    """All knobs for one nightly run.

    Required:
      store_path: path to the JSON candidate store (created on first run).
      source:    "alerce_ztf" | "synthetic" | "synthetic_kepler"

    Sky / time window (defaults reasonable for first-run testing):
      ra, dec, radius_deg : single cone centre + radius (degrees) -- used when `cones` is None.
      cones               : optional list of (ra, dec, radius_deg) -- sweeps multiple
                            patches in one invocation. Each cone is pulled + filtered
                            separately, but they share ONE store (so dedupe is global).
      mjd_start, mjd_end  : detection MJD window (None = unbounded for source)
      max_alerts          : cap per cone

    Filter tuning:
      cluster_pos_tol_arcsec, rate_window_arcsec_hr, pair_dt_hours, rms_threshold_arcsec
      max_position_gap_arcsec, max_rate_change_pct (for multi-night chaining)
      use_helio_linc:  True -> use the validated HelioLinC sweep (full (r,rdot) grid
                       hypothesis search) instead of the simpler extrapolation chainer.
                       Slower per-cone but recovers MORE genuine multi-night links.

    Behaviour:
      stale_after_days: candidates not seen in N days move to status=stale
      do_xmatch:        SkyBoT cross-match accepted candidates (recommended True)
      dry_run:          run the pipeline but don't fire alerts or save store
    """
    store_path: str
    source: str = "alerce_ztf"

    ra: float = 180.0
    dec: float = 20.0
    radius_deg: float = 5.0
    cones: list | None = None       # optional list of (ra, dec, radius_deg)
    mjd_start: float | None = None
    mjd_end: float | None = None
    max_alerts: int = 1000

    cluster_pos_tol_arcsec: float = 2.0
    rate_window_arcsec_hr: tuple = (0.05, 30.0)
    pair_dt_hours: tuple = (0.1, 6.0)
    rms_threshold_arcsec: float = 15.0
    max_position_gap_arcsec: float = 60.0
    max_rate_change_pct: float = 50.0
    use_helio_linc: bool = False

    stale_after_days: float = 60.0
    do_xmatch: bool = True
    dry_run: bool = False

    alert_sinks: list = field(default_factory=list)


def _pull_alerts(cfg: NightlyConfig, ra: float, dec: float, radius_deg: float):
    """Pull alerts from the configured source for ONE cone. Returns a list of Alert objects."""
    if cfg.source == "alerce_ztf":
        from ..brokers.alerce import AlerceZTFBroker
        from ..brokers.base import collect
        broker = AlerceZTFBroker(class_name="asteroid")
        return collect(broker.query_cone(ra, dec, radius_deg,
                                          cfg.mjd_start or 0,
                                          cfg.mjd_end or 1e7,
                                          max_alerts=cfg.max_alerts),
                       max_n=cfg.max_alerts)
    if cfg.source == "synthetic":
        from ..brokers.base import synthesise_alerts
        return synthesise_alerts(n_real_objects=3, n_interlopers=200,
                                  ra_center=ra, dec_center=dec,
                                  width_deg=radius_deg, seed=0)
    if cfg.source == "synthetic_kepler":
        from ..brokers.base import synthesise_keplerian_alerts
        return synthesise_keplerian_alerts(orbits=[
            {"a_au": 80, "e": 0.05, "i": 8,  "Omega": 30, "omega": 50, "M": 180},
            {"a_au": 45, "e": 0.10, "i": 12, "Omega": 90, "omega": 20, "M": 150},
        ], n_interlopers=150)
    raise ValueError(f"unknown source: {cfg.source!r}")


def _process_one_cone(cfg: NightlyConfig, ra: float, dec: float, radius_deg: float,
                      *, store, run_id: str, cone_idx: int) -> dict:
    """Pull + filter ONE sky cone. Returns per-cone summary; caller aggregates."""
    from .. import realtime
    t0 = time.time()
    print(f"\n-- cone {cone_idx} @ ({ra}, {dec}, r={radius_deg}) --")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alerts = _pull_alerts(cfg, ra, dec, radius_deg)
    print(f"  fetched {len(alerts)} alerts in {time.time()-t0:.1f}s")
    if not alerts:
        return {"alerts": 0, "accepted": 0, "new": 0}

    res = realtime.run_pipeline(
        alerts,
        cluster_pos_tol_arcsec=cfg.cluster_pos_tol_arcsec,
        rate_window_arcsec_hr=cfg.rate_window_arcsec_hr,
        pair_dt_hours=cfg.pair_dt_hours,
        rms_threshold_arcsec=cfg.rms_threshold_arcsec,
        do_xmatch=cfg.do_xmatch and not cfg.dry_run,
        use_helio_linc=cfg.use_helio_linc,
    )
    accepted = [r for r in res if r.get("status") == "accepted"]

    new_alerts = 0
    for tr in accepted:
        ra_deg = math.degrees(tr["ra"])
        dec_deg = math.degrees(tr["dec"])
        rate = float(tr["rate_arcsec_hr"])
        mjd = float(tr["jd"] - 2400000.5)
        rms = float(tr["rms_arcsec"])
        skybot_names = tr.get("xmatch", {}).get("names", [])
        if cfg.dry_run:
            continue
        # Persist epoch alongside state so the follow-up predictor knows when
        # the orbit was anchored (otherwise it falls back to last_seen_mjd).
        t_ref_et = float(tr.get("t", 0.0))
        cand, is_new = store.upsert(
            ra=ra_deg, dec=dec_deg, rate_arcsec_hr=rate, mjd=mjd,
            rms_arcsec=rms,
            orbit_state=(tr.get("x_fit_km", []) + tr.get("v_fit_kms", []))
                        if "x_fit_km" in tr else None,
            skybot_names=skybot_names,
            meta={"survey": "ZTF", "run_id": run_id,
                  "t_ref_et": t_ref_et, "cone_idx": cone_idx},
        )
        if is_new and not skybot_names:
            new_alerts += 1
            fire_all(cfg.alert_sinks, cand, run_id=run_id)

    return {"alerts": len(alerts), "accepted": len(accepted), "new": new_alerts}


def run_nightly(cfg: NightlyConfig) -> dict:
    """Execute one nightly discovery run (one or more sky cones). Returns summary dict."""
    run_id = f"run-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    t0 = time.time()
    cones = cfg.cones if cfg.cones else [(cfg.ra, cfg.dec, cfg.radius_deg)]
    print(f"=== Ariadne nightly run {run_id} ({len(cones)} cone(s)) ===")
    print(f"  source={cfg.source}, mjd=[{cfg.mjd_start}, {cfg.mjd_end}], "
          f"use_helio_linc={cfg.use_helio_linc}")

    store = CandidateStore(cfg.store_path) if not cfg.dry_run else None

    agg = {"alerts": 0, "accepted": 0, "new": 0}
    per_cone = []
    for i, (ra, dec, r) in enumerate(cones):
        s = _process_one_cone(cfg, ra, dec, r, store=store, run_id=run_id, cone_idx=i)
        for k in agg:
            agg[k] += s[k]
        per_cone.append({"cone": (ra, dec, r), **s})

    n_stale = 0
    if not cfg.dry_run and store is not None:
        n_stale = store.mark_stale(max_age_days=cfg.stale_after_days)
        store.save()
        print(f"\n  store: {len(store)} total candidates "
              f"({len(store.discovery_candidates())} active discovery leads), "
              f"{n_stale} marked stale")

    elapsed = time.time() - t0
    print(f"  fired {agg['new']} new-candidate alerts across {len(cones)} cone(s); "
          f"total {elapsed:.0f}s")
    return {"run_id": run_id, "n_cones": len(cones),
            "alerts": agg["alerts"], "accepted": agg["accepted"],
            "new": agg["new"], "stale_marked": n_stale,
            "elapsed_s": elapsed, "per_cone": per_cone}
