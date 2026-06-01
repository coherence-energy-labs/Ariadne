"""DB chain <-> IOD ensemble adapter.

The persistent detection_db schema uses (ra, dec) in degrees and MJD
times. The IOD ensemble (iod_advanced / iod_robust / iod_bayesian)
expects chain entries in radians + SPICE ET seconds with extra fields
(dra, ddec, rate_arcsec_hr) the linker historically attaches.

This module bridges the two: pull a chain from the DB, build an IOD-
shaped chain, run the IOD ensemble, store the result back in the DB
chain row.

Public API:
  load_chain_for_iod(db, chain_id) -> list[dict]
    Fetch chain detections from DB, format as IOD chain entries.
  run_iod_on_chain(db, chain_id, ...) -> EnsembleFit
    Load + run robust_iod + persist the (x, v, rms, strategy) back to
    the DB chain row. Returns the EnsembleFit.
  run_iod_on_all_open_chains(db, ...) -> list[(chain_id, EnsembleFit)]
    Bulk: run IOD on every open chain with >= min_tracklets.
"""
from __future__ import annotations

import math
import time
from typing import Sequence

import numpy as np


SEC_PER_DAY = 86400.0
J2000_MJD = 51544.5


def _mjd_to_spice_et(mjd: float) -> float:
    return (mjd - J2000_MJD) * SEC_PER_DAY


def load_chain_for_iod(db, chain_id: int) -> list[dict]:
    """Pull `chain_id`'s detections from the DB and return them in the
    chain-entry format the IOD ensemble expects:
        {t, jd, ra (rad), dec (rad), dra, ddec, rate_arcsec_hr, mag,
         source_pair: (), night}
    """
    dets = db.get_chain_detections(chain_id)
    if not dets:
        return []
    tracklets = db.get_chain_tracklets(chain_id)
    # Build rate lookup by (mean_mjd) for the chain's tracklets so each
    # detection inherits the tracklet's rate value.
    rate_by_mjd = []
    for tr in tracklets:
        rate_by_mjd.append((float(tr["mean_mjd"]),
                              float(tr["rate_arcsec_hr"] or 0.0)))

    def _rate_at(mjd: float) -> float:
        if not rate_by_mjd:
            return 0.0
        # Pick the tracklet whose mean_mjd is closest to this detection
        return min(rate_by_mjd, key=lambda p: abs(p[0] - mjd))[1]

    out = []
    for d in dets:
        mjd = float(d["mjd"])
        out.append({
            "t": _mjd_to_spice_et(mjd),
            "jd": mjd + 2400000.5,
            "ra": math.radians(float(d["ra"])),
            "dec": math.radians(float(d["dec"])),
            "dra": 1e-9,
            "ddec": 1e-9,
            "rate_arcsec_hr": _rate_at(mjd),
            "mag": float(d.get("mag") or -99.0),
            "source_pair": (),
            "night": int(mjd),
        })
    out.sort(key=lambda e: e["t"])
    return out


def run_iod_on_chain(db, chain_id: int,
                       *,
                       rms_acceptance_arcsec: float = 5.0,
                       n_draws: int = 5,
                       use_monte_carlo: bool = True,
                       neural_weights: dict | None = None,
                       persist: bool = True,
                       ) -> dict:
    """Load a chain from the DB, run robust_iod, optionally persist the
    fit back to the chain row.

    Returns a dict with fields:
      success, strategy, rms_arcsec, x_fit, v_fit, t_ref, n_observations,
      wall_seconds.
    """
    from ..iod_robust import robust_iod
    t0 = time.time()
    chain_entries = load_chain_for_iod(db, chain_id)
    if not chain_entries or len(chain_entries) < 3:
        return {
            "success": False,
            "reason": (f"insufficient observations ({len(chain_entries)})"),
            "wall_seconds": time.time() - t0,
        }
    try:
        ens = robust_iod(
            chain_entries,
            rms_acceptance_arcsec=rms_acceptance_arcsec,
            n_draws=n_draws,
            use_monte_carlo=use_monte_carlo,
            neural_weights=neural_weights)
    except Exception as exc:
        return {
            "success": False,
            "reason": f"exception: {exc!s}"[:200],
            "wall_seconds": time.time() - t0,
        }
    out = {
        "success": bool(ens.success),
        "strategy": ens.winning_strategy,
        "rms_arcsec": float(ens.rms_arcsec),
        "x_fit": [float(v) for v in ens.x_fit],
        "v_fit": [float(v) for v in ens.v_fit],
        "t_ref": float(ens.t_ref),
        "n_observations": len(chain_entries),
        "wall_seconds": time.time() - t0,
    }
    if persist and ens.success:
        db.update_chain(chain_id, {
            "iod_strategy": ens.winning_strategy,
            "iod_rms_arcsec": float(ens.rms_arcsec),
            "iod_x_km_x": float(ens.x_fit[0]),
            "iod_x_km_y": float(ens.x_fit[1]),
            "iod_x_km_z": float(ens.x_fit[2]),
            "iod_v_kms_x": float(ens.v_fit[0]),
            "iod_v_kms_y": float(ens.v_fit[1]),
            "iod_v_kms_z": float(ens.v_fit[2]),
            "iod_t_ref": float(ens.t_ref),
        })
    return out


def run_iod_on_all_open_chains(db,
                                  *,
                                  min_tracklets: int = 3,
                                  rms_acceptance_arcsec: float = 5.0,
                                  n_draws: int = 5,
                                  max_chains: int = 500,
                                  neural_weights: dict | None = None,
                                  ) -> list[tuple[int, dict]]:
    """Run IOD on every open chain with >= `min_tracklets`.

    Returns list of (chain_id, iod_result_dict).
    """
    open_chains = db.query_open_chains(max_chains=max_chains)
    candidates = [c for c in open_chains
                    if int(c.get("n_tracklets") or 0) >= min_tracklets
                    and not c.get("iod_strategy")]
    results = []
    for chain in candidates:
        cid = int(chain["id"])
        res = run_iod_on_chain(
            db, cid,
            rms_acceptance_arcsec=rms_acceptance_arcsec,
            n_draws=n_draws,
            neural_weights=neural_weights)
        results.append((cid, res))
    return results
