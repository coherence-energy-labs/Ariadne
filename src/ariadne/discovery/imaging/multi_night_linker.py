"""Multi-night linking against the persistent detection database.

This is the operational counterpart to the in-memory chain linker:
when tonight's exposures land, we don't just look at tonight's
tracklets in isolation. We:

  1. Query the DB for OPEN chains whose last_mjd is within `link_window_days`
  2. Predict where each open chain SHOULD be tonight via its rate vector
  3. Cross-match tonight's tracklets against those predictions
  4. Extend matching chains; for non-matching tonight-tracklets, attempt
     to seed NEW chains by cross-matching with DB tracklets from the past
     `seed_window_days`

This implements the MOPS-style "tracklet attribution" step on a budget
that's bounded by the time window (default 30 nights).

Public API:
  link_tonight(db, tonight_tracklets, ...)
    Returns LinkingReport with counts + IDs of extended / seeded chains.

  predict_chain_at_mjd(chain_dict, target_mjd) -> (ra, dec)
    Linear extrapolation along the chain's mean rate vector.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .detection_db import (
    DetectionDB, ChainRow, TrackletRow)


@dataclass
class LinkingReport:
    """Outcome of one night's linking pass."""
    n_tonight_tracklets: int = 0
    n_open_chains_considered: int = 0
    n_chains_extended: int = 0
    n_chains_seeded: int = 0
    n_tracklets_unmatched: int = 0
    chain_extensions: list[tuple[int, int]] = field(default_factory=list)
    chain_seeds: list[int] = field(default_factory=list)
    wall_seconds: float = 0.0


def predict_chain_at_mjd(chain: dict, target_mjd: float) -> tuple[float, float]:
    """Linear extrapolation of a chain's predicted (ra, dec) at target_mjd.

    Uses `chain['mean_rate_arcsec_hr']` and `chain['mean_pa_deg']` and
    the last known (mean_ra, mean_dec) at `chain['last_mjd']`. Good
    enough for short forward extrapolation (a few nights); for longer
    extrapolation, the IOD-derived state vector should be used.
    """
    last_mjd = float(chain.get("last_mjd") or 0.0)
    dt_hr = (target_mjd - last_mjd) * 24.0
    rate_arcsec_hr = float(chain.get("mean_rate_arcsec_hr") or 0.0)
    pa_rad = math.radians(float(chain.get("mean_pa_deg") or 0.0))
    motion_arcsec = rate_arcsec_hr * dt_hr
    cos_dec = max(math.cos(math.radians(
        float(chain.get("mean_dec") or 0.0))), 1e-6)
    dra_deg = motion_arcsec * math.sin(pa_rad) / 3600.0 / cos_dec
    ddec_deg = motion_arcsec * math.cos(pa_rad) / 3600.0
    return (float(chain.get("mean_ra") or 0.0) + dra_deg,
            float(chain.get("mean_dec") or 0.0) + ddec_deg)


def _angular_distance_deg(ra1: float, dec1: float,
                            ra2: float, dec2: float) -> float:
    cos_dec = math.cos(math.radians(0.5 * (dec1 + dec2)))
    dra = (ra1 - ra2) * cos_dec
    ddec = dec1 - dec2
    return math.hypot(dra, ddec)


def link_tonight(db: DetectionDB,
                   tonight_tracklets: Sequence[dict],
                   *,
                   link_window_days: float = 30.0,
                   seed_window_days: float = 6.0,
                   position_tol_arcsec: float = 60.0,
                   rate_tol_pct: float = 50.0,
                   min_arc_hours_to_seed: float = 1.0,
                   ) -> LinkingReport:
    """Attribute tonight's tracklets to DB chains; seed new chains from
    unmatched ones that pair with past tracklets.

    `tonight_tracklets`: list of dicts with keys
      tracklet_id, mean_mjd, mean_ra, mean_dec, rate_arcsec_hr, pa_deg

    `link_window_days`: how far back to look for OPEN chains.
    `seed_window_days`: how far back to look for tracklets that could
        pair with tonight's to seed a new chain.

    Returns LinkingReport with counts.
    """
    t0 = time.time()
    report = LinkingReport(n_tonight_tracklets=len(tonight_tracklets))
    if not tonight_tracklets:
        return report
    tonight_mjd = float(max(t["mean_mjd"] for t in tonight_tracklets))
    # ---------------------------------------------------------------
    # 1. Attempt to attribute each tonight-tracklet to an OPEN chain
    # ---------------------------------------------------------------
    open_chains = db.query_open_chains(min_arc_hours=0,
                                          max_chains=50000)
    open_chains = [c for c in open_chains
                     if (c.get("last_mjd") or 0.0) >= tonight_mjd - link_window_days
                     and (c.get("last_mjd") or 0.0) < tonight_mjd]
    report.n_open_chains_considered = len(open_chains)

    pos_tol_deg = position_tol_arcsec / 3600.0
    used_tonight = set()
    for chain in open_chains:
        pred_ra, pred_dec = predict_chain_at_mjd(chain, tonight_mjd)
        chain_rate = float(chain.get("mean_rate_arcsec_hr") or 0.0)
        # Look for tonight's tracklet within the position+rate window
        best_idx, best_d = -1, pos_tol_deg
        for j, t in enumerate(tonight_tracklets):
            if j in used_tonight:
                continue
            d = _angular_distance_deg(t["mean_ra"], t["mean_dec"],
                                          pred_ra, pred_dec)
            if d > best_d:
                continue
            # Rate-coherence check
            t_rate = float(t.get("rate_arcsec_hr") or 0.0)
            if chain_rate > 0 and abs(t_rate - chain_rate) / chain_rate > rate_tol_pct / 100.0:
                continue
            best_d = d
            best_idx = j
        if best_idx >= 0:
            t = tonight_tracklets[best_idx]
            used_tonight.add(best_idx)
            chain_id = int(chain["id"])
            tracklet_id = int(t["tracklet_id"])
            db.attach_tracklet_to_chain(tracklet_id, chain_id)
            # Recompute chain rollups
            tracks = db.get_chain_tracklets(chain_id)
            if tracks:
                first_mjd = min(tr["mean_mjd"] for tr in tracks)
                last_mjd = max(tr["mean_mjd"] for tr in tracks)
                arc_days = last_mjd - first_mjd
                mean_rate = sum(tr["rate_arcsec_hr"] for tr in tracks) / len(tracks)
                db.update_chain(chain_id, {
                    "n_tracklets": len(tracks),
                    "first_mjd": first_mjd,
                    "last_mjd": last_mjd,
                    "arc_days": arc_days,
                    "mean_rate_arcsec_hr": mean_rate,
                    "mean_ra": t["mean_ra"],
                    "mean_dec": t["mean_dec"],
                })
            report.n_chains_extended += 1
            report.chain_extensions.append((chain_id, tracklet_id))

    # ---------------------------------------------------------------
    # 2. For unmatched tonight-tracklets, try to SEED a new chain by
    #    pairing with DB tracklets in the seed_window_days BEFORE tonight
    # ---------------------------------------------------------------
    for j, t in enumerate(tonight_tracklets):
        if j in used_tonight:
            continue
        cos_dec = math.cos(math.radians(float(t["mean_dec"])))
        ra_box = (t["mean_ra"] - 0.5 / cos_dec, t["mean_ra"] + 0.5 / cos_dec)
        dec_box = (t["mean_dec"] - 0.5, t["mean_dec"] + 0.5)
        # Look back seed_window_days
        prev = db.query_tracklets_by_window(
            t["mean_mjd"] - seed_window_days,
            t["mean_mjd"] - 1.0 / 24.0,    # must be a different night
            ra_range=ra_box, dec_range=dec_box, limit=200)
        # Filter to those NOT already part of a chain (or with same rate band)
        for p in prev:
            if p.get("chain_id"):
                continue
            # Rate-coherence + position-trajectory check
            t_rate = float(t.get("rate_arcsec_hr") or 0.0)
            p_rate = float(p["rate_arcsec_hr"] or 0.0)
            if p_rate <= 0 or abs(t_rate - p_rate) / max(p_rate, 1e-6) > rate_tol_pct / 100.0:
                continue
            # Verify that t's position is roughly consistent with p's rate
            dt_hr = (t["mean_mjd"] - p["mean_mjd"]) * 24.0
            if dt_hr <= 0:
                continue
            pred_ra, pred_dec = (
                p["mean_ra"]
                  + p_rate * dt_hr * math.sin(math.radians(p["pa_deg"]))
                  / 3600.0 / cos_dec,
                p["mean_dec"]
                  + p_rate * dt_hr * math.cos(math.radians(p["pa_deg"])) / 3600.0)
            offset = _angular_distance_deg(
                t["mean_ra"], t["mean_dec"], pred_ra, pred_dec)
            if offset > pos_tol_deg * 2:
                continue
            # Found a valid seed pair; create a chain
            mean_rate = 0.5 * (t_rate + p_rate)
            mean_pa = float(p["pa_deg"])
            arc_days = (t["mean_mjd"] - p["mean_mjd"])
            if arc_days * 24.0 < min_arc_hours_to_seed:
                continue
            chain = ChainRow(
                n_tracklets=2, n_detections=4,
                first_mjd=p["mean_mjd"], last_mjd=t["mean_mjd"],
                arc_days=arc_days,
                mean_ra=t["mean_ra"], mean_dec=t["mean_dec"],
                mean_rate_arcsec_hr=mean_rate,
                mean_pa_deg=mean_pa,
                status="open",
            )
            chain_id = db.insert_chain(chain)
            db.attach_tracklet_to_chain(p["id"], chain_id)
            db.attach_tracklet_to_chain(int(t["tracklet_id"]), chain_id)
            used_tonight.add(j)
            report.n_chains_seeded += 1
            report.chain_seeds.append(chain_id)
            break

    report.n_tracklets_unmatched = (
        report.n_tonight_tracklets - report.n_chains_extended
        - report.n_chains_seeded)
    report.wall_seconds = time.time() - t0
    return report
