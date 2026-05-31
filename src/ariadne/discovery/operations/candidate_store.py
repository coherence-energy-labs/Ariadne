"""Candidate persistence -- JSON-backed store with dedupe + history tracking.

Each Candidate has a stable canonical key (rounded RA/Dec/rate so the same
object surfaces under the same key across nights). The store tracks first-seen
MJD, last-seen MJD, number of independent runs that surfaced it, and RMS
history. Used by the nightly orchestrator to dedupe re-detections and only
fire an alert the FIRST time something surfaces (or when something previously
absent reappears).

JSON-on-disk is intentionally low-tech: no DB dependency, easy to grep/inspect,
easy to back up. Atomic writes via tmp-file + rename.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator


def _canonical_key(ra_deg: float, dec_deg: float, rate_arcsec_hr: float,
                   pos_round_arcmin: float = 5.0, rate_round_arcsec_hr: float = 0.5) -> str:
    """Round position + rate to make a stable key under small re-observation drift.

    Default tolerances: 5 arcmin for sky position (a candidate that re-appears
    within ~5' counts as the same object); 0.5 arcsec/hr for rate.
    """
    ra_r = round(ra_deg * 60.0 / pos_round_arcmin) * pos_round_arcmin / 60.0
    dec_r = round(dec_deg * 60.0 / pos_round_arcmin) * pos_round_arcmin / 60.0
    rate_r = round(rate_arcsec_hr / rate_round_arcsec_hr) * rate_round_arcsec_hr
    return f"{ra_r:08.4f}_{dec_r:+08.4f}_{rate_r:05.2f}"


@dataclass
class Candidate:
    """One candidate orbit in the store.

    Fields:
      key:            canonical position/rate key (stable across re-observations)
      ra, dec:        latest reported position (degrees)
      rate_arcsec_hr: latest reported rate (arcsec / hour)
      first_seen_mjd: MJD of the first time we surfaced this candidate
      last_seen_mjd:  MJD of the most recent surfacing
      n_runs:         how many independent nightly runs have surfaced this
      rms_history:    list of (mjd, rms_arcsec) for each surfacing
      orbit_state:    latest 6D heliocentric state [x,y,z,vx,vy,vz] km/(km/s)
      skybot_names:   names of nearby known objects (empty if truly unmatched)
      status:         "new" | "active" | "stale" | "rejected"
      meta:           free-form bag of extra fields (last_survey, alert_id, ...)
    """
    key: str
    ra: float
    dec: float
    rate_arcsec_hr: float
    first_seen_mjd: float
    last_seen_mjd: float
    n_runs: int = 1
    rms_history: list = field(default_factory=list)
    orbit_state: list | None = None
    skybot_names: list = field(default_factory=list)
    status: str = "new"
    meta: dict = field(default_factory=dict)


class CandidateStore:
    """JSON-backed store of candidate orbits with dedupe + history."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._cands: dict[str, Candidate] = {}
        if self.path.exists():
            self._load()

    def _load(self):
        with open(self.path, encoding="utf-8") as f:
            raw = json.load(f)
        for rec in raw.get("candidates", []):
            c = Candidate(**rec)
            self._cands[c.key] = c

    def save(self):
        """Atomic write: dump to tmp, then rename."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = {
            "version": 1,
            "saved_at_unix": time.time(),
            "n_candidates": len(self._cands),
            "candidates": [asdict(c) for c in self._cands.values()],
        }
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, self.path)

    def __len__(self) -> int:
        return len(self._cands)

    def __iter__(self) -> Iterator[Candidate]:
        return iter(self._cands.values())

    def __contains__(self, key: str) -> bool:
        return key in self._cands

    def get(self, key: str) -> Candidate | None:
        return self._cands.get(key)

    def upsert(self, *, ra: float, dec: float, rate_arcsec_hr: float, mjd: float,
               rms_arcsec: float | None = None, orbit_state: list | None = None,
               skybot_names: list | None = None, meta: dict | None = None
               ) -> tuple[Candidate, bool]:
        """Insert or update a candidate. Returns (candidate, is_new).

        is_new = True ONLY on first sighting; subsequent re-detections update
        last_seen_mjd, append to rms_history, bump n_runs, and return False.
        Use is_new to decide whether to fire an alert.
        """
        key = _canonical_key(ra, dec, rate_arcsec_hr)
        if key in self._cands:
            c = self._cands[key]
            c.ra = ra; c.dec = dec; c.rate_arcsec_hr = rate_arcsec_hr
            c.last_seen_mjd = mjd
            c.n_runs += 1
            if rms_arcsec is not None:
                c.rms_history.append([float(mjd), float(rms_arcsec)])
            if orbit_state is not None:
                c.orbit_state = list(orbit_state)
            if skybot_names:
                # Refresh: re-detection may add/change known matches
                c.skybot_names = list(skybot_names)
            if meta:
                c.meta.update(meta)
            # Status transition: any re-detection counts as 'active'
            if c.status == "new":
                c.status = "active"
            return c, False
        c = Candidate(
            key=key, ra=ra, dec=dec, rate_arcsec_hr=rate_arcsec_hr,
            first_seen_mjd=mjd, last_seen_mjd=mjd,
            rms_history=[[float(mjd), float(rms_arcsec)]] if rms_arcsec is not None else [],
            orbit_state=list(orbit_state) if orbit_state is not None else None,
            skybot_names=list(skybot_names) if skybot_names else [],
            meta=dict(meta) if meta else {},
        )
        self._cands[key] = c
        return c, True

    def mark_stale(self, max_age_days: float = 60.0, current_mjd: float | None = None):
        """Mark candidates not seen in `max_age_days` as 'stale'. Returns count marked."""
        import time as _time
        if current_mjd is None:
            current_mjd = _time.time() / 86400.0 + 40587.0   # unix -> MJD
        n = 0
        for c in self._cands.values():
            if c.status in ("active", "new") and (current_mjd - c.last_seen_mjd) > max_age_days:
                c.status = "stale"; n += 1
        return n

    def discovery_candidates(self) -> list[Candidate]:
        """Candidates with no SkyBoT match, status active or new -- the leads."""
        return [c for c in self._cands.values()
                if c.status in ("new", "active")
                and not c.skybot_names]
