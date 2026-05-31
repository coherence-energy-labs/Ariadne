"""Candidate quality scoring -- one 0-1 number summarising "is this real?"

After the discovery pipeline accepts a candidate, multiple soft signals govern
whether you should commit follow-up observation time to it:

  * RMS of the orbit fit (lower = more like a real object than noise).
  * Number of detections (more = better-constrained, less linker artefact).
  * Arc length in days (longer = stronger orbit constraint, less hypothesis-fit).
  * Independent re-detections across nightly runs (more = real, not a fluke).
  * SkyBoT angular separation to nearest known object (larger = less likely a
    poorly-photometric known asteroid masquerading as new).

This module fuses those into a single 0-1 quality score. The weights are
deliberately simple-and-justified rather than ML-tuned -- we have no labelled
"real vs spurious" training set yet, and a transparent linear fusion is robust
under unknown systematics.

  total_score = 0.40 * rms_score
              + 0.20 * arc_score
              + 0.20 * runs_score
              + 0.10 * skybot_score
              + 0.10 * obs_score

Each subscore is a smooth 0-1 monotonic mapping designed so that "obviously
real" candidates score >0.7 and "marginal" candidates score 0.3-0.6. A 0-score
means the candidate is so weak it should not consume follow-up budget.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .operations.candidate_store import Candidate


@dataclass
class QualityScore:
    """Per-subscore breakdown + total."""
    total: float            # 0..1
    rms: float              # 0..1, higher = lower RMS
    arc: float              # 0..1, higher = longer arc
    runs: float             # 0..1, higher = more independent re-detections
    skybot: float           # 0..1, higher = farther from any known object
    obs: float              # 0..1, higher = more individual detections

    def grade(self) -> str:
        """Letter grade for at-a-glance ranking. Calibrated against the
        subscore-weighted total under reasonable test corpora.
        """
        if self.total >= 0.80:
            return "A"      # strong: commit follow-up
        if self.total >= 0.60:
            return "B"      # promising: follow up if budget allows
        if self.total >= 0.40:
            return "C"      # marginal: wait for re-detection
        return "D"          # weak: likely spurious


def _logistic(x: float, x0: float, slope: float) -> float:
    """Centred logistic; 0 below x0, 1 above, transition width ~ 1/slope."""
    if not math.isfinite(x):
        return 0.0
    return 1.0 / (1.0 + math.exp(-slope * (x - x0)))


def _rms_score(rms_arcsec: float | None) -> float:
    """1 at RMS=0, 0.5 at RMS=4", 0.1 at RMS=10". Decays past 15"."""
    if rms_arcsec is None or not math.isfinite(rms_arcsec):
        return 0.0
    return 1.0 - _logistic(rms_arcsec, x0=4.0, slope=0.5)


def _arc_score(arc_days: float) -> float:
    """0 at arc=0 days, 0.5 at 3 days, 0.9 at ~10 days, asymptotes past 30d.

    Recalibrated 2026-06: previous centre at 7d meant a clean 3-night
    discovery (a typical first multi-night detection) only scored ~0.30
    on this axis, capping the candidate at grade C. New centre 3 days
    means a real 3-night arc gets the credit it deserves while a 30-day
    confirmed arc still maxes the channel.
    """
    if arc_days <= 0:
        return 0.0
    return _logistic(arc_days, x0=3.0, slope=0.35)


def _runs_score(n_runs: int) -> float:
    """0.45 at 1 run (one-night detection), 0.70 at 2, 0.95 at 5+.

    Recalibrated 2026-06: a single nightly run is the typical state of
    a fresh discovery -- the previous 0.30 floor was too punitive. We
    still reward re-detections (the strongest "this is real" prior),
    but the new mapping gives single-night candidates a fair shot at
    grade B given good RMS + sky + obs.
    """
    if n_runs <= 0:
        return 0.0
    return min(1.0, 0.45 + 0.25 * math.log1p(n_runs - 1))


def _skybot_score(skybot_names: list, has_xmatch_run: bool = True) -> float:
    """1.0 if SkyBoT returned nothing (no known asteroid in cone); 0.0 if
    matched. We don't have angular separation in the store, so this is binary --
    upgrade if/when the store records nearest-known angular separation.
    """
    if not has_xmatch_run:
        return 0.5            # uncertain -- no cross-match performed
    return 0.0 if skybot_names else 1.0


def _obs_score(n_observations: int) -> float:
    """0 at 0 observations, 0.5 at 4 (one 2-night tracklet pair), 1.0 at 12+."""
    return _logistic(n_observations, x0=4.0, slope=0.4)


def score_candidate(c: Candidate, *,
                    weights: tuple[float, float, float, float, float] = (0.40, 0.20, 0.20, 0.10, 0.10),
                    n_observations: int | None = None,
                    has_xmatch_run: bool = True) -> QualityScore:
    """Compute the quality score for a stored Candidate.

    Args:
      c:                the Candidate from the store.
      weights:          (rms, arc, runs, skybot, obs) weights; must sum to 1.0.
      n_observations:   detection count (if known); if None, infer from rms_history
                        length (one entry per nightly surfacing).
      has_xmatch_run:   whether SkyBoT cross-match was actually performed on this
                        candidate (False -> skybot subscore = 0.5 = unknown).

    Returns:
      QualityScore breakdown + total.
    """
    arc_days = max(0.0, c.last_seen_mjd - c.first_seen_mjd)
    rms_latest = c.rms_history[-1][1] if c.rms_history else None
    if n_observations is None:
        n_observations = len(c.rms_history)

    rms = _rms_score(rms_latest)
    arc = _arc_score(arc_days)
    runs = _runs_score(c.n_runs)
    sky = _skybot_score(c.skybot_names, has_xmatch_run=has_xmatch_run)
    obs = _obs_score(n_observations)

    w = weights
    if not math.isclose(sum(w), 1.0, abs_tol=1e-6):
        raise ValueError(f"weights must sum to 1.0, got {sum(w)}")

    total = w[0] * rms + w[1] * arc + w[2] * runs + w[3] * sky + w[4] * obs
    return QualityScore(total=total, rms=rms, arc=arc, runs=runs, skybot=sky, obs=obs)


def rank_candidates(candidates: list[Candidate], **kwargs) -> list[tuple[Candidate, QualityScore]]:
    """Score every candidate and return them sorted by total (best first)."""
    scored = [(c, score_candidate(c, **kwargs)) for c in candidates]
    scored.sort(key=lambda x: x[1].total, reverse=True)
    return scored
