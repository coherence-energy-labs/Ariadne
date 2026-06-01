"""Photometric chain identifier via lightcurve coherence.

The standard chain linkers in `advanced_linking.py` score candidate
pairs on POSITION + RATE. Photometry is included as a soft term but
isn't used as the PRIMARY identifier. That's fine when stars are
sparse — position alone disambiguates — but in a crowded field with
~80 background sources per image, the photometric SIGNATURE of an
asteroid (its magnitude and any rotational variability) is a powerful
secondary identifier that current linkers ignore.

This module adds:

  * `photometric_chain_score`
        Score a candidate chain on its photometric coherence. Higher
        means more likely a true single-object chain.

  * `lightcurve_features`
        Extract feature vector from a chain's magnitude time series
        (median, std, rolling-std, period if detectable).

  * `match_chains_photometrically`
        Cluster a set of chains into groups likely to be the same
        object based on their lightcurve features. Useful for
        identifying duplicate chains from different linkers.

  * `correlate_lightcurves`
        Cross-correlate two chains' lightcurves to estimate whether
        they could be the same object with a time-shifted rotation
        period.

These complement the position-based linkers, especially for slow
movers (TNOs, distant MBAs) where the position changes slowly across
nights but the photometric signature should be steady.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class LightcurveFeatures:
    """Extracted photometric features for one chain."""
    n_observations: int
    mag_median: float
    mag_std: float            # std across observations
    mag_range: float          # p95 - p5
    arc_hours: float
    n_outliers: int           # observations > 3-sigma from median
    smoothness: float         # 0..1, higher means more constant brightness
    period_hours: float       # estimated rotation period (0 if not found)
    period_significance: float   # 0..1 strength of the periodic signal
    notes: str = ""


def _extract_mags_and_times(chain: Sequence[dict]) -> tuple[list[float], list[float]]:
    """Return (mags, times_hours) tuples, sorted by time."""
    rows = []
    for e in chain:
        m = float(e.get("mag", -99.0))
        if m > -50:
            rows.append((float(e["t"]) / 3600.0, m))
            continue
        # Try source_pair members
        pair = e.get("source_pair") or ()
        for s in pair:
            m_s = getattr(s, "mag", None)
            if m_s is not None and m_s > -50:
                rows.append((float(e["t"]) / 3600.0, float(m_s)))
                break
    rows.sort()
    if not rows:
        return [], []
    times = [r[0] for r in rows]
    mags = [r[1] for r in rows]
    return mags, times


def lightcurve_features(chain: Sequence[dict]) -> LightcurveFeatures:
    """Extract photometric features from a chain's magnitude time series."""
    mags, times = _extract_mags_and_times(chain)
    if not mags:
        return LightcurveFeatures(
            n_observations=0, mag_median=0.0, mag_std=0.0,
            mag_range=0.0, arc_hours=0.0,
            n_outliers=0, smoothness=0.0,
            period_hours=0.0, period_significance=0.0,
            notes="no usable photometry")
    n = len(mags)
    mag_med = float(np.median(mags))
    mag_std = float(np.std(mags)) if n >= 2 else 0.0
    mag_range = float(np.percentile(mags, 95) - np.percentile(mags, 5))
    arc_hours = (max(times) - min(times)) if n >= 2 else 0.0
    # Outliers: > 3-sigma from median
    mad = float(np.median(np.abs(np.array(mags) - mag_med)))
    sigma_robust = mad * 1.4826
    n_outliers = (int(np.sum(np.abs(np.array(mags) - mag_med)
                              > 3 * sigma_robust))
                   if sigma_robust > 0 else 0)
    # Smoothness: 1 if mag_std small relative to typical photometric noise
    typical_phot_noise = 0.05   # mag
    smoothness = max(0.0, 1.0 - mag_std / (5 * typical_phot_noise))
    # Periodogram for rotational period (only if we have enough points)
    period_hours, period_sig = _detect_period(times, mags) if n >= 4 else (0.0, 0.0)
    return LightcurveFeatures(
        n_observations=n, mag_median=mag_med, mag_std=mag_std,
        mag_range=mag_range, arc_hours=arc_hours,
        n_outliers=n_outliers, smoothness=smoothness,
        period_hours=period_hours, period_significance=period_sig,
        notes=f"{n} mags, MAD-sigma {sigma_robust:.3f}")


def _detect_period(times_hours: Sequence[float],
                     mags: Sequence[float],
                     *, p_min: float = 1.0, p_max: float = 100.0,
                     n_periods: int = 200) -> tuple[float, float]:
    """Simple Lomb-Scargle-like periodogram for rotational period.

    Returns (period_hours, significance). Significance is the ratio of
    the peak power to the median power. > 5 typically means a real
    periodic signal.
    """
    if len(times_hours) < 4:
        return 0.0, 0.0
    t = np.array(times_hours) - np.min(times_hours)
    m = np.array(mags) - np.mean(mags)
    periods = np.linspace(p_min, p_max, n_periods)
    powers = []
    for P in periods:
        omega = 2.0 * math.pi / P
        # Lomb-Scargle simplified: sin/cos projection
        c = np.cos(omega * t); s = np.sin(omega * t)
        cc = float(np.sum(c * c)); ss = float(np.sum(s * s)); cs = float(np.sum(c * s))
        ym_c = float(np.sum(m * c)); ym_s = float(np.sum(m * s))
        # Standard LS power formula
        if cc * ss - cs ** 2 < 1e-10:
            powers.append(0.0)
            continue
        det = cc * ss - cs ** 2
        power = (ss * ym_c ** 2 - 2 * cs * ym_c * ym_s + cc * ym_s ** 2) / det
        powers.append(power)
    powers = np.array(powers)
    peak_idx = int(np.argmax(powers))
    peak_power = float(powers[peak_idx])
    median_power = float(np.median(powers))
    if median_power <= 0 or peak_power <= 0:
        return 0.0, 0.0
    return float(periods[peak_idx]), peak_power / median_power


def photometric_chain_score(chain: Sequence[dict],
                              *, max_mag_std_expected: float = 0.3,
                              ) -> float:
    """Return a 0..1 photometric coherence score.

    Higher = more likely a true single-object chain. Components:
      * mag_std penalty: penalises chains where mag_std > expected
      * smoothness: high when brightness is stable
      * outlier penalty: each 3-sigma outlier knocks 0.1 off
    """
    feat = lightcurve_features(chain)
    if feat.n_observations < 2:
        return 0.5    # neutral when no info
    std_score = max(0.0, 1.0 - feat.mag_std / max(max_mag_std_expected, 0.01))
    outlier_score = max(0.0, 1.0 - 0.1 * feat.n_outliers)
    return float(0.5 * std_score + 0.3 * feat.smoothness + 0.2 * outlier_score)


def correlate_lightcurves(chain_a: Sequence[dict],
                            chain_b: Sequence[dict]) -> float:
    """Return a 0..1 similarity score between two chains' lightcurves.

    Uses the brightness AND any detected periodic signal. Two chains
    of the same physical object should score near 1.0; chains of
    different objects should score near 0.
    """
    a = lightcurve_features(chain_a)
    b = lightcurve_features(chain_b)
    if a.n_observations < 2 or b.n_observations < 2:
        return 0.5    # not enough info
    # Magnitude similarity (Gaussian in mag space)
    dmag = abs(a.mag_median - b.mag_median)
    mag_sim = math.exp(-0.5 * (dmag / 0.3) ** 2)
    # Period similarity (if both have a detected period)
    if a.period_significance > 5 and b.period_significance > 5:
        d_period = abs(a.period_hours - b.period_hours) / max(
            a.period_hours, b.period_hours, 1.0)
        period_sim = math.exp(-0.5 * (d_period / 0.1) ** 2)
    else:
        period_sim = 0.5
    # Smoothness similarity
    smoothness_sim = math.exp(-0.5 * ((a.smoothness - b.smoothness) / 0.3) ** 2)
    return float(0.6 * mag_sim + 0.2 * period_sim + 0.2 * smoothness_sim)


def match_chains_photometrically(chains: Sequence[Sequence[dict]],
                                    *, similarity_threshold: float = 0.7
                                    ) -> dict[int, list[int]]:
    """Group chain indices by photometric similarity.

    Returns {representative_idx: [member_idx, ...]} where every member
    has correlate_lightcurves >= similarity_threshold with the
    representative. Useful for collapsing chains that came from
    different linker strategies but represent the same object.
    """
    n = len(chains)
    visited = set()
    groups: dict[int, list[int]] = {}
    for i in range(n):
        if i in visited:
            continue
        groups[i] = [i]
        visited.add(i)
        for j in range(i + 1, n):
            if j in visited:
                continue
            sim = correlate_lightcurves(chains[i], chains[j])
            if sim >= similarity_threshold:
                groups[i].append(j)
                visited.add(j)
    return groups
