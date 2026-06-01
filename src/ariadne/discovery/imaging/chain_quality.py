"""Chain-quality filters for the image-pipeline.

The advanced linkers in `advanced_linking.py` produce chains by spatial+
temporal matching of tracklets across nights. They do well on yield (32
chains from 3 truth objects in the standard DECam synth test) but the
PURITY is low: most of those chains glue together different objects.

This module filters chains on PHYSICAL and STATISTICAL coherence cues
that should hold for a true single-object chain but fail for false
linkages:

  * `rate_coherence_filter`
        Within a real chain, the within-night rate should be
        approximately constant (the object moves smoothly). If the rate
        varies by more than `max_rate_spread` x the median, the chain
        is mixing different objects.

  * `photometric_coherence_filter`
        Same magnitude across all entries of a true chain (the same
        physical object). If `mag_std > max_mag_std`, the chain is
        mixing different brightnesses, ie different objects.

  * `epoch_coverage_filter`
        Require >= `min_unique_epochs` distinct observation epochs
        before allowing the chain to attempt IOD. Two-epoch chains
        are short-arc by definition and Gauss-class IOD fails on them
        regardless of strategy.

  * `arclength_filter`
        Require >= `min_arc_hours` time span between the chain's
        earliest and latest detection. Very short arcs (< 6 hours)
        rarely yield converged orbits.

  * `chain_purity_score`
        Composite 0..1 score derived from rate + photometric + epoch +
        arc tests. Useful as input to a Bayesian linker that wants a
        scalar quality.

  * `filter_chains`
        Convenience: apply all four filters in sequence and return
        (kept, dropped) lists plus a per-chain diagnostic.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ChainQualityVerdict:
    """Per-chain result of the quality battery."""
    chain_idx: int
    n_entries: int
    n_unique_epochs: int
    arc_hours: float
    rate_median: float
    rate_spread_rel: float
    mag_median: float
    mag_std: float
    purity_score: float
    passes_rate: bool
    passes_photometric: bool
    passes_epoch: bool
    passes_arc: bool
    passes_all: bool
    reasons: tuple[str, ...]


def _epoch_key(entry: dict) -> int:
    """Coerce a chain-entry 't' into a discrete epoch day. The image-pipeline
    uses SPICE-ET seconds, which is ~ 8e8. Floor-divide by 86400 to bin to
    days; same-night observations share an epoch_key."""
    return int(entry["t"] / 86400.0)


def _chain_rates(chain: Sequence[dict]) -> list[float]:
    return [float(e.get("rate_arcsec_hr", 0.0)) for e in chain
            if e.get("rate_arcsec_hr") is not None]


def _chain_mags(chain: Sequence[dict]) -> list[float]:
    """Extract magnitudes -- chain entries inherit mag from their source_pair
    members. If no usable photometry, returns empty."""
    mags = []
    for e in chain:
        # Try direct mag first
        if "mag" in e and e["mag"] not in (None, -99.0, 0.0):
            mags.append(float(e["mag"]))
            continue
        # Fall back to source_pair members
        pair = e.get("source_pair")
        if not pair:
            continue
        for s in pair:
            m = getattr(s, "mag", None)
            if m not in (None, -99.0, 0.0):
                mags.append(float(m))
    return mags


def rate_coherence_score(chain: Sequence[dict]) -> tuple[float, float]:
    """Return (median_rate, rate_spread_relative).

    rate_spread_relative = (rate_p95 - rate_p5) / median_rate.
    For a true single-object chain this is < ~0.3; for a mis-linked
    chain it is often > 1.0.
    """
    rates = _chain_rates(chain)
    if len(rates) < 2:
        return (float(rates[0]) if rates else 0.0, 0.0)
    med = float(statistics.median(rates))
    if med <= 0:
        return (med, 0.0)
    p5 = float(min(rates))
    p95 = float(max(rates))
    return (med, (p95 - p5) / med)


def photometric_coherence_score(chain: Sequence[dict]) -> tuple[float, float]:
    """Return (median_mag, mag_std). Empty -> (0, 0)."""
    mags = _chain_mags(chain)
    if not mags:
        return (0.0, 0.0)
    if len(mags) == 1:
        return (mags[0], 0.0)
    med = float(statistics.median(mags))
    std = float(statistics.stdev(mags))
    return (med, std)


def epoch_coverage(chain: Sequence[dict]) -> tuple[int, float]:
    """Return (n_unique_epoch_days, arc_hours)."""
    if not chain:
        return (0, 0.0)
    epoch_days = {_epoch_key(e) for e in chain}
    ts = [e["t"] for e in chain]
    arc_hours = (max(ts) - min(ts)) / 3600.0
    return (len(epoch_days), arc_hours)


def chain_purity_score(chain: Sequence[dict]) -> float:
    """A 0..1 quality score for use as a Bayesian prior or sort key.

    Higher = more likely a real single-object chain. Computed as the
    geometric mean of four components, each clipped to [0, 1]:
      r_score = 1 - min(1, rate_spread / 1.0)
      p_score = 1 - min(1, mag_std / 1.0)
      e_score = min(1, n_unique_epochs / 4)
      a_score = min(1, arc_hours / 48)
    """
    _, spread = rate_coherence_score(chain)
    _, mag_std = photometric_coherence_score(chain)
    n_epochs, arc_hours = epoch_coverage(chain)
    r_score = max(0.0, 1.0 - min(1.0, spread / 1.0))
    p_score = max(0.0, 1.0 - min(1.0, mag_std / 1.0))
    e_score = min(1.0, n_epochs / 4.0)
    a_score = min(1.0, arc_hours / 48.0)
    # Geometric mean is harsher on weakest component (1 weak -> low score)
    return (r_score * p_score * e_score * a_score) ** 0.25


def rate_coherence_filter(chain: Sequence[dict], *,
                            max_rate_spread: float = 0.5) -> bool:
    """True if the chain's within-chain rate spread is below threshold.
    `max_rate_spread = 0.5` means rate p95-p5 is allowed to be at most
    50% of the median (=> kept), otherwise reject as mis-linked."""
    _, spread = rate_coherence_score(chain)
    return spread <= max_rate_spread


def photometric_coherence_filter(chain: Sequence[dict], *,
                                   max_mag_std: float = 0.6,
                                   require_photometry: bool = False) -> bool:
    """True if the chain's magnitude std is below threshold (the same object
    has roughly constant brightness across a few-day arc). If chain entries
    have no usable magnitudes, returns True unless require_photometry=True."""
    mags = _chain_mags(chain)
    if not mags:
        return not require_photometry
    if len(mags) == 1:
        return True
    std = statistics.stdev(mags)
    return std <= max_mag_std


def epoch_coverage_filter(chain: Sequence[dict], *,
                            min_unique_epochs: int = 3) -> bool:
    """True if the chain covers at least `min_unique_epochs` distinct
    epoch days. Gauss-class IOD fundamentally needs 3+ epochs."""
    n_epochs, _ = epoch_coverage(chain)
    return n_epochs >= min_unique_epochs


def arclength_filter(chain: Sequence[dict], *,
                      min_arc_hours: float = 12.0) -> bool:
    """True if the chain spans at least `min_arc_hours` between earliest
    and latest detection. Very short arcs almost never yield converged
    orbits regardless of how clean the data is."""
    _, arc_hours = epoch_coverage(chain)
    return arc_hours >= min_arc_hours


def filter_chains(chains: Sequence[Sequence[dict]],
                   *,
                   max_rate_spread: float = 0.5,
                   max_mag_std: float = 0.6,
                   min_unique_epochs: int = 3,
                   min_arc_hours: float = 12.0,
                   require_photometry: bool = False,
                   ) -> tuple[list, list, list[ChainQualityVerdict]]:
    """Apply all four quality filters in sequence.

    Returns (kept_chains, dropped_chains, verdicts) where verdicts is one
    ChainQualityVerdict per input chain (in input order).
    """
    kept, dropped, verdicts = [], [], []
    for idx, ch in enumerate(chains):
        n_epochs, arc_hours = epoch_coverage(ch)
        rate_med, rate_spread = rate_coherence_score(ch)
        mag_med, mag_std = photometric_coherence_score(ch)
        purity = chain_purity_score(ch)

        passes_rate = rate_spread <= max_rate_spread
        passes_photo = (photometric_coherence_filter(
                          ch, max_mag_std=max_mag_std,
                          require_photometry=require_photometry))
        passes_epoch = n_epochs >= min_unique_epochs
        passes_arc = arc_hours >= min_arc_hours

        reasons = []
        if not passes_rate:
            reasons.append(f"rate spread {rate_spread:.2f}>{max_rate_spread}")
        if not passes_photo:
            reasons.append(f"mag std {mag_std:.2f}>{max_mag_std}")
        if not passes_epoch:
            reasons.append(f"epochs {n_epochs}<{min_unique_epochs}")
        if not passes_arc:
            reasons.append(f"arc {arc_hours:.1f}h<{min_arc_hours}")

        passes_all = passes_rate and passes_photo and passes_epoch and passes_arc
        verdict = ChainQualityVerdict(
            chain_idx=idx, n_entries=len(ch),
            n_unique_epochs=n_epochs, arc_hours=arc_hours,
            rate_median=rate_med, rate_spread_rel=rate_spread,
            mag_median=mag_med, mag_std=mag_std,
            purity_score=purity,
            passes_rate=passes_rate, passes_photometric=passes_photo,
            passes_epoch=passes_epoch, passes_arc=passes_arc,
            passes_all=passes_all, reasons=tuple(reasons))
        verdicts.append(verdict)
        if passes_all:
            kept.append(ch)
        else:
            dropped.append(ch)
    return kept, dropped, verdicts
