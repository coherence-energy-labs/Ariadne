"""RANSAC-style chain refinement: outlier removal before IOD.

A mixed/polluted chain has 1-2 stray observations (background stars or
cross-tracklets) that dominate the centroid residuals when an orbit
is fit. Removing those stray observations BEFORE IOD lets the
strategy ensemble converge on the real orbit.

Approach: for each chain:

  1. Fit the full chain with one of the cheap strategies (Vaisala,
     gauss-3pt seed + LM) to get a baseline orbit + RMS.
  2. Leave-one-out (LOO): for each chain entry, refit without it.
     The entry whose removal LOWERS RMS the most is the most likely
     outlier.
  3. If removing the worst entry lowers RMS by more than
     `outlier_drop_factor`, drop it and repeat. Stop when no single
     removal improves things, or chain length falls below `min_keep`.
  4. Return the refined chain.

This is a coarse RANSAC — not the full Monte Carlo random-subsample
variant. For chains with > 1 outlier, multiple LOO passes converge
on the clean subset.

Robust to clean chains: when the chain is already pure, every LOO
fit gives similar RMS, the algorithm exits without removing anything.

Public API:
  refine_chain_ransac(chain, ...) -> (cleaned_chain, n_removed, base_rms,
                                        final_rms)
  refine_chains(chains, ...) -> list[(cleaned_chain, ...)]
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _fit_rms(chain: Sequence[dict]) -> float:
    """Run a cheap IOD attempt on `chain` and return the post-LM RMS in
    arcsec. Returns inf when no strategy converges or the chain is too
    short."""
    if len(chain) < 3:
        return float("inf")
    from .iod_advanced import (
        _strategy_gauss, _strategy_vaisala, _refine_with_lm)
    t_ref = float(np.median([e["t"] for e in chain]))
    best = float("inf")
    for strat in (_strategy_gauss, _strategy_vaisala):
        try:
            seed = strat(chain, t_ref)
        except Exception:
            continue
        if not seed.success:
            continue
        try:
            rms, _, _, _, ok = _refine_with_lm(
                chain, t_ref, seed.x_init, seed.v_init)
        except Exception:
            continue
        if ok and rms < best:
            best = rms
    return float(best)


def refine_chain_ransac(chain: Sequence[dict],
                          *,
                          outlier_drop_factor: float = 1.5,
                          min_keep: int = 3,
                          max_passes: int = 5,
                          ) -> dict:
    """Run LOO outlier removal on `chain`.

    Returns a dict with:
      cleaned_chain   list[dict] -- the chain after outlier removal
      n_removed       int -- number of observations dropped
      base_rms        float -- RMS of the full chain
      final_rms       float -- RMS of the cleaned chain
      removed_idx     list[int] -- indices into ORIGINAL chain of removed obs

    `outlier_drop_factor` controls aggressiveness: if removing an entry
    reduces RMS by at least `1 / outlier_drop_factor` (i.e. RMS becomes
    1.5x lower for default), keep going. Larger = more conservative.
    """
    chain = list(chain)
    original_chain = list(chain)
    base_rms = _fit_rms(chain)
    final_rms = base_rms
    removed_idx: list[int] = []
    removed_signatures = set()

    for pass_idx in range(max_passes):
        if len(chain) <= min_keep:
            break
        current_rms = _fit_rms(chain)
        if not math.isfinite(current_rms):
            break
        best_drop = 0.0
        best_remove_local_idx = -1
        for i in range(len(chain)):
            trial = chain[:i] + chain[i + 1:]
            trial_rms = _fit_rms(trial)
            if not math.isfinite(trial_rms):
                continue
            drop = current_rms - trial_rms
            if drop > best_drop:
                best_drop = drop
                best_remove_local_idx = i
        if best_remove_local_idx < 0:
            break
        # Threshold: only remove if the drop is significant
        improvement_ratio = (current_rms / max(current_rms - best_drop, 1e-6))
        if improvement_ratio < outlier_drop_factor:
            break
        # Map local idx back to original-chain idx
        removed_entry = chain[best_remove_local_idx]
        # Find original idx by identity (safe since we copy-list, not deep copy)
        for orig_i, orig_e in enumerate(original_chain):
            if orig_e is removed_entry and orig_i not in removed_idx:
                removed_idx.append(orig_i)
                break
        chain = chain[:best_remove_local_idx] + chain[best_remove_local_idx + 1:]
        final_rms = current_rms - best_drop

    return {
        "cleaned_chain": chain,
        "n_removed": len(removed_idx),
        "base_rms": base_rms,
        "final_rms": final_rms,
        "removed_idx": removed_idx,
    }


def refine_chains(chains: Sequence[Sequence[dict]],
                   **kwargs) -> list[dict]:
    """Apply RANSAC outlier removal to every chain in `chains`."""
    return [refine_chain_ransac(ch, **kwargs) for ch in chains]
