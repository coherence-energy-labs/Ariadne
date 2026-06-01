"""Robust IOD wrapper: noise-aware Monte Carlo + rate-class-aware strategy.

The deterministic IOD ensemble in `iod_advanced.py` is hyper-sensitive to
astrometric noise -- 0.5 arcsec of centroid jitter can push a Gauss-method
8th-degree polynomial root past the convergence basin. On clean simulated
input it achieves 100% recovery; on real image-derived chains with PSF-fit
noise it falls to 0%.

This module wraps the deterministic ensemble with two upgrades:

  monte_carlo_iod
    Perturb each observation by its astrometric sigma N times, run the
    underlying ensemble on each perturbation, and report the MEDIAN
    orbit + posterior covariance. The strategy that converges most
    reliably across perturbations wins.

  rate_class_aware_iod
    Branch on the chain's median rate to pick a strategy ordering most
    likely to converge:
      slow movers (< 5"/hr): TNOs / distant outer-system   -> BK first
                                                              then Vaisala
                                                              then HelioLinC
                                                              (skip Gauss)
      medium movers (5-50): main belt / inner-system        -> Adaptive HelioLinC
                                                              then Gauss
                                                              then Vaisala
      fast movers (> 50):   NEOs / near-Earth               -> Gauss first
                                                              then Vaisala
                                                              then HelioLinC

Combining both:

  robust_iod
    Rate-class-aware strategy ordering wrapped in Monte Carlo for noise
    robustness. The top-level entry point for the image-pipeline.

All wrappers preserve the EnsembleFit return type so downstream code
(smart_annotate, MPC submission gate) doesn't need to change.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from . import iod_advanced as IODA


@dataclass
class MonteCarloResult:
    """Result of `monte_carlo_iod`."""
    success: bool
    median_x: np.ndarray = field(default_factory=lambda: np.zeros(3))
    median_v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    cov_x: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    cov_v: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    rms_arcsec_median: float = float("inf")
    rms_arcsec_p84: float = float("inf")
    winning_strategy: str = "none"
    n_attempts: int = 0
    n_successes: int = 0
    consensus_fraction: float = 0.0
    fit: IODA.EnsembleFit | None = None
    notes: str = ""


def _perturb_chain(chain: Sequence[dict],
                    sigma_arcsec: float,
                    rng: np.random.Generator) -> list[dict]:
    """Return a deep copy of `chain` with each (ra, dec) Gaussian-perturbed
    by `sigma_arcsec` arcsec in both axes. Times and rates are preserved.
    """
    sigma_rad = math.radians(sigma_arcsec / 3600.0)
    new_chain = []
    for entry in chain:
        e = copy.copy(entry)   # shallow copy of the dict
        dec_orig = e["dec"]
        cos_dec = max(math.cos(dec_orig), 1e-4)
        e["ra"] = e["ra"] + (rng.standard_normal() * sigma_rad / cos_dec)
        e["dec"] = e["dec"] + (rng.standard_normal() * sigma_rad)
        new_chain.append(e)
    return new_chain


def estimate_chain_sigma_arcsec(chain: Sequence[dict],
                                  default_sigma: float = 0.30) -> float:
    """Estimate the per-observation astrometric sigma of a chain.

    Best path: read PSF-fit residuals if the chain entries carry them.
    Fall back to a default of 0.30 arcsec (typical DECam-like seeing-
    limited centroid uncertainty).
    """
    sigmas = []
    for e in chain:
        # The image-pipeline source_pair members may carry psf_sigma
        pair = e.get("source_pair") or ()
        for s in pair:
            ps = getattr(s, "psf_sigma_arcsec", None)
            if ps is not None and ps > 0:
                sigmas.append(float(ps))
    if sigmas:
        return float(np.median(sigmas))
    return float(default_sigma)


def chain_median_rate(chain: Sequence[dict]) -> float:
    rates = [float(e.get("rate_arcsec_hr", 0.0)) for e in chain
             if e.get("rate_arcsec_hr") is not None]
    if not rates:
        return 0.0
    return float(np.median(rates))


def rate_class_strategy_order(median_rate_arcsec_hr: float) -> list[str]:
    """Return strategy names in best-first order for the rate class."""
    if median_rate_arcsec_hr < 5.0:
        # Slow: TNOs / distant outer system
        return ["bernstein_khushalani", "vaisala",
                 "adaptive_linker", "gauss"]
    if median_rate_arcsec_hr < 50.0:
        # Medium: main belt
        return ["adaptive_linker", "gauss", "vaisala",
                 "bernstein_khushalani"]
    # Fast: NEO
    return ["gauss", "vaisala", "adaptive_linker",
             "bernstein_khushalani"]


def monte_carlo_iod(chain: Sequence[dict], *,
                     n_draws: int = 16,
                     sigma_arcsec: float | None = None,
                     rms_acceptance_arcsec: float = 5.0,
                     seed: int = 0,
                     **ensemble_kwargs) -> MonteCarloResult:
    """Run the deterministic ensemble N times on noise-perturbed copies of
    `chain` and return the consensus orbit.

    A perturbation is "successful" if `EnsembleFit.success` is True AND
    `EnsembleFit.rms_arcsec <= rms_acceptance_arcsec`. The CONSENSUS orbit
    is the median (x, v) across successful perturbations; the covariance
    is the sample covariance across those same successes.

    Returns MonteCarloResult with:
      success                  True if any perturbation converged
      median_x, median_v       median state across successful perturbations
      cov_x, cov_v             sample covariance of those states
      rms_arcsec_median        median RMS across successful perturbations
      rms_arcsec_p84           84th-percentile RMS (tail)
      consensus_fraction       n_successes / n_draws
      winning_strategy         most-frequent winning strategy among successes
    """
    if sigma_arcsec is None:
        sigma_arcsec = estimate_chain_sigma_arcsec(chain)

    rng = np.random.default_rng(seed)
    xs, vs, rmss, strats = [], [], [], []
    best_fit: IODA.EnsembleFit | None = None
    best_rms = float("inf")

    for k in range(n_draws):
        perturbed = _perturb_chain(chain, sigma_arcsec, rng)
        try:
            fit = IODA.fit_candidate_ensemble(
                perturbed, rms_acceptance_arcsec=rms_acceptance_arcsec,
                **ensemble_kwargs)
        except Exception as exc:
            continue
        if fit.success and fit.rms_arcsec <= rms_acceptance_arcsec:
            xs.append(np.asarray(fit.x_fit, dtype=float))
            vs.append(np.asarray(fit.v_fit, dtype=float))
            rmss.append(float(fit.rms_arcsec))
            strats.append(fit.winning_strategy)
            if fit.rms_arcsec < best_rms:
                best_rms = fit.rms_arcsec
                best_fit = fit

    if not xs:
        return MonteCarloResult(success=False, n_attempts=n_draws,
                                  notes=(f"no MC draw converged at "
                                          f"sigma={sigma_arcsec:.2f}\""))

    xs_arr = np.array(xs); vs_arr = np.array(vs)
    median_x = np.median(xs_arr, axis=0)
    median_v = np.median(vs_arr, axis=0)
    cov_x = (np.cov(xs_arr.T, ddof=1) if xs_arr.shape[0] > 1
              else np.zeros((3, 3)))
    cov_v = (np.cov(vs_arr.T, ddof=1) if vs_arr.shape[0] > 1
              else np.zeros((3, 3)))
    rms_median = float(np.median(rmss))
    rms_p84 = float(np.percentile(rmss, 84))
    winning = max(set(strats), key=strats.count)
    consensus = len(xs) / n_draws

    return MonteCarloResult(
        success=True, median_x=median_x, median_v=median_v,
        cov_x=cov_x, cov_v=cov_v,
        rms_arcsec_median=rms_median, rms_arcsec_p84=rms_p84,
        winning_strategy=f"mc_{winning}",
        n_attempts=n_draws, n_successes=len(xs),
        consensus_fraction=consensus, fit=best_fit,
        notes=f"MC sigma={sigma_arcsec:.2f}\" {len(xs)}/{n_draws} converged",
    )


def _try_neural_seed(chain: Sequence[dict], *,
                       neural_weights: dict,
                       rms_acceptance_arcsec: float) -> IODA.EnsembleFit | None:
    """Try LM refining the neural-prior prediction as a single seed.

    Returns an EnsembleFit if the refinement succeeded (RMS <= acceptance),
    else None so the caller falls through to the multi-strategy ensemble.
    """
    try:
        from .imaging.neural_orbit_prior import (
            build_chain_features, predict_initial_state_normalised)
    except Exception:
        return None
    feats = build_chain_features(chain)
    try:
        x_init, v_init = predict_initial_state_normalised(feats, neural_weights)
    except Exception:
        return None
    t_ref = float(np.median([t["t"] for t in chain]))
    try:
        rms, x_fit, v_fit, nfev, ok = IODA._refine_with_lm(
            chain, t_ref, x_init, v_init)
    except Exception:
        return None
    if not ok or rms > rms_acceptance_arcsec:
        return None
    return IODA.EnsembleFit(
        success=True, x_fit=x_fit, v_fit=v_fit,
        rms_arcsec=float(rms), t_ref=float(t_ref),
        winning_strategy="neural_prior", strategy_results=[],
        seed_rms_arcsec=float(rms), nfev=int(nfev),
        notes="seeded by neural orbit prior")


def robust_iod(chain: Sequence[dict], *,
                n_draws: int = 16,
                sigma_arcsec: float | None = None,
                rms_acceptance_arcsec: float = 5.0,
                use_monte_carlo: bool = True,
                use_rate_class: bool = True,
                neural_weights: dict | None = None,
                seed: int = 0,
                **ensemble_kwargs) -> IODA.EnsembleFit:
    """Top-level robust-IOD entry point.

    Combines rate-class-aware strategy ordering with optional Monte Carlo
    noise perturbation. Returns an EnsembleFit so callers don't need to
    distinguish "robust" from "vanilla" downstream.

    When `use_rate_class=True`, the deterministic ensemble is told to
    short-circuit on the rate-appropriate strategy first (early-exit at
    the first one to converge).

    When `use_monte_carlo=True`, the rate-class-ordered ensemble is then
    wrapped in a Monte Carlo loop with the chain's estimated astrometric
    sigma.
    """
    # Step 0: try the neural prior as a single fast seed.
    # When the prior is well-trained, this converges in one LM call
    # without needing to grind through the 4-strategy ensemble.
    if neural_weights is not None:
        seeded = _try_neural_seed(chain,
                                    neural_weights=neural_weights,
                                    rms_acceptance_arcsec=rms_acceptance_arcsec)
        if seeded is not None:
            return seeded

    if use_rate_class:
        median_rate = chain_median_rate(chain)
        strategy_order = tuple(rate_class_strategy_order(median_rate))
        # Pass strategies in the rate-class-best order, and disable
        # cheap_first so the ensemble respects our ordering instead of
        # forcing its hardcoded "gauss, adaptive_linker, ..." sequence.
        ensemble_kwargs.setdefault("strategies", strategy_order)
        ensemble_kwargs.setdefault("cheap_first", False)
        ensemble_kwargs.setdefault("early_exit_rms_arcsec",
                                      rms_acceptance_arcsec * 0.7)

    if not use_monte_carlo:
        return IODA.fit_candidate_ensemble(
            chain, rms_acceptance_arcsec=rms_acceptance_arcsec,
            **ensemble_kwargs)

    mc = monte_carlo_iod(chain, n_draws=n_draws,
                          sigma_arcsec=sigma_arcsec,
                          rms_acceptance_arcsec=rms_acceptance_arcsec,
                          seed=seed, **ensemble_kwargs)
    if not mc.success:
        # Fall back to one deterministic run so we still return a fit
        # object with diagnostics, even if it's a failed one.
        return IODA.fit_candidate_ensemble(
            chain, rms_acceptance_arcsec=rms_acceptance_arcsec,
            **ensemble_kwargs)
    # Build an EnsembleFit using the median MC state, but stamp the
    # winning_strategy as mc_<strategy> so it's visible in reports.
    base_fit = mc.fit
    if base_fit is None:
        return IODA.fit_candidate_ensemble(
            chain, rms_acceptance_arcsec=rms_acceptance_arcsec,
            **ensemble_kwargs)
    return IODA.EnsembleFit(
        success=True,
        x_fit=mc.median_x,
        v_fit=mc.median_v,
        rms_arcsec=mc.rms_arcsec_median,
        t_ref=base_fit.t_ref,
        winning_strategy=mc.winning_strategy,
        strategy_results=base_fit.strategy_results,
        seed_rms_arcsec=base_fit.seed_rms_arcsec,
        nfev=base_fit.nfev,
        refined_with_nbody=base_fit.refined_with_nbody,
        notes=mc.notes,
    )
