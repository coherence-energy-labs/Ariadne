"""Bayesian IOD with orbital-class priors.

The deterministic IOD strategies in iod_advanced.py (Gauss, adaptive
HelioLinC, Vaisala, Bernstein-Khushalani) all bootstrap from generic
geometric assumptions. They work when the chain has enough independent
observations to constrain the orbit; they fail on short / noisy chains
where the geometric degeneracies dominate.

This module adds a Bayesian IOD that uses ORBITAL CLASS PRIORS to
regularize the fit. For each candidate orbital class (NEO, MBA,
Centaur, TNO), it:

  1. Samples N seed states from the class prior (heliocentric distance,
     near-circular eccentricity, inclination).
  2. Initialises one seed per sampled (r_au, rdot, tangential direction).
  3. Runs LM refinement on each seed.
  4. Scores each refined orbit by: log P(chain | orbit) + log P(orbit | class)
  5. Returns the highest-posterior orbit across all classes.

This works on 2-night / 4-observation chains where pure-geometry IOD
fails, because the class prior breaks the degeneracy (a TNO won't fit
a 1-AU orbit even if the geometric fit is OK).

Public API:
  bayesian_iod(chain, ...) -> EnsembleFit
    Same return type as iod_advanced.fit_candidate_ensemble; drop-in
    replacement for chains where the deterministic ensemble failed.

References:
  Bernstein & Khushalani 2000, AJ 120:3323 (BK prior parametrisation)
  Holman+ 2018, AJ 156:135 (HelioLinC)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .iod_advanced import EnsembleFit, StrategyResult, _refine_with_lm


# ---------------------------------------------------------------------------
# Orbital class priors (mixture of Gaussians over heliocentric distance,
# eccentricity, inclination)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrbitalClassPrior:
    """One orbital-class prior component (log-weighted in mixture)."""
    name: str
    log_weight: float        # mixture weight (log)
    r_au_mean: float         # mean heliocentric distance
    r_au_sigma: float        # 1-sigma in r
    e_max: float             # eccentricity ceiling (truncated uniform)
    i_max_deg: float         # inclination ceiling (truncated uniform)


ORBITAL_PRIORS = (
    OrbitalClassPrior("neo",      math.log(0.05),  1.2,  0.4, 0.5, 35.0),
    OrbitalClassPrior("mba",      math.log(0.45),  2.7,  0.6, 0.2, 25.0),
    OrbitalClassPrior("hilda",    math.log(0.05),  4.0,  0.3, 0.2, 15.0),
    OrbitalClassPrior("centaur",  math.log(0.10),  10.0, 5.0, 0.3, 25.0),
    OrbitalClassPrior("tno_class", math.log(0.20), 45.0, 8.0, 0.1, 15.0),
    OrbitalClassPrior("tno_scatter", math.log(0.15), 70.0, 25.0, 0.4, 35.0),
)


def log_orbital_class_prior(r_au: float) -> tuple[float, str]:
    """Return (log marginal P(r), dominant class name) for a given
    heliocentric distance. Used to ASSESS a candidate orbit's prior
    probability after the fit."""
    log_terms = []
    for prior in ORBITAL_PRIORS:
        if prior.r_au_sigma <= 0:
            continue
        z = (r_au - prior.r_au_mean) / prior.r_au_sigma
        log_pdf = -0.5 * z * z - math.log(
            prior.r_au_sigma * math.sqrt(2.0 * math.pi))
        log_terms.append((prior.log_weight + log_pdf, prior.name))
    if not log_terms:
        return (-50.0, "none")
    log_max = max(t[0] for t in log_terms)
    log_marginal = log_max + math.log(
        sum(math.exp(t[0] - log_max) for t in log_terms))
    dominant = max(log_terms, key=lambda t: t[0])[1]
    return (log_marginal, dominant)


# ---------------------------------------------------------------------------
# Seed generation from a class prior
# ---------------------------------------------------------------------------

def _seed_from_class(prior: OrbitalClassPrior, chain: Sequence[dict],
                       n_samples: int = 8,
                       rng: np.random.Generator | None = None
                       ) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Generate `n_samples` (r0, v0, t_ref) seeds from this class prior.

    Each seed places the object at heliocentric distance r ~ N(mu, sigma)
    along the line-of-sight to the chain's first observation, with a
    near-circular tangential velocity perturbed by a random tilt within
    the class's inclination ceiling.
    """
    from ..data.constants import GM_SUN, AU_KM
    from ..data.ephemeris import body_state
    rng = rng if rng is not None else np.random.default_rng(0)

    if not chain:
        return []
    sorted_ch = sorted(chain, key=lambda e: e["t"])
    obs = sorted_ch[len(sorted_ch) // 2]   # middle observation as reference
    t_ref = float(obs["t"])
    R_e_ref = np.array(body_state("EARTH", t_ref, "J2000", "SUN")[:3])
    ra = float(obs["ra"]); dec = float(obs["dec"])
    d_los = np.array([math.cos(dec) * math.cos(ra),
                       math.cos(dec) * math.sin(ra),
                       math.sin(dec)])

    seeds = []
    for _ in range(n_samples):
        # Sample r from truncated normal at the prior mean
        r_au = max(0.5, float(rng.normal(prior.r_au_mean,
                                            prior.r_au_sigma)))
        rho_km = r_au * AU_KM
        r0 = R_e_ref + rho_km * d_los
        r0_norm = float(np.linalg.norm(r0))
        v_circ = math.sqrt(GM_SUN / r0_norm)
        r_hat = r0 / r0_norm
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(r_hat, up)) > 0.95:
            up = np.array([1.0, 0.0, 0.0])
        tan_e = np.cross(r_hat, up)
        tan_e = tan_e / float(np.linalg.norm(tan_e))
        tan_n = np.cross(r_hat, tan_e)
        # Random tilt within the class's inclination ceiling
        tilt_deg = float(rng.uniform(-prior.i_max_deg, prior.i_max_deg))
        tilt = math.radians(tilt_deg)
        # Allow eccentricity to shift the speed: e=0.1 -> +/-5%
        e_factor = 1.0 + float(rng.uniform(-prior.e_max, prior.e_max)) * 0.5
        v_tangent = (math.cos(tilt) * tan_e
                       + math.sin(tilt) * tan_n) * v_circ * e_factor
        seeds.append((r0, v_tangent, t_ref))
    return seeds


# ---------------------------------------------------------------------------
# Likelihood (Gaussian over angular residuals)
# ---------------------------------------------------------------------------

def _chain_log_likelihood(chain: Sequence[dict],
                            x0: np.ndarray, v0: np.ndarray,
                            t_ref: float,
                            astrometric_sigma_arcsec: float = 0.3) -> float:
    """log P(chain | orbit) under Gaussian astrometric noise."""
    from ..data.constants import GM_SUN
    from ..data.ephemeris import body_state
    from ..dynamics.secular import kepler_step

    log_l = 0.0
    sigma_rad = math.radians(astrometric_sigma_arcsec / 3600.0)
    log_norm = -math.log(sigma_rad * math.sqrt(2 * math.pi))
    for entry in chain:
        dt_s = float(entry["t"]) - float(t_ref)
        try:
            r_t, _ = kepler_step(x0, v0, GM_SUN, dt_s)
        except Exception:
            return -1e9
        R_e = np.array(body_state("EARTH", entry["t"], "J2000", "SUN")[:3])
        geo = r_t - R_e
        rho = float(np.linalg.norm(geo))
        if rho < 1.0:
            return -1e9
        ra_pred = math.atan2(geo[1], geo[0]) % (2 * math.pi)
        dec_pred = math.asin(geo[2] / rho)
        d_ra = (ra_pred - entry["ra"])
        # Wrap-around
        if d_ra > math.pi:
            d_ra -= 2 * math.pi
        elif d_ra < -math.pi:
            d_ra += 2 * math.pi
        d_ra *= math.cos(entry["dec"])
        d_dec = dec_pred - entry["dec"]
        z = (d_ra * d_ra + d_dec * d_dec) / (sigma_rad * sigma_rad)
        log_l += -0.5 * z + 2 * log_norm   # 2 angular components
    return log_l


# ---------------------------------------------------------------------------
# Top-level Bayesian IOD
# ---------------------------------------------------------------------------

def bayesian_iod(chain: Sequence[dict], *,
                  n_seeds_per_class: int = 4,
                  astrometric_sigma_arcsec: float = 0.5,
                  rms_acceptance_arcsec: float = 30.0,
                  seed: int = 0) -> EnsembleFit:
    """Run the Bayesian IOD on a chain. Returns an EnsembleFit.

    For each orbital class prior, generates `n_seeds_per_class` seeds
    from the class prior, runs LM refinement on each, and scores the
    refined orbit by:
        log P(chain | orbit) + log P(orbit | class prior)

    Returns the highest-posterior orbit across all classes/seeds. The
    `winning_strategy` field is set to "bayesian_<class>" so callers can
    distinguish results from the deterministic ensemble.
    """
    if not chain or len(chain) < 2:
        return EnsembleFit(
            success=False, x_fit=np.zeros(3), v_fit=np.zeros(3),
            rms_arcsec=float("inf"), t_ref=0.0,
            winning_strategy="bayesian_none", strategy_results=[],
            notes="empty chain or fewer than 2 observations")

    sorted_ch = sorted(chain, key=lambda e: e["t"])
    t_ref = float(np.median([e["t"] for e in sorted_ch]))
    rng = np.random.default_rng(seed)

    seeds_attempted: list[StrategyResult] = []
    best = None     # (posterior_score, rms, x, v, class_name)
    for prior in ORBITAL_PRIORS:
        prior_seeds = _seed_from_class(prior, chain,
                                          n_samples=n_seeds_per_class,
                                          rng=rng)
        for r0, v0, seed_t_ref in prior_seeds:
            try:
                rms, x_fit, v_fit, nfev, ok = _refine_with_lm(
                    chain, t_ref, r0, v0)
            except Exception:
                continue
            if not ok:
                continue
            log_l = _chain_log_likelihood(
                chain, x_fit, v_fit, t_ref,
                astrometric_sigma_arcsec=astrometric_sigma_arcsec)
            from ..data.constants import AU_KM
            r_au = float(np.linalg.norm(x_fit)) / AU_KM
            log_prior, dom_class = log_orbital_class_prior(r_au)
            posterior = log_l + log_prior
            seeds_attempted.append(StrategyResult(
                strategy=f"bayesian_{prior.name}",
                success=True, x_init=r0, v_init=v0,
                t_ref=t_ref, r_au=r_au, rdot=0.0,
                scatter_km=0.0,
                notes=f"rms={rms:.2f} log_l={log_l:.1f} log_prior={log_prior:.1f}"))
            if best is None or posterior > best[0]:
                best = (posterior, rms, x_fit, v_fit, prior.name)

    if best is None:
        return EnsembleFit(
            success=False, x_fit=np.zeros(3), v_fit=np.zeros(3),
            rms_arcsec=float("inf"), t_ref=t_ref,
            winning_strategy="bayesian_none",
            strategy_results=seeds_attempted,
            notes="no LM convergence across any class prior")

    posterior, rms, x_fit, v_fit, class_name = best
    accepted = math.isfinite(rms) and rms < rms_acceptance_arcsec
    return EnsembleFit(
        success=accepted,
        x_fit=x_fit, v_fit=v_fit,
        rms_arcsec=float(rms), t_ref=float(t_ref),
        winning_strategy=f"bayesian_{class_name}",
        strategy_results=seeds_attempted,
        seed_rms_arcsec=float(rms), nfev=0,
        notes=f"posterior log-prob={posterior:.1f}",
    )
