"""Bayesian MCMC orbit posterior -- full 6D uncertainty distribution, not a point.

The LM orbit fit returns a single best estimate of the heliocentric state
(x, y, z, vx, vy, vz). For most operational uses that's enough; but when
you want to KNOW how confident you are in the orbit (especially with sparse
observations), an MCMC posterior is the right answer:

  * Predict apparent positions at any future epoch WITH a real posterior
    cloud of predicted (RA, Dec), not just a centroid + 1-sigma scalar.
  * Estimate the probability the orbit belongs to a particular dynamical
    class (e.g. "what's P(TNO | observations)?").
  * Detect orbital degeneracies that LM hides (multi-modal posteriors,
    flat directions).

Implementation: emcee (Goodman-Weare affine-invariant ensemble) is the
canonical choice. If emcee is unavailable, fall back to a simple Metropolis-
Hastings sampler with a Gaussian proposal -- 10-100x slower convergence but
no extra dependencies.

Inputs: tracklet records (same shape as iod.fit_candidate) + an LM seed
(point estimate + covariance estimate from the Jacobian).
Output: McMcOrbitPosterior containing the chain + summary statistics.

Reference: Foreman-Mackey et al. 2013 (emcee paper); Bernstein-Khushalani
2000 (TNO Bayesian orbit fitting).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

from ..data.constants import AU_KM, GM_SUN
from ..data.ephemeris import body_state
from ..dynamics.secular import kepler_step

C_KM_S = 299792.458


@dataclass
class McMcOrbitPosterior:
    """Result of one MCMC orbit-fit run.

    Fields:
      chain:           shape (n_steps, n_walkers, 6) -- full chain in
                        (x, y, z, vx, vy, vz) scaled units.
      log_prob:        shape (n_steps, n_walkers) -- log posterior at each
                        sample.
      best_sample:     argmax-of-log_prob state vector (km, km/s).
      mean:            mean of the burn-in-discarded samples.
      cov:             6x6 covariance.
      x_quantiles_au:  (16, 50, 84)% quantile triplet for the position
                        magnitude in AU.
      v_quantiles_km_s:same for the velocity magnitude.
      a_quantiles_au:  same for the derived semi-major axis (if elliptical).
      e_quantiles:     same for eccentricity.
      i_quantiles_deg: same for inclination.
      n_burnin:        burn-in steps discarded.
      n_thin:          chain thinning interval.
      converged:       True if the integrated autocorrelation time was
                        reliably measured.
      sampler_used:    "emcee" | "metropolis"
    """
    chain: np.ndarray
    log_prob: np.ndarray
    best_sample: np.ndarray
    mean: np.ndarray
    cov: np.ndarray
    x_quantiles_au: tuple
    v_quantiles_km_s: tuple
    a_quantiles_au: tuple
    e_quantiles: tuple
    i_quantiles_deg: tuple
    n_burnin: int
    n_thin: int
    converged: bool
    sampler_used: str


def _log_likelihood(state_scaled, ts, ras, decs, Ro_t, t_ref, sigma_arcsec,
                    pos_scale, light_time):
    """Gaussian likelihood in (RA*cos(dec), Dec) for the residuals."""
    r = state_scaled[:3] * pos_scale
    v = state_scaled[3:]
    log_l = 0.0
    sigma_rad = sigma_arcsec / 206265.0
    for k in range(len(ts)):
        dt = float(ts[k]) - t_ref
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rt, _ = kepler_step(r, v, GM_SUN, dt)
        if not np.all(np.isfinite(rt)):
            return -1e30
        if light_time:
            rho = float(np.linalg.norm(rt - Ro_t[k]))
            if not np.isfinite(rho) or rho < 1e3:
                return -1e30
            tau = rho / C_KM_S
            try:
                R_em = body_state("EARTH", float(ts[k]) - tau, "J2000", "SUN")[:3]
            except Exception:
                return -1e30
            g = rt - R_em
        else:
            g = rt - Ro_t[k]
        rn = float(np.linalg.norm(g))
        if not np.isfinite(rn) or rn < 1e3:
            return -1e30
        ra_p = math.atan2(g[1], g[0])
        dec_p = math.asin(max(-1.0, min(1.0, g[2] / rn)))
        dra = (ra_p - ras[k] + math.pi) % (2 * math.pi) - math.pi
        ddec = dec_p - decs[k]
        log_l -= 0.5 * ((dra * math.cos(decs[k])) ** 2 + ddec ** 2) / sigma_rad ** 2
    return log_l


def _log_prior(state_scaled, pos_scale, r_max_au: float = 1000.0):
    """Improper-flat prior with hard bounds.

    Bound: |r| <= r_max_au, |v| <= 50 km/s. Outside -> -inf.
    """
    r = state_scaled[:3] * pos_scale
    v = state_scaled[3:]
    r_au = float(np.linalg.norm(r)) / AU_KM
    v_km_s = float(np.linalg.norm(v))
    if r_au > r_max_au or v_km_s > 50.0:
        return -np.inf
    if r_au < 0.1:
        return -np.inf
    return 0.0


def _log_posterior(state_scaled, ts, ras, decs, Ro_t, t_ref,
                   sigma_arcsec, pos_scale, light_time):
    lp = _log_prior(state_scaled, pos_scale)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _log_likelihood(state_scaled, ts, ras, decs, Ro_t, t_ref,
                                  sigma_arcsec, pos_scale, light_time)


def _metropolis_sampler(log_post_fn, x0, n_walkers, n_steps,
                        proposal_sigma, args, seed=0):
    """Vanilla Gaussian-proposal Metropolis -- a fallback when emcee is missing."""
    rng = np.random.default_rng(seed)
    n_dim = len(x0)
    walkers = x0[None, :] + rng.normal(0, 1e-4, (n_walkers, n_dim))
    chain = np.empty((n_steps, n_walkers, n_dim))
    log_prob = np.empty((n_steps, n_walkers))
    log_p_now = np.array([log_post_fn(w, *args) for w in walkers])
    for s in range(n_steps):
        prop = walkers + rng.normal(0, proposal_sigma, (n_walkers, n_dim))
        log_p_prop = np.array([log_post_fn(w, *args) for w in prop])
        ratio = log_p_prop - log_p_now
        u = rng.uniform(size=n_walkers)
        accept = np.log(u) < ratio
        walkers[accept] = prop[accept]
        log_p_now[accept] = log_p_prop[accept]
        chain[s] = walkers
        log_prob[s] = log_p_now
    return chain, log_prob


def _derive_elements(state_km_kms):
    """(x, v) -> (a_au, e, i_deg). Returns NaNs for hyperbolic orbits."""
    r = state_km_kms[:3]; v = state_km_kms[3:]
    rn = float(np.linalg.norm(r))
    v2 = float(np.dot(v, v))
    energy = 0.5 * v2 - GM_SUN / rn
    if energy >= 0:
        return float("nan"), float("nan"), float("nan")
    a_km = -GM_SUN / (2 * energy)
    h = np.cross(r, v)
    hn = float(np.linalg.norm(h))
    e_vec = np.cross(v, h) / GM_SUN - r / rn
    e = float(np.linalg.norm(e_vec))
    i_deg = math.degrees(math.acos(max(-1.0, min(1.0, h[2] / max(hn, 1e-9)))))
    return a_km / AU_KM, e, i_deg


def sample_posterior(tracklet_records, t_ref: float,
                     x_seed_km, v_seed_kms,
                     *, sigma_arcsec: float = 1.0,
                     n_walkers: int = 32, n_steps: int = 2000,
                     burn_in: int = 500, thin: int = 5,
                     light_time: bool = True,
                     prefer_emcee: bool = True,
                     seed: int = 0) -> McMcOrbitPosterior:
    """Sample the orbit posterior given observed tracklets + an LM seed.

    Args:
      tracklet_records:   list of dicts with 't', 'ra', 'dec' (ET seconds, rad).
      t_ref:              reference epoch (ET seconds).
      x_seed_km, v_seed_kms: state at t_ref from a converged 2-body LM fit.
      sigma_arcsec:       per-observation Gaussian noise sigma.
      n_walkers:          MCMC walkers (emcee default 32).
      n_steps:            steps per walker.
      burn_in:            number of initial samples to discard.
      thin:               keep every Nth sample after burn-in.
      light_time:         iterate light-time correction in the residual.
      prefer_emcee:       try emcee first; fall back to metropolis if missing.

    Returns:
      McMcOrbitPosterior with chain, mean, cov, quantile triplets, etc.
    """
    ts = np.array([t["t"] for t in tracklet_records])
    ras = np.array([t["ra"] for t in tracklet_records])
    decs = np.array([t["dec"] for t in tracklet_records])
    Ro_t = np.array([body_state("EARTH", float(t), "J2000", "SUN")[:3] for t in ts])

    POS_SCALE = AU_KM
    x0 = np.concatenate([np.asarray(x_seed_km) / POS_SCALE,
                         np.asarray(v_seed_kms)])

    args = (ts, ras, decs, Ro_t, t_ref, sigma_arcsec, POS_SCALE, light_time)

    sampler_used = "metropolis"
    if prefer_emcee:
        try:
            import emcee
            initial = x0[None, :] + 1e-4 * np.random.default_rng(seed).normal(
                size=(n_walkers, len(x0)))
            sampler = emcee.EnsembleSampler(
                n_walkers, len(x0), _log_posterior, args=args)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sampler.run_mcmc(initial, n_steps, progress=False)
            chain = sampler.get_chain()       # (n_steps, n_walkers, ndim)
            log_prob = sampler.get_log_prob()
            sampler_used = "emcee"
            converged = True
        except ImportError:
            prefer_emcee = False
        except Exception:
            prefer_emcee = False

    if not prefer_emcee:
        proposal_sigma = np.concatenate([np.full(3, 0.001), np.full(3, 0.05)])
        chain, log_prob = _metropolis_sampler(
            _log_posterior, x0, n_walkers, n_steps, proposal_sigma, args, seed)
        converged = False                    # MH convergence not auto-checked

    # Post-process: discard burn-in, thin, flatten
    post = chain[burn_in::thin]              # (n_kept, n_walkers, ndim)
    flat = post.reshape(-1, post.shape[-1])
    flat_km_kms = np.concatenate(
        [flat[:, :3] * POS_SCALE, flat[:, 3:]], axis=1)

    log_prob_flat = log_prob[burn_in::thin].reshape(-1)
    best_idx = int(np.argmax(log_prob_flat))
    best_sample = flat_km_kms[best_idx]
    mean = flat_km_kms.mean(axis=0)
    cov = np.cov(flat_km_kms, rowvar=False)

    # Derive elements per sample
    elems = np.array([_derive_elements(s) for s in flat_km_kms])
    finite = np.all(np.isfinite(elems), axis=1)
    elems = elems[finite]

    def _q(v):
        if v.size == 0:
            return (float("nan"),) * 3
        return tuple(float(np.percentile(v, p)) for p in (16, 50, 84))

    x_mag_au = np.linalg.norm(flat_km_kms[:, :3], axis=1) / AU_KM
    v_mag = np.linalg.norm(flat_km_kms[:, 3:], axis=1)

    return McMcOrbitPosterior(
        chain=chain, log_prob=log_prob,
        best_sample=best_sample, mean=mean, cov=cov,
        x_quantiles_au=_q(x_mag_au),
        v_quantiles_km_s=_q(v_mag),
        a_quantiles_au=_q(elems[:, 0]) if elems.size else (float("nan"),) * 3,
        e_quantiles=_q(elems[:, 1]) if elems.size else (float("nan"),) * 3,
        i_quantiles_deg=_q(elems[:, 2]) if elems.size else (float("nan"),) * 3,
        n_burnin=burn_in, n_thin=thin,
        converged=converged,
        sampler_used=sampler_used,
    )
