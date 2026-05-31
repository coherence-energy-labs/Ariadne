"""N-body orbit fit -- use the full solar-system propagator in the LM residuals.

The existing `iod.fit_orbit_lm` uses a 2-body Kepler propagator. For NEOs and
inner-main-belt asteroids, the 2-body assumption is fine over months; for
TNOs and Centaurs near Neptune, planetary perturbations DOMINATE the orbit
evolution over multi-year arcs. A long-arc orbit fit with 2-body propagation
will produce systematically wrong elements -- typically several arcseconds of
mismatch on the residuals, hiding the true astrometric precision.

This module repeats the IOD seed -> LM differential correction loop, but with
the test-particle N-body propagator from `dynamics.ephemeris_nbody` doing the
propagation in the residual function. Default perturbers: SUN (central),
+EARTH +MOON +VENUS +MARS +JUPITER +SATURN +URANUS +NEPTUNE.

The fit is more expensive (~10-20x per residual evaluation) but produces
elements consistent with JPL Horizons to within the astrometric noise floor.
Use for: long-arc orbits, anything beyond Mars, candidates that already have
a clean 2-body fit and need final-quality elements before MPC submission.

Reference: Bowell+ 1989 (asteroid orbit refinement w/ planetary perturbations);
Bernstein-Khushalani 2000 (TNO 6-D orbit fitting).
"""
from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.optimize import least_squares

from ..data.constants import AU_KM, GM_SUN
from ..data.ephemeris import body_state
from ..dynamics.ephemeris_nbody import propagate_test_particle
from . import iod as IOD


DEFAULT_PERTURBERS = ("EARTH", "MOON", "VENUS", "MARS", "JUPITER",
                       "SATURN", "URANUS", "NEPTUNE")

C_KM_S = 299792.458


def _residuals_nbody(state_scaled, ts, ras, decs, Ro_t, t_ref,
                     perturbers, light_time, pos_scale):
    """LM residual function with N-body propagation between t_ref and each obs."""
    r = state_scaled[:3] * pos_scale
    v = state_scaled[3:]
    out = np.empty(2 * len(ts))
    for k in range(len(ts)):
        dt = float(ts[k]) - t_ref
        # propagate from t_ref to obs time using N-body
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sol = propagate_test_particle(
                    r, v, t_ref, (0.0, dt) if dt > 0 else (dt, 0.0),
                    central="SUN", perturbers=perturbers,
                    t_eval=[dt])
            rt = sol.y[:3, -1] if sol.t.size > 0 else np.array([0.0, 0.0, 0.0])
        except Exception:
            out[2 * k] = 1e-3; out[2 * k + 1] = 1e-3
            continue
        if not np.all(np.isfinite(rt)):
            out[2 * k] = 1e-3; out[2 * k + 1] = 1e-3
            continue
        if light_time:
            rho = float(np.linalg.norm(rt - Ro_t[k]))
            if not np.isfinite(rho) or rho < 1e3:
                out[2 * k] = 1e-3; out[2 * k + 1] = 1e-3
                continue
            tau = rho / C_KM_S
            R_em = body_state("EARTH", float(ts[k]) - tau, "J2000", "SUN")[:3]
            g = rt - R_em
        else:
            g = rt - Ro_t[k]
        rn = float(np.linalg.norm(g))
        if not np.isfinite(rn) or rn < 1e3:
            out[2 * k] = 1e-3; out[2 * k + 1] = 1e-3
            continue
        ra_p = math.atan2(g[1], g[0])
        dec_p = math.asin(max(-1.0, min(1.0, g[2] / rn)))
        dra = (ra_p - ras[k] + math.pi) % (2 * math.pi) - math.pi
        ddec = dec_p - decs[k]
        out[2 * k] = dra * math.cos(decs[k])
        out[2 * k + 1] = ddec
    return out


def fit_orbit_nbody(tracklet_records, t_ref: float, x_init, v_init,
                    *, perturbers: tuple = DEFAULT_PERTURBERS,
                    light_time: bool = True,
                    max_nfev: int = 200):
    """N-body LM differential correction starting from an IOD seed.

    Slow (each residual = one full N-body integration). Use only when the
    2-body fit has converged well enough to be IN THE SAME BASIN as the
    true orbit -- typically initial guess RMS < ~5 arcsec.

    Args:
      tracklet_records: list of dicts with keys 't' (ET seconds), 'ra', 'dec'.
      t_ref:            reference epoch (ET seconds past J2000).
      x_init, v_init:   heliocentric initial state at t_ref (km, km/s).
      perturbers:       tuple of body names to include as perturbers.
      light_time:       iterate light-time correction at each residual eval.
      max_nfev:         LM iteration cap (smaller default than 2-body because
                         each iteration is much more expensive).

    Returns:
      dict with x_fit, v_fit, rms_arcsec, nfev, success, perturbers_used.
    """
    ts = np.array([t["t"] for t in tracklet_records])
    ras = np.array([t["ra"] for t in tracklet_records])
    decs = np.array([t["dec"] for t in tracklet_records])

    Ro_t = np.array([body_state("EARTH", float(t), "J2000", "SUN")[:3] for t in ts])
    POS_SCALE = AU_KM
    x0 = np.concatenate([np.asarray(x_init) / POS_SCALE, np.asarray(v_init)])

    try:
        r = least_squares(
            _residuals_nbody, x0, method="lm",
            args=(ts, ras, decs, Ro_t, t_ref, perturbers, light_time, POS_SCALE),
            xtol=1e-12, ftol=1e-12, max_nfev=max_nfev,
        )
        rms_rad = math.sqrt((r.fun ** 2).mean())
        x_fit_km = r.x[:3] * POS_SCALE
        v_fit = r.x[3:]
        return {
            "x_fit": x_fit_km, "v_fit": v_fit,
            "rms_arcsec": float(rms_rad * 206265),
            "nfev": int(r.nfev), "success": bool(r.success),
            "perturbers_used": list(perturbers),
        }
    except Exception as e:
        return {
            "x_fit": x_init, "v_fit": v_init,
            "rms_arcsec": float("inf"),
            "nfev": 0, "success": False,
            "perturbers_used": list(perturbers),
            "err": str(e)[:120],
        }


def fit_candidate_nbody(tracklet_records, *, t_ref=None,
                        perturbers: tuple = DEFAULT_PERTURBERS):
    """One-shot: IOD seed (via 2-body, fast) + N-body LM refinement.

    The IOD seed uses 2-body propagation (fast); the final refinement uses
    N-body. The IOD must produce a basin-of-attraction-quality seed before
    the N-body LM converges -- if the seed RMS is huge, the N-body fit will
    waste expensive iterations chasing a wrong basin.

    Returns the same dict shape as iod.fit_candidate, with additional
    'perturbers_used' field.
    """
    if t_ref is None:
        t_ref = float(np.median([t["t"] for t in tracklet_records]))
    seed = IOD.iod_hypothesis_search(tracklet_records, t_ref=t_ref)
    if seed is None:
        return None
    # FIRST do a 2-body LM refinement to get into the correct basin
    fit_2body = IOD.fit_orbit_lm(tracklet_records, t_ref,
                                  seed["x_init"], seed["v_init"])
    if fit_2body["rms_arcsec"] > 30.0 or not fit_2body["success"]:
        return None
    # NOW do the N-body refinement starting from the 2-body fit
    fit = fit_orbit_nbody(tracklet_records, t_ref,
                           fit_2body["x_fit"], fit_2body["v_fit"],
                           perturbers=perturbers)
    fit["t_ref"] = t_ref
    fit["iod"] = {"r_au": seed["r_au"], "rdot": seed["rdot"],
                  "scatter_km": seed["scatter_km"], "n_valid": seed["n_valid"]}
    fit["rms_2body_arcsec"] = fit_2body["rms_arcsec"]
    return fit
