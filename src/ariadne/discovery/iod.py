"""Initial Orbit Determination (IOD) -- a real, validated seed for orbit-fit differential correction.

The linker's (r, rdot) hypothesis-search is the natural IOD here: under (r, rdot) each tracklet maps to
a full heliocentric (x, v); propagate all to a common epoch; the (r, rdot) that makes them cluster
tightest IS the best initial orbit. The cluster centroid (x, v) at the reference epoch is then a clean
seed for LM differential correction.

This avoids the classical Gauss-method failure modes (range root-finding, degenerate coplanar geometry,
ambiguous solutions) by using the linker's own constrained geometry. Validated to recover real TNO
orbits from their MPC astrometry within a few percent in (a, e, i).

References:
- Holman, Payne et al. 2018 (HelioLinC -- the (r, rdot) hypothesis is itself an IOD step).
- Bate-Mueller-White, Fundamentals of Astrodynamics, Ch. 5 (orbit determination).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares

from . import linkage as L
from ..data.constants import AU_KM, GM_SUN
from ..dynamics.secular import kepler_step
from ..data.ephemeris import body_state


def iod_hypothesis_search(tracklets, t_ref=None,
                          r_grid_au=None, rdot_grid=None,
                          refine_iters=2):
    """Re-derive the best (r_au, rdot) hypothesis on a candidate's own tracklets and return the
    centroid heliocentric (x, v) at t_ref as the IOD seed.

    Returns dict with x_init, v_init, t_ref, r_au, rdot, scatter_km, n_valid.
    """
    if len(tracklets) < 3:
        return None
    geom = L.precompute_geometry(tracklets)
    if t_ref is None:
        t_ref = float(np.median(geom.t))

    if r_grid_au is None:
        r_grid_au = np.concatenate([np.linspace(30, 80, 51), np.linspace(82, 200, 60),
                                    np.linspace(205, 400, 40)])
    if rdot_grid is None:
        rdot_grid = np.linspace(-1.5, 1.5, 31)            # km/s, step 0.1

    best = {"scatter_km": np.inf}
    for r_au in r_grid_au:
        r_km = r_au * AU_KM
        for rdot in rdot_grid:
            x, v, valid = L.transform(geom, r_km, rdot)
            if valid.sum() < 3:
                continue
            idx = np.where(valid)[0]
            with np.errstate(all="ignore"):
                xref, vref = kepler_step(x[idx], v[idx], GM_SUN, t_ref - geom.t[idx])
            fin = np.all(np.isfinite(xref), axis=1) & np.all(np.isfinite(vref), axis=1)
            if fin.sum() < 3:
                continue
            xref, vref = xref[fin], vref[fin]
            # robust centroid via median (resistant to outlier tracklets)
            x_med = np.median(xref, axis=0)
            scatter = float(np.sum(np.linalg.norm(xref - x_med, axis=1) ** 2))
            if scatter < best["scatter_km"]:
                v_med = np.median(vref, axis=0)
                best = {"x_init": x_med, "v_init": v_med, "t_ref": t_ref,
                        "r_au": float(r_au), "rdot": float(rdot),
                        "scatter_km": scatter, "n_valid": int(fin.sum())}

    if not np.isfinite(best["scatter_km"]):
        return None

    # local refinement: zoom into the (r, rdot) basin
    zooms = [(4.0, 0.20, 25, 21), (1.0, 0.05, 25, 21)]    # (Δr, Δrdot, n_r, n_rdot) per pass
    for (dr, drd, nr, nrd) in zooms[:refine_iters]:
        r0 = best["r_au"]; rd0 = best["rdot"]
        r_grid = np.linspace(max(20, r0 - dr), r0 + dr, nr)
        rd_grid = np.linspace(rd0 - drd, rd0 + drd, nrd)
        for r_au in r_grid:
            r_km = r_au * AU_KM
            for rdot in rd_grid:
                x, v, valid = L.transform(geom, r_km, rdot)
                if valid.sum() < 3:
                    continue
                idx = np.where(valid)[0]
                with np.errstate(all="ignore"):
                    xref, vref = kepler_step(x[idx], v[idx], GM_SUN, t_ref - geom.t[idx])
                fin = np.all(np.isfinite(xref), axis=1) & np.all(np.isfinite(vref), axis=1)
                if fin.sum() < 3:
                    continue
                xref, vref = xref[fin], vref[fin]
                x_med = np.median(xref, axis=0)
                scatter = float(np.sum(np.linalg.norm(xref - x_med, axis=1) ** 2))
                if scatter < best["scatter_km"]:
                    v_med = np.median(vref, axis=0)
                    best = {"x_init": x_med, "v_init": v_med, "t_ref": t_ref,
                            "r_au": float(r_au), "rdot": float(rdot),
                            "scatter_km": scatter, "n_valid": int(fin.sum())}
    return best


def _predict_radec(state, t_ref, t, R_obs):
    r, v = state[:3], state[3:]
    with np.errstate(all="ignore"):
        rt, _ = kepler_step(r, v, GM_SUN, t - t_ref)
    g = rt - R_obs
    rn = float(np.linalg.norm(g))
    if rn < 1e3 or not np.isfinite(rn):
        return float("nan"), float("nan")
    return math.atan2(g[1], g[0]), math.asin(g[2] / rn)


def _light_time_correct_observer(t, R_obs, target_pos_helio):
    """Iterate light-time: the photon we observe at t left the target at t - rho/c.

    For TNOs this is 5-15 hours -- a few arcsec systematic if ignored. Two iterations is enough.
    """
    C_KM_S = 299792.458
    rho = float(np.linalg.norm(target_pos_helio - R_obs))
    for _ in range(2):
        tau = rho / C_KM_S
        R_at_emit = body_state("EARTH", t - tau, "J2000", "SUN")[:3]
        rho = float(np.linalg.norm(target_pos_helio - R_at_emit))
    return tau, R_at_emit


def fit_orbit_lm(tracklet_records, t_ref, x_init, v_init,
                 light_time=True, max_nfev=2000):
    """LM differential correction from an IOD seed. Returns dict with x_fit, v_fit, rms_arcsec, nfev.

    Uses scipy LM (linear loss, full residuals -- not soft_l1; we want LM to drive residuals all the
    way to ~arcsec). State is internally rescaled so position-component (km, ~1e10) and velocity-
    component (km/s, ~few) have comparable scale to the optimiser's step-size logic. With light-time
    correction (~ few arcsec systematic if omitted).
    """
    ts = np.array([t["t"] for t in tracklet_records])
    ras = np.array([t["ra"] for t in tracklet_records])
    decs = np.array([t["dec"] for t in tracklet_records])

    Ro_t = np.array([body_state("EARTH", float(t), "J2000", "SUN")[:3] for t in ts])
    # state rescale: pos in AU, vel in km/s -- comparable magnitudes for LM step heuristics
    POS_SCALE = AU_KM

    def residuals(state_scaled):
        out = np.empty(2 * len(ts))
        r = state_scaled[:3] * POS_SCALE                       # back to km
        v = state_scaled[3:]
        for k in range(len(ts)):
            with np.errstate(all="ignore"):
                rt, _ = kepler_step(r, v, GM_SUN, float(ts[k]) - t_ref)
            if not np.all(np.isfinite(rt)):
                out[2 * k] = 1e-3; out[2 * k + 1] = 1e-3        # ~200 arcsec, large but non-pathological
                continue
            if light_time:
                rho = float(np.linalg.norm(rt - Ro_t[k]))
                if not np.isfinite(rho) or rho < 1e3:
                    out[2 * k] = 1e-3; out[2 * k + 1] = 1e-3
                    continue
                tau = rho / 299792.458
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

    x0 = np.concatenate([np.asarray(x_init) / POS_SCALE, np.asarray(v_init)])
    try:
        r = least_squares(residuals, x0, method="lm",
                          xtol=1e-14, ftol=1e-14, max_nfev=max_nfev)
        rms_rad = math.sqrt((r.fun ** 2).mean())
        x_fit_km = r.x[:3] * POS_SCALE
        v_fit = r.x[3:]
        return {"x_fit": x_fit_km, "v_fit": v_fit,
                "rms_arcsec": float(rms_rad * 206265),
                "nfev": int(r.nfev), "success": bool(r.success)}
    except Exception as e:
        return {"x_fit": x_init, "v_fit": v_init,
                "rms_arcsec": float("inf"), "nfev": 0, "success": False, "err": str(e)[:120]}


def fit_candidate(tracklet_records, t_ref=None):
    """One-shot: IOD + LM. Returns full fit dict or None on geometric/IOD failure."""
    if t_ref is None:
        t_ref = float(np.median([t["t"] for t in tracklet_records]))
    seed = iod_hypothesis_search(tracklet_records, t_ref=t_ref)
    if seed is None:
        return None
    fit = fit_orbit_lm(tracklet_records, t_ref, seed["x_init"], seed["v_init"])
    fit["t_ref"] = t_ref
    fit["iod"] = {"r_au": seed["r_au"], "rdot": seed["rdot"],
                  "scatter_km": seed["scatter_km"], "n_valid": seed["n_valid"]}
    return fit
