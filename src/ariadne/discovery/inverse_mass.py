"""Inverse hidden-mass localizer + two-tier discovery pipeline (MASTER_PLAN.md - Stage 28).

Stage 27 was the FORWARD model (given a body, the residual it makes). This is the INVERSE problem
and the user's two-tier vision: a broad sensitive overview flags an anomaly, then we hone in to a
location -- exactly how Neptune was found (Le Verrier turned Uranus's residuals into a sky position).

  Tier 1 (overview)  -- residual accelerations across many tracked bodies flag an anomaly.
  Tier 2 (localize)  -- weighted nonlinear least-squares solves for the hidden body's (position, GM)
                        from those residual VECTORS; diverse tracked bodies triangulate the source.
                        Returns a best fit + a covariance -> a confidence region (NOT a magic point).
  Honing             -- the confidence region SHRINKS as more tracked bodies are added.
  Handoff            -- the localization yields a sky search-box (ecliptic lon/lat + distance + 1-sigma)
                        = where to point IR/optical surveys (the multi-band fusion step).

HONEST limits, baked in:
  - DETECTABILITY FLOOR: a body whose GM/d^2 is below the noise floor is unrecoverable (Stage 27 map).
  - DEGENERACY: a single tracked body gives only a direction + a mass/distance^2 product; you need
    >= 2 geometrically-diverse bodies to triangulate. We demonstrate the single-body failure.
  - GM only: gravity yields GM, never density/composition (shell theorem) -- that needs a size
    measurement from the IR/optical handoff.
  - SIMULATION: recovery here inverts the forward model under measurement noise. A real detection
    additionally needs secular (Myr) accumulation, non-gravitational force modelling, and real data.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares

from ..fields.hidden_mass import residual_accel, AU_KM, GM_EARTH


def simulate_observations(tracked_positions, hidden_gm, hidden_pos, noise_ms2=1e-14, seed=0):
    """Residual-acceleration 'observations' (km/s^2) at each tracked body + Gaussian noise."""
    rng = np.random.default_rng(seed)
    sig_kms = noise_ms2 / 1000.0
    obs = []
    for x in tracked_positions:
        a = residual_accel(np.asarray(x, float), hidden_gm, np.asarray(hidden_pos, float))
        obs.append(a + rng.normal(0.0, sig_kms, 3))
    return np.array(obs)


def localize(tracked_positions, observations, noise_ms2=1e-14, x0_au=None):
    """Weighted nonlinear least-squares for the hidden body's (position, GM) from residual vectors.

    Params = [x_au, y_au, z_au, log10(m_earth)]. Returns best-fit position (km), GM, and the
    position covariance (km^2) -> the confidence region. Diverse tracked bodies triangulate it.
    """
    P = [np.asarray(x, float) for x in tracked_positions]
    obs = np.asarray(observations, float)
    sig_kms = noise_ms2 / 1000.0

    if 3 * len(P) < 4:        # underdetermined: a single tracked body gives only a direction
        return {"position": np.full(3, np.nan), "gm": np.nan, "m_earth": np.nan,
                "cost": np.nan, "cov_pos_km2": None, "pos_sigma_km": math.inf,
                "success": False, "degenerate": True}

    def resid(p):
        r = p[:3] * AU_KM
        gm = (10.0 ** p[3]) * GM_EARTH
        pred = np.array([residual_accel(x, gm, r) for x in P])
        return ((pred - obs) / sig_kms).ravel()

    if x0_au is None:
        # crude Tier-1 seed: mean tracked distance, modest mass
        x0_au = [np.mean([np.linalg.norm(x) for x in P]) / AU_KM, 10.0, 10.0, 0.5]
    res = least_squares(resid, np.asarray(x0_au, float), method="lm", max_nfev=4000)

    r_fit = res.x[:3] * AU_KM
    gm_fit = (10.0 ** res.x[3]) * GM_EARTH
    cov_pos = None
    try:
        cov = np.linalg.inv(res.jac.T @ res.jac)        # weighted residuals -> unit-variance cov
        cov_pos = cov[:3, :3] * AU_KM ** 2               # back to km^2
    except np.linalg.LinAlgError:
        pass
    pos_sigma_km = float(np.sqrt(np.trace(cov_pos))) if cov_pos is not None else math.inf
    return {"position": r_fit, "gm": gm_fit, "m_earth": gm_fit / GM_EARTH,
            "cost": float(res.cost), "cov_pos_km2": cov_pos,
            "pos_sigma_km": pos_sigma_km, "success": bool(res.success)}


def sky_box(position_km, pos_sigma_km):
    """Telescope search-box from a localization: ecliptic lon/lat (deg), distance (AU), 1-sigma (deg)."""
    x, y, z = position_km
    r = math.sqrt(x * x + y * y + z * z)
    lon = math.degrees(math.atan2(y, x)) % 360.0
    lat = math.degrees(math.asin(z / r)) if r > 0 else 0.0
    ang_sigma = math.degrees(pos_sigma_km / r) if r > 0 and math.isfinite(pos_sigma_km) else float("inf")
    return {"ecliptic_lon_deg": lon, "ecliptic_lat_deg": lat,
            "distance_au": r / AU_KM, "angular_sigma_deg": ang_sigma}


def localization_vs_n(tracked_positions, hidden_gm, hidden_pos, noise_ms2=1e-14, seed=0):
    """How the localization tightens as tracked bodies are added (the two-tier honing)."""
    rows = []
    for n in range(2, len(tracked_positions) + 1):
        P = tracked_positions[:n]
        obs = simulate_observations(P, hidden_gm, hidden_pos, noise_ms2, seed)
        r = localize(P, obs, noise_ms2)
        err = float(np.linalg.norm(r["position"] - np.asarray(hidden_pos, float))) / AU_KM
        rows.append({"n": n, "pos_sigma_au": r["pos_sigma_km"] / AU_KM,
                     "pos_error_au": err, "m_earth": r["m_earth"]})
    return rows


def sensitivity_skymap(tracked_positions, distance_au, noise_ms2=1e-14, n_lon=24, n_lat=12):
    """Minimum detectable mass (Earth masses) vs hidden-body sky direction at fixed distance.

    For each direction, the body's residual at the BEST-placed tracked body must exceed the noise;
    high min-mass = a blind spot. Reveals where a hidden body could hide from this network.
    """
    lons = np.linspace(0, 360, n_lon, endpoint=False)
    lats = np.linspace(-80, 80, n_lat)
    floor_kms = noise_ms2 / 1000.0
    grid = np.zeros((n_lat, n_lon))
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            la, lo = math.radians(lat), math.radians(lon)
            d = distance_au * AU_KM
            pos = np.array([d * math.cos(la) * math.cos(lo),
                            d * math.cos(la) * math.sin(lo), d * math.sin(la)])
            # smallest GM s.t. residual at the closest tracked body exceeds the noise floor
            dmin = min(np.linalg.norm(pos - np.asarray(x, float)) for x in tracked_positions)
            gm_min = floor_kms * dmin ** 2               # GM with residual = floor at closest body
            grid[i, j] = gm_min / GM_EARTH
    return {"lons": lons, "lats": lats, "min_mass_earth": grid, "distance_au": distance_au}
