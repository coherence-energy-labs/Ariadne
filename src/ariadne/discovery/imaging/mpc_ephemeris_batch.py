"""Vectorized batch ephemeris computation for the MPC catalog cross-match.

The serial loop in mpc_catalog.flag_known_in_db is ~1 ms / orbit, so
1.5M orbits at one epoch takes ~25 minutes. That's untenable for
operational survey use.

This module batches every step into numpy so 1.5M orbits propagate +
project in O(seconds). Key optimisations:

  1. Bulk Kepler elements -> heliocentric state (already in dynamics
     but called per-orbit in the catalog module).
  2. Bulk kepler_step propagation: solve N Kepler equations at once
     via vectorized Newton iteration.
  3. Single Earth ephemeris call per epoch (was being repeated).
  4. Bulk geocentric -> sky projection via numpy.

For multi-epoch queries (e.g., one orbit per detection time), we batch
by epoch since the propagation is the cost driver.

Public API:
  bulk_ephemeris_at_mjd(records, target_mjd)
    Return (N, 4) array of (ra_deg, dec_deg, mag_est, geocentric_au).
    Fully vectorized; ~1000x faster than the per-orbit serial loop.

  bulk_cross_match(detections, records, target_mjd, ...)
    Predict every record's position, build a KD-tree on the sky, look
    up each detection's nearest known.
"""
from __future__ import annotations

import math
import time
from typing import Sequence

import numpy as np


def _solve_kepler_E_vec(M: np.ndarray, e: np.ndarray,
                         *, n_iter: int = 8, tol: float = 1e-10
                         ) -> np.ndarray:
    """Bulk Newton solver for Kepler's equation M = E - e sin(E).

    Returns the eccentric anomaly E (radians). Vectorized over all
    input M/e (same shape). Stable for e < 0.95.
    """
    # Initial guess: M for low-e, M + e*sin(M) for higher
    E = np.where(e < 0.3, M, M + e * np.sin(M))
    for _ in range(n_iter):
        f = E - e * np.sin(E) - M
        fp = 1 - e * np.cos(E)
        dE = f / fp
        E = E - dE
        if np.max(np.abs(dE)) < tol:
            break
    return E


def bulk_elements_to_state(a_au: np.ndarray, e: np.ndarray,
                              i_deg: np.ndarray, Omega_deg: np.ndarray,
                              omega_deg: np.ndarray, M_deg: np.ndarray
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Keplerian elements -> heliocentric state (r, v) in km, km/s.

    All inputs are 1-D arrays of length N. Returns r (N, 3) and v (N, 3).
    """
    from ...data.constants import GM_SUN, AU_KM
    N = len(a_au)
    a_km = a_au * AU_KM
    inc = np.radians(i_deg)
    Omega = np.radians(Omega_deg)
    omega = np.radians(omega_deg)
    M = np.radians(M_deg)
    E = _solve_kepler_E_vec(M, e)
    # True anomaly
    cos_E = np.cos(E); sin_E = np.sin(E)
    sqrt_one_minus_e2 = np.sqrt(np.maximum(1 - e * e, 0))
    cos_nu = (cos_E - e) / (1 - e * cos_E)
    sin_nu = (sqrt_one_minus_e2 * sin_E) / (1 - e * cos_E)
    # Distance from focus
    r_orb = a_km * (1 - e * cos_E)
    # Position in orbital plane (perifocal frame): (cos_nu, sin_nu, 0)
    x_orb = r_orb * cos_nu
    y_orb = r_orb * sin_nu
    # Velocity in perifocal frame: derivative of (r, theta)
    mu_over_p = GM_SUN / (a_km * (1 - e * e))
    sqrt_mu_over_p = np.sqrt(np.maximum(mu_over_p, 0))
    vx_orb = -sqrt_mu_over_p * sin_nu
    vy_orb = sqrt_mu_over_p * (e + cos_nu)
    # Rotation: perifocal -> ECI
    cos_O = np.cos(Omega); sin_O = np.sin(Omega)
    cos_w = np.cos(omega); sin_w = np.sin(omega)
    cos_i = np.cos(inc); sin_i = np.sin(inc)
    # Rotation matrix elements
    P11 = cos_O * cos_w - sin_O * sin_w * cos_i
    P12 = -cos_O * sin_w - sin_O * cos_w * cos_i
    P21 = sin_O * cos_w + cos_O * sin_w * cos_i
    P22 = -sin_O * sin_w + cos_O * cos_w * cos_i
    P31 = sin_w * sin_i
    P32 = cos_w * sin_i
    r = np.column_stack([
        P11 * x_orb + P12 * y_orb,
        P21 * x_orb + P22 * y_orb,
        P31 * x_orb + P32 * y_orb,
    ])
    v = np.column_stack([
        P11 * vx_orb + P12 * vy_orb,
        P21 * vx_orb + P22 * vy_orb,
        P31 * vx_orb + P32 * vy_orb,
    ])
    return r, v


def bulk_kepler_step(r0: np.ndarray, v0: np.ndarray,
                       dt_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized 2-body propagation via universal-variable Stumpff
    method.

    Each row of r0, v0 is one object; dt_s is per-object (or scalar).
    Returns (r, v) at t0 + dt for each.

    For our cross-match use case where dt is typically days to years
    and orbits are bound (e < 1), we use a simpler approach: re-derive
    the elements -> propagate M -> elements -> state. This is O(N) per
    propagation step and avoids the universal-variable convergence
    issues.
    """
    from ...data.constants import GM_SUN
    # Compute current orbital elements per object
    r_norm = np.linalg.norm(r0, axis=1)
    v_norm = np.linalg.norm(v0, axis=1)
    # Specific energy + angular momentum
    eps = 0.5 * v_norm ** 2 - GM_SUN / np.maximum(r_norm, 1.0)
    a = -GM_SUN / (2 * eps)   # negative for bound orbits; positive
    # Mean motion
    n = np.sqrt(GM_SUN / np.maximum(np.abs(a) ** 3, 1.0))
    # Current eccentric anomaly via:
    # r dot v = sqrt(GM_SUN * a) * e * sin(E)
    # 1 - r/a = e * cos(E)
    rv_dot = np.sum(r0 * v0, axis=1)
    sin_E_e = rv_dot / np.sqrt(GM_SUN * np.maximum(np.abs(a), 1.0))
    cos_E_e = 1 - r_norm / np.maximum(np.abs(a), 1.0)
    E0 = np.arctan2(sin_E_e, cos_E_e)
    e = np.sqrt(np.maximum(sin_E_e ** 2 + cos_E_e ** 2, 0))
    # Current mean anomaly
    M0 = E0 - e * np.sin(E0)
    # New mean anomaly
    M = M0 + n * dt_s
    # Solve Kepler's equation for new E
    E = _solve_kepler_E_vec(M, e)
    # Lagrangian coefficients (Vallado §2.3)
    f = 1 - (a / np.maximum(r_norm, 1.0)) * (1 - np.cos(E - E0))
    sqrt_a3_over_mu = np.sqrt(np.maximum(np.abs(a), 1.0) ** 3 / GM_SUN)
    g = dt_s - sqrt_a3_over_mu * ((E - E0) - np.sin(E - E0))
    # New position
    r_new = (f[:, None] * r0 + g[:, None] * v0)
    r_new_norm = np.linalg.norm(r_new, axis=1)
    # New velocity via fdot/gdot
    fdot = (-np.sqrt(GM_SUN * np.maximum(np.abs(a), 1.0))
             / (r_norm * r_new_norm)
             * np.sin(E - E0))
    gdot = 1 - (a / np.maximum(r_new_norm, 1.0)) * (1 - np.cos(E - E0))
    v_new = (fdot[:, None] * r0 + gdot[:, None] * v0)
    return r_new, v_new


# Mean obliquity of the ecliptic at J2000 (IAU 2006), radians.
OBLIQUITY_J2000_RAD = math.radians(23.4392911)
_C_KM_S = 299792.458   # speed of light, km/s


def ecliptic_to_equatorial(r: np.ndarray) -> np.ndarray:
    """Rotate (N,3) vectors from the J2000 ECLIPTIC frame to the J2000
    EQUATORIAL (ICRF) frame by the obliquity about the X axis.

    MPCORB orbital elements (i, Omega, omega) are referenced to the
    ecliptic, so `bulk_elements_to_state` yields ecliptic-frame positions.
    SPICE body_state in the "J2000" frame is EQUATORIAL. Subtracting them
    directly mixes frames and throws RA/Dec off by up to the obliquity
    (~23 deg) -- this rotation is mandatory before forming the geocentric
    vector and computing RA/Dec.
    """
    ce = math.cos(OBLIQUITY_J2000_RAD)
    se = math.sin(OBLIQUITY_J2000_RAD)
    out = np.empty_like(r)
    out[:, 0] = r[:, 0]
    out[:, 1] = r[:, 1] * ce - r[:, 2] * se
    out[:, 2] = r[:, 1] * se + r[:, 2] * ce
    return out


def _apply_stellar_aberration(geo: np.ndarray, v_obs_km_s: np.ndarray
                                 ) -> np.ndarray:
    """Shift geocentric direction vectors for stellar aberration given the
    observer's velocity (km/s, equatorial). Returns vectors of the same
    magnitudes with apparent-place directions."""
    rho = np.linalg.norm(geo, axis=1, keepdims=True)
    u = geo / np.maximum(rho, 1.0)
    beta = v_obs_km_s / _C_KM_S            # (3,)
    u_app = u + beta[None, :]              # first-order aberration
    u_app = u_app / np.linalg.norm(u_app, axis=1, keepdims=True)
    return u_app * rho


def bulk_ephemeris_at_mjd(records: Sequence,
                             target_mjd: float,
                             *, observer_geo_km: np.ndarray | None = None,
                             light_time: bool = True,
                             stellar_aberration: bool = False,
                             ) -> np.ndarray:
    """Compute predicted (ra_deg, dec_deg, mag, geocentric_au) for every
    record at `target_mjd`.

    Applies the ecliptic->equatorial rotation (mandatory; MPCORB elements
    are ecliptic, SPICE Earth is equatorial), planetary-aberration
    (light-time) correction, and an optional topocentric observer offset.

    Args:
      observer_geo_km: (3,) observer position relative to geocenter in the
                       J2000 equatorial frame (km). None -> geocentric.
      light_time:      iterate the asteroid position back by rho/c so the
                       predicted direction is where the object was when the
                       observed light left it.

    Returns array shape (N, 4). Records is a sequence of OrbitalElements.
    """
    N = len(records)
    if N == 0:
        return np.empty((0, 4))
    a_au = np.fromiter((r.a_au for r in records), float, N)
    e = np.fromiter((r.e for r in records), float, N)
    i_deg = np.fromiter((r.i_deg for r in records), float, N)
    Omega_deg = np.fromiter((r.Omega_deg for r in records), float, N)
    omega_deg = np.fromiter((r.omega_deg for r in records), float, N)
    M_deg = np.fromiter((r.M_deg for r in records), float, N)
    epoch_mjd = np.fromiter((r.epoch_mjd for r in records), float, N)
    H_mag = np.fromiter((r.H_mag for r in records), float, N)
    return bulk_ephemeris_from_arrays(
        a_au, e, i_deg, Omega_deg, omega_deg, M_deg, epoch_mjd, H_mag,
        target_mjd, observer_geo_km=observer_geo_km,
        light_time=light_time, stellar_aberration=stellar_aberration)


def bulk_ephemeris_from_arrays(a_au, e, i_deg, Omega_deg, omega_deg, M_deg,
                                 epoch_mjd, H_mag, target_mjd,
                                 *, observer_geo_km: np.ndarray | None = None,
                                 light_time: bool = True,
                                 stellar_aberration: bool = False
                                 ) -> np.ndarray:
    """Array-native core of `bulk_ephemeris_at_mjd`.

    Takes parallel numpy arrays of orbital elements directly, avoiding the
    per-record Python attribute access + object materialisation. The
    operational catalog cross-match loads element columns straight into
    arrays and calls this -- ~2x faster than going through OrbitalElements.
    """
    from ...data.constants import AU_KM
    from ...data.ephemeris import body_state

    from ...data.constants import GM_SUN
    N = len(a_au)
    if N == 0:
        return np.empty((0, 4))

    # Analytic 2-body advance: when starting from known elements, propagate
    # by advancing the mean anomaly (M = M0 + n*dt) and re-deriving the
    # state -- ONE Kepler solve per evaluation. (The old path went
    # elements->state->kepler_step, and kepler_step re-derived the elements
    # FROM the state vector to propagate -- a second, redundant Kepler
    # solve. Same result, ~2x faster on the 1.5M-orbit catalog scan.)
    a_km = a_au * AU_KM
    n_rad_s = np.sqrt(GM_SUN / np.maximum(a_km, 1.0) ** 3)  # mean motion
    dt_s = (target_mjd - epoch_mjd) * 86400.0

    # Observer position + velocity (heliocentric, equatorial): Earth state.
    et_target = (target_mjd - 51544.5) * 86400.0
    earth_state = np.array(body_state("EARTH", et_target, "J2000", "SUN"),
                             dtype=float)
    R_e = earth_state[:3]
    V_obs = earth_state[3:6]   # km/s, equatorial -- for stellar aberration
    R_obs = R_e if observer_geo_km is None else (R_e + observer_geo_km)

    def _geo_equ(extra_dt_s):
        M_t = M_deg + np.degrees(n_rad_s * (dt_s + extra_dt_s))
        r_t, _ = bulk_elements_to_state(a_au, e, i_deg, Omega_deg,
                                           omega_deg, M_t)
        r_t_equ = ecliptic_to_equatorial(r_t)
        return r_t_equ - R_obs[None, :], r_t_equ

    # Light-time iteration (planetary aberration)
    geo, r_t_equ = _geo_equ(0.0)
    if light_time:
        for _ in range(2):
            rho_km = np.linalg.norm(geo, axis=1)
            geo, r_t_equ = _geo_equ(-rho_km / _C_KM_S)

    # Stellar aberration (~20.6" max). Default OFF: survey detections are
    # calibrated against Gaia/ICRF (an astrometric frame with aberration
    # already removed), so predictions must also be astrometric. Enable
    # only when matching against apparent-place positions.
    if stellar_aberration:
        geo = _apply_stellar_aberration(geo, V_obs)

    rho_km = np.linalg.norm(geo, axis=1)
    valid = rho_km > 1.0
    out = np.full((N, 4), np.nan)
    ra_deg = np.degrees(np.arctan2(geo[:, 1], geo[:, 0])) % 360.0
    dec_deg = np.degrees(np.arcsin(geo[:, 2] / np.maximum(rho_km, 1.0)))
    r_helio_au = np.linalg.norm(r_t_equ, axis=1) / AU_KM
    rho_au = rho_km / AU_KM
    mag_est = np.where(
        (rho_au > 0) & (r_helio_au > 0),
        H_mag + 5.0 * np.log10(np.maximum(rho_au * r_helio_au, 1e-6)),
        np.nan)
    out[valid, 0] = ra_deg[valid]
    out[valid, 1] = dec_deg[valid]
    out[valid, 2] = mag_est[valid]
    out[valid, 3] = rho_au[valid]
    return out


def bulk_cross_match(detections: Sequence[dict],
                       records: Sequence,
                       target_mjd: float,
                       *, match_radius_arcsec: float = 3.0,
                       force_2body: bool = False,
                       observer_geo_km: np.ndarray | None = None,
                       ) -> dict:
    """Vectorized cross-match: bulk-ephemeris every record, build KD-tree
    on the sky, look up each detection's nearest known within tolerance.

    By default selects 2-body vs N-body propagation automatically based
    on max |epoch_delta|. Pass force_2body=True to skip N-body (e.g.,
    for unit tests where you control epoch=target).

    Returns dict with:
      'matches'           {det_id: designation, ...}
      'ephem_array'        (N, 4) predicted (ra, dec, mag, rho)
      'nearest_offset_arcsec' median nearest-neighbor distance (diagnostic)
    """
    from scipy.spatial import cKDTree
    if force_2body:
        eph = bulk_ephemeris_at_mjd(records, target_mjd,
                                      observer_geo_km=observer_geo_km)
    else:
        from .mpc_ephemeris_nbody import auto_ephemeris_at_mjd
        eph = auto_ephemeris_at_mjd(records, target_mjd,
                                      observer_geo_km=observer_geo_km)
    # Drop NaN rows
    valid = ~np.isnan(eph[:, 0])
    if not np.any(valid):
        return {"matches": {}, "ephem_array": eph,
                  "nearest_offset_arcsec": float("inf")}
    pred_ra = eph[valid, 0]
    pred_dec = eph[valid, 1]
    # Map sky coords to 3D unit vectors so we can use KD-tree on the sphere
    cos_dec = np.cos(np.radians(pred_dec))
    pred_xyz = np.column_stack([
        cos_dec * np.cos(np.radians(pred_ra)),
        cos_dec * np.sin(np.radians(pred_ra)),
        np.sin(np.radians(pred_dec)),
    ])
    tree = cKDTree(pred_xyz)
    radius_rad = math.radians(match_radius_arcsec / 3600.0)
    # Convert det positions to 3D
    det_ra = np.array([d["ra"] for d in detections])
    det_dec = np.array([d["dec"] for d in detections])
    det_cos_dec = np.cos(np.radians(det_dec))
    det_xyz = np.column_stack([
        det_cos_dec * np.cos(np.radians(det_ra)),
        det_cos_dec * np.sin(np.radians(det_ra)),
        np.sin(np.radians(det_dec)),
    ])
    # Chord distance = 2 sin(angle/2). For small angles, chord ≈ angle.
    # tree.query returns the chord distance directly.
    dists, idxs = tree.query(det_xyz, distance_upper_bound=2 * math.sin(radius_rad / 2))
    matches = {}
    valid_records = [r for i, r in enumerate(records) if valid[i]]
    for det, dist, idx in zip(detections, dists, idxs):
        if math.isinf(dist) or idx >= len(valid_records):
            continue
        matches[int(det["id"])] = valid_records[idx].designation
    # Diagnostic: median nearest-neighbor distance across all detections
    finite = dists[np.isfinite(dists)]
    if finite.size:
        median_dist_arcsec = (
            2 * math.degrees(math.asin(np.median(finite) / 2)) * 3600.0)
    else:
        median_dist_arcsec = float("inf")
    return {
        "matches": matches,
        "ephem_array": eph,
        "nearest_offset_arcsec": median_dist_arcsec,
    }
