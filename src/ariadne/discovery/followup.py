"""Active follow-up predictor -- where will this candidate be tomorrow?

Once a candidate has a fitted orbit (stored as 6D heliocentric x,v at t_ref),
we can predict its apparent (RA, Dec) at any future epoch by:
  1. Kepler-propagating the orbit forward.
  2. Subtracting the Earth's heliocentric position at that epoch.
  3. Converting the topocentric vector to (RA, Dec).
  4. (Optional) Light-time correction.

We also estimate the ephemeris uncertainty by Monte-Carlo-perturbing the initial
state by 1-sigma orbit-fit errors, propagating each draw, and reporting the
spread on the predicted sky position. This is what tells you whether the next
night's observation needs a 5-arcsec window or a 5-arcmin search box.

The output is two-fold:
  * `predict_ephemeris(candidate, mjd)` -> single best-estimate (RA, Dec).
  * `predict_with_uncertainty(candidate, mjd, n_samples=50)` -> (mu, sigma) on the sky.

A nightly orchestrator wraps this to emit a per-candidate "look here next" list:
  `next_night_targets(store, mjd_next)` -> [(key, ra, dec, sigma_arcsec), ...]

Use this to auto-schedule confirmation observations and self-extend candidate arcs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..data.constants import GM_SUN
from ..data.ephemeris import body_state
from ..dynamics.secular import kepler_step
from .operations.candidate_store import Candidate


SEC_PER_DAY = 86400.0
C_KM_S = 299792.458


@dataclass
class Ephemeris:
    """Predicted apparent position + uncertainty.

    Fields:
      mjd:               epoch of prediction (UTC).
      ra_deg, dec_deg:   apparent equatorial position.
      sigma_arcsec:      1-sigma uncertainty radius on the sky (max axis of the
                         projection of the Monte-Carlo cloud).
      sun_distance_au:   heliocentric distance at the predicted epoch.
      earth_distance_au: geocentric distance (range) at the predicted epoch.
      phase_angle_deg:   Sun-target-observer angle.
      v_sky_arcsec_hr:   apparent on-sky rate at the predicted epoch.
    """
    mjd: float
    ra_deg: float
    dec_deg: float
    sigma_arcsec: float
    sun_distance_au: float
    earth_distance_au: float
    phase_angle_deg: float
    v_sky_arcsec_hr: float


def _mjd_to_et(mjd: float) -> float:
    """MJD (UTC) -> ephemeris time seconds past J2000."""
    jd = mjd + 2400000.5
    return (jd - 2451545.0) * SEC_PER_DAY


def _radec_from_geocentric(rx: float, ry: float, rz: float) -> tuple[float, float, float]:
    """Geocentric inertial (km) -> (RA deg, Dec deg, range km)."""
    rn = math.hypot(math.hypot(rx, ry), rz)
    ra = math.atan2(ry, rx) % (2 * math.pi)
    dec = math.asin(max(-1.0, min(1.0, rz / rn)))
    return math.degrees(ra), math.degrees(dec), rn


def _propagate_apparent(x_helio: np.ndarray, v_helio: np.ndarray,
                        t_ref_et: float, t_obs_et: float,
                        light_time: bool = True) -> tuple[float, float, float, float, float, float]:
    """Propagate one heliocentric state to t_obs and return apparent geometry.

    Returns: ra_deg, dec_deg, range_km, sun_dist_km, phase_deg, v_sky_rad_s
    """
    dt = t_obs_et - t_ref_et
    rt, vt = kepler_step(x_helio, v_helio, GM_SUN, dt)
    if light_time:
        # iterate twice: photon left target at t - rho/c
        R_obs = body_state("EARTH", t_obs_et, "J2000", "SUN")[:3]
        rho = float(np.linalg.norm(rt - R_obs))
        for _ in range(2):
            tau = rho / C_KM_S
            rt_em, _ = kepler_step(x_helio, v_helio, GM_SUN, dt - tau)
            R_em = body_state("EARTH", t_obs_et - tau, "J2000", "SUN")[:3]
            rho = float(np.linalg.norm(rt_em - R_em))
        geo = rt_em - R_em
        sun_at_em = rt_em
        # velocity for on-sky rate: same kepler step, same epoch
        vt_em = vt                                          # tangent vector at emission ~= vt for short dt
    else:
        R_obs = body_state("EARTH", t_obs_et, "J2000", "SUN")[:3]
        geo = rt - R_obs
        sun_at_em = rt
        vt_em = vt

    ra_deg, dec_deg, range_km = _radec_from_geocentric(geo[0], geo[1], geo[2])
    sun_dist_km = float(np.linalg.norm(sun_at_em))

    # Phase angle (Sun-target-observer)
    s_to_obs = -sun_at_em + (R_obs if light_time else R_obs)
    cos_phase = float(np.dot(sun_at_em, R_obs - sun_at_em)
                      / max(sun_dist_km * float(np.linalg.norm(R_obs - sun_at_em)), 1.0))
    cos_phase = max(-1.0, min(1.0, cos_phase))
    phase_deg = math.degrees(math.acos(cos_phase))

    # On-sky rate: ra' = dRA/dt, dec' = dDec/dt from the geocentric velocity
    # apparent velocity = d(rt - R_obs)/dt = vt - V_earth
    V_obs = body_state("EARTH", t_obs_et, "J2000", "SUN")[3:]
    apv = vt_em - V_obs
    rn = range_km
    # spherical-derivative formulas
    cos_dec = math.cos(math.radians(dec_deg))
    dra_dt = (-apv[0] * geo[1] + apv[1] * geo[0]) / (rn ** 2 * max(cos_dec ** 2, 1e-12))
    ddec_dt = (apv[2] * (geo[0] ** 2 + geo[1] ** 2) - geo[2] * (apv[0] * geo[0] + apv[1] * geo[1])) \
              / (rn ** 2 * math.sqrt(max(geo[0] ** 2 + geo[1] ** 2, 1e-6)))
    v_sky_rad_s = math.hypot(dra_dt * cos_dec, ddec_dt)

    return ra_deg, dec_deg, range_km, sun_dist_km, phase_deg, v_sky_rad_s


def predict_ephemeris(candidate: Candidate, mjd: float,
                      *, light_time: bool = True) -> Ephemeris | None:
    """Predict (RA, Dec) at MJD from the candidate's fitted orbit.

    Returns None if the candidate has no fitted state. Single best-estimate, no
    uncertainty (use `predict_with_uncertainty` for that).
    """
    if candidate.orbit_state is None or len(candidate.orbit_state) != 6:
        return None
    x = np.asarray(candidate.orbit_state[:3], dtype=float)
    v = np.asarray(candidate.orbit_state[3:], dtype=float)
    t_ref_et = candidate.meta.get("t_ref_et")
    if t_ref_et is None:
        # fall back: use last_seen_mjd as the epoch of the stored state
        t_ref_et = _mjd_to_et(candidate.last_seen_mjd)

    t_obs_et = _mjd_to_et(mjd)
    try:
        ra, dec, rng, sun_d, phase, v_sky = _propagate_apparent(
            x, v, t_ref_et, t_obs_et, light_time=light_time)
    except Exception:
        return None

    return Ephemeris(
        mjd=mjd, ra_deg=ra, dec_deg=dec,
        sigma_arcsec=float("nan"),                                  # set by uncertainty fn
        sun_distance_au=sun_d / 149597870.7,
        earth_distance_au=rng / 149597870.7,
        phase_angle_deg=phase,
        v_sky_arcsec_hr=v_sky * 206265 * 3600.0,
    )


def predict_with_uncertainty(candidate: Candidate, mjd: float,
                             *, n_samples: int = 50,
                             pos_sigma_km: float = 5e4,
                             vel_sigma_km_s: float = 0.05,
                             seed: int = 0) -> Ephemeris | None:
    """Predict + propagate Monte-Carlo to estimate the ephemeris uncertainty.

    Default 1-sigma errors: 50,000 km in position, 50 m/s in velocity -- honest
    1-sigma for a 3-night IOD+LM fit on real ZTF/DECam arcsec-quality astrometry
    (a multi-month arc with hundreds of observations tightens the velocity sigma
    by 100x or more; pass tighter sigmas in that case).

    Returns Ephemeris with sigma_arcsec populated (max sky-plane axis of the
    Monte-Carlo cloud's covariance).
    """
    base = predict_ephemeris(candidate, mjd, light_time=True)
    if base is None:
        return None

    x = np.asarray(candidate.orbit_state[:3], dtype=float)
    v = np.asarray(candidate.orbit_state[3:], dtype=float)
    t_ref_et = candidate.meta.get("t_ref_et") or _mjd_to_et(candidate.last_seen_mjd)
    t_obs_et = _mjd_to_et(mjd)

    rng_rs = np.random.default_rng(seed)
    pts = []
    for _ in range(n_samples):
        dx = rng_rs.normal(0, pos_sigma_km, 3)
        dv = rng_rs.normal(0, vel_sigma_km_s, 3)
        try:
            ra, dec, *_ = _propagate_apparent(x + dx, v + dv, t_ref_et, t_obs_et,
                                              light_time=False)
        except Exception:
            continue
        # store sky-plane deltas, in arcsec, relative to base
        dra = (ra - base.ra_deg) * math.cos(math.radians(base.dec_deg))
        ddec = dec - base.dec_deg
        pts.append([dra * 3600.0, ddec * 3600.0])

    if not pts:
        return base
    pts = np.asarray(pts)
    # max-axis 1-sigma of the cloud (eigenvalue of cov matrix)
    cov = np.cov(pts.T) if len(pts) > 2 else np.diag([np.std(pts[:, 0]) ** 2,
                                                       np.std(pts[:, 1]) ** 2])
    eigvals = np.linalg.eigvalsh(cov)
    sigma_arcsec = float(math.sqrt(max(float(eigvals.max()), 0.0)))

    return Ephemeris(
        mjd=base.mjd, ra_deg=base.ra_deg, dec_deg=base.dec_deg,
        sigma_arcsec=sigma_arcsec,
        sun_distance_au=base.sun_distance_au,
        earth_distance_au=base.earth_distance_au,
        phase_angle_deg=base.phase_angle_deg,
        v_sky_arcsec_hr=base.v_sky_arcsec_hr,
    )


def next_night_targets(candidates: list[Candidate], mjd_next: float,
                       *, max_sigma_arcsec: float = 600.0,
                       n_samples: int = 30) -> list[dict]:
    """Emit a "look here tonight" list ranked by ephemeris precision.

    For each candidate with a fitted orbit, predict where it'll be at mjd_next
    and how big the sky-window needs to be (4*sigma is a safe radius for a
    confirmation observation -- captures the 99% region).

    Args:
      candidates:       store.discovery_candidates() output.
      mjd_next:         the night you're scheduling for.
      max_sigma_arcsec: drop candidates whose predicted position is too uncertain
                        to bother with (default 10 arcmin -- a degree-scale search
                        box is the practical limit on a 1-deg DECam pointing).

    Returns:
      list of dicts (sorted by sigma): {key, ra_deg, dec_deg, sigma_arcsec,
                                        search_radius_arcsec, range_au, v_sky_arcsec_hr}
    """
    targets = []
    for c in candidates:
        eph = predict_with_uncertainty(c, mjd_next, n_samples=n_samples)
        if eph is None or not math.isfinite(eph.sigma_arcsec):
            continue
        if eph.sigma_arcsec > max_sigma_arcsec:
            continue
        targets.append({
            "key": c.key,
            "ra_deg": eph.ra_deg, "dec_deg": eph.dec_deg,
            "sigma_arcsec": eph.sigma_arcsec,
            "search_radius_arcsec": 4.0 * eph.sigma_arcsec,
            "range_au": eph.earth_distance_au,
            "v_sky_arcsec_hr": eph.v_sky_arcsec_hr,
            "n_runs": c.n_runs,
            "last_seen_mjd": c.last_seen_mjd,
        })
    targets.sort(key=lambda t: t["sigma_arcsec"])
    return targets
