"""N-body batch ephemeris with planetary perturbations.

The 2-body (Sun-only) propagator in `mpc_ephemeris_batch` is exact for
the central force but drifts ~5"/year for typical main-belt asteroids
because it ignores Jupiter and Saturn. For multi-year epoch_deltas
(historical backtests, archived imagery) this drift exceeds typical
match radii and recall drops.

This module adds N-body (Sun + giant-planet perturbations) propagation
vectorised across the full 1.5M-orbit catalog. Uses scipy DOP853 on a
flattened state of length 6N with vectorised gravity.

Public API:
  bulk_propagate_nbody(r0, v0, t0_mjd, t1_mjd, perturbers=(...))
  bulk_ephemeris_at_mjd_nbody(records, target_mjd, ...)
  auto_ephemeris_at_mjd(records, target_mjd, ...)
    Auto-select 2-body vs N-body based on max |epoch_delta|. Default
    threshold: 1 year.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.integrate import solve_ivp

from ...data.constants import GM_SUN, AU_KM
from ...data.ephemeris import body_gm, body_pos
from .mpc_ephemeris_batch import bulk_elements_to_state


# Full planetary perturber set. Once the integration is done in a frame
# consistent with the perturbers (equatorial), adding the inner planets
# tightens main-belt ephemerides from ~6.7" to ~1.9" median over a 1.2 yr
# arc (measured vs JPL Horizons). The terrestrial planets matter.
DEFAULT_PERTURBERS = ("MERCURY BARYCENTER", "VENUS BARYCENTER", "EARTH",
                      "MARS BARYCENTER", "JUPITER BARYCENTER",
                      "SATURN BARYCENTER", "URANUS BARYCENTER",
                      "NEPTUNE BARYCENTER")


def _hg_phase_term(alpha_rad, G):
    """IAU H-G phase-function dimming term (mag), >= 0, zero at opposition.

    V = H + 5 log10(r * delta) + [this term]. The two-parameter (H, G) model
    of Bowell et al. 1989:
        term = -2.5 log10[(1 - G) Phi1(alpha) + G Phi2(alpha)]
        Phi_i = exp(-A_i * tan(alpha/2) ** B_i)
    with (A1,B1)=(3.33,0.63), (A2,B2)=(1.87,1.22). Vectorised over alpha, G.
    """
    alpha_rad = np.asarray(alpha_rad, float)
    G = np.asarray(G, float)
    tan_half = np.tan(np.clip(alpha_rad, 0.0, math.pi - 1e-6) / 2.0)
    tan_half = np.maximum(tan_half, 0.0)
    phi1 = np.exp(-3.33 * np.power(tan_half, 0.63))
    phi2 = np.exp(-1.87 * np.power(tan_half, 1.22))
    return -2.5 * np.log10(np.maximum((1.0 - G) * phi1 + G * phi2, 1e-9))


def _mjd_to_et(mjd: float) -> float:
    """MJD -> SPICE ET seconds (J2000 epoch)."""
    return (mjd - 51544.5) * 86400.0


def _vector_rhs(perturbers_gm: list,
                  perturbers: list,
                  et0: float):
    """Build the vectorised acceleration RHS for a flattened state of
    shape (6N,) = [rN_x, rN_y, rN_z, vN_x, ..., vN_z] (interleaved).
    Sun-centered J2000. Each perturber's position is queried once per
    timestep from SPICE.
    """
    def rhs(t, y):
        N = y.size // 6
        ry = y[:3 * N].reshape(N, 3)
        # Sun gravity: a = -GM_SUN * r / |r|^3
        r_norm = np.linalg.norm(ry, axis=1)
        a = -GM_SUN * ry / np.maximum(r_norm, 1.0)[:, None] ** 3
        # Planet perturbations: a += GM * ((s-r)/|s-r|^3 - s/|s|^3)
        et_t = et0 + t
        for body, gmb in zip(perturbers, perturbers_gm):
            s = body_pos(body, et_t, "J2000", "SUN")
            sr = ry - s
            sr_norm = np.linalg.norm(sr, axis=1)
            s_norm = np.linalg.norm(s)
            a += gmb * (-sr / np.maximum(sr_norm, 1.0)[:, None] ** 3
                          - s / s_norm ** 3)
        out = np.empty_like(y)
        out[:3 * N] = y[3 * N:]
        out[3 * N:] = a.ravel()
        return out
    return rhs


def bulk_propagate_nbody(r0: np.ndarray, v0: np.ndarray,
                            t0_mjd: float, t1_mjd: float,
                            *, perturbers: Sequence[str] = DEFAULT_PERTURBERS,
                            rtol: float = 1e-9,
                            atol: float = 1e-3,
                            max_step: float | None = None,
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Batch propagate N orbits under Sun + perturbers from t0_mjd to t1_mjd.

    r0, v0 : (N, 3) heliocentric J2000 in km, km/s
    Returns (rN, vN) of the same shape.
    """
    N = r0.shape[0]
    if N == 0:
        return r0, v0
    dt = (t1_mjd - t0_mjd) * 86400.0
    if abs(dt) < 1e-3:
        return r0.copy(), v0.copy()
    et0 = _mjd_to_et(t0_mjd)
    perturbers_gm = [body_gm(b) for b in perturbers]
    y0 = np.concatenate([r0.ravel(), v0.ravel()])
    rhs = _vector_rhs(perturbers_gm, list(perturbers), et0)
    integ_kwargs = dict(method="DOP853", rtol=rtol, atol=atol,
                          dense_output=False)
    if max_step is not None:
        integ_kwargs["max_step"] = max_step
    sol = solve_ivp(rhs, (0.0, dt), y0, t_eval=[dt], **integ_kwargs)
    if not sol.success:
        raise RuntimeError(f"N-body propagation failed: {sol.message}")
    y_final = sol.y[:, -1]
    r1 = y_final[:3 * N].reshape(N, 3)
    v1 = y_final[3 * N:].reshape(N, 3)
    return r1, v1


def bulk_ephemeris_at_mjd_nbody(records: Sequence,
                                     target_mjd: float,
                                     *, perturbers: Sequence[str]
                                          = DEFAULT_PERTURBERS,
                                     rtol: float = 1e-7,
                                     atol: float = 1.0,
                                     max_step: float | None = 432000.0,
                                     observer_geo_km: np.ndarray | None = None,
                                     light_time: bool = True,
                                     stellar_aberration: bool = False,
                                     ) -> np.ndarray:
    """Like bulk_ephemeris_at_mjd but uses N-body propagation.

    Applies the same ecliptic->equatorial rotation, light-time, and
    optional topocentric observer offset as the 2-body path.

    Default tolerances (rtol=1e-7, atol=1 km, max_step=5 d) are tuned for
    the arcsec cross-match: at ~1.5 AU, 1" ~ 1100 km, so 1 km absolute
    tolerance is far below the match threshold. These are ~4.5x faster
    than nm-level tolerances and stay sub-arcsec vs them (median 0.0",
    worst case ~0.9" over a 1.2 yr arc). Tighten for orbit-fitting work.

    All records must share the same epoch_mjd (or close to it) for the
    batch integration to be coherent.
    """
    from ...data.ephemeris import body_state
    from .mpc_ephemeris_batch import ecliptic_to_equatorial, _C_KM_S
    N = len(records)
    if N == 0:
        return np.empty((0, 4))
    # All-same-epoch assumption: use median if there's any spread
    epochs = np.array([r.epoch_mjd for r in records])
    t0 = float(np.median(epochs))
    a_au = np.array([r.a_au for r in records])
    e = np.array([r.e for r in records])
    i_deg = np.array([r.i_deg for r in records])
    Omega_deg = np.array([r.Omega_deg for r in records])
    omega_deg = np.array([r.omega_deg for r in records])
    M_deg = np.array([r.M_deg for r in records])
    H_mag = np.array([r.H_mag for r in records])

    # Elements -> state at each record's catalog epoch (ECLIPTIC frame)
    r0, v0 = bulk_elements_to_state(a_au, e, i_deg, Omega_deg,
                                          omega_deg, M_deg)
    # If records have spread of epochs, first 2-body propagate each to t0
    # (2-body is frame-agnostic, so do it before the frame rotation)
    if (epochs.max() - epochs.min()) > 0.01:
        from .mpc_ephemeris_batch import bulk_kepler_step
        dt_to_t0 = (t0 - epochs) * 86400.0
        r0, v0 = bulk_kepler_step(r0, v0, dt_to_t0)

    # CRITICAL: rotate the initial state to the EQUATORIAL frame BEFORE the
    # N-body integration. The planetary perturbers (body_pos, SPICE "J2000")
    # are equatorial; integrating an ecliptic-frame state against equatorial
    # perturbers rotates every perturbation by the obliquity and corrupts the
    # result per-orbit. Doing the whole integration in the equatorial frame
    # keeps the asteroid and the perturbers consistent. (A rotation is linear,
    # so it commutes with the dynamics; applying it to (r0, v0) is exact.)
    r0 = ecliptic_to_equatorial(r0)
    v0 = ecliptic_to_equatorial(v0)

    # Earth state (pos+vel) + optional topocentric observer (equatorial)
    et_target = _mjd_to_et(target_mjd)
    earth_state = np.array(body_state("EARTH", et_target, "J2000", "SUN"),
                             dtype=float)
    R_e = earth_state[:3]
    V_obs = earth_state[3:6]
    R_obs = R_e if observer_geo_km is None else (R_e + observer_geo_km)

    # N-body propagate t0 -> target ONCE (r0,v0 are equatorial; the
    # integration stays equatorial, so no further frame rotation).
    r_t_equ, v_t_equ = bulk_propagate_nbody(
        r0, v0, t0, target_mjd, perturbers=perturbers,
        rtol=rtol, atol=atol, max_step=max_step)
    geo = r_t_equ - R_obs[None, :]

    # Light-time (planetary aberration), applied PER OBJECT via its own
    # velocity. Over the light-time (minutes) the orbit is locally linear,
    # so r(t - rho/c) ~ r(t) - v * (rho/c). This avoids re-propagating the
    # whole batch to a single batch-median time -- which was wrong for any
    # object whose distance differed from the batch median (an inner-belt
    # object among outer ones drifted tens of arcsec).
    if light_time:
        for _ in range(2):
            rho_km = np.linalg.norm(geo, axis=1)
            lt_s = rho_km / _C_KM_S
            r_lt = r_t_equ - v_t_equ * lt_s[:, None]
            geo = r_lt - R_obs[None, :]
        r_t_equ = r_lt

    if stellar_aberration:
        from .mpc_ephemeris_batch import _apply_stellar_aberration
        geo = _apply_stellar_aberration(geo, V_obs)

    rho_km = np.linalg.norm(geo, axis=1)
    valid = rho_km > 1.0
    out = np.full((N, 4), np.nan)
    ra_deg = np.degrees(np.arctan2(geo[:, 1], geo[:, 0])) % 360.0
    dec_deg = np.degrees(np.arcsin(geo[:, 2] / np.maximum(rho_km, 1.0)))
    r_helio_au = np.linalg.norm(r_t_equ, axis=1) / AU_KM
    rho_au = rho_km / AU_KM
    # Apparent V mag with the IAU H-G phase function (NOT just H + 5log(r*delta)).
    # Phase angle alpha = angle Sun-object-observer: object->Sun is -r_t_equ,
    # object->observer is -geo, so cos(alpha) = (r_t_equ . geo)/(|r_t_equ||geo|).
    # Omitting the phase term over-predicts brightness by up to ~1 mag away
    # from opposition -- which made faint objects look detectable when they are
    # not. JPL Horizons' V mag includes this term.
    G_arr = np.array([float(getattr(r, "G_param", 0.15) or 0.15) for r in records])
    dot = np.einsum("ij,ij->i", r_t_equ, geo)
    cos_alpha = np.clip(dot / np.maximum(r_helio_au * AU_KM * rho_km, 1e-9), -1.0, 1.0)
    mag_est = np.where(
        (rho_au > 0) & (r_helio_au > 0),
        H_mag + 5.0 * np.log10(np.maximum(rho_au * r_helio_au, 1e-6))
        + _hg_phase_term(np.arccos(cos_alpha), G_arr),
        np.nan)
    out[valid, 0] = ra_deg[valid]
    out[valid, 1] = dec_deg[valid]
    out[valid, 2] = mag_est[valid]
    out[valid, 3] = rho_au[valid]
    return out


def auto_ephemeris_at_mjd(records: Sequence,
                              target_mjd: float,
                              *, nbody_threshold_days: float = 60.0,
                              observer_geo_km: np.ndarray | None = None,
                              ) -> np.ndarray:
    """Auto-select 2-body vs N-body propagation based on epoch_delta.

    If the median |target_mjd - epoch_mjd| exceeds nbody_threshold_days,
    use N-body. Otherwise use the fast 2-body path. The threshold is low
    (60 d) because perturbations reach the arcsec level within weeks for
    main-belt objects -- N-body is required for any real cross-match where
    arcsec accuracy matters.
    """
    from .mpc_ephemeris_batch import bulk_ephemeris_at_mjd
    if not records:
        return np.empty((0, 4))
    epochs = np.array([r.epoch_mjd for r in records])
    max_dt = float(np.max(np.abs(target_mjd - epochs)))
    if max_dt < nbody_threshold_days:
        return bulk_ephemeris_at_mjd(records, target_mjd,
                                       observer_geo_km=observer_geo_km)
    return bulk_ephemeris_at_mjd_nbody(records, target_mjd,
                                         observer_geo_km=observer_geo_km)
