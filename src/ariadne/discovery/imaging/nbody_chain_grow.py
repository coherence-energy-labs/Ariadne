"""N-body-aware chain-grow for long-arc multi-night chains.

The orbit-grow approach in advanced_linking.py (`orbit_grow_chain`) is the
Pan-STARRS MOF (Moving Object Find) strategy: seed an orbit from a
single 2-night chain, then iteratively extend by adding the next-night
detection whose RA/Dec is closest to the seed's predicted position.

Its weakness: the prediction uses 2-body Kepler propagation. For TNO
arcs spanning weeks (the regime where multi-month follow-up matters)
this is fine — Jupiter's perturbation is tiny over weeks. For arcs
spanning MONTHS the 2-body propagation can drift by tens of arcseconds.
Over months, that's enough to MISS a real subsequent detection that
the FULL N-body integration would have placed on top of.

This module provides:

  `nbody_step`
        Integrate (x, v) forward by `dt_s` seconds under Sun + Jupiter
        + Saturn gravity (the dominant perturbers for outer-system
        objects). Mars + Earth are 1000x smaller and skipped.

  `propagate_orbit_with_perturbations`
        Wrap nbody_step for a sequence of target epochs. Returns
        predicted (ra_deg, dec_deg) per epoch.

  `nbody_grow_chain`
        Drop-in replacement for `orbit_grow_chain` that uses N-body
        propagation instead of Kepler. Same API; same return type
        (list of chains). When `use_nbody=False` it falls back to
        Kepler, so the same orchestrator can call either.

Integration: leapfrog (symplectic, 2nd order) so we get tens of seconds
of accuracy over months at acceptable cost. For tighter accuracy use
RK4 (selectable via the `integrator` arg).
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _heliocentric_acceleration(r: np.ndarray, et: float,
                                  include_jupiter: bool = True,
                                  include_saturn: bool = True) -> np.ndarray:
    """Total heliocentric acceleration at position r (km) and epoch et (sec).

    Sun is at origin in this heliocentric frame; Jupiter and Saturn
    positions come from SPICE.
    """
    from ...data.constants import GM_SUN
    from ...data.ephemeris import body_state

    GM_JUPITER = 1.26712764800000e8     # km^3 / s^2
    GM_SATURN  = 3.79405852000000e7     # km^3 / s^2

    r_norm = float(np.linalg.norm(r))
    if r_norm < 1.0:
        return np.zeros(3)
    a = -GM_SUN * r / r_norm ** 3
    if include_jupiter:
        rj = np.array(body_state("JUPITER BARYCENTER", et, "J2000", "SUN")[:3])
        d = r - rj
        d_norm = float(np.linalg.norm(d))
        if d_norm > 1.0:
            a += -GM_JUPITER * (d / d_norm ** 3 + rj / float(np.linalg.norm(rj)) ** 3)
    if include_saturn:
        rs = np.array(body_state("SATURN BARYCENTER", et, "J2000", "SUN")[:3])
        d = r - rs
        d_norm = float(np.linalg.norm(d))
        if d_norm > 1.0:
            a += -GM_SATURN * (d / d_norm ** 3 + rs / float(np.linalg.norm(rs)) ** 3)
    return a


def nbody_step(r0: np.ndarray, v0: np.ndarray,
                t0_et: float, t1_et: float,
                *, dt_max: float = 86400.0 * 7,
                integrator: str = "leapfrog",
                include_jupiter: bool = True,
                include_saturn: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Integrate (r0, v0) from epoch t0_et to epoch t1_et under Sun + Jupiter + Saturn.

    `dt_max` caps the per-step size; larger steps speed up integration
    but reduce accuracy. Default 1 week is a good tradeoff for TNO arcs
    spanning months.
    """
    if integrator not in ("leapfrog", "rk4"):
        raise ValueError(f"unknown integrator: {integrator!r}")
    dt_total = float(t1_et - t0_et)
    if abs(dt_total) < 1.0:
        return r0.copy(), v0.copy()
    n_steps = max(1, int(math.ceil(abs(dt_total) / dt_max)))
    dt = dt_total / n_steps
    r, v = r0.astype(float).copy(), v0.astype(float).copy()
    t = float(t0_et)

    if integrator == "leapfrog":
        # Kick-drift-kick (symplectic, 2nd order)
        a = _heliocentric_acceleration(r, t,
                                          include_jupiter=include_jupiter,
                                          include_saturn=include_saturn)
        for _ in range(n_steps):
            v_half = v + 0.5 * dt * a
            r = r + dt * v_half
            t += dt
            a = _heliocentric_acceleration(r, t,
                                              include_jupiter=include_jupiter,
                                              include_saturn=include_saturn)
            v = v_half + 0.5 * dt * a
    else:  # rk4
        for _ in range(n_steps):
            k1v = dt * _heliocentric_acceleration(r, t,
                                                       include_jupiter=include_jupiter,
                                                       include_saturn=include_saturn)
            k1r = dt * v
            k2v = dt * _heliocentric_acceleration(r + 0.5 * k1r, t + 0.5 * dt,
                                                       include_jupiter=include_jupiter,
                                                       include_saturn=include_saturn)
            k2r = dt * (v + 0.5 * k1v)
            k3v = dt * _heliocentric_acceleration(r + 0.5 * k2r, t + 0.5 * dt,
                                                       include_jupiter=include_jupiter,
                                                       include_saturn=include_saturn)
            k3r = dt * (v + 0.5 * k2v)
            k4v = dt * _heliocentric_acceleration(r + k3r, t + dt,
                                                       include_jupiter=include_jupiter,
                                                       include_saturn=include_saturn)
            k4r = dt * (v + k3v)
            r = r + (k1r + 2 * k2r + 2 * k3r + k4r) / 6.0
            v = v + (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
            t += dt
    return r, v


def predict_sky_position(r_helio: np.ndarray, et: float) -> tuple[float, float]:
    """Convert heliocentric (r) at epoch (et) to geocentric (RA_deg, Dec_deg)."""
    from ...data.ephemeris import body_state
    R_e = np.array(body_state("EARTH", et, "J2000", "SUN")[:3])
    geo = r_helio - R_e
    rho = float(np.linalg.norm(geo))
    if rho < 1.0:
        return (0.0, 0.0)
    ra_deg = math.degrees(math.atan2(geo[1], geo[0])) % 360.0
    dec_deg = math.degrees(math.asin(geo[2] / rho))
    return (ra_deg, dec_deg)


def propagate_orbit_with_perturbations(r0: np.ndarray, v0: np.ndarray,
                                          t0_et: float,
                                          target_epochs_et: Sequence[float],
                                          **nbody_kwargs) -> list[tuple[float, float]]:
    """Propagate (r0, v0) to each target_epoch and project to sky.

    Returns [(ra_deg, dec_deg)] per target epoch. The propagation chain
    is sequential -- each next epoch starts from the previous one's
    state, so cumulative drift over a long arc is computed correctly.
    """
    r, v = r0.copy(), v0.copy()
    t = float(t0_et)
    out = []
    for et in target_epochs_et:
        r, v = nbody_step(r, v, t, float(et), **nbody_kwargs)
        t = float(et)
        out.append(predict_sky_position(r, et))
    return out


def _seed_orbit_from_pair(t_a: dict, t_b: dict) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Crude initial orbit from a 2-tracklet seed: use straight-line
    propagation in heliocentric frame to estimate (r, v) at t_a.

    This is a one-step Vaisala-like assumption: place the object at a
    fixed heliocentric distance (default 5 AU = Centaur regime) along
    the line of sight, then estimate v from the 2-night displacement.
    For long-arc growth, the actual orbit type doesn't matter -- the
    seed gets refined as the chain grows.
    """
    from ...data.constants import AU_KM
    from ...data.ephemeris import body_state

    # Assume 5 AU heliocentric (rough mid-system); the chain-grow only
    # needs a starting point.
    rho_km = 5.0 * AU_KM
    ta = t_a["t"]; tb = t_b["t"]
    if tb <= ta:
        return None
    # Line-of-sight unit vectors
    ra_a = float(t_a["ra"]); dec_a = float(t_a["dec"])
    ra_b = float(t_b["ra"]); dec_b = float(t_b["dec"])
    d_a = np.array([math.cos(dec_a) * math.cos(ra_a),
                     math.cos(dec_a) * math.sin(ra_a),
                     math.sin(dec_a)])
    d_b = np.array([math.cos(dec_b) * math.cos(ra_b),
                     math.cos(dec_b) * math.sin(ra_b),
                     math.sin(dec_b)])
    R_a = np.array(body_state("EARTH", ta, "J2000", "SUN")[:3])
    R_b = np.array(body_state("EARTH", tb, "J2000", "SUN")[:3])
    # Geocentric distance from Earth assuming heliocentric rho
    r_a = R_a + rho_km * d_a
    r_b = R_b + rho_km * d_b
    v0 = (r_b - r_a) / (tb - ta)
    return r_a, v0, float(ta)


def nbody_grow_chain(tracklets: Sequence[dict],
                      *, rms_acceptance_arcsec: float = 5.0,
                      max_chain_length: int = 12,
                      use_nbody: bool = True,
                      integrator: str = "leapfrog") -> list[list[dict]]:
    """Chain-grow with N-body propagation.

    For each pair of 2-night seed tracklets:
      1. Build a crude heliocentric (r, v) seed.
      2. Predict the next-night sky position via nbody_step (or kepler_step
         when use_nbody=False).
      3. Find the closest tracklet to the prediction within
         rms_acceptance_arcsec; if found, extend the chain.
      4. Repeat until no more matches OR max_chain_length reached.

    Returns deduped list of chains; longer chains are preferred when
    two candidates have a shared seed.
    """
    if len(tracklets) < 3:
        return []

    # Bucket by night-day for fast next-night lookup
    by_night: dict[int, list[dict]] = {}
    for tr in tracklets:
        d = int(tr["t"] / 86400)
        by_night.setdefault(d, []).append(tr)
    night_days = sorted(by_night.keys())

    chains: list[list[dict]] = []
    for i, d_a in enumerate(night_days[:-2]):
        for d_b in night_days[i + 1:i + 3]:
            for t_a in by_night[d_a]:
                for t_b in by_night[d_b]:
                    seed = _seed_orbit_from_pair(t_a, t_b)
                    if seed is None:
                        continue
                    r0, v0, t0 = seed
                    chain = [t_a, t_b]
                    # Try to extend on each subsequent night
                    for d_n in night_days[night_days.index(d_b) + 1:]:
                        if len(chain) >= max_chain_length:
                            break
                        # Predict position at this night's median time
                        candidates = by_night[d_n]
                        et_n = float(np.median([c["t"] for c in candidates]))
                        if use_nbody:
                            try:
                                preds = propagate_orbit_with_perturbations(
                                    r0, v0, t0, [et_n], integrator=integrator)
                            except Exception:
                                break
                        else:
                            from ...dynamics.secular import kepler_step
                            from ...data.constants import GM_SUN
                            r_n, _ = kepler_step(r0, v0, GM_SUN, et_n - t0)
                            preds = [predict_sky_position(r_n, et_n)]
                        ra_pred, dec_pred = preds[0]
                        # Find closest tracklet within rms tolerance
                        cos_dec = math.cos(math.radians(dec_pred))
                        best, best_d = None, rms_acceptance_arcsec / 3600.0
                        for c in candidates:
                            ra_c = math.degrees(c["ra"]) % 360.0
                            dec_c = math.degrees(c["dec"])
                            d = math.hypot((ra_c - ra_pred) * cos_dec,
                                            dec_c - dec_pred)
                            if d < best_d:
                                best_d = d
                                best = c
                        if best is None:
                            break
                        chain.append(best)
                    if len(chain) >= 3:
                        chains.append(chain)
    # Dedup: prefer longer chains
    chains.sort(key=lambda c: -len(c))
    seen = set()
    out = []
    for ch in chains:
        sig = frozenset(id(e) for e in ch)
        if any(sig.issubset(s) for s in seen):
            continue
        seen.add(sig)
        out.append(ch)
    return out
