"""Analytic single-snapshot orbit geometry.

The backbone cue: near opposition, an object's apparent sky rate is an
(almost) deterministic, invertible function of its heliocentric distance,
because the dominant motion is Earth's parallactic reflex. For a circular,
coplanar orbit observed at exact opposition the relative transverse speed
is

    v_rel = v_earth * (1 - 1/sqrt(r))        [r in AU, prograde-prograde]

and the geocentric distance is Delta = r - 1, so the angular rate is

    omega = v_rel / Delta
          = K * (1 - 1/sqrt(r)) / (r - 1)     [K = v_earth in "/day units]

with K = v_earth * 86400 * 206264.806 / AU_km = 3548.4 "/day.

omega(r) is monotonic decreasing, so a measured opposition rate inverts to
a unique heliocentric distance. This is the strongest single-snapshot
distance cue (it is the engine inside MPC's `digest2`). Real objects are
eccentric, inclined, and not exactly at opposition, so the inversion has
scatter -- which is exactly what the full statistical-ranging posterior
(statistical_ranging.py) marginalises over. This module is the analytic
core + initialiser + a sanity check we can validate against known objects.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

V_EARTH_KM_S = 29.7847
AU_KM = 149597870.7
RATE_DISTANCE_K = V_EARTH_KM_S * 86400.0 * 206264.806 / AU_KM   # ~3548 "/day


def opposition_rate(r_au: float) -> float:
    """Apparent sky rate ("/day) at exact opposition for a circular,
    coplanar orbit at heliocentric distance r_au (> 1)."""
    if r_au <= 1.0:
        return float("inf")
    return RATE_DISTANCE_K * (1.0 - 1.0 / math.sqrt(r_au)) / (r_au - 1.0)


def opposition_rate_to_distance(rate_arcsec_day: float,
                                  *, r_lo: float = 1.001, r_hi: float = 1000.0
                                  ) -> float:
    """Invert the opposition rate-distance relation: given a sky rate
    ("/day), return the heliocentric distance (AU). Monotonic, so a robust
    bisection. Returns NaN for non-physical rates."""
    if rate_arcsec_day <= 0:
        return float("nan")
    # omega is decreasing in r; find r with omega(r) = rate.
    hi_rate = opposition_rate(r_lo)     # large (fast, near Earth)
    lo_rate = opposition_rate(r_hi)     # small (slow, distant)
    if rate_arcsec_day >= hi_rate:
        return r_lo
    if rate_arcsec_day <= lo_rate:
        return r_hi
    a, b = r_lo, r_hi
    for _ in range(80):
        m = 0.5 * (a + b)
        if opposition_rate(m) > rate_arcsec_day:
            a = m       # too fast -> object is farther
        else:
            b = m
    return 0.5 * (a + b)


def implied_absolute_magnitude(v_mag: float, r_au: float, delta_au: float,
                                 phase_deg: float = 0.0,
                                 g_slope: float = 0.15) -> float:
    """IAU H-G: H = V - 5 log10(r * Delta) - phase term. Near opposition
    the phase angle ~0 so the phase term ~0."""
    if r_au <= 0 or delta_au <= 0:
        return float("nan")
    # H-G phase function (Bowell): Phi at phase alpha
    alpha = math.radians(phase_deg)
    if alpha > 0:
        phi1 = math.exp(-3.33 * math.tan(alpha / 2) ** 0.63)
        phi2 = math.exp(-1.87 * math.tan(alpha / 2) ** 1.22)
        phase_term = -2.5 * math.log10((1 - g_slope) * phi1 + g_slope * phi2)
    else:
        phase_term = 0.0
    return v_mag - 5.0 * math.log10(r_au * delta_au) - phase_term


def solar_elongation_deg(observer_helio_km, los_unit) -> float:
    """Angle (deg) between the Sun direction and the line of sight, as seen
    by the observer. ~180 deg at opposition (where the rate-distance
    relation is sharpest)."""
    import numpy as _np
    R = _np.asarray(observer_helio_km, float)
    sun_from_obs = -R / (_np.linalg.norm(R) + 1e-12)   # Sun direction
    los = _np.asarray(los_unit, float)
    los = los / (_np.linalg.norm(los) + 1e-12)
    c = float(_np.clip(sun_from_obs @ los, -1, 1))
    return math.degrees(math.acos(c))


@dataclass
class SnapshotEstimate:
    distance_au: float          # geocentric distance point estimate
    distance_sigma_au: float    # 1-sigma (calibrated from real data)
    helio_r_au: float
    orbit_class: str
    elongation_deg: float
    rate_arcsec_hr: float
    implied_H: float
    near_opposition: bool       # is the rate->distance relation trustworthy?
    incomer_flag: bool          # bright + untrailed -> possible nearby/incoming
    note: str = ""


# 1-sigma scatter of the opposition inversion vs truth, calibrated on real
# DECam asteroids (median |err| ~0.36 AU in heliocentric distance).
_OPP_DISTANCE_SIGMA_AU = 0.4


def single_snapshot_estimate(rate_arcsec_hr: float, v_mag: float,
                               observer_helio_km, los_unit,
                               *, opposition_tol_deg: float = 35.0,
                               bright_neo_H: float = 22.0) -> "SnapshotEstimate":
    """Single-snapshot distance + orbit-class estimate from rate + brightness.

    Uses the validated opposition rate->distance relation when the field is
    near opposition. Flags a possible nearby/incoming object when the rate
    is low (=> nominally distant) but the brightness would require an
    implausibly large body at that distance -- i.e. the "no trail = slow OR
    coming-at-us" ambiguity resolved by photometry.
    """
    elong = solar_elongation_deg(observer_helio_km, los_unit)
    near_opp = elong >= (180.0 - opposition_tol_deg)
    rate_day = rate_arcsec_hr * 24.0
    r_helio = opposition_rate_to_distance(rate_day) if rate_day > 0 else float("nan")
    delta = r_helio - 1.0 if (r_helio == r_helio and r_helio > 1) else float("nan")
    H = implied_absolute_magnitude(v_mag, r_helio, max(delta, 1e-3)) \
        if (delta == delta and delta > 0) else float("nan")
    cls = classify_by_distance(r_helio) if r_helio == r_helio else "unknown"
    # Incomer logic: a slow (low-rate) object inverts to LARGE distance; if
    # it is also bright, the implied H is very negative (huge object) ->
    # implausible -> it is more likely NEARBY with mostly-radial motion.
    incomer = bool(delta == delta and delta > 3.0 and v_mag < 21.0
                     and H == H and H < 12.0)
    note = ""
    if not near_opp:
        note = (f"elongation {elong:.0f} deg: off opposition, rate->distance "
                "degraded (use full geometry).")
    if incomer:
        note += " INCOMER CANDIDATE: bright + low-rate implies an implausibly large distant body; likely nearby/radial."
    return SnapshotEstimate(
        distance_au=delta, distance_sigma_au=_OPP_DISTANCE_SIGMA_AU,
        helio_r_au=r_helio, orbit_class=cls, elongation_deg=elong,
        rate_arcsec_hr=rate_arcsec_hr, implied_H=H,
        near_opposition=near_opp, incomer_flag=incomer, note=note.strip())


def classify_by_distance(r_au: float, e: float | None = None) -> str:
    """Coarse orbit class from heliocentric distance (+ eccentricity if
    known). Single-snapshot classes."""
    if r_au < 1.3:
        return "NEO/inner"
    if r_au < 2.0:
        return "Mars-crosser/inner-belt"
    if r_au < 3.3:
        return "main-belt"
    if r_au < 6.0:
        return "outer-belt/Hilda/Trojan"
    if r_au < 30.0:
        return "Centaur"
    return "TNO/distant"
