"""Hidden-mass detection from trajectory residuals (MASTER_PLAN.md - Stage 27).

The principle (the user's insight, and how Neptune was found in 1846): a tracked trajectory is
pulled by the TOTAL gravitational field. If the path deviates from what the complete KNOWN-mass
model predicts -- a residual acceleration "in some direction with nothing known there" -- that
residual is the fingerprint of an UNMODELED mass. Drive the residual to zero by adding the right
mass, and you have both improved the model AND located the hidden body.

    a_residual(x) = a_observed(x) - a_known_model(x)  ->  points to the unmodeled mass

This module quantifies, rigorously and honestly:
  - the residual acceleration a hypothesized hidden body (e.g. Planet 9) imparts at a test location,
  - the NOISE FLOOR from the unmodeled small-body population (Kuiper belt) that any real residual
    must rise above (Monte-Carlo RMS), and
  - the comparison to the dominant (solar) acceleration, which sets why detection needs long
    secular baselines rather than an instantaneous snapshot.

Real precedent: Cassini range tracking of Saturn already CONSTRAINS Planet 9 exactly this way --
the absence of an unexplained residual on Saturn's distance rules out close/massive configurations.
"""
from __future__ import annotations

import math

import numpy as np

from .tau_c import newtonian_accel, potential, C2

AU_KM = 149597870.7
GM_EARTH = 398600.435436

#: Real clustered extreme trans-Neptunian objects (the Batygin & Brown 2016 evidence set).
#: Published osculating elements (a in AU, e, i/Omega/omega in deg) from the JPL SBDB / MPC.
#: These are the objects whose apsidal clustering motivates the Planet 9 hypothesis.
CLUSTERED_ETNOS = [
    {"name": "Sedna",      "a_au": 506.8, "e": 0.8590, "i": 11.93, "Omega": 144.40, "omega": 311.29},
    {"name": "2012 VP113", "a_au": 263.1, "e": 0.7036, "i": 24.05, "Omega": 90.81,  "omega": 293.80},
    {"name": "2004 VN112", "a_au": 317.7, "e": 0.8513, "i": 25.55, "Omega": 66.00,  "omega": 327.10},
    {"name": "2007 TG422", "a_au": 485.3, "e": 0.9291, "i": 18.59, "Omega": 112.91, "omega": 285.70},
    {"name": "2010 GB174", "a_au": 370.9, "e": 0.8665, "i": 21.56, "Omega": 130.60, "omega": 347.30},
    {"name": "2013 RF98",  "a_au": 325.8, "e": 0.8849, "i": 29.60, "Omega": 67.60,  "omega": 316.50},
]

#: Published Planet 9 hypothesis parameters (Batygin & Brown 2016 + 2021 refinements):
#: ~5-10 Earth masses, semimajor axis ~400-800 AU, moderate eccentricity/inclination, with
#: perihelion roughly ANTI-aligned to the clustered eTNO perihelia.
PLANET9 = {"m_earth": 6.0, "a_au": 500.0, "e": 0.25, "i": 20.0, "Omega": 90.0, "omega": 150.0}


def elements_to_position(a_au, e, i_deg, Omega_deg, omega_deg, nu_deg):
    """Heliocentric Cartesian position (km) from Keplerian elements at true anomaly nu (3-1-3)."""
    a = a_au * AU_KM
    i, Om, om, nu = (math.radians(v) for v in (i_deg, Omega_deg, omega_deg, nu_deg))
    r = a * (1 - e * e) / (1 + e * math.cos(nu))
    xp, yp = r * math.cos(nu), r * math.sin(nu)         # perifocal
    co, so, cO, sO, ci, si = (math.cos(om), math.sin(om), math.cos(Om),
                              math.sin(Om), math.cos(i), math.sin(i))
    x = (cO * co - sO * so * ci) * xp + (-cO * so - sO * co * ci) * yp
    y = (sO * co + cO * so * ci) * xp + (-sO * so + cO * co * ci) * yp
    z = (so * si) * xp + (co * si) * yp
    return np.array([x, y, z])


def residual_accel(x, hidden_gm, hidden_pos):
    """Acceleration (km/s^2) an unmodeled mass adds at x -- the trajectory residual it would cause."""
    x = np.asarray(x, float)
    d = np.asarray(hidden_pos, float) - x
    dn = np.linalg.norm(d)
    return hidden_gm * d / dn ** 3 if dn > 0 else np.zeros(3)


def coherence_anomaly(x, hidden_gm, hidden_pos):
    """The hidden mass's contribution to the coherence field, delta tau_c = Phi_hidden / c^2."""
    return potential(x, [(hidden_gm, hidden_pos)]) / C2


def kuiper_noise_floor(x, m_total_earth=0.02, r_in_au=30.0, r_out_au=50.0,
                       n_bodies=1200, n_mc=40, seed=0):
    """Monte-Carlo RMS residual acceleration (km/s^2) from an UNMODELED Kuiper-belt population.

    The belt is modelled as `n_bodies` equal masses summing to `m_total_earth` Earth masses, drawn
    uniformly in an annulus (mild +-10 deg inclination). The net acceleration at x largely cancels;
    the surviving RMS over `n_mc` realizations is the floor a real hidden-mass residual must beat.
    """
    rng = np.random.default_rng(seed)
    gm_body = m_total_earth * GM_EARTH / n_bodies
    x = np.asarray(x, float)
    mags = []
    for _ in range(n_mc):
        rr = rng.uniform(r_in_au, r_out_au, n_bodies) * AU_KM
        th = rng.uniform(0, 2 * np.pi, n_bodies)
        inc = rng.uniform(-np.pi / 18, np.pi / 18, n_bodies)
        pos = np.stack([rr * np.cos(th) * np.cos(inc),
                        rr * np.sin(th) * np.cos(inc),
                        rr * np.sin(inc)], axis=1)
        a = np.zeros(3)
        for p in pos:
            d = p - x
            dn = np.linalg.norm(d)
            a += gm_body * d / dn ** 3
        mags.append(np.linalg.norm(a))
    return float(np.mean(mags))


#: Reference gravitating bodies (mass in Earth masses, typical heliocentric distance in AU).
#: The residual method is GENERAL -- it detects/constrains any of these, not just Planet 9.
REFERENCE_BODIES = [
    {"name": "10 km asteroid", "m_earth": 1.7e-10, "d_au": 2.7},
    {"name": "Halley comet",   "m_earth": 3.7e-11, "d_au": 5.0},
    {"name": "Ceres",          "m_earth": 1.57e-4, "d_au": 2.77},
    {"name": "Pluto",          "m_earth": 2.20e-3, "d_au": 39.5},
    {"name": "Eris",           "m_earth": 2.78e-3, "d_au": 67.8},
    {"name": "Planet 9 (hyp.)", "m_earth": 6.0,    "d_au": 500.0},
]


def residual_magnitude(m_earth, d_au):
    """Residual acceleration magnitude (m/s^2) from a body of mass m_earth at range d_au: GM/d^2."""
    gm = m_earth * GM_EARTH                      # km^3/s^2
    d = d_au * AU_KM                             # km
    return (gm / d ** 2) * 1000.0                # m/s^2


def detectability_map(m_earth_grid, d_au_grid, floor_ms2, threshold_ms2):
    """Is a body of given (mass, distance) detectable? Returns residual grid + detectable mask.

    Detectable when its residual GM/d^2 exceeds BOTH the unmodeled-small-body floor and a tracking
    threshold. This generalizes the detector to ANY gravitating body (asteroid, comet, planet)."""
    res = np.array([[residual_magnitude(m, d) for m in m_earth_grid] for d in d_au_grid])
    detectable = res > max(floor_ms2, threshold_ms2)
    return {"mass_grid": list(m_earth_grid), "dist_grid": list(d_au_grid),
            "residual_ms2": res, "detectable": detectable,
            "floor_ms2": floor_ms2, "threshold_ms2": threshold_ms2}


def analyze_hidden_body(test_pos, known_masses, hidden_gm, hidden_pos,
                        m_kuiper_earth=0.02):
    """Full honest detectability analysis of a hypothesized hidden body at a tracked test location."""
    sig = residual_accel(test_pos, hidden_gm, hidden_pos)
    sig_mag = float(np.linalg.norm(sig))
    known = newtonian_accel(test_pos, known_masses)
    known_mag = float(np.linalg.norm(known))
    floor = kuiper_noise_floor(test_pos)
    return {
        "signal_ms2": sig_mag * 1000.0,                 # m/s^2
        "known_accel_ms2": known_mag * 1000.0,
        "kuiper_floor_ms2": floor * 1000.0,
        "signal_over_floor": sig_mag / floor if floor > 0 else np.inf,
        "signal_fraction_of_known": sig_mag / known_mag if known_mag > 0 else np.inf,
        "coherence_anomaly": coherence_anomaly(test_pos, hidden_gm, hidden_pos),
        "above_floor": sig_mag > floor,
    }
