"""Doubly-averaged (secular) dynamics -- MASTER_PLAN.md Stage 31.

The direct symplectic map (`secular.py`) resolves every orbit (it must take a
timestep that is a small fraction of Jupiter's 12-yr period), so a billion-year
integration would need ~1e9 steps -- days of compute. The genuine tool for
Gyr-scale questions is SECULAR PERTURBATION THEORY: analytically average the
dynamics over the fast orbital phases (the mean anomalies), which removes the
12-yr timescale entirely. The semi-major axes then become constants of motion
(a theorem of the doubly-averaged problem) and the slow angles -- eccentricity,
inclination, and the perihelion/node longitudes -- evolve on the secular
timescale, so a 50,000-yr step is perfectly stable. That buys a factor of ~1e4
in reach: Gyr integrations in minutes.

Method (Gauss's "ring" averaging, done numerically -- robust, no disturbing-
function derivatives):
  * each perturber (the giant planets, a hypothesized Planet 9) is smeared into
    a massive RING -- its mass spread along its Keplerian orbit in proportion to
    the time spent there (uniform in MEAN anomaly);
  * the secular rate of change of a test particle's elements is the Gauss
    planetary equations, evaluated with the ring-averaged perturbing
    acceleration and then averaged over the test particle's own orbit.

The perturbing acceleration uses the SAME heliocentric form (direct minus the
Sun's indirect reflex) as the direct integrator, so the two are directly
comparable -- and Stage 31 VALIDATES the secular precession rates against the
direct symplectic integrator before trusting it at Gyr.

Standard Newtonian gravity throughout (the credibility firewall).
"""
from __future__ import annotations

import math

import numpy as np

from ..data.constants import GM_SUN, AU_KM
from ..data.ephemeris import et as _et, body_gm, body_state
from .secular import state_to_elements, GIANTS, YEAR_S


# --------------------------------------------------------------------------- #
# Kepler's equation + orbit sampling (uniform in MEAN anomaly = time-weighted)
# --------------------------------------------------------------------------- #
def _solve_kepler_eq(M, e, tol=1e-13, max_iter=60):
    """Eccentric anomaly E from mean anomaly M (vectorised Newton)."""
    M = np.asarray(M, float)
    E = M + e * np.sin(M)
    for _ in range(max_iter):
        dE = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E -= dE
        if np.max(np.abs(dE)) < tol:
            break
    return E


def _rot_313(i, Om, om):
    co, so = math.cos(om), math.sin(om)
    cO, sO = math.cos(Om), math.sin(Om)
    ci, si = math.cos(i), math.sin(i)
    return np.array([
        [cO * co - sO * so * ci, -cO * so - sO * co * ci, sO * si],
        [sO * co + cO * so * ci, -sO * so + cO * co * ci, -cO * si],
        [so * si, co * si, ci],
    ])


def sample_orbit(a_au, e, i_deg, Om_deg, om_deg, n=64, mu=GM_SUN, with_vel=False):
    """Positions (and optionally velocities, true anomalies) sampled UNIFORM in mean anomaly.

    Uniform-in-M sampling is time-weighted, i.e. the physical mass density of the
    ring. Returns pos (n,3) km [, vel (n,3) km/s, nu (n,) rad].
    """
    a = a_au * AU_KM
    i, Om, om = math.radians(i_deg), math.radians(Om_deg), math.radians(om_deg)
    M = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    E = _solve_kepler_eq(M, e)
    nu = 2.0 * np.arctan2(math.sqrt(1 + e) * np.sin(E / 2),
                          math.sqrt(1 - e) * np.cos(E / 2))
    r = a * (1.0 - e * np.cos(E))
    Rm = _rot_313(i, Om, om)
    pf = np.stack([r * np.cos(nu), r * np.sin(nu), np.zeros_like(nu)], axis=1)
    pos = pf @ Rm.T
    if not with_vel:
        return pos
    p = a * (1.0 - e * e)
    s = math.sqrt(mu / p)
    vpf = np.stack([-s * np.sin(nu), s * (e + np.cos(nu)), np.zeros_like(nu)], axis=1)
    vel = vpf @ Rm.T
    return pos, vel, nu


# --------------------------------------------------------------------------- #
# Ring-averaged perturbing acceleration (heliocentric: direct + indirect)
# --------------------------------------------------------------------------- #
def ring_accel(field_pts, ring_pos, gm):
    """Heliocentric perturbing acceleration at field_pts (m,3) from a perturber RING.

    a = <gm[(r_p - r)/|r_p - r|^3  -  r_p/|r_p|^3]>  averaged over the ring samples
    (the bracket's 2nd term is the Sun's indirect reflex). Returns (m,3) km/s^2.
    """
    field_pts = np.atleast_2d(field_pts)                      # (m,3)
    d = ring_pos[None, :, :] - field_pts[:, None, :]          # (m,n,3)
    dn = np.linalg.norm(d, axis=2)                            # (m,n)
    direct = (d / dn[:, :, None] ** 3).mean(axis=1)           # (m,3)
    rp = ring_pos
    indirect = (rp / np.linalg.norm(rp, axis=1)[:, None] ** 3).mean(axis=0)  # (3,)
    return gm * (direct - indirect[None, :])


# --------------------------------------------------------------------------- #
# Gauss planetary equations, orbit-averaged over the test particle
# --------------------------------------------------------------------------- #
def secular_rates(elem, perturbers, n_tp=96):
    """Secular d/dt of (a,e,i,Omega,omega) for a test particle under ring perturbers.

    elem       : dict a_au,e,i_deg,Omega_deg,omega_deg
    perturbers : list of (ring_pos (n,3), gm) tuples
    Returns dict of rates in (AU/s, 1/s, rad/s, rad/s, rad/s).
    """
    a_au, e = elem["a_au"], elem["e"]
    i = math.radians(elem["i_deg"])
    om = math.radians(elem["omega_deg"])
    a = a_au * AU_KM
    mu = GM_SUN
    n = math.sqrt(mu / a ** 3)                                 # mean motion (rad/s)
    p = a * (1.0 - e * e)

    pos, vel, nu = sample_orbit(a_au, e, elem["i_deg"], elem["Omega_deg"],
                                elem["omega_deg"], n=n_tp, with_vel=True)
    r = np.linalg.norm(pos, axis=1)                            # (n_tp,)

    # perturbing acceleration at each test-orbit sample
    acc = np.zeros_like(pos)
    for ring_pos, gm in perturbers:
        acc += ring_accel(pos, ring_pos, gm)

    # local orbital frame: radial, transverse (in-plane), normal
    r_hat = pos / r[:, None]
    h_vec = np.cross(pos, vel)
    w_hat = h_vec / np.linalg.norm(h_vec, axis=1)[:, None]
    t_hat = np.cross(w_hat, r_hat)
    aR = np.einsum("ij,ij->i", acc, r_hat)
    aT = np.einsum("ij,ij->i", acc, t_hat)
    aW = np.einsum("ij,ij->i", acc, w_hat)

    sqrt1e = math.sqrt(1.0 - e * e)
    theta = om + nu                                            # argument of latitude
    cosE = (e + np.cos(nu)) / (1.0 + e * np.cos(nu))

    da = (2.0 / (n * sqrt1e)) * (e * np.sin(nu) * aR + (p / r) * aT)
    de = (sqrt1e / (n * a)) * (np.sin(nu) * aR + (np.cos(nu) + cosE) * aT)
    di = (r * np.cos(theta)) / (n * a ** 2 * sqrt1e) * aW
    dOm = (r * np.sin(theta)) / (n * a ** 2 * sqrt1e * math.sin(i)) * aW
    dom = (sqrt1e / (n * a * e)) * (-np.cos(nu) * aR
                                    + np.sin(nu) * (1.0 + r / p) * aT) - dOm * math.cos(i)

    # secular = time-average over the orbit (uniform-in-M samples => simple mean)
    return {"da_au": float(da.mean()) / AU_KM, "de": float(de.mean()),
            "di": float(di.mean()), "dOmega": float(dOm.mean()),
            "domega": float(dom.mean())}


# --------------------------------------------------------------------------- #
# Perturber set (real DE440 orbits) + secular integration
# --------------------------------------------------------------------------- #
def giant_rings(epoch_utc, bodies=GIANTS, n=64, extra=None):
    """Ring samples + GM for the giant planets (and optional extra bodies) from DE440.

    `extra` is a list of dicts {name,gm,a_au,e,i,Omega,omega} (e.g. a Planet 9).
    Returns (rings, info) where rings is a list of (ring_pos (n,3), gm).
    """
    e0 = _et(epoch_utc)
    rings, info = [], []
    for b in bodies:
        st = body_state(b, e0, "J2000", "SUN")
        el = state_to_elements(st[:3], st[3:])
        rp = sample_orbit(el["a_au"], el["e"], el["i_deg"], el["Omega_deg"],
                          el["omega_deg"], n=n)
        rings.append((rp, body_gm(b)))
        info.append({"name": b, **el})
    for x in (extra or []):
        rp = sample_orbit(x["a_au"], x["e"], x["i"], x["Omega"], x["omega"], n=n)
        rings.append((rp, x["gm"]))
        info.append({"name": x["name"], "a_au": x["a_au"], "e": x["e"]})
    return rings, info


def integrate_secular(elements, rings, dt_yr, n_steps, record_every=0, n_tp=96):
    """RK4-integrate the secular elements of test particles under fixed perturber rings.

    elements : list of element dicts (a_au,e,i_deg,Omega_deg,omega_deg). a is held
               (the doubly-averaged theorem); we still track da/dt for the check.
    Returns dict with final elements and (if recording) time series.
    """
    dt = dt_yr * YEAR_S
    state = [dict(el) for el in elements]
    rec_t, rec = [], []

    def deriv(el):
        rt = secular_rates(el, rings, n_tp=n_tp)
        return np.array([rt["de"], rt["di"], rt["dOmega"], rt["domega"]]), rt["da_au"]

    da_max = 0.0
    for k in range(n_steps):
        if record_every and (k % record_every == 0):
            rec_t.append(k * dt_yr)
            rec.append([dict(s) for s in state])
        for s in state:
            y = np.array([s["e"], math.radians(s["i_deg"]),
                          math.radians(s["Omega_deg"]), math.radians(s["omega_deg"])])

            def f(yv):
                e = dict(s)
                e["e"] = max(1e-6, min(0.999, yv[0]))
                e["i_deg"] = math.degrees(yv[1])
                e["Omega_deg"] = math.degrees(yv[2])
                e["omega_deg"] = math.degrees(yv[3])
                d, da = deriv(e)
                return d * dt, da

            k1, da1 = f(y)
            k2, _ = f(y + 0.5 * k1)
            k3, _ = f(y + 0.5 * k2)
            k4, _ = f(y + k3)
            y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            da_max = max(da_max, abs(da1 / dt * (dt_yr)))   # |da/dt| in AU/yr (for the check)
            s["e"] = max(1e-6, min(0.999, y[0]))
            s["i_deg"] = math.degrees(y[1])
            s["Omega_deg"] = math.degrees(y[2]) % 360.0
            s["omega_deg"] = math.degrees(y[3]) % 360.0
            s["varpi_deg"] = (s["Omega_deg"] + s["omega_deg"]) % 360.0

    out = {"elements": state, "da_au_per_yr_max": da_max,
           "span_yr": n_steps * dt_yr}
    if record_every:
        out["times_yr"] = np.array(rec_t)
        out["history"] = rec                                # list[T] of list[ntp] elem dicts
    return out


def perihelion_resultant_deg(elem_row):
    """Mean resultant length R of the perihelion longitudes varpi (1=clustered, 0=dispersed)."""
    ang = np.radians([e["varpi_deg"] for e in elem_row])
    return float(math.hypot(np.cos(ang).mean(), np.sin(ang).mean()))


def varpi_dispersion_deg(elem_row):
    ang = np.radians([e["varpi_deg"] for e in elem_row])
    R = math.hypot(np.cos(ang).mean(), np.sin(ang).mean())
    return math.degrees(math.sqrt(-2.0 * math.log(R))) if R > 1e-9 else float("inf")
