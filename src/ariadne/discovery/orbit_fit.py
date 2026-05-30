"""Orbit-fit verification of a candidate cluster -- residuals tell real from false.

Given a candidate's tracklets (each with time, RA, Dec, observer-Earth position) and an initial state
guess (the linker's (r, rdot)-hypothesis state at t_ref), fit a single Keplerian orbit by minimising
the sky-position residuals. A REAL object's tracklets fit to ~arcsec residuals (consistent with the
astrometric precision); a FALSE cluster has degree-level residuals -- a sharp filter beyond
known-object cross-match.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares

from ..dynamics.secular import kepler_step
from ..data.constants import GM_SUN


def predict_radec(state, t_ref, t, R_obs):
    """Propagate (r, v) at t_ref to t, subtract observer, return predicted (RA, Dec) in radians."""
    r, v = state[:3], state[3:]
    with np.errstate(all="ignore"):
        rt, _ = kepler_step(r, v, GM_SUN, t - t_ref)
    g = rt - R_obs
    rn = np.linalg.norm(g)
    return math.atan2(g[1], g[0]), math.asin(g[2] / rn)


def fit_orbit(tracklets, geom, t_ref, x_init, v_init):
    """LSQ fit a heliocentric state at t_ref to the tracklets' sky positions. Returns dict.

    `geom` is the precomputed observer geometry (linkage.Geometry). `tracklets` are members of one
    candidate cluster (subset by index). Initial state from the linker hypothesis.
    """
    idx = [tracklets[k] for k in range(len(tracklets))]
    Ro = geom.Ro                                              # (N,3) observer pos at each tracklet time
    obs_ra = geom.s.copy()                                    # use the geom for observed unit vectors
    # but geom is precomputed for ALL tracklets in the bin; we need only the cluster's
    # we will use tracklet's t/ra/dec directly
    ts = np.array([t["t"] for t in idx])
    ras = np.array([t["ra"] for t in idx])
    decs = np.array([t["dec"] for t in idx])
    # observer Ro at each tracklet's time (lookup from geom by matching t -- assume tracklets ARE
    # the members so we get them via map)
    # Simpler: re-evaluate observer positions from ephemeris
    from ..data.ephemeris import body_state
    Ro_t = np.array([body_state("EARTH", float(t), "J2000", "SUN")[:3] for t in ts])

    def residuals(state):
        out = np.empty(2 * len(idx))
        for k in range(len(idx)):
            ra_p, dec_p = predict_radec(state, t_ref, float(ts[k]), Ro_t[k])
            # cosine-Dec weighting for RA difference
            out[2 * k] = (ra_p - ras[k]) * math.cos(decs[k])
            out[2 * k + 1] = dec_p - decs[k]
        return out

    x0 = np.concatenate([x_init, v_init])
    try:
        r = least_squares(residuals, x0, method="lm", xtol=1e-12, ftol=1e-12, max_nfev=200)
        rms_rad = math.sqrt((r.fun ** 2).mean())
        return {"x_fit": r.x[:3], "v_fit": r.x[3:], "rms_arcsec": rms_rad * 206265,
                "nfev": r.nfev, "success": True}
    except Exception as e:                                   # pragma: no cover
        return {"rms_arcsec": float("inf"), "success": False, "err": str(e)[:120]}


def multi_opposition_search(x_state, v_state, t_ref, all_tracks, cone_arcsec=180.0,
                            year_offsets=(-3, -2, -1, 1, 2, 3)):
    """Propagate a candidate orbit to other oppositions and search ALL tracklets for matches.

    A genuine distant object should appear in adjacent oppositions too (in the ITF or known catalogs).
    Finding even one matching tracklet at a different year extends the arc and dramatically increases
    confidence. Returns matches keyed by year-offset.
    """
    from ..data.ephemeris import body_state
    YEAR = 365.25 * 86400.0
    cone_rad = cone_arcsec / 206265.0
    matches = {}
    for dy in year_offsets:
        t_test = t_ref + dy * YEAR
        with np.errstate(all="ignore"):
            r_t, _ = kepler_step(np.asarray(x_state), np.asarray(v_state), GM_SUN, dy * YEAR)
        R_obs = body_state("EARTH", t_test, "J2000", "SUN")[:3]
        g = r_t - R_obs
        rn = float(np.linalg.norm(g))
        ra_pred = math.atan2(g[1], g[0]) % (2 * math.pi)
        dec_pred = math.asin(g[2] / rn)
        hits = []
        for tr in all_tracks:
            if abs(tr["t"] - t_test) > 30 * 86400.0:
                continue
            dra = (tr["ra"] - ra_pred + math.pi) % (2 * math.pi) - math.pi
            ddec = tr["dec"] - dec_pred
            sep = math.hypot(dra * math.cos(dec_pred), ddec)
            if sep <= cone_rad:
                hits.append({"desig": tr["desig"], "jd": tr["jd"],
                             "sep_arcsec": float(sep * 206265.0)})
        if hits:
            matches[dy] = sorted(hits, key=lambda h: h["sep_arcsec"])[:5]
    return matches


def verify_candidate_state(t_ref, x_state, v_state, tracklet_records):
    """Back-compat shim. Uses the validated IOD+LM pathway (discovery.iod) for robust residuals.

    The (x_state, v_state) is ignored as a seed (the IOD finds its own from the candidate's tracklets
    via the linker's (r, rdot) hypothesis search); kept in the signature for callers.
    """
    from . import iod as IOD
    fit = IOD.fit_candidate(tracklet_records, t_ref=t_ref)
    if fit is None:
        return {"x_fit": np.asarray(x_state), "v_fit": np.asarray(v_state),
                "rms_arcsec": float("inf"), "nfev": 0}
    return {"x_fit": fit["x_fit"], "v_fit": fit["v_fit"],
            "rms_arcsec": fit["rms_arcsec"], "nfev": fit["nfev"]}
