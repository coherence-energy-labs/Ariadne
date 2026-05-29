"""Apsidal/nodal clustering significance on the REAL distant-TNO catalog (Stage 34).

This is the actual Batygin & Brown analysis, run on the CURRENT JPL Small-Body Database
rather than the 6-object 2016 sample. The question: are the perihelion longitudes (varpi),
arguments of perihelion (omega) and node longitudes (Omega) of the extreme trans-Neptunian
objects clustered more than chance -- the observational hint behind the Planet 9 hypothesis?

We compute proper circular statistics (mean resultant length R, the Rayleigh test, and a
Monte-Carlo null) and report the significance HONESTLY, including the dominant caveat that
this population is subject to strong OBSERVATIONAL SELECTION BIAS (surveys look where it is
dark/accessible), so a low p-value is necessary but NOT sufficient evidence of a perturber.

Data: data/distant_tnos.json (cached from the JPL SBDB Query API; refresh=True re-fetches).
Standard astrometry only -- no coherence, no new physics.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

_CACHE = os.path.join("data", "distant_tnos.json")
_SBDB = ("https://ssd-api.jpl.nasa.gov/sbdb_query.api?"
         "fields=full_name,a,e,i,om,w,q,per&sb-cdata=%7B%22AND%22:%5B%22a%7CGT%7C150%22%5D%7D")


def load_distant_tnos(path=_CACHE, refresh=False):
    """Load the distant-TNO catalog as a list of dicts. refresh=True re-fetches from JPL SBDB."""
    if refresh or not os.path.exists(path):
        import urllib.request
        with urllib.request.urlopen(_SBDB, timeout=60) as r:
            doc = json.loads(r.read().decode())
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(doc, f)
    else:
        with open(path) as f:
            doc = json.load(f)
    out = []
    for row in doc["data"]:
        name, a, e, i, om, w, q, per = row
        if a is None or e is None or om is None or w is None:
            continue
        a = float(a); e = float(e)
        out.append({"name": name.strip(), "a_au": a, "e": e,
                    "i_deg": float(i), "Omega_deg": float(om), "omega_deg": float(w),
                    "q_au": float(q) if q is not None else a * (1 - e),
                    "varpi_deg": (float(om) + float(w)) % 360.0})
    return out


_CACHE_SIGMA = os.path.join("data", "distant_tnos_sigma.json")


def load_with_uncertainty(path=_CACHE_SIGMA):
    """Load the catalog WITH per-element 1-sigma uncertainties (varpi sigma included).

    sigma_varpi ~ sqrt(sigma_Omega^2 + sigma_omega^2) (ignoring element correlation -- a
    conservative-ish combination). Objects missing angle sigmas get a large sigma (treated as
    unconstrained). Used to propagate observational uncertainty into the clustering significance.
    """
    with open(path) as f:
        doc = json.load(f)
    out = []
    for row in doc["data"]:
        name, a, e, i, om, w, q, s_a, s_e, s_i, s_om, s_w = row
        if a is None or om is None or w is None:
            continue
        a = float(a); e = float(e)
        s_om = float(s_om) if s_om is not None else 180.0
        s_w = float(s_w) if s_w is not None else 180.0
        out.append({"name": name.strip(), "a_au": a, "e": e, "i_deg": float(i),
                    "Omega_deg": float(om), "omega_deg": float(w),
                    "q_au": float(q) if q is not None else a * (1 - e),
                    "varpi_deg": (float(om) + float(w)) % 360.0,
                    "sigma_varpi_deg": min(180.0, math.hypot(s_om, s_w))})
    return out


def resampled_clustering_p(rows, n_real=3000, seed=0):
    """Distribution of the varpi-clustering Rayleigh p under observational uncertainty.

    For each of n_real catalog realizations, resample each object's varpi from
    N(varpi, sigma_varpi) (wrapped), recompute the analytic Rayleigh p. Returns the array of
    p-values -- its median/spread shows whether the clustering significance is driven (or not)
    by measurement error vs being intrinsic (small N + selection bias).
    """
    rng = np.random.default_rng(seed)
    mu = np.array([r["varpi_deg"] for r in rows])
    sig = np.array([r["sigma_varpi_deg"] for r in rows])
    ps = np.empty(n_real)
    for k in range(n_real):
        vp = (mu + rng.normal(0, sig)) % 360.0
        ps[k] = circular_stats(vp)["p_analytic"]
    return ps


def filter_population(rows, a_min=250.0, q_min=42.0):
    """The dynamically-detached extreme population: a >= a_min AND perihelion q >= q_min.

    q >= ~40 AU excludes objects whose perihelia are coupled to Neptune (scattered disk),
    isolating the bodies whose clustering the Planet 9 hypothesis is about.
    """
    return [r for r in rows if r["a_au"] >= a_min and r["q_au"] >= q_min]


def circular_stats(angles_deg):
    """Mean resultant length R, mean direction, and the Rayleigh test (analytic p)."""
    ang = np.radians(np.asarray(angles_deg, float))
    n = len(ang)
    C, S = np.cos(ang).mean(), np.sin(ang).mean()
    R = math.hypot(C, S)
    mean_dir = math.degrees(math.atan2(S, C)) % 360.0
    Z = n * R * R                                            # Rayleigh statistic
    # standard analytic approximation (Zar / Mardia)
    p = math.exp(-Z) * (1 + (2 * Z - Z * Z) / (4 * n) -
                        (24 * Z - 132 * Z ** 2 + 76 * Z ** 3 - 9 * Z ** 4) / (288 * n * n))
    return {"n": n, "R": R, "mean_dir_deg": mean_dir, "rayleigh_Z": Z,
            "p_analytic": max(0.0, min(1.0, p))}


def rayleigh_mc(angles_deg, n_mc=200000, seed=0):
    """Monte-Carlo p-value: fraction of uniform random samples with R >= observed."""
    n = len(angles_deg)
    R_obs = circular_stats(angles_deg)["R"]
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, size=(n_mc, n))
    R = np.hypot(np.cos(ang).mean(1), np.sin(ang).mean(1))
    return float((R >= R_obs).mean())


def clustering_report(rows, n_mc=200000, seed=0):
    """Full circular-clustering report for varpi, omega, Omega of a population."""
    out = {"n": len(rows)}
    for key, label in (("varpi_deg", "varpi"), ("omega_deg", "omega"), ("Omega_deg", "Omega")):
        ang = [r[key] for r in rows]
        st = circular_stats(ang)
        st["p_mc"] = rayleigh_mc(ang, n_mc=n_mc, seed=seed)
        out[label] = st
    return out
