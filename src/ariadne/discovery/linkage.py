"""Moving-object orbit linkage -- MASTER_PLAN.md Stage 39 (the discovery core).

This is the step that crosses from "validated engine" to "could find something new": linking
unassociated detections into orbits. Surveys publish millions of single-epoch detections; most are
never linked. The HelioLinC method (Holman, Payne et al. 2018) makes the search tractable: HYPOTHESISE
a heliocentric distance r and radial velocity rdot; under that hypothesis every tracklet (a short
on-sky position+rate) maps to a full heliocentric state; propagate each to a common reference epoch
with a real 2-body integrator; tracklets that belong to the SAME object then COLLAPSE to a tight
cluster, while unrelated detections scatter. Cluster -> candidate orbit.

Honest validation (Stage 39): synthesise tracklets from the REAL orbits of known distant objects plus
a large field of interloper tracklets, and show the linker RECOVERS the known objects from the haystack
with few false positives. If it recovers known objects, the same machinery can find unknown ones; if it
cannot, it is not a discovery engine -- and we would know honestly.

Uses the validated Kepler propagator (vectorised) and real Earth ephemeris. Standard gravity only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..data.constants import GM_SUN, AU_KM
from ..data.ephemeris import et, body_state
from ..dynamics.secular import kepler_step, elements_to_state

MU = GM_SUN


def _unit(ra, dec):
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


@dataclass
class Geometry:
    """Per-tracklet observer geometry, precomputed once (independent of the hypothesis)."""
    t: np.ndarray          # (N,) ephemeris time
    Ro: np.ndarray         # (N,3) observer (Earth) heliocentric position, km
    Vo: np.ndarray         # (N,3) observer heliocentric velocity, km/s
    s: np.ndarray          # (N,3) line-of-sight unit vectors
    sdot: np.ndarray       # (N,3) on-sky angular rate of s, 1/s
    obj: np.ndarray        # (N,) truth label for validation (-1 = interloper)


def precompute_geometry(tracklets):
    """Build the per-tracklet observer geometry (one SPICE query per tracklet, not per hypothesis)."""
    n = len(tracklets)
    t = np.array([tr["t"] for tr in tracklets])
    Ro = np.empty((n, 3)); Vo = np.empty((n, 3)); s = np.empty((n, 3)); sdot = np.empty((n, 3))
    obj = np.array([tr.get("obj", -1) for tr in tracklets])
    for i, tr in enumerate(tracklets):
        st = body_state("EARTH", tr["t"], "J2000", "SUN")
        Ro[i] = st[:3]; Vo[i] = st[3:]
        ra, dec, dra, ddec = tr["ra"], tr["dec"], tr["dra"], tr["ddec"]
        s[i] = _unit(ra, dec)
        sdot[i] = [-np.cos(dec) * np.sin(ra) * dra - np.sin(dec) * np.cos(ra) * ddec,
                   np.cos(dec) * np.cos(ra) * dra - np.sin(dec) * np.sin(ra) * ddec,
                   np.cos(dec) * ddec]
    return Geometry(t=t, Ro=Ro, Vo=Vo, s=s, sdot=sdot, obj=obj)


def transform(geom, r_km, rdot, mu=MU):
    """Map every tracklet to a heliocentric state under (r, rdot). Returns (x, v, valid)."""
    Ros = np.einsum("ij,ij->i", geom.Ro, geom.s)
    disc = Ros ** 2 - np.einsum("ij,ij->i", geom.Ro, geom.Ro) + r_km ** 2
    valid = disc >= 0
    rho = -Ros + np.sqrt(np.where(valid, disc, 0.0))
    valid &= rho > 0
    x = geom.Ro + rho[:, None] * geom.s
    x_dot_Vo = np.einsum("ij,ij->i", x, geom.Vo)
    x_dot_sdot = np.einsum("ij,ij->i", x, geom.sdot)
    denom = Ros + rho
    rho_dot = (r_km * rdot - x_dot_Vo - rho * x_dot_sdot) / np.where(denom != 0, denom, 1e30)
    v = geom.Vo + rho_dot[:, None] * geom.s + rho[:, None] * geom.sdot
    return x, v, valid


def link(geom, t_ref, r_grid_au, rdot_grid, cluster_au=20.0, min_obs=4, min_nights=3, mu=MU):
    """Sweep (r, rdot) hypotheses; cluster reference-epoch positions; return candidate tracklet sets."""
    from scipy.spatial import cKDTree
    candidates = []
    night = np.round((geom.t - geom.t.min()) / 86400.0).astype(int)
    for r_au in r_grid_au:
        r_km = r_au * AU_KM
        for rdot in rdot_grid:
            x, v, valid = transform(geom, r_km, rdot, mu)
            if valid.sum() < min_obs:
                continue
            idx = np.where(valid)[0]
            dt = t_ref - geom.t[idx]
            with np.errstate(all="ignore"):
                xref, _ = kepler_step(x[idx], v[idx], mu, dt)
            fin = np.all(np.isfinite(xref), axis=1)
            idx, xref = idx[fin], xref[fin] / AU_KM
            if len(idx) < min_obs:
                continue
            tree = cKDTree(xref)
            for grp in tree.query_ball_tree(tree, cluster_au):
                if len(grp) >= min_obs:
                    members = idx[grp]
                    if len(set(night[members])) >= min_nights:
                        candidates.append(frozenset(int(m) for m in members))
    # dedupe nested sets, keep maximal
    uniq = []
    for c in sorted(candidates, key=len, reverse=True):
        if not any(c <= u for u in uniq):
            uniq.append(c)
    return uniq


def recovery_report(candidates, geom, min_members=4, purity=0.8):
    """How many TRUE objects were recovered, and how clean the candidates are."""
    recovered = set()
    pure = 0
    for c in candidates:
        labels = [int(geom.obj[i]) for i in c]
        dom = max(set(labels), key=labels.count)
        cnt = labels.count(dom)
        if dom >= 0 and cnt >= min_members:
            recovered.add(dom)
            if cnt / len(c) >= purity:
                pure += 1
    n_true = len(set(int(o) for o in geom.obj if o >= 0))
    return {"recovered": sorted(recovered), "n_recovered": len(recovered), "n_true": n_true,
            "n_candidates": len(candidates), "n_pure": pure}


# --------------------------------------------------------------------------- #
# REAL detections: build tracklets from the JPL/MPC astrometric record
# --------------------------------------------------------------------------- #
def tracklets_from_mpc(designation, window_days=120, min_per_night=2, obj_label=0):
    """Build nightly tracklets from a known object's REAL recorded MPC astrometry.

    Fetches the actual telescope observations (RA, Dec, epoch) via astroquery, restricts to the
    DENSEST `window_days` opposition window, and groups same-night detections into tracklets
    (position + on-sky rate). Observatory-vs-geocenter parallax is ignored (negligible for distant
    objects). Returns (tracklets, e0). Requires astroquery + network.
    """
    from astroquery.mpc import MPC
    o = MPC.get_observations(designation)
    jd = np.array([float(r["epoch"].value) for r in o])
    ra = np.radians([float(r["RA"].value) for r in o])
    dec = np.radians([float(r["DEC"].value) for r in o])
    # densest window
    best_n, best_t0 = 0, jd.min()
    for t0 in np.sort(jd):
        n = ((jd >= t0) & (jd < t0 + window_days)).sum()
        if n > best_n:
            best_n, best_t0 = n, t0
    m = (jd >= best_t0) & (jd < best_t0 + window_days)
    from collections import defaultdict
    nights = defaultdict(list)
    for j, r, d in zip(jd[m], ra[m], dec[m]):
        nights[int(round(j))].append((j, r, d))
    tracks = []
    for pts in nights.values():
        if len(pts) < min_per_night:
            continue
        pts = sorted(pts)
        (j1, r1, d1), (j2, r2, d2) = pts[0], pts[-1]
        if j2 - j1 < 1e-4:
            continue
        et = (0.5 * (j1 + j2) - 2451545.0) * 86400.0
        tracks.append({"t": et, "ra": 0.5 * (r1 + r2), "dec": 0.5 * (d1 + d2),
                       "dra": (r2 - r1) / ((j2 - j1) * 86400.0),
                       "ddec": (d2 - d1) / ((j2 - j1) * 86400.0), "obj": obj_label})
    e0 = tracks[0]["t"] if tracks else None
    return tracks, e0


def add_interlopers(tracks, n, seed=0, dec_max=0.5):
    """Append n random interloper tracklets at the existing observation times (for a haystack)."""
    rng = np.random.default_rng(seed)
    times = [tr["t"] for tr in tracks]
    out = list(tracks)
    for _ in range(n):
        out.append({"t": float(rng.choice(times)), "ra": rng.uniform(0, 2 * np.pi),
                    "dec": float(np.arcsin(rng.uniform(-dec_max, dec_max))),
                    "dra": rng.normal(0, 5e-9), "ddec": rng.normal(0, 5e-9), "obj": -1})
    rng.shuffle(out)
    return out


# --------------------------------------------------------------------------- #
# Synthetic-from-real tracklet generator (for honest validation)
# --------------------------------------------------------------------------- #
def synthesize_tracklets(orbits, epoch="2026-01-01T00:00:00",
                         night_offsets_days=(0, 90, 200, 380, 600, 900),
                         n_interlopers=300, noise_arcsec=0.2, pair_dt_s=3600.0, seed=0):
    """Tracklets from REAL orbits (propagated with the validated integrator) + interlopers.

    Each orbit dict needs a_au,e,i,Omega,omega; observed from Earth across several nights, with
    Gaussian astrometric noise. Interlopers are random sky positions+rates. Returns (tracklets, e0).
    """
    rng = np.random.default_rng(seed)
    e0 = et(epoch)
    states = [elements_to_state(o["a_au"], o["e"], o["i"], o["Omega"], o["omega"], 180.0)
              for o in orbits]
    states = [(np.array(p), np.array(v)) for p, v in states]
    tracks = []
    for oi, (r0, v0) in enumerate(states):
        for off in night_offsets_days:
            dt = off * 86400.0
            sub = []
            for ddt in (0.0, pair_dt_s):
                xs, _ = kepler_step(r0, v0, MU, dt + ddt)
                R_e = body_state("EARTH", e0 + dt + ddt, "J2000", "SUN")[:3]
                geo = xs - R_e
                rr = np.linalg.norm(geo)
                ra = math.atan2(geo[1], geo[0]) + rng.normal(0, noise_arcsec / 206265 / max(1e-6, math.cos(math.asin(geo[2] / rr))))
                dec = math.asin(geo[2] / rr) + rng.normal(0, noise_arcsec / 206265)
                sub.append((e0 + dt + ddt, ra, dec))
            (t1, ra1, d1), (t2, ra2, d2) = sub
            tracks.append({"t": 0.5 * (t1 + t2), "ra": 0.5 * (ra1 + ra2), "dec": 0.5 * (d1 + d2),
                           "dra": (ra2 - ra1) / (t2 - t1), "ddec": (d2 - d1) / (t2 - t1), "obj": oi})
    nights_et = [e0 + o * 86400.0 for o in night_offsets_days]
    for _ in range(n_interlopers):
        tracks.append({"t": float(rng.choice(nights_et)), "ra": rng.uniform(0, 2 * np.pi),
                       "dec": math.asin(rng.uniform(-0.6, 0.6)),
                       "dra": rng.normal(0, 3e-9), "ddec": rng.normal(0, 3e-9), "obj": -1})
    rng.shuffle(tracks)
    return tracks, e0
