"""Fit a real TNO orbit from MPC astrometry -- the discovery-engine filter that decides
whether a candidate tracklet cluster is a single Keplerian orbit or a false-positive.

Given the recorded MPC astrometry (RA, Dec, JD) for a known Trans-Neptunian Object, Ariadne:
1. Builds nightly tracklets from the dense-opposition window.
2. Re-derives the (heliocentric-distance, range-rate) hypothesis on the candidate's own data
   (the linker's IOD trick).
3. Refines a full 6D heliocentric state via LM differential correction with light-time
   correction, achieving few-arcsecond residuals on real orbits and recovering a, e, i within
   a few percent.

This is the filter that turns the linker's candidate-cluster output into a verified orbit --
the key discrimination between "real new object" and "false-positive linkage of mixed sources."

Run:  PYTHONPATH=src python examples/04_tno_orbit_fit.py
"""
import math, warnings
warnings.filterwarnings("ignore")
import numpy as np

from ariadne.discovery import linkage as L, iod as IOD
from ariadne.data.constants import GM_SUN, AU_KM

# Known TNOs (designation, JPL reference elements)
TARGETS = [
    ("Sedna",      "90377",  {"a": 506.0, "e": 0.85, "i": 11.93}),
    ("Eris",       "136199", {"a":  67.7, "e": 0.44, "i": 44.04}),
    ("Quaoar",     "50000",  {"a":  43.2, "e": 0.04, "i":  7.99}),
]

print("Fitting real TNO orbits from MPC astrometry\n")
for label, desig, jpl in TARGETS:
    print(f"--- {label} (JPL: a={jpl['a']} AU, e={jpl['e']}, i={jpl['i']} deg)")
    tracks, e0 = L.tracklets_from_mpc(desig, window_days=720, min_per_night=2)
    print(f"  fetched {len(tracks)} real MPC tracklets (densest opposition window)")
    if len(tracks) < 4:
        print(f"  too few tracklets, skipping\n"); continue
    t_ref = float(np.median([t["t"] for t in tracks]))
    fit = IOD.fit_candidate(tracks, t_ref=t_ref)
    # Convert state to elements
    r, v = np.asarray(fit["x_fit"]), np.asarray(fit["v_fit"])
    rn, vn = float(np.linalg.norm(r)), float(np.linalg.norm(v))
    a_km = 1.0 / (2.0 / rn - vn ** 2 / GM_SUN)
    a_au = a_km / AU_KM
    h = np.cross(r, v); hn = float(np.linalg.norm(h))
    e_vec = np.cross(v, h) / GM_SUN - r / rn
    ecc = float(np.linalg.norm(e_vec))
    inc = math.degrees(math.acos(max(-1, min(1, h[2] / hn))))
    a_err = abs(a_au - jpl["a"]) / jpl["a"] * 100
    grade = "EXCELLENT" if fit["rms_arcsec"] < 1 else "GOOD" if fit["rms_arcsec"] < 10 else "POOR"
    print(f"  IOD seed: r={fit['iod']['r_au']:.1f} AU, rdot={fit['iod']['rdot']:+.2f} km/s")
    print(f"  FIT: a={a_au:.1f} AU (err {a_err:.1f}%), e={ecc:.3f}, i={inc:.2f} deg, "
          f"RMS={fit['rms_arcsec']:.2f}\"  [{grade}]\n")
