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
import matplotlib.pyplot as plt

from ariadne.discovery import linkage as L, iod as IOD
from ariadne.data.constants import GM_SUN, AU_KM

# Known TNOs (designation, JPL reference elements)
TARGETS = [
    ("Sedna",      "90377",  {"a": 506.0, "e": 0.85, "i": 11.93}),
    ("Eris",       "136199", {"a":  67.7, "e": 0.44, "i": 44.04}),
    ("Quaoar",     "50000",  {"a":  43.2, "e": 0.04, "i":  7.99}),
]

print("Fitting real TNO orbits from MPC astrometry\n")
results = []
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
    results.append((label, jpl, a_au, ecc, inc, fit["rms_arcsec"], a_err))

# Visualise fit-vs-JPL recovery
fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))
labels = [r[0] for r in results]
ax[0].bar(labels, [r[6] for r in results], color='steelblue')
ax[0].set_ylabel("|a_fit - a_JPL| / a_JPL  (%)")
ax[0].set_title("Semi-major-axis recovery error")
ax[0].grid(axis='y', alpha=0.3)
ax[1].bar(labels, [r[5] for r in results], color='crimson')
ax[1].set_ylabel("RMS residual (arcsec)")
ax[1].set_title("Sky-position fit residual (real MPC data)")
ax[1].axhline(10, color='gray', ls='--', label='10\" filter threshold')
ax[1].grid(axis='y', alpha=0.3); ax[1].legend()
xs_jpl = [r[1]["a"] for r in results]; xs_fit = [r[2] for r in results]
ax[2].loglog(xs_jpl, xs_fit, 'o', markersize=12, color='forestgreen')
for r in results:
    ax[2].annotate(r[0], (r[1]["a"], r[2]), xytext=(8, -5), textcoords='offset points')
lo, hi = min(xs_jpl + xs_fit) * 0.5, max(xs_jpl + xs_fit) * 2
ax[2].plot([lo, hi], [lo, hi], '--', color='gray', label='ideal: a_fit = a_JPL')
ax[2].set_xlabel("JPL a (AU)"); ax[2].set_ylabel("Fit a (AU)")
ax[2].set_title("Recovered semi-major axis"); ax[2].legend(); ax[2].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("examples_out/04_tno_orbit_fit.png", dpi=120)
print("Wrote examples_out/04_tno_orbit_fit.png")
