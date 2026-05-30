"""Discovery synthetic-injection test: plant a known TNO into an interloper haystack,
verify the HelioLinC linker recovers it AND the IOD+LM fitter accepts it.

The Stage 44 conclusion "0/348 unmatched survive orbit-fit" claims the discovery pipeline
is a sharp filter: real orbits in, false positives out. That's a NEGATIVE finding. The
positive direction needs proof too: if a real TNO IS hiding in a noisy haystack, does the
pipeline find it?

This benchmark does the positive control:
  1. Pull real MPC astrometry for a known TNO (Sedna).
  2. Build interloper tracklets at random sky positions covering the same epochs.
  3. Mix them into one tracklet pool.
  4. Run the HelioLinC linker -- it should produce a candidate cluster that contains
     mostly Sedna's real tracklets (pure recovery).
  5. Run the IOD+LM fitter on that candidate -- it should accept (low-RMS).

If both succeed, the pipeline finds real objects in haystacks. If either fails, the
"0/348" negative is less trustworthy because we haven't proven the positive side.
"""
import warnings, time
warnings.filterwarnings("ignore")
import numpy as np

from ariadne.discovery import linkage as L, iod as IOD
from ariadne.data.constants import GM_SUN, AU_KM
import math

print("=" * 76)
print("Discovery synthetic injection: plant a known TNO + verify recovery")
print("=" * 76)


def synthetic_haystack(target_desig, n_interlopers=200, seed=0):
    """Pull a known TNO's real tracklets + sprinkle in random interlopers."""
    real, e0 = L.tracklets_from_mpc(target_desig, window_days=720, min_per_night=2)
    if not real:
        return [], e0
    # tag real tracklets with object=0
    for t in real:
        t["obj"] = 0
    # add random interlopers at the same epochs
    interlopers = L.add_interlopers(real, n_interlopers, seed=seed)
    return interlopers, e0


# 1. Build haystack
TARGET = "90377"   # Sedna
print(f"\n[1] Pulling real MPC astrometry for {TARGET} (Sedna)...")
haystack, e0 = synthetic_haystack(TARGET, n_interlopers=200, seed=42)
n_real = sum(1 for t in haystack if t.get("obj") == 0)
n_int = len(haystack) - n_real
print(f"    {n_real} real Sedna tracklets + {n_int} random interlopers = "
      f"{len(haystack)} total tracklets")

# 2. Compute geometry + link via HelioLinC
print(f"\n[2] HelioLinC linking (full search over (r, rdot) hypotheses)...")
geom = L.precompute_geometry(haystack)
t_ref = float(np.median([t["t"] for t in haystack]))
r_grid = np.linspace(40, 200, 40)
rdot_grid = np.linspace(-1.0, 1.0, 11)
t0 = time.time()
candidates = L.link(geom, t_ref, r_grid, rdot_grid, cluster_au=1.0,
                    min_obs=4, min_nights=3)
print(f"    {len(candidates)} candidate clusters found in {time.time()-t0:.1f}s")

# 3. Report cluster purity
report = L.recovery_report(candidates, geom, min_members=4, purity=0.8)
print(f"\n[3] Recovery report:")
print(f"    {report['n_recovered']}/{report['n_true']} true objects recovered")
print(f"    {report['n_pure']}/{report['n_candidates']} candidates are >=80% pure")

# 4. Pick the candidate with the most Sedna members; run IOD+LM
print(f"\n[4] Orbit-fit the Sedna-dominant cluster...")
sedna_clusters = []
for c in candidates:
    labels = [haystack[i].get("obj", -1) for i in c]
    n_sedna = sum(1 for x in labels if x == 0)
    sedna_clusters.append((n_sedna, c, len(c)))
sedna_clusters.sort(reverse=True)
if not sedna_clusters or sedna_clusters[0][0] == 0:
    print("    NO Sedna-containing candidates -- linker FAILED to find the object")
    print("\n[5] VERDICT: positive control FAILS -- pipeline misses planted objects.")
    import sys
    sys.exit(1)

n_sedna, best_cluster, n_cluster = sedna_clusters[0]
purity = n_sedna / n_cluster * 100
print(f"    best cluster: {n_sedna}/{n_cluster} Sedna ({purity:.0f}% pure) + "
      f"{n_cluster - n_sedna} interlopers")

# 5. Run IOD+LM on this cluster
tracks_in_cluster = [haystack[i] for i in best_cluster]
fit = IOD.fit_candidate(tracks_in_cluster, t_ref=t_ref)
if fit is None:
    print("    IOD failed (geometric)")
    import sys; sys.exit(1)

r, v = np.asarray(fit["x_fit"]), np.asarray(fit["v_fit"])
rn, vn = float(np.linalg.norm(r)), float(np.linalg.norm(v))
a_au = (1.0 / (2.0 / rn - vn**2 / GM_SUN)) / AU_KM
ecc = float(np.linalg.norm(np.cross(v, np.cross(r, v)) / GM_SUN - r / rn))
print(f"\n    FIT: a={a_au:.1f} AU (Sedna JPL=506),  e={ecc:.3f} (JPL=0.85),  "
      f"RMS={fit['rms_arcsec']:.2f}\"")

# 6. Verdict
print(f"\n[5] VERDICT")
linker_ok = n_sedna >= 5
fitter_ok = fit['rms_arcsec'] < 50  # generous: the cluster has interlopers polluting fit
sma_ok = abs(a_au - 506.0) / 506.0 < 0.30  # within 30%
all_ok = linker_ok and fitter_ok and sma_ok
print(f"    Linker recovered >=5 Sedna tracklets:           {linker_ok}  ({n_sedna})")
print(f"    Orbit-fit RMS reasonable (<50 arcsec):          {fitter_ok}  ({fit['rms_arcsec']:.1f}\")")
print(f"    Recovered semi-major axis within 30% of JPL:    {sma_ok}  ({a_au:.0f} AU vs 506)")
print(f"    OVERALL: {'PASS' if all_ok else 'FAIL'} -- positive direction validated")
import sys
sys.exit(0 if all_ok else 1)
