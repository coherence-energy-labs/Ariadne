"""Honest heteroclinic-patch Delta-v: include the position-gap closure cost.

The earlier examples reported `patch Delta-v = ||vx_src - vx_dst||` at the closest 4D
approach of two manifold tubes on a Poincare section. This counts ONLY the velocity
mismatch and assumes the position mismatch (often tens of thousands of km in the y or z
component) is closed for free. That's an honest patch only when the position gap is
within tracking tolerance, which it usually isn't.

This benchmark computes the HONEST total per-patch Delta-v by adding a rendezvous burn
to close the position gap:

    Delta-v_total  =  Delta-v_velocity  +  ||position_gap|| / dt_correction

where dt_correction is the chosen time over which to make the correction. For a typical
midcourse-correction window of 1 day, even small position gaps become several-hundred-
m/s burns.

This is the honest discrimination between "exact heteroclinic" (position gap = 0,
true ballistic) and "near-miss patch" (a real maneuver in disguise).
"""
import math, warnings, time
warnings.filterwarnings("ignore")
import numpy as np

import ariadne
from ariadne.orbits.halo import halo_family
from ariadne.connections.poincare_3d import tube_section_cut_3d, closest_approach_4d
from ariadne.dynamics.cr3bp import pseudo_potential

em = ariadne.system("EARTH_MOON")
MU, L_STAR, V_STAR = em.mu, em.L_star, em.V_star

DT_CORRECTION_DAYS = 1.0       # midcourse correction window
DT_CORRECTION_S = DT_CORRECTION_DAYS * 86400.0

print("=" * 76)
print("Heteroclinic-patch HONEST Delta-v: velocity + position-gap rendezvous")
print(f"Position-correction window: {DT_CORRECTION_DAYS} day = {DT_CORRECTION_S:.0f} s")
print("=" * 76)


def honest_patch_dv(orbit_src, orbit_dst, x_sec, axis, n_seeds=200, branch_src=+1, branch_dst=+1):
    """Compute honest patch Delta-v including position-gap closure."""
    cu = tube_section_cut_3d(MU, orbit_src, x_sec=x_sec, stable=False, branch=branch_src,
                              n_seeds=n_seeds, t_max=12.0, axis=axis)
    cs = tube_section_cut_3d(MU, orbit_dst, x_sec=x_sec, stable=True, branch=branch_dst,
                              n_seeds=n_seeds, t_max=12.0, axis=axis)
    r = closest_approach_4d(cu['yzvyvz'], cs['yzvyvz'])
    pa, pb = r['point_a'], r['point_b']
    # crossing in the (axis-perp position 1, axis-perp position 2, vel 1, vel 2) 4D
    if axis == 0:
        # section x = x_sec; 4-tuple is (y, z, vy, vz); vx is energy-determined
        y_int = 0.5 * (pa[0] + pb[0]); z_int = 0.5 * (pa[1] + pb[1])
        v1_a, v2_a = float(pa[2]), float(pa[3])
        v1_b, v2_b = float(pb[2]), float(pb[3])
        om = pseudo_potential([x_sec, y_int, z_int, 0, 0, 0], MU)
        arg_a = 2 * om - orbit_src.jacobi - v1_a**2 - v2_a**2
        arg_b = 2 * om - orbit_dst.jacobi - v1_b**2 - v2_b**2
        if arg_a < 0 or arg_b < 0: return None
        vx_a, vx_b = math.sqrt(arg_a), math.sqrt(arg_b)
        dvx, dvy, dvz = vx_a - vx_b, v1_a - v1_b, v2_a - v2_b
    elif axis == 1:
        # section y = x_sec; 4-tuple is (x, z, vx, vz); vy is energy-determined
        x_int = 0.5 * (pa[0] + pb[0]); z_int = 0.5 * (pa[1] + pb[1])
        v1_a, v2_a = float(pa[2]), float(pa[3])
        v1_b, v2_b = float(pb[2]), float(pb[3])
        om = pseudo_potential([x_int, x_sec, z_int, 0, 0, 0], MU)
        arg_a = 2 * om - orbit_src.jacobi - v1_a**2 - v2_a**2
        arg_b = 2 * om - orbit_dst.jacobi - v1_b**2 - v2_b**2
        if arg_a < 0 or arg_b < 0: return None
        vy_a, vy_b = math.sqrt(arg_a), math.sqrt(arg_b)
        dvx, dvy, dvz = v1_a - v1_b, vy_a - vy_b, v2_a - v2_b
    else:
        return None
    dv_vel = math.sqrt(dvx*dvx + dvy*dvy + dvz*dvz) * V_STAR * 1000  # m/s
    pos_gap_nondim = float(np.linalg.norm(pa[:2] - pb[:2]))
    pos_gap_km = pos_gap_nondim * L_STAR
    dv_rendezvous = (pos_gap_km * 1000) / DT_CORRECTION_S  # m/s
    return {"dv_velocity_ms": dv_vel,
            "pos_gap_km": pos_gap_km,
            "dv_rendezvous_ms": dv_rendezvous,
            "dv_total_ms": dv_vel + dv_rendezvous,
            "energy_diff": orbit_dst.jacobi - orbit_src.jacobi}


# Build the orbits
print("\nBuilding L1 halo, L2 halo, Gateway NRHO...")
l1 = halo_family(MU, point="L1", n=10, dz=4e-3, fam_n=40, lyap_amp0=2e-3, lyap_dx=4e-3)[5]
l2 = halo_family(MU, point="L2", n=10, dz=4e-3, fam_n=40, lyap_amp0=2e-3, lyap_dx=4e-3)[5]
nrho = ariadne.gateway_nrho()
print(f"  L1 halo C={l1.jacobi:.4f}, L2 halo C={l2.jacobi:.4f}, NRHO C={nrho.jacobi:.4f}")

print("\n[1] L1 halo -> L2 halo (x = 1-mu section)")
r = honest_patch_dv(l1, l2, x_sec=1 - MU, axis=0)
if r:
    print(f"  velocity-mismatch Delta-v  = {r['dv_velocity_ms']:6.1f} m/s")
    print(f"  position gap               = {r['pos_gap_km']:6.0f} km")
    print(f"  position-correction burn   = {r['dv_rendezvous_ms']:6.1f} m/s   ({DT_CORRECTION_DAYS}d window)")
    print(f"  HONEST TOTAL               = {r['dv_total_ms']:6.1f} m/s")
    print(f"  energy diff (C_dst - C_src) = {r['energy_diff']:+.4f}")

print("\n[2] NRHO -> L2 halo (y = 0 section)")
r = honest_patch_dv(nrho, l2, x_sec=0.0, axis=1, branch_src=-1, branch_dst=+1)
if r:
    print(f"  velocity-mismatch Delta-v  = {r['dv_velocity_ms']:6.1f} m/s")
    print(f"  position gap               = {r['pos_gap_km']:6.0f} km")
    print(f"  position-correction burn   = {r['dv_rendezvous_ms']:6.1f} m/s   ({DT_CORRECTION_DAYS}d window)")
    print(f"  HONEST TOTAL               = {r['dv_total_ms']:6.1f} m/s")
    print(f"  energy diff (C_dst - C_src) = {r['energy_diff']:+.4f}")

print("\n[3] L2 halo -> L2 halo (same orbit, same energy: should be ~ballistic)")
r = honest_patch_dv(l2, l2, x_sec=1 - MU, axis=0)
if r:
    print(f"  velocity-mismatch Delta-v  = {r['dv_velocity_ms']:6.1f} m/s   (should be ~0 same-energy)")
    print(f"  position gap               = {r['pos_gap_km']:6.0f} km")
    print(f"  HONEST TOTAL               = {r['dv_total_ms']:6.1f} m/s")

print("\nINTERPRETATION:")
print("  - Same-energy same-orbit patches are ~exact (tiny velocity + tiny position).")
print("  - Different-energy patches have BOTH velocity mismatch AND position gap; honest")
print("    accounting must include the rendezvous burn to close the gap. The earlier")
print("    quote of '119 m/s for NRHO->L2 patch' was velocity-only and omitted the")
print("    several-hundred-m/s burn that would actually be needed for a 1-day correction.")
print("  - Real mission-design: pick a longer correction window and re-target via DE440.")
print("    The certified-route promotion (Stage 46) does exactly this.")
