"""Re-converge a CR3BP libration trajectory in the full DE440 ephemeris (Stage 18).

The one fidelity rung Stage 15 left open: take a CR3BP orbit/route, embed it as a
dimensional inertial trajectory via the synodic<->inertial transform, and re-establish it
in the real ephemeris (Earth central + Sun + Moon) by MULTIPLE SHOOTING.

We use position-continuous multiple shooting: the trajectory is split into segments between
patch points; for each segment we Newton-correct the departure velocity so the DE440 arc hits
the next patch POSITION exactly (the same single-segment corrector as Stage 8, generalized).
The velocity mismatch left at each patch is a real maneuver -- the **stationkeeping / transfer
Delta-v** needed to fly the CR3BP shape in the true ephemeris. Position continuity is enforced
to < ~tol km; the reported Delta-v is the honest cost of the real-perturbation correction.

This is exactly how flight-grade libration trajectories (e.g. Gateway NRHO) are re-targeted
from a CR3BP seed. A maneuver-FREE natural quasi-periodic ephemeris orbit (full-state shooting)
is the further refinement; here we quantify the maneuver budget, which is what a mission needs.
"""
from __future__ import annotations

import numpy as np

from ..dynamics.ephemeris_nbody import propagate_test_particle
from ..dynamics.frames import synodic_frame, synodic_to_inertial
from ..dynamics.cr3bp import propagate as cr3bp_propagate

PERTURBERS = ("SUN", "MOON")


def _prop(r, v, et0, dt, perturbers=PERTURBERS):
    return propagate_test_particle(np.asarray(r, float), np.asarray(v, float),
                                   et0, (0.0, dt), perturbers=perturbers).y[:, -1]


def target_position(r0, v_guess, et0, dt, target_r, perturbers=PERTURBERS,
                    tol=1e-2, max_iter=40):
    """Newton-correct departure velocity so the DE440 arc from r0 reaches target_r.

    Returns (v_corrected, arrival_state6, miss_km).
    """
    v = np.asarray(v_guess, float).copy()
    h = 1e-4                       # km/s finite-difference step
    rf = _prop(r0, v, et0, dt, perturbers)
    for _ in range(max_iter):
        miss = rf[:3] - target_r
        if np.linalg.norm(miss) < tol:
            break
        J = np.zeros((3, 3))
        for i in range(3):
            vp = v.copy(); vp[i] += h
            J[:, i] = (_prop(r0, vp, et0, dt, perturbers)[:3] - rf[:3]) / h
        v = v + np.linalg.solve(J, -miss)
        rf = _prop(r0, v, et0, dt, perturbers)
    return v, rf, float(np.linalg.norm(rf[:3] - target_r))


def embed_trajectory(states_nd, taus, et0, mu, frame_epoch_rate=None):
    """Embed nondim synodic states (sampled at nondim times `taus`) as inertial states.

    Each patch is embedded in the instantaneous synodic frame at its OWN epoch, so the
    frame's rotation over the arc is captured. Returns (positions, vel_guesses, epochs).
    """
    f0 = synodic_frame(et0)
    t_star = 1.0 / f0["omega"] if frame_epoch_rate is None else frame_epoch_rate
    epochs = et0 + np.asarray(taus, float) * t_star
    pos, vel = [], []
    for k, e_k in enumerate(epochs):
        S = synodic_to_inertial(states_nd[k], e_k, mu)
        pos.append(S[:3]); vel.append(S[3:])
    return np.array(pos), np.array(vel), epochs


def retarget_orbit(orbit, et0, mu, system, n_patch=8, periodic=False,
                   perturbers=PERTURBERS):
    """Re-converge a CR3BP periodic orbit in DE440 by position-continuous multiple shooting.

    Returns total stationkeeping Delta-v (m/s), max position residual (km), and per-patch data.
    Default is an OPEN one-revolution arc (periodic=False): the reported Delta-v is the sum of
    interior velocity corrections (the real stationkeeping to track the CR3BP shape in DE440).
    periodic=True additionally forces loop closure, whose wrap maneuver is the (large, non-physical)
    cost of making a naturally non-closing orbit close -- not stationkeeping -- so it is not the
    metric of interest here.
    """
    taus = np.linspace(0.0, orbit.period, n_patch + 1)
    sol = cr3bp_propagate(orbit.s0, (0.0, orbit.period), mu, t_eval=taus)
    states_nd = sol.y.T
    pos, vguess, epochs = embed_trajectory(states_nd, taus, et0, mu)

    n_seg = len(pos) - 1
    starts, arrivals, resid = [], [], []
    for i in range(n_seg):
        dt = float(epochs[i + 1] - epochs[i])
        target = pos[0] if (periodic and i == n_seg - 1) else pos[i + 1]
        v, arr, miss = target_position(pos[i], vguess[i], epochs[i], dt, target, perturbers)
        starts.append(v); arrivals.append(arr[3:]); resid.append(miss)

    dvs = []
    for i in range(n_seg):
        v_in = arrivals[(i - 1) % n_seg] if periodic else (arrivals[i - 1] if i > 0 else starts[0])
        dvs.append(float(np.linalg.norm(starts[i] - v_in)))
    return {"total_dv_ms": sum(dvs) * 1000.0, "max_resid_km": max(resid),
            "n_segments": n_seg, "dvs_ms": [d * 1000.0 for d in dvs],
            "positions": pos, "epochs": epochs}


def retarget_heteroclinic(mu, system, conn, et0, n_patch=10, t_leg=2.6,
                          perturbers=PERTURBERS):
    """Re-converge a CR3BP L1<->L2 heteroclinic connecting arc in DE440.

    `conn` is a find_heteroclinic() result. We propagate the unstable-tube seed forward to
    the section and the stable-tube seed backward from it, stitch the connecting arc, embed
    it, and multiple-shoot it in DE440. Returns the connection Delta-v + residual.
    """
    from ..manifolds.manifold import manifold_seeds, manifold_trajectory

    o_src, o_tgt = conn["orbit_source"], conn["orbit_target"]
    bu, bs = conn["branch_unstable"], conn["branch_stable"]
    seeds_u, _ = manifold_seeds(mu, o_src, n_seeds=80, stable=False, branch=bu)
    seeds_s, _ = manifold_seeds(mu, o_tgt, n_seeds=80, stable=True, branch=bs)
    # pick the seeds whose section crossing is closest to the connection point
    from ..connections.poincare import first_section_crossing
    yv = conn["connection_yv"]
    x_sec = conn["x_section"]

    def closest(seeds, stable):
        best, bd = None, np.inf
        for s in seeds:
            st = first_section_crossing(mu, s, x_sec, stable, t_max=10.0)
            if st is None:
                continue
            d = (st[1] - yv[0]) ** 2 + (st[4] - yv[1]) ** 2
            if d < bd:
                bd, best = d, s
        return best

    su, ss = closest(seeds_u, False), closest(seeds_s, True)
    if su is None or ss is None:
        return None
    tu, Yu = manifold_trajectory(mu, su, stable=False, t_max=t_leg, n=n_patch // 2 + 1)
    ts, Ys = manifold_trajectory(mu, ss, stable=True, t_max=t_leg, n=n_patch // 2 + 1)
    # forward arc: unstable (orbit->section) then stable reversed (section->orbit)
    arc = np.hstack([Yu, Ys[:, ::-1]]).T
    taus = np.linspace(0.0, 2.0 * t_leg, arc.shape[0])
    pos, vguess, epochs = embed_trajectory(arc, taus, et0, mu)

    n_seg = len(pos) - 1
    starts, arrivals, resid = [], [], []
    for i in range(n_seg):
        dt = float(epochs[i + 1] - epochs[i])
        v, a, miss = target_position(pos[i], vguess[i], epochs[i], dt, pos[i + 1], perturbers)
        starts.append(v); arrivals.append(a[3:]); resid.append(miss)
    dvs = [float(np.linalg.norm(starts[i] - arrivals[i - 1])) for i in range(1, n_seg)]
    return {"total_dv_ms": sum(dvs) * 1000.0, "max_resid_km": max(resid),
            "n_segments": n_seg, "positions": pos, "epochs": epochs}
