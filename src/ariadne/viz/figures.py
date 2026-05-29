"""Figure generation for Ariadne (MASTER_PLAN.md §12).

Renders the headline Stage 1-3 results:
  - the Earth-Moon L1<->L2 invariant-manifold tubes + Poincare section (the
    heteroclinic "highway"),
  - the L1 planar Lyapunov family colored by Jacobi constant with the halo
    bifurcation marked.

Run:  PYTHONPATH=src python -m ariadne.viz.figures
Figures are written to docs/figures/.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..data.constants import EARTH_MOON
from ..dynamics.cr3bp import propagate
from ..orbits.lagrange import lagrange_points
from ..orbits.families import lyapunov_family, find_halo_bifurcation
from ..manifolds.manifold import manifold_seeds
from ..connections.poincare import propagate_until_section
from ..connections.heteroclinic import find_heteroclinic

_OUT = os.path.join("docs", "figures")


def _orbit_xy(mu, orbit, npts=500):
    sol = propagate(orbit.s0, (0.0, orbit.period), mu,
                    t_eval=np.linspace(0.0, orbit.period, npts))
    return sol.y[0], sol.y[1]


def figure_heteroclinic(mu=EARTH_MOON.mu, c_target=3.15, n_seeds=120):
    conn = find_heteroclinic(mu, c_target, "L1", "L2", n_seeds=n_seeds)
    if conn is None:
        raise RuntimeError("no heteroclinic connection found")
    o1, o2 = conn["orbit_source"], conn["orbit_target"]
    bu, bs, x_sec = conn["branch_unstable"], conn["branch_stable"], conn["x_section"]
    L = lagrange_points(mu)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.2))

    # Panel 1: rotating-frame tubes
    ax1.plot(-mu, 0, "o", color="tab:blue", ms=11, label="Earth")
    ax1.plot(1 - mu, 0, "o", color="0.5", ms=6, label="Moon")
    for k in ("L1", "L2"):
        ax1.plot(L[k][0], 0, "k+", ms=10)
        ax1.annotate(k, (L[k][0], 0.004), fontsize=9)

    seeds_u, _ = manifold_seeds(mu, o1, n_seeds=n_seeds, stable=False, branch=bu)
    for s in seeds_u:
        _, Y, _ = propagate_until_section(mu, s, x_sec, stable=False, t_max=10.0)
        ax1.plot(Y[0], Y[1], color="tab:green", alpha=0.18, lw=0.5)
    seeds_s, _ = manifold_seeds(mu, o2, n_seeds=n_seeds, stable=True, branch=bs)
    for s in seeds_s:
        _, Y, _ = propagate_until_section(mu, s, x_sec, stable=True, t_max=10.0)
        ax1.plot(Y[0], Y[1], color="tab:red", alpha=0.18, lw=0.5)

    x, y = _orbit_xy(mu, o1); ax1.plot(x, y, color="darkgreen", lw=2, label="L1 Lyapunov")
    x, y = _orbit_xy(mu, o2); ax1.plot(x, y, color="darkred", lw=2, label="L2 Lyapunov")
    ax1.axvline(x_sec, color="k", ls=":", lw=0.8)
    ax1.plot([], [], color="tab:green", lw=2, label="L1 unstable tube W$^u$")
    ax1.plot([], [], color="tab:red", lw=2, label="L2 stable tube W$^s$")
    ax1.set_xlabel("x  (rotating frame, nondim)")
    ax1.set_ylabel("y")
    ax1.set_title(f"Earth–Moon L1↔L2 transport tubes  (C = {c_target})")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_aspect("equal")
    ax1.set_xlim(0.75, 1.22)

    # Panel 2: Poincare section. Drop near-Moon crossings (|y| < ~7700 km),
    # which sit on the singularity at x=1-mu and have huge v_y.
    def _clean(c):
        return c[np.abs(c[:, 0]) > 0.02]
    cu, cs = _clean(conn["unstable_cut"]), _clean(conn["stable_cut"])
    ax2.plot(cu[:, 0], cu[:, 1], ".-", color="tab:green", ms=3, lw=0.8,
             label="L1 unstable cut")
    ax2.plot(cs[:, 0], cs[:, 1], ".-", color="tab:red", ms=3, lw=0.8,
             label="L2 stable cut")
    for (yy, vv) in conn["intersections"]:
        ax2.plot(yy, vv, "k*", ms=16, label="heteroclinic connection")
    both = np.vstack([cu, cs]) if len(cu) and len(cs) else np.vstack([c for c in (cu, cs) if len(c)])
    if len(both):
        ymar = 0.02 * (np.ptp(both[:, 0]) + 1e-9)
        vmar = 0.10 * (np.ptp(both[:, 1]) + 1e-9)
        ax2.set_xlim(both[:, 0].min() - ymar, both[:, 0].max() + ymar)
        ax2.set_ylim(both[:, 1].min() - vmar, both[:, 1].max() + vmar)
    ax2.set_xlabel("y")
    ax2.set_ylabel("$v_y$")
    ax2.set_title("Poincaré section x = 1−μ : tube cuts → heteroclinic ⋆")
    ax2.legend(fontsize=8)

    fig.suptitle("Ariadne — the interplanetary highway between L1 and L2", fontsize=12)
    fig.tight_layout()
    os.makedirs(_OUT, exist_ok=True)
    path = os.path.join(_OUT, "heteroclinic_L1_L2.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_family(mu=EARTH_MOON.mu, n=40):
    fam = lyapunov_family(mu, "L1", amplitude0=1e-3, dx=2e-3, n=n)
    bif = find_halo_bifurcation(fam)
    L1x = lagrange_points(mu)["L1"][0]

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    cmap = plt.get_cmap("viridis")
    cmin = min(m.orbit.jacobi for m in fam)
    cmax = max(m.orbit.jacobi for m in fam)
    for m in fam:
        x, y = _orbit_xy(mu, m.orbit, npts=300)
        col = cmap((m.orbit.jacobi - cmin) / (cmax - cmin + 1e-12))
        ax.plot(x, y, color=col, lw=0.9)
    ax.plot(L1x, 0, "k+", ms=11)
    ax.annotate("L1", (L1x, 0.004), fontsize=10)
    ax.plot(1 - mu, 0, "o", color="0.5", ms=6)
    ax.annotate("Moon", (1 - mu, 0.004), fontsize=8)

    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=cmin, vmax=cmax))
    cb = fig.colorbar(sm, ax=ax)
    cb.set_label("Jacobi constant C")
    title = "L1 planar Lyapunov family (Earth–Moon)"
    if bif:
        title += f"\nhalo bifurcation at C = {bif['jacobi']:.4f}"
    ax.set_title(title)
    ax.set_xlabel("x  (rotating frame, nondim)")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.tight_layout()
    os.makedirs(_OUT, exist_ok=True)
    path = os.path.join(_OUT, "L1_lyapunov_family.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_budget():
    from ..optimize.budget import earth_moon_budget
    b = earth_moon_budget()
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    labels = ["Direct\n(Apollo-class)", "Low-energy\n(ballistic capture)"]
    tli = [b["dv_tli"] * 1000, b["dv_tli"] * 1000]
    loi = [b["dv_loi_direct"] * 1000, b["dv_loi_ballistic"] * 1000]
    ax.bar(labels, tli, color="tab:blue", label="TLI (trans-lunar injection)")
    ax.bar(labels, loi, bottom=tli, color="tab:orange", label="LOI (lunar capture)")
    for i, tot in enumerate([b["total_direct"], b["total_low_energy"]]):
        ax.text(i, tot * 1000 + 30, f"{tot*1000:.0f} m/s", ha="center", fontweight="bold")
    ax.axhline(3925, color="k", ls="--", lw=1)
    ax.text(1.4, 3925 + 30, "Coimbra 3925 m/s", ha="right", fontsize=9)
    ax.set_ylabel("Δv  (m/s)")
    ax.set_title("Earth→Moon Δv budget (LEO→LLO)\nballistic capture saves the LOI burn")
    ax.legend(loc="lower center")
    ax.set_ylim(0, 4400)
    fig.tight_layout()
    os.makedirs(_OUT, exist_ok=True)
    path = os.path.join(_OUT, "delta_v_budget.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_ephemeris_validation(epoch="2025-06-01T00:00:00"):
    """Real-data validation: n-body integration error vs JPL DE440 over time."""
    from ..data.ephemeris import et, body_state
    from ..dynamics.ephemeris_nbody import propagate_nbody
    bodies = ["SUN", "EARTH", "MOON"]
    ext = ["JUPITER BARYCENTER", "VENUS BARYCENTER", "MARS BARYCENTER", "SATURN BARYCENTER"]
    e0 = et(epoch)
    spans = [2, 5, 10, 20, 30, 45, 60]
    errs = {b: [] for b in bodies}
    for d in spans:
        sol, _ = propagate_nbody(bodies, e0, (0.0, d * 86400.0), external=ext)
        for i, b in enumerate(bodies):
            integ = sol.y[3 * i:3 * i + 3, -1]
            truth = body_state(b, e0 + d * 86400.0, "J2000", "SSB")[:3]
            errs[b].append(np.linalg.norm(integ - truth))

    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    for b, c in zip(bodies, ("tab:orange", "tab:blue", "0.4")):
        ax.semilogy(spans, errs[b], "o-", color=c, label=b.title())
    ax.set_xlabel("propagation time (days)")
    ax.set_ylabel("position error vs DE440 (km)")
    ax.set_title("Ariadne n-body propagator vs JPL DE440\n(Sun-Earth-Moon + 4 planet perturbers)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs(_OUT, exist_ok=True)
    path = os.path.join(_OUT, "ephemeris_validation.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_moon_orbit(epoch="2025-06-01T00:00:00", days=28.0, n=400):
    """The real Moon orbit about Earth over ~one month, straight from DE440."""
    from ..data.ephemeris import et, body_pos
    e0 = et(epoch)
    ts = np.linspace(0.0, days * 86400.0, n)
    P = np.array([body_pos("MOON", e0 + t, "J2000", "EARTH") for t in ts])
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.plot(P[:, 0], P[:, 1], color="0.4", lw=1.2)
    ax.plot(0, 0, "o", color="tab:blue", ms=12, label="Earth")
    ax.plot(P[0, 0], P[0, 1], "o", color="0.5", ms=6, label="Moon (start)")
    ax.set_aspect("equal")
    ax.set_xlabel("x (km, J2000)")
    ax.set_ylabel("y (km, J2000)")
    ax.set_title(f"Real Moon orbit from JPL DE440 ({days:.0f} days)")
    ax.legend()
    fig.tight_layout()
    os.makedirs(_OUT, exist_ok=True)
    path = os.path.join(_OUT, "moon_orbit_de440.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_low_energy_transfer(mu=None, C=3.15):
    from ..data.constants import EARTH_MOON
    from ..orbits.families import lyapunov_orbit_at_jacobi
    from ..orbits.lagrange import lagrange_points
    from ..transfers.lunar_capture import ballistic_capture
    from ..manifolds.manifold import manifold_trajectory
    mu = EARTH_MOON.mu if mu is None else mu
    orb = lyapunov_orbit_at_jacobi(mu, "L1", C)
    cap = ballistic_capture(orb, llo_alt=100.0)
    t, Y = manifold_trajectory(mu, cap["seed"], stable=False, t_max=6.0, n=1200)
    i = cap["peri_index"]
    L = lagrange_points(mu)

    fig, (ax, axz) = plt.subplots(1, 2, figsize=(13.5, 6.2))
    for a in (ax, axz):
        a.plot(-mu, 0, "o", color="tab:blue", ms=11, label="Earth")
        a.plot(1 - mu, 0, "o", color="0.5", ms=7, label="Moon")
        a.plot(L["L1"][0], 0, "k+", ms=10)
        xo, yo = _orbit_xy(mu, orb)
        a.plot(xo, yo, color="darkgreen", lw=1.8, label="L1 Lyapunov orbit")
        a.plot(Y[0, :i + 1], Y[1, :i + 1], color="tab:red", lw=1.4,
               label="ballistic capture coast")
        a.plot(Y[0, i], Y[1, i], "r*", ms=14, label="LLO capture")
        a.set_aspect("equal")
    ax.annotate("L1", (L["L1"][0], 0.01), fontsize=9)
    ax.set_xlim(0.78, 1.08)
    ax.set_xlabel("x (rotating, nondim)")
    ax.set_ylabel("y")
    ax.set_title(f"Low-energy Earth–Moon transfer (C={C}): "
                 f"manifold ballistic capture")
    ax.legend(loc="upper left", fontsize=8)
    # zoom near the Moon
    axz.set_xlim(1 - mu - 0.04, 1 - mu + 0.04)
    axz.set_ylim(-0.04, 0.04)
    axz.set_title(f"Zoom at the Moon: capture to {cap['periapsis_alt_km']:.0f} km, "
                  f"LOI {cap['dv_capture_kms']*1000:.0f} m/s")
    axz.set_xlabel("x")
    fig.suptitle("Ariadne — ballistic lunar capture via invariant manifold "
                 "(real CR3BP dynamics)", fontsize=12)
    fig.tight_layout()
    os.makedirs(_OUT, exist_ok=True)
    path = os.path.join(_OUT, "low_energy_transfer.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_halo_3d(mu=None):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from ..data.constants import EARTH_MOON
    from ..dynamics.cr3bp import propagate
    from ..orbits.lagrange import lagrange_points
    from ..orbits.halo import halo_family
    mu = EARTH_MOON.mu if mu is None else mu
    halos = halo_family(mu, "L1", n=16, dz=2e-3)
    L = lagrange_points(mu)
    fig = plt.figure(figsize=(8.5, 7))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("plasma")
    for i, h in enumerate(halos):
        sol = propagate(h.s0, (0.0, h.period), mu,
                        t_eval=np.linspace(0, h.period, 300))
        ax.plot(sol.y[0], sol.y[1], sol.y[2],
                color=cmap(i / len(halos)), lw=0.9)
    ax.scatter([1 - mu], [0], [0], color="0.5", s=60, label="Moon")
    ax.scatter([L["L1"][0]], [0], [0], color="k", marker="+", s=80)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("Earth–Moon L1 halo family (3D)\nbranching from the Lyapunov bifurcation")
    fig.tight_layout()
    os.makedirs(_OUT, exist_ok=True)
    path = os.path.join(_OUT, "halo_family_3d.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_genesis():
    from ..data.constants import SUN_EARTH as SE
    from ..orbits.lagrange import lagrange_points
    from ..manifolds.manifold import manifold_seeds, manifold_trajectory
    from ..transfers.genesis import genesis_halo
    mu, L = SE.mu, SE.L_star
    h, _ = genesis_halo()
    # find the manifold trajectory that comes closest to Earth
    best_d, best_Y = np.inf, None
    for stable in (True, False):
        for br in (+1, -1):
            seeds, _ = manifold_seeds(mu, h, n_seeds=100, displacement=1e-6,
                                      stable=stable, branch=br)
            for s in seeds:
                _, Y = manifold_trajectory(mu, s, stable=stable, t_max=8.0, n=1500)
                d = (np.sqrt((Y[0] - (1 - mu)) ** 2 + Y[1] ** 2 + Y[2] ** 2).min()) * L
                if d < best_d:
                    best_d, best_Y = d, Y
    ex = (np.asarray([s.s0 for s in [h]])[0])
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    # plot in km relative to Earth
    ex_e = (1 - mu)
    ax.plot((best_Y[0] - ex_e) * L, best_Y[1] * L, color="tab:purple", lw=0.9,
            label="manifold (superhighway)")
    ax.plot(0, 0, "o", color="tab:blue", ms=12, label="Earth")
    ax.plot((lagrange_points(mu)["L1"][0] - ex_e) * L, 0, "k+", ms=11, label="Sun–Earth L1")
    ax.add_patch(plt.Circle((0, 0), 6378.0, color="tab:blue", alpha=0.3))
    ax.set_aspect("equal")
    ax.set_xlabel("x − Earth (km)"); ax.set_ylabel("y (km)")
    ax.set_title(f"Genesis mechanism: Sun–Earth L1 halo manifold\n"
                 f"reaches {best_d:,.0f} km from Earth (halo period 178 d)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    os.makedirs(_OUT, exist_ok=True)
    path = os.path.join(_OUT, "genesis_superhighway.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_ephemeris_transfer(epoch="2025-06-01T00:00:00", tof_days=5.5):
    from ..data.ephemeris import et, body_pos
    from ..dynamics.ephemeris_nbody import propagate_test_particle
    from ..transfers.ephemeris_transfer import design_transfer
    e0 = et(epoch)
    d = design_transfer(e0, tof_days)
    tof = tof_days * 86400.0
    ts = np.linspace(0.0, tof, 400)
    sc = propagate_test_particle(d["r1"], d["v1"], e0, (0.0, tof),
                                 perturbers=("SUN",), t_eval=ts).y
    moon = np.array([body_pos("MOON", e0 + t, "J2000", "EARTH") for t in ts])

    fig, ax = plt.subplots(figsize=(8.0, 7.2))
    ax.plot(moon[:, 0], moon[:, 1], color="0.6", lw=1.0, ls="--", label="Moon orbit (DE440)")
    ax.plot(sc[0], sc[1], color="tab:red", lw=1.6, label="spacecraft (DE440 propagation)")
    ax.plot(0, 0, "o", color="tab:blue", ms=12, label="Earth")
    ax.plot(sc[0, 0], sc[1, 0], "g^", ms=9, label="TLI (LEO departure)")
    ax.plot(moon[-1, 0], moon[-1, 1], "o", color="0.4", ms=9, label="Moon at arrival")
    ax.set_aspect("equal")
    ax.set_xlabel("x (km, J2000)")
    ax.set_ylabel("y (km, J2000)")
    ax.set_title(f"Full-ephemeris Earth→Moon transfer (DE440)\n"
                 f"TOF {tof_days:.1f} d, TLI {d['dv_tli_ms']:.0f} + LOI "
                 f"{d['dv_loi_ms']:.0f} = {d['total_ms']:.0f} m/s, Moon miss {d['miss_km']*1000:.0f} m")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    os.makedirs(_OUT, exist_ok=True)
    path = os.path.join(_OUT, "ephemeris_transfer.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_wsb_transfer():
    from ..data.ephemeris import et, body_pos
    from ..transfers.wsb import evaluate_transfer, transfer_trajectory, SOLUTION_PARAMS
    b = evaluate_transfer(SOLUTION_PARAMS)       # deterministic canonical solution
    t, Y = transfer_trajectory(b)               # forward: LEO -> WSB loop -> Moon
    e0 = et(b["epoch"])
    # Moon orbit over the transfer window (for context)
    ts = np.linspace(t[0], t[-1], 300)
    moon = np.array([body_pos("MOON", e0 + tt, "J2000", "EARTH") for tt in ts])

    fig, ax = plt.subplots(figsize=(8.4, 7.6))
    ax.plot(moon[:, 0], moon[:, 1], color="0.7", lw=1.0, ls="--", label="Moon orbit (DE440)")
    ax.plot(Y[0], Y[1], color="tab:purple", lw=1.3, label="WSB low-energy transfer")
    ax.plot(0, 0, "o", color="tab:blue", ms=12, label="Earth")
    ax.plot(Y[0, 0], Y[1, 0], "g^", ms=10, label="LEO departure")
    ax.plot(Y[0, -1], Y[1, -1], "r*", ms=15, label="ballistic lunar capture")
    ax.set_aspect("equal")
    ax.set_xlabel("x (km, J2000)")
    ax.set_ylabel("y (km, J2000)")
    ax.set_title(f"Sun-assisted low-energy (WSB) Earth→Moon transfer\n"
                 f"{b['total_ms']:.0f} m/s (< direct 3953, < Coimbra 3925), "
                 f"TOF {b['tof_days']:.0f} d, v∞ {b['v_inf']:.2f} km/s")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    os.makedirs(_OUT, exist_ok=True)
    path = os.path.join(_OUT, "wsb_transfer.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_coherence_frontier():
    from ..data.ephemeris import et
    from ..data.constants import R_MOON
    from ..dynamics.ephemeris_nbody import propagate_test_particle
    from ..transfers.ephemeris_transfer import design_transfer
    from ..transfers.wsb import _capture_state, _frame, SOLUTION_PARAMS, evaluate_transfer
    from ..analysis.coherence import endpoint_sensitivity

    e0 = et("2025-06-01T00:00:00")
    dv, sens, lab = [], [], []
    for tof in (3.0, 4.0, 5.0, 6.0):
        d = design_transfer(e0, tof)
        s0 = np.concatenate([d["r1"], d["v1"]]); T = tof * 86400.0
        prop = lambda s, T=T: propagate_test_particle(
            s[:3], s[3:], e0, (0, T), perturbers=("SUN", "MOON")).y[:, -1]
        dv.append(d["total_ms"]); sens.append(endpoint_sensitivity(prop, s0))
        lab.append(f"direct {tof:.0f} d")

    ec = et("2025-11-12T00:00:00"); fr = _frame(ec)
    fa, al, be, ph = SOLUTION_PARAMS
    pos, vel, _ = _capture_state(ec, fa, al, be, ph, R_MOON + 100.0, *fr)
    s0w = np.concatenate([pos, vel]); Tw = 48.8 * 86400.0
    propw = lambda s: propagate_test_particle(
        s[:3], s[3:], ec, (0, -Tw), perturbers=("SUN", "MOON")).y[:, -1]
    bw = evaluate_transfer(SOLUTION_PARAMS)
    dv.append(bw["total_ms"]); sens.append(endpoint_sensitivity(propw, s0w))
    lab.append("WSB 49 d")

    from ..transfers.coherence_optimizer import pareto_front, knee
    pts = [{"label": t, "dv_ms": x, "sensitivity": y} for x, y, t in zip(dv, sens, lab)]
    kp = knee(pareto_front(pts))

    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    colors = ["tab:green"] * 4 + ["tab:purple"]
    ax.scatter(dv, sens, c=colors, s=90, zorder=3)
    if kp is not None:
        ax.scatter([kp["dv_ms"]], [kp["sensitivity"]], s=320, facecolors="none",
                   edgecolors="k", linewidths=2, zorder=4)
        ax.annotate("knee\n(best compromise)", (kp["dv_ms"], kp["sensitivity"]),
                    textcoords="offset points", xytext=(10, -28), fontsize=9, fontweight="bold")
    for x, y, t in zip(dv, sens, lab):
        ax.annotate(t, (x, y), textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("total Δv  (m/s)  —  cheaper →")
    ax.set_ylabel("endpoint sensitivity  km per m/s  (↑ = more fragile / less coherent)")
    ax.set_title("Robustness costs fuel: the Δv–coherence frontier\n"
                 "(the cheapest WSB path is ~8x more fragile than the fast transfer)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    os.makedirs(_OUT, exist_ok=True)
    path = os.path.join(_OUT, "coherence_frontier.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_low_thrust_spiral(mu=None):
    from ..data.constants import EARTH_MOON
    from ..dynamics.low_thrust import propagate_low_thrust
    mu = EARTH_MOON.mu if mu is None else mu
    # start on a bound orbit in the Earth realm; continuous tangential thrust -> spiral out
    s0 = np.array([-mu + 0.05, 0.0, 0.0, 0.0, 4.2, 0.0])
    coast = propagate_low_thrust(s0, (0.0, 12.0), mu, 0.0, t_eval=np.linspace(0, 12, 3000))
    burn = propagate_low_thrust(s0, (0.0, 12.0), mu, 7e-3, "tangential",
                                t_eval=np.linspace(0, 12, 3000))
    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    ax.plot(coast.y[0], coast.y[1], color="0.75", lw=0.8, label="ballistic (no thrust)")
    ax.plot(burn.y[0], burn.y[1], color="tab:red", lw=0.7, label="low-thrust spiral")
    ax.plot(-mu, 0, "o", color="tab:blue", ms=11, label="Earth")
    lim = float(np.max(np.abs(burn.y[:2]))) * 1.1
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (rotating, nondim)"); ax.set_ylabel("y")
    ax.set_title("Low-thrust regime (CR3BP): continuous tangential thrust\n"
                 "raises energy and spirals outward (vs ballistic coast)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(_OUT, exist_ok=True)
    path = os.path.join(_OUT, "low_thrust_spiral.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_transport_graph():
    """The Earth-Moon transport graph with the minimum-Delta-v route highlighted (Stage 14)."""
    from ..data.constants import EARTH_MOON
    from ..transport_graph.graph import build_transport_graph
    from ..transport_graph.search import dijkstra, reconstruct_path

    energies = [3.120, 3.140, 3.160, 3.172]
    g = build_transport_graph(EARTH_MOON, energies, points=("L1", "L2"),
                              n_seeds=120)        # converged resolution (see Stage 14 note)
    source = f"L1@{energies[0]:.3f}"
    target = f"L2@{energies[-1]:.3f}"
    dj = dijkstra(g, source)
    route = reconstruct_path(dj["prev"], source, target) or []
    route_edges = set(zip(route[:-1], route[1:]))

    def pos(node):
        return (node.jacobi, 0.0 if node.point == "L1" else 1.0)

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    # all patch edges, faint; near-ballistic ones dashed green; route bold blue
    for elist in g.edges.values():
        for e in elist:
            x0, y0 = pos(g.nodes[e.src]); x1, y1 = pos(g.nodes[e.dst])
            dv_ms = g.dv_ms(e.dv)
            if (e.src, e.dst) in route_edges:
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=2.4))
            elif dv_ms < 50.0:
                ax.plot([x0, x1], [y0, y1], color="tab:green", ls="--", lw=1.0, alpha=0.7)
            else:
                ax.plot([x0, x1], [y0, y1], color="0.8", lw=0.5, alpha=0.6, zorder=0)
    for node in g.nodes.values():
        x, y = pos(node)
        on = node.key in route
        ax.plot(x, y, "o", ms=13, color="tab:blue" if on else "0.5", zorder=3)
        ax.annotate(node.key, (x, y + 0.08), fontsize=8, ha="center")

    total = g.dv_ms(dj["dist"].get(target, float("nan")))
    ax.plot([], [], color="tab:blue", lw=2.4, label=f"min-Δv route ({total:.0f} m/s)")
    ax.plot([], [], color="tab:green", ls="--", label="near-ballistic patch (<50 m/s)")
    ax.plot([], [], color="0.8", lw=0.8, label="other patch edges")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["L1", "L2"])
    ax.set_ylim(-0.4, 1.5)
    ax.set_xlabel("Jacobi constant C  (energy level)")
    ax.set_title("Ariadne transport graph — Dijkstra/A* route over the L1/L2 manifold network\n"
                 f"{source}  →  {target}   (SSSP = the shortest-path search that beats brute force)")
    ax.legend(loc="upper center", fontsize=8, ncol=3)
    ax.invert_xaxis()      # lower C (more energy) to the right reads naturally
    fig.tight_layout()
    os.makedirs(_OUT, exist_ok=True)
    path = os.path.join(_OUT, "transport_graph.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_coherence_field():
    """The coherence (FLI) field of the Earth-Moon rotating frame with the manifold tubes overlaid."""
    from ..data.constants import EARTH_MOON
    from ..fields.coherence_field import coherence_map
    from ..orbits.families import lyapunov_orbit_at_jacobi
    from ..manifolds.manifold import manifold_seeds, manifold_trajectory
    mu = EARTH_MOON.mu; C = 3.15
    xs, ys, F = coherence_map(mu, C, (0.78, 1.18), (-0.3, 0.3), n=44, t_max=3.0)

    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    im = ax.pcolormesh(xs, ys, F, shading="auto", cmap="viridis")
    fig.colorbar(im, label="FLI  (high = chaotic / low coherence;  low = regular / coherent)")
    for pt, col in (("L1", "white"), ("L2", "magenta")):
        orb = lyapunov_orbit_at_jacobi(mu, pt, C)
        for br in (+1, -1):
            seeds, _ = manifold_seeds(mu, orb, n_seeds=40, stable=False, branch=br)
            for s in seeds[::3]:
                _, Y = manifold_trajectory(mu, s, stable=False, t_max=3.0, n=200)
                m = (np.abs(Y[1]) < 0.3) & (Y[0] > 0.78) & (Y[0] < 1.18)
                ax.plot(Y[0][m], Y[1][m], color=col, lw=0.4, alpha=0.5)
        ax.plot([], [], color=col, lw=1.5, label=f"{pt} unstable tube")
    ax.plot(1 - mu, 0, "o", color="0.7", ms=6)
    ax.set_xlabel("x (rotating frame, nondim)"); ax.set_ylabel("y")
    ax.set_title("Coherence (FLI) field at C=3.15 with invariant-manifold tubes\n"
                 "test: are the transport tubes the field's chaos ridges?  (NO -- they are the "
                 "orderly skeleton)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0.78, 1.18); ax.set_ylim(-0.3, 0.3)
    fig.tight_layout()
    os.makedirs(_OUT, exist_ok=True)
    path = os.path.join(_OUT, "coherence_field.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_grand_tradeoff():
    """The unified time/energy/robustness trade space with coherence-balanced picks (Stage 23)."""
    from ..data.ephemeris import et
    from ..interplanetary.grand import build_tradeoff, most_coherent_route
    e0 = et("2026-01-01T00:00:00")
    tr = build_tradeoff("EARTH", "MARS BARYCENTER", e0, dep_days=540, tof_range=(120, 400),
                        n_dep=45, n_tof=35)
    tof = np.array([p["tof_days"] for p in tr])
    dv = np.array([p["total_ms"] for p in tr]) / 1000.0
    sens = np.array([p["sensitivity_ms_per_day"] for p in tr])

    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    sc = ax.scatter(tof, dv, c=sens, cmap="plasma", s=60, zorder=3)
    fig.colorbar(sc, label="launch-window sensitivity (m/s per day)  -- lower = more robust")
    ax.plot(tof, dv, "k-", lw=0.5, alpha=0.4, zorder=1)
    picks = [((1, 1, 1), "balanced", "tab:green", "*"),
             ((3, 1, 0.5), "energy-first", "tab:blue", "P"),
             ((0.5, 3, 0.5), "time-first", "tab:red", "D")]
    for w, name, col, mk in picks:
        c = most_coherent_route(tr, w)
        ax.scatter([c["tof_days"]], [c["total_ms"] / 1000.0], marker=mk, s=230,
                   edgecolor="k", facecolor=col, zorder=5, label=f"{name} pick")
    ax.set_xlabel("time of flight (days)")
    ax.set_ylabel("total Delta-v (km/s)")
    ax.set_title("Ariadne grand optimizer -- Earth->Mars time x energy x robustness\n"
                 "coherence balances all three; the weights are the dial")
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(_OUT, exist_ok=True)
    path = os.path.join(_OUT, "grand_tradeoff.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_veega():
    """Heliocentric flight path of the Galileo-class VEEGA to Jupiter (Stage 22)."""
    from scipy.integrate import solve_ivp
    from ..data.constants import GM_SUN, AU_KM
    from ..data.ephemeris import body_pos, body_state
    from ..interplanetary.flyby import reference_veega, GALILEO_VEEGA
    from ..data.ephemeris import et
    from ..optimize.lambert import lambert
    v = reference_veega()
    epochs = v["epochs"]; bodies = v["bodies"]

    def helio(t, s):
        r = s[:3]; rn = np.linalg.norm(r)
        return np.concatenate([s[3:], -GM_SUN * r / rn ** 3])

    fig, ax = plt.subplots(figsize=(8.4, 8.0))
    ax.plot(0, 0, "o", color="gold", ms=14, label="Sun")
    for body, col, yr in [("VENUS", "tab:orange", 225), ("EARTH", "tab:blue", 365),
                          ("JUPITER BARYCENTER", "tab:red", 4333)]:
        ts = epochs[0] + np.linspace(0, yr * 86400.0, 500)
        P = np.array([body_pos(body, t, "J2000", "SUN") for t in ts])
        ax.plot(P[:, 0] / AU_KM, P[:, 1] / AU_KM, color=col, lw=0.6, alpha=0.7)
    # each leg's transfer arc
    for i in range(len(bodies) - 1):
        s0 = body_state(bodies[i], epochs[i], "J2000", "SUN")
        s1 = body_state(bodies[i + 1], epochs[i + 1], "J2000", "SUN")
        tof = epochs[i + 1] - epochs[i]
        vv, _ = lambert(s0[:3], s1[:3], tof, GM_SUN)
        sol = solve_ivp(helio, (0, tof), np.concatenate([s0[:3], vv]),
                        t_eval=np.linspace(0, tof, 300), method="DOP853", rtol=1e-9, atol=1e-9)
        ax.plot(sol.y[0] / AU_KM, sol.y[1] / AU_KM, "k-", lw=1.4)
        # flyby marker
        ax.plot(s0[0] / AU_KM, s0[1] / AU_KM, "o", color="0.3", ms=6)
    ax.plot([], [], "k-", lw=1.4, label="transfer legs")
    ax.plot([], [], "o", color="0.3", ms=6, label="flyby bodies (V/E/E)")
    ax.set_aspect("equal")
    ax.set_xlabel("x (AU)"); ax.set_ylabel("y (AU)")
    ax.set_title(f"Galileo-class VEEGA to Jupiter (Venus-Earth-Earth)\n"
                 f"launch C3 {v['c3']:.1f} km$^2$/s$^2$ (vs direct ~85), "
                 f"{v['tof_total_days']/365.25:.1f} yr")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    os.makedirs(_OUT, exist_ok=True)
    path = os.path.join(_OUT, "veega_jupiter.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_porkchop():
    """Earth->Mars porkchop (total Delta-v over launch date x TOF) with the global optimum (Stage 21)."""
    from ..data.ephemeris import et, utc
    from ..interplanetary.porkchop import porkchop, optimize_window
    e0 = et("2026-01-01T00:00:00")
    pk = porkchop("EARTH", "MARS BARYCENTER", e0, dep_days=540, tof_range=(120, 400),
                  n_dep=70, n_tof=55)
    opt = optimize_window("EARTH", "MARS BARYCENTER", e0, dep_days=540,
                          tof_range=(120, 400), maxiter=50)
    dep_days = (pk["dep_grid"] - e0) / 86400.0
    Z = pk["total_ms"] / 1000.0       # km/s
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    levels = np.linspace(np.nanmin(Z), np.nanpercentile(Z, 85), 25)
    cs = ax.contourf(dep_days, pk["tof_grid"], Z, levels=levels, cmap="viridis_r", extend="max")
    fig.colorbar(cs, label="total Delta-v  (km/s, LEO injection + Mars capture)")
    ax.contour(dep_days, pk["tof_grid"], Z, levels=levels[::4], colors="k", linewidths=0.3, alpha=0.4)
    od = (opt["et_dep"] - e0) / 86400.0
    ax.plot(od, opt["tof_days"], "r*", ms=18,
            label=f"optimum {opt['utc_dep'][:10]}  {opt['total_ms']/1000:.2f} km/s")
    ax.set_xlabel("departure (days after 2026-01-01)")
    ax.set_ylabel("time of flight (days)")
    ax.set_title("Ariadne Earth->Mars porkchop (DE440) -- launch epoch is a free variable")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    os.makedirs(_OUT, exist_ok=True)
    path = os.path.join(_OUT, "porkchop_earth_mars.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_mars_transfer():
    """Top-down heliocentric flight path of the optimal Earth->Mars transfer (Stage 21)."""
    from scipy.integrate import solve_ivp
    from ..data.constants import GM_SUN, AU_KM
    from ..data.ephemeris import et, body_pos
    from ..interplanetary.porkchop import optimize_window
    e0 = et("2026-01-01T00:00:00")
    opt = optimize_window("EARTH", "MARS BARYCENTER", e0, dep_days=540,
                          tof_range=(120, 400), maxiter=50)
    tof = opt["tof_days"] * 86400.0

    def helio(t, s):
        r = s[:3]; rn = np.linalg.norm(r)
        return np.concatenate([s[3:], -GM_SUN * r / rn ** 3])
    s0 = np.concatenate([opt["r1"], opt["v1"]])
    sol = solve_ivp(helio, (0, tof), s0, t_eval=np.linspace(0, tof, 400),
                    method="DOP853", rtol=1e-10, atol=1e-10)

    def orbit(body):
        ts = opt["et_dep"] + np.linspace(0, 687 * 86400.0, 400)   # ~Mars year
        P = np.array([body_pos(body, t, "J2000", "SUN") for t in ts])
        return P[:, 0] / AU_KM, P[:, 1] / AU_KM

    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    ax.plot(0, 0, "o", color="gold", ms=15, label="Sun")
    ex, ey = orbit("EARTH"); ax.plot(ex, ey, color="tab:blue", lw=0.7, label="Earth orbit")
    mx, my = orbit("MARS BARYCENTER"); ax.plot(mx, my, color="tab:red", lw=0.7, label="Mars orbit")
    ax.plot(sol.y[0] / AU_KM, sol.y[1] / AU_KM, color="k", lw=1.8, label="transfer")
    ax.plot(opt["r1"][0] / AU_KM, opt["r1"][1] / AU_KM, "o", color="tab:blue", ms=9)
    rf = body_pos("MARS BARYCENTER", opt["et_arr"], "J2000", "SUN")
    ax.plot(rf[0] / AU_KM, rf[1] / AU_KM, "o", color="tab:red", ms=9)
    ax.set_aspect("equal")
    ax.set_xlabel("x (AU)"); ax.set_ylabel("y (AU)")
    ax.set_title(f"Optimal Earth->Mars transfer  ({opt['utc_dep'][:10]} -> {opt['utc_arr'][:10]},"
                 f" {opt['tof_days']:.0f} d)\nC3 {opt['c3']:.1f} km$^2$/s$^2$, total {opt['total_ms']/1000:.2f} km/s")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    os.makedirs(_OUT, exist_ok=True)
    path = os.path.join(_OUT, "mars_transfer.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_moon_tour():
    """Galilean gravity-assist tour: v_inf profile + the assist saving vs Hohmann (Stage 20)."""
    from ..transfers.tisserand import moon_tour
    t = moon_tour(flyby_alt_km=200.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    # left: v_inf at each moon along the tour
    labels, vis, vos = [], [], []
    for leg in t["legs"]:
        labels.append(f"{leg['from'][:3]}->{leg['to'][:3]}")
        vis.append(leg["vinf_inner_kms"]); vos.append(leg["vinf_outer_kms"])
    x = np.arange(len(labels))
    ax1.bar(x - 0.2, vis, 0.4, label="v$_\\infty$ at inner moon", color="tab:blue")
    ax1.bar(x + 0.2, vos, 0.4, label="v$_\\infty$ at outer moon", color="tab:orange")
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylabel("v$_\\infty$ (km/s)")
    ax1.set_title("Galilean tour: flyby v$_\\infty$ per leg")
    ax1.legend(fontsize=8)

    # right: gravity-assist deterministic Delta-v vs Hohmann baseline
    ga, hoh = t["ga_deterministic_dv_ms"], t["hohmann_dv_ms"]
    ax2.bar(["gravity-assist\n(deterministic)", "Hohmann\nbaseline"], [ga, hoh],
            color=["tab:green", "0.6"])
    ax2.set_ylabel("Io -> Callisto tour Delta-v (m/s)")
    ax2.set_title(f"Gravity assists save {t['saving_ms']:.0f} m/s "
                  f"({hoh / max(ga, 1e-9):.0f}x)")
    for i, val in enumerate([ga, hoh]):
        ax2.annotate(f"{val:.0f}", (i, val), ha="center", va="bottom", fontsize=9)
    fig.suptitle("Ariadne multi-moon tour mining (Tisserand graph)", fontsize=12)
    fig.tight_layout()
    os.makedirs(_OUT, exist_ok=True)
    path = os.path.join(_OUT, "moon_tour.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_nrho():
    """The Gateway-class L2 NRHO (3D) with halo-family context (Stage 19)."""
    from ..data.constants import EARTH_MOON, R_MOON
    from ..dynamics.cr3bp import propagate
    from ..orbits.nrho import nrho_family
    mu = EARTH_MOON.mu
    Ts = EARTH_MOON.T_star / 86400.0
    nrho, fam = nrho_family(mu, "L2", t_star_days=Ts, l_star=EARTH_MOON.L_star,
                            target_period_d=6.56, ds=4e-3)
    L = EARTH_MOON.L_star

    def xyz(orb, npts=600):
        sol = propagate(orb.s0, (0.0, orb.period), mu, t_eval=np.linspace(0, orb.period, npts))
        return ((sol.y[0] - (1 - mu)) * L, sol.y[1] * L, sol.y[2] * L)   # Moon-centered km

    fig = plt.figure(figsize=(8.2, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    # a few family members for context (every ~80th, the rounding family)
    for orb in fam[::max(1, len(fam) // 6)]:
        x, y, z = xyz(orb, 300)
        ax.plot(x, y, z, color="0.8", lw=0.5)
    if nrho is not None:
        x, y, z = xyz(nrho)
        ax.plot(x, y, z, color="tab:red", lw=2.0, label=f"NRHO  (period {nrho.period*Ts:.2f} d)")
    # the Moon
    u, v = np.mgrid[0:2*np.pi:24j, 0:np.pi:12j]
    ax.plot_surface(R_MOON*np.cos(u)*np.sin(v), R_MOON*np.sin(u)*np.sin(v),
                    R_MOON*np.cos(v), color="0.5", alpha=0.5, linewidth=0)
    ax.set_xlabel("x (km, Moon-centered)"); ax.set_ylabel("y (km)"); ax.set_zlabel("z (km)")
    ax.set_title("Earth-Moon L2 Near-Rectilinear Halo Orbit (Gateway class)\n"
                 "built by pseudo-arclength continuation of the halo family")
    ax.legend(fontsize=9)
    fig.tight_layout()
    os.makedirs(_OUT, exist_ok=True)
    path = os.path.join(_OUT, "nrho.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_solar_atlas():
    """The whole solar system on one plot: L1 distance vs mass ratio for all 23 systems (Stage 26)."""
    from ..data.constants import SOLAR_SYSTEM
    from ..transfers.jovian import moon_libration
    mus, ratios, names = [], [], []
    for S in SOLAR_SYSTEM:
        m = moon_libration(S)
        mus.append(S.mu)
        ratios.append(m["L1_km"] / S.L_star)
        names.append(S.name)
    mus = np.array(mus); ratios = np.array(ratios)

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    mm = np.logspace(np.log10(mus.min()) - 0.3, np.log10(mus.max()) + 0.3, 200)
    ax.loglog(mm, (mm / 3.0) ** (1.0 / 3.0), "--", color="0.5", label=r"Hill radius $(\mu/3)^{1/3}$")
    # color Sun-planet vs moon vs binary
    for x, y, n in zip(mus, ratios, names):
        col = ("tab:orange" if n.startswith("Sun-") else
               "tab:purple" if n in ("Pluto-Charon", "Didymos-Dimorphos") else "tab:blue")
        ax.loglog(x, y, "o", color=col, ms=8, zorder=3)
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(6, 3), fontsize=6.5)
    ax.loglog([], [], "o", color="tab:orange", label="Sun-planet")
    ax.loglog([], [], "o", color="tab:blue", label="planet-moon")
    ax.loglog([], [], "o", color="tab:purple", label="binary")
    ax.set_xlabel(r"mass ratio $\mu = m_2/(m_1+m_2)$")
    ax.set_ylabel("L1 distance from secondary / separation")
    ax.set_title("Ariadne across the entire solar system -- 23 CR3BP systems, 7 orders of magnitude\n"
                 "(periodic L1 Lyapunov orbits everywhere; all track the Hill-radius scaling)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    os.makedirs(_OUT, exist_ok=True)
    path = os.path.join(_OUT, "solar_atlas.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_atlas_systems():
    """The engine across the mass-ratio spectrum: L1 distance vs mu, with Hill scaling (Stage 16)."""
    from ..data.constants import EARTH_MOON, ATLAS_SYSTEMS
    from ..transfers.jovian import moon_libration

    systems = [EARTH_MOON] + list(ATLAS_SYSTEMS)
    mus, ratios, names = [], [], []
    for S in systems:
        m = moon_libration(S)
        mus.append(S.mu)
        ratios.append(m["L1_km"] / S.L_star)     # L1 distance from secondary / separation
        names.append(S.name)
    mus = np.array(mus); ratios = np.array(ratios)

    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    mm = np.logspace(np.log10(mus.min()) - 0.3, np.log10(mus.max()) + 0.3, 200)
    ax.loglog(mm, (mm / 3.0) ** (1.0 / 3.0), "--", color="0.5",
              label=r"Hill radius $(\mu/3)^{1/3}$")
    ax.loglog(mus, ratios, "o", color="tab:blue", ms=9, zorder=3)
    for x, y, n in zip(mus, ratios, names):
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(7, 4), fontsize=8)
    ax.set_xlabel(r"mass ratio $\mu = m_2/(m_1+m_2)$")
    ax.set_ylabel(r"L1 distance from secondary  /  separation")
    ax.set_title("Ariadne generalizes across 6 orders of magnitude in mass ratio\n"
                 "(periodic L1 Lyapunov orbits; L1 tracks the Hill-radius scaling)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    os.makedirs(_OUT, exist_ok=True)
    path = os.path.join(_OUT, "atlas_systems.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main():
    print("Rendering L1 Lyapunov family ...")
    print("  ->", figure_family())
    print("Rendering Earth->Moon Delta-v budget ...")
    print("  ->", figure_budget())
    print("Rendering real Moon orbit (DE440) ...")
    print("  ->", figure_moon_orbit())
    print("Rendering ephemeris validation (n-body vs DE440) ...")
    print("  ->", figure_ephemeris_validation())
    print("Rendering low-energy lunar transfer (ballistic capture) ...")
    print("  ->", figure_low_energy_transfer())
    print("Rendering Earth-Moon L1 halo family (3D) ...")
    print("  ->", figure_halo_3d())
    print("Rendering Genesis Sun-Earth superhighway ...")
    print("  ->", figure_genesis())
    print("Rendering full-ephemeris Earth-Moon transfer (DE440) ...")
    print("  ->", figure_ephemeris_transfer())
    print("Rendering Sun-assisted low-energy WSB transfer ...")
    print("  ->", figure_wsb_transfer())
    print("Rendering Delta-v vs coherence frontier ...")
    print("  ->", figure_coherence_frontier())
    print("Rendering low-thrust spiral ...")
    print("  ->", figure_low_thrust_spiral())
    print("Rendering transport graph + route ...")
    print("  ->", figure_transport_graph())
    print("Rendering atlas systems (mass-ratio spectrum) ...")
    print("  ->", figure_atlas_systems())
    print("Rendering solar-system atlas (23 systems) ...")
    print("  ->", figure_solar_atlas())
    print("Rendering Gateway-class NRHO (3D) ...")
    print("  ->", figure_nrho())
    print("Rendering Galilean gravity-assist moon tour ...")
    print("  ->", figure_moon_tour())
    print("Rendering Earth->Mars porkchop ...")
    print("  ->", figure_porkchop())
    print("Rendering Earth->Mars optimal transfer (heliocentric) ...")
    print("  ->", figure_mars_transfer())
    print("Rendering Galileo-class VEEGA to Jupiter ...")
    print("  ->", figure_veega())
    print("Rendering grand multi-objective trade space ...")
    print("  ->", figure_grand_tradeoff())
    print("Rendering coherence (FLI) field + manifold tubes ...")
    print("  ->", figure_coherence_field())
    print("Rendering L1<->L2 heteroclinic tubes ...")
    print("  ->", figure_heteroclinic())


if __name__ == "__main__":
    main()
