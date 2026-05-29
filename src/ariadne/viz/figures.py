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
    print("Rendering L1<->L2 heteroclinic tubes ...")
    print("  ->", figure_heteroclinic())


if __name__ == "__main__":
    main()
