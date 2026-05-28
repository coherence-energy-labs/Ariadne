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


def main():
    print("Rendering L1 Lyapunov family ...")
    print("  ->", figure_family())
    print("Rendering L1<->L2 heteroclinic tubes ...")
    print("  ->", figure_heteroclinic())


if __name__ == "__main__":
    main()
