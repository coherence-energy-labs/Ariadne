"""Solar-system transfer corridor atlas.

This module builds a first-cut map of heliocentric Lambert corridors between
major gravity bodies on real ephemerides.  The rendered "tubes" are not claimed
to be CR3BP invariant manifolds; they are launch-window sampled Lambert
corridors, useful as a whole-system route atlas and as a visual front end for
the higher-fidelity engines.
"""
from __future__ import annotations

import heapq
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..data.constants import AU_KM, GM_SUN
from ..data.ephemeris import body_state, et, utc
from .porkchop import DAY, lambert_transfer


@dataclass(frozen=True)
class AtlasBody:
    """One mapped solar-system node."""
    name: str
    ephemeris_name: str
    a_au: float
    color: str
    radius_scale: float = 1.0


@dataclass(frozen=True)
class TransferCorridor:
    """Best sampled transfer corridor from one body to another."""
    origin: str
    target: str
    et_dep: float
    et_arr: float
    tof_days: float
    total_dv_ms: float
    c3_km2_s2: float
    dep_vinf_kms: float
    arr_vinf_kms: float
    score: float
    feasible: bool = True
    samples: int = 0


@dataclass(frozen=True)
class SolarTransferAtlas:
    """A complete route graph plus visualization metadata."""
    schema: str
    epoch_start_utc: str
    departure_window_days: float
    bodies: tuple[AtlasBody, ...]
    corridors: tuple[TransferCorridor, ...]
    route_origin: str | None = None
    route_target: str | None = None
    optimal_route: tuple[str, ...] = ()
    optimal_route_score: float | None = None
    certificate_hash: str = ""
    artifacts: dict = field(default_factory=dict)


MAJOR_BODIES: tuple[AtlasBody, ...] = (
    AtlasBody("MERCURY", "MERCURY BARYCENTER", 0.387, "#9a8f82", 0.75),
    AtlasBody("VENUS", "VENUS BARYCENTER", 0.723, "#d79a36", 0.95),
    AtlasBody("EARTH", "EARTH", 1.000, "#3f7fd2", 1.0),
    AtlasBody("MARS", "MARS BARYCENTER", 1.524, "#c95d3d", 0.9),
    AtlasBody("JUPITER", "JUPITER BARYCENTER", 5.203, "#d0aa74", 1.55),
    AtlasBody("SATURN", "SATURN BARYCENTER", 9.537, "#d8c28b", 1.45),
    AtlasBody("URANUS", "URANUS BARYCENTER", 19.191, "#79c7c7", 1.25),
    AtlasBody("NEPTUNE", "NEPTUNE BARYCENTER", 30.07, "#4f70d8", 1.25),
)


def _stable_hash(payload) -> str:
    import hashlib
    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _patch_matplotlib_deepcopy_bug():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.path as mpath
    mpath.Path.__deepcopy__ = lambda self, memo: self


def _hohmann_tof_days(a1_au: float, a2_au: float) -> float:
    sma_km = 0.5 * (a1_au + a2_au) * AU_KM
    return math.pi * math.sqrt(sma_km ** 3 / GM_SUN) / DAY


def _tof_range(a1_au: float, a2_au: float) -> tuple[float, float]:
    h = _hohmann_tof_days(a1_au, a2_au)
    lo = max(45.0, 0.52 * h)
    hi = min(18000.0, max(lo + 45.0, 1.85 * h))
    return lo, hi


def _corridor_score(rec: dict) -> float:
    # Low-energy first, with a small time term so multi-leg routes do not win
    # merely by chaining many slow cheap arcs.
    return float(rec["total_ms"] / 1000.0 + 0.0035 * rec["tof_days"] + 0.035 * rec["c3"])


def best_corridor(origin: AtlasBody, target: AtlasBody, epoch_start_et: float,
                  departure_window_days: float, *, n_dep: int = 14,
                  n_tof: int = 10) -> TransferCorridor | None:
    """Sample a launch-window grid and return the best corridor."""
    if origin.name == target.name:
        return None
    dep_grid = epoch_start_et + np.linspace(0.0, departure_window_days, n_dep) * DAY
    tof_grid = np.linspace(*_tof_range(origin.a_au, target.a_au), n_tof)
    best = None
    samples = 0
    for ed in dep_grid:
        for tof in tof_grid:
            samples += 1
            rec = lambert_transfer(
                origin.ephemeris_name, target.ephemeris_name, float(ed), float(tof),
                capture=False,
            )
            if rec is None:
                continue
            score = _corridor_score(rec)
            if best is None or score < best[0]:
                best = (score, rec)
    if best is None:
        return None
    score, rec = best
    return TransferCorridor(
        origin=origin.name,
        target=target.name,
        et_dep=float(rec["et_dep"]),
        et_arr=float(rec["et_arr"]),
        tof_days=float(rec["tof_days"]),
        total_dv_ms=float(rec["total_ms"]),
        c3_km2_s2=float(rec["c3"]),
        dep_vinf_kms=float(rec["dep_vinf_kms"]),
        arr_vinf_kms=float(rec["arr_vinf_kms"]),
        score=float(score),
        samples=samples,
    )


def build_solar_transfer_atlas(
    *,
    epoch_start: str = "2028-01-01T00:00:00",
    departure_window_days: float = 900.0,
    bodies: tuple[AtlasBody, ...] = MAJOR_BODIES,
    n_dep: int = 14,
    n_tof: int = 10,
    route_origin: str | None = "EARTH",
    route_target: str | None = "SATURN",
) -> SolarTransferAtlas:
    """Build a directed corridor graph across major solar-system bodies."""
    e0 = et(epoch_start)
    corridors = []
    for origin in bodies:
        for target in bodies:
            rec = best_corridor(origin, target, e0, departure_window_days, n_dep=n_dep, n_tof=n_tof)
            if rec is not None:
                corridors.append(rec)
    path, score = shortest_corridor_route(corridors, route_origin, route_target)
    shell = {
        "schema": "ariadne.solar_transfer_atlas.v1",
        "epoch_start_utc": epoch_start,
        "departure_window_days": departure_window_days,
        "bodies": [asdict(b) for b in bodies],
        "corridors": [asdict(c) for c in corridors],
        "route_origin": route_origin,
        "route_target": route_target,
        "optimal_route": path,
        "optimal_route_score": score,
    }
    return SolarTransferAtlas(
        schema=shell["schema"],
        epoch_start_utc=epoch_start,
        departure_window_days=float(departure_window_days),
        bodies=tuple(bodies),
        corridors=tuple(corridors),
        route_origin=route_origin,
        route_target=route_target,
        optimal_route=tuple(path),
        optimal_route_score=score,
        certificate_hash=_stable_hash(shell),
    )


def shortest_corridor_route(corridors, origin: str | None, target: str | None):
    """Dijkstra route over the sampled corridor graph."""
    if not origin or not target:
        return (), None
    graph: dict[str, list[tuple[float, str]]] = {}
    for c in corridors:
        graph.setdefault(c.origin, []).append((float(c.score), c.target))
    dist = {origin.upper(): 0.0}
    prev: dict[str, str] = {}
    heap = [(0.0, origin.upper())]
    target = target.upper()
    while heap:
        cost, node = heapq.heappop(heap)
        if node == target:
            break
        if cost > dist.get(node, math.inf):
            continue
        for w, nxt in graph.get(node, []):
            cand = cost + w
            if cand < dist.get(nxt, math.inf):
                dist[nxt] = cand
                prev[nxt] = node
                heapq.heappush(heap, (cand, nxt))
    if target not in dist:
        return (), None
    path = [target]
    while path[-1] != origin.upper():
        path.append(prev[path[-1]])
    path.reverse()
    return tuple(path), float(dist[target])


def _body_positions(atlas: SolarTransferAtlas):
    e0 = et(atlas.epoch_start_utc)
    out = {}
    for b in atlas.bodies:
        s = body_state(b.ephemeris_name, e0, "J2000", "SUN")
        out[b.name] = s[:3] / AU_KM
    return out


def _sqrt_xy(xy_au):
    xy = np.asarray(xy_au, float)
    r = np.linalg.norm(xy[..., :2], axis=-1)
    scale = np.where(r > 0.0, np.sqrt(r) / r, 0.0)
    return xy[..., :2] * scale[..., None]


def _corridor_curve(p0, p1, bend=0.18, n=90):
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    mid = 0.5 * (p0 + p1)
    v = p1 - p0
    nrm = np.linalg.norm(v)
    if nrm == 0.0:
        return np.repeat(p0[None, :], n, axis=0)
    normal = np.array([-v[1], v[0]]) / nrm
    control = mid + normal * bend * nrm
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * control + t ** 2 * p1


def _corridor_lookup(atlas: SolarTransferAtlas) -> dict[tuple[str, str], TransferCorridor]:
    return {(c.origin, c.target): c for c in atlas.corridors}


def _propagated_lambert_arc(c: TransferCorridor, bodies: dict[str, AtlasBody], n=420):
    """Recompute and propagate one Lambert corridor for route-focused plotting."""
    from scipy.integrate import solve_ivp

    rec = lambert_transfer(
        bodies[c.origin].ephemeris_name,
        bodies[c.target].ephemeris_name,
        c.et_dep,
        c.tof_days,
        capture=False,
    )
    if rec is None:
        return None

    def helio(_, state):
        r = state[:3]
        rn = np.linalg.norm(r)
        return np.concatenate([state[3:], -GM_SUN * r / rn ** 3])

    s0 = np.concatenate([rec["r1"], rec["v1"]])
    sol = solve_ivp(
        helio,
        (0.0, c.tof_days * DAY),
        s0,
        t_eval=np.linspace(0.0, c.tof_days * DAY, n),
        method="DOP853",
        rtol=2e-9,
        atol=2e-9,
    )
    if not sol.success:
        return None
    return sol.y[:3].T / AU_KM


def _orbit_xy(body: AtlasBody, epoch_et: float, n=540):
    period_days = math.sqrt(body.a_au ** 3) * 365.25
    ts = epoch_et + np.linspace(0.0, period_days * DAY, n)
    try:
        pts = np.array([body_state(body.ephemeris_name, t, "J2000", "SUN")[:3] for t in ts]) / AU_KM
        return pts[:, :2]
    except Exception:
        theta = np.linspace(0.0, 2.0 * math.pi, n)
        return np.column_stack([body.a_au * np.cos(theta), body.a_au * np.sin(theta)])


# ===========================================================================
# Spatial cost-field heatmap (top-down solar system, colour = transfer cost)
# ===========================================================================

def render_solar_cost_field(atlas: SolarTransferAtlas,
                             outpath: str | Path,
                             *, n_grid: int = 320,
                             max_au: float | None = None,
                             dv_clip_kms: float = 18.0) -> Path:
    """Top-down heatmap of the solar system showing transfer cost to every (x, y).

    For each grid cell (x_au, y_au), estimate the dv cost of a Hohmann-like
    Earth-to-(x, y) transfer at the atlas epoch. Build a spatial cost field
    that you can READ like a topographic map: dark = cheap, bright = expensive.
    Overlay planet orbits, current body positions, and the selected optimal
    route as a glowing trail through the field.

    This is the "where is it cheap to fly?" view the corridor matrix can't
    show. Operators can see at a glance which sectors of the solar system
    are within reach for the epoch.
    """
    _patch_matplotlib_deepcopy_bug()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from . import nasa_plate as NP

    NP.apply_style()
    e0 = et(atlas.epoch_start_utc)
    body_by_name = {b.name: b for b in atlas.bodies}
    if max_au is None:
        if atlas.optimal_route:
            max_au = max(body_by_name[n].a_au for n in atlas.optimal_route
                          if n in body_by_name) * 1.18
        else:
            max_au = max(b.a_au for b in atlas.bodies) * 1.08
    max_au = float(max_au)

    # Earth's heliocentric state at epoch -- the launch site for the cost field
    earth_state = body_state("EARTH", e0, "J2000", "SUN")
    earth_pos = np.array(earth_state[:3], dtype=float)
    earth_v = np.array(earth_state[3:], dtype=float)
    v_earth_kms = float(np.linalg.norm(earth_v))

    # Build the cost field via Hohmann approximation. For each grid cell
    # at radius R, treat the cell as a circular orbit at R and a Hohmann
    # transfer FROM Earth's orbit (a=1 AU) TO R. The total dv is then:
    #   dv = |v_perihelion(transfer) - v_earth_circular|        (departure)
    #      + |v_target_circular     - v_aphelion(transfer)|    (insertion)
    # Units: GM_SUN is in km^3/s^2 and AU_KM is in km, so velocities come
    # out in km/s directly. Do NOT divide by 1000.
    # This is a fast analytic proxy -- not exact Lambert, but a faithful
    # topology of which regions are reachable cheaply at this epoch.
    xs = np.linspace(-max_au, max_au, n_grid)
    ys = np.linspace(-max_au, max_au, n_grid)
    X, Y = np.meshgrid(xs, ys)
    R = np.sqrt(X ** 2 + Y ** 2)
    a_earth_au = 1.0
    # Avoid div-by-zero at the Sun + near Earth's orbit
    R_safe = np.maximum(R, 0.25)
    # Hohmann semi-major axis from Earth (a=1 AU) to target r (in AU)
    a_xfer = 0.5 * (a_earth_au + R_safe)
    v_earth_orb = math.sqrt(GM_SUN / (a_earth_au * AU_KM))
    v_target_orb_circ = np.sqrt(GM_SUN / (R_safe * AU_KM))
    # Outbound transfers (R > 1): perihelion = Earth side
    # Inbound transfers (R < 1): aphelion = Earth side -- formula symmetric
    v_at_earth_xfer = np.sqrt(GM_SUN * (2 / (a_earth_au * AU_KM)
                                          - 1 / (a_xfer * AU_KM)))
    v_at_target_xfer = np.sqrt(GM_SUN * (2 / (R_safe * AU_KM)
                                            - 1 / (a_xfer * AU_KM)))
    # Inject burn at Earth + circularise at target
    dv_kms = (np.abs(v_at_earth_xfer - v_earth_orb)
              + np.abs(v_target_orb_circ - v_at_target_xfer))
    # Saturate at the clip level so colour map renders the useful band
    cost = np.clip(dv_kms, 0, dv_clip_kms)

    # Heat-trail palette: deep navy -> cyan -> gold (cheap to expensive)
    trail_cmap = LinearSegmentedColormap.from_list("ariadne_trail", [
        (0.0,  "#04101a"),
        (0.18, "#0c2336"),
        (0.38, "#176aa0"),
        (0.58, "#3ec5d4"),
        (0.78, "#f0c674"),
        (1.0,  "#f85149"),
    ])

    fig = plt.figure(figsize=(13.6, 12.4), facecolor=NP.DEEP_SPACE_BG)
    fig.subplots_adjust(left=0.06, right=0.92, top=0.91, bottom=0.07)
    ax = fig.add_subplot(111)
    ax.set_facecolor(NP.DEEP_SPACE_BG)

    # Cost field (raster)
    im = ax.imshow(cost, extent=(-max_au, max_au, -max_au, max_au),
                    origin="lower", cmap=trail_cmap, alpha=0.92,
                    interpolation="bilinear", zorder=1)
    cb = fig.colorbar(im, ax=ax, pad=0.015, shrink=0.82)
    cb.set_label("EARTH -> (X, Y) HOHMANN COST (km/s)",
                  color=NP.TEXT_SECONDARY, fontsize=8.5)
    cb.ax.tick_params(colors=NP.TEXT_SECONDARY)
    cb.outline.set_edgecolor(NP.GRID_LINE)

    # Iso-cost contours so the topology is readable
    cs = ax.contour(X, Y, cost,
                     levels=np.linspace(2, dv_clip_kms, 6),
                     colors=NP.TEXT_PRIMARY, linewidths=0.4, alpha=0.30,
                     zorder=2)
    ax.clabel(cs, inline=True, fontsize=6.5, fmt="%.0f km/s",
               colors=NP.TEXT_PRIMARY)

    # Orbit lines (thin dashed circles -- top-down approximation)
    for body in atlas.bodies:
        if body.a_au > max_au * 1.05:
            continue
        circle = plt.Circle((0, 0), body.a_au, fill=False,
                              color=NP.TEXT_SECONDARY, alpha=0.45,
                              linewidth=0.6, linestyle=(0, (4, 3)), zorder=3)
        ax.add_patch(circle)

    # Sun
    ax.scatter([0], [0], s=520, color=NP.ACCENT_GOLD,
                edgecolor=NP.DEEP_SPACE_BG, linewidths=1.5, zorder=10)

    # Body positions + labels with halos
    pos3 = _body_positions(atlas)
    for body in atlas.bodies:
        if body.a_au > max_au * 1.05:
            continue
        x, y = pos3[body.name][:2]
        ax.scatter([x], [y], s=140 * body.radius_scale,
                    color=body.color, edgecolor=NP.TEXT_PRIMARY,
                    linewidths=1.0, zorder=11)
        NP.halo_text(ax, x, y + max_au * 0.025,
                      body.name.title(),
                      color=NP.TEXT_PRIMARY, fontsize=9, weight="bold",
                      ha="center", halo_width=2.5)

    # Selected route as a glowing trail through the cost field
    if atlas.optimal_route and len(atlas.optimal_route) >= 2:
        route_xs, route_ys = [], []
        for name in atlas.optimal_route:
            if name in body_by_name:
                route_xs.append(pos3[name][0])
                route_ys.append(pos3[name][1])
        if len(route_xs) >= 2:
            # Multi-pass glow
            for lw, alpha in ((8.5, 0.22), (5.5, 0.42), (3.0, 0.88)):
                ax.plot(route_xs, route_ys, color=NP.ACCENT_CYAN, lw=lw,
                          alpha=alpha, solid_capstyle="round", zorder=12)
            ax.scatter(route_xs, route_ys, s=180, marker="o",
                        facecolor=NP.ACCENT_CYAN,
                        edgecolor=NP.DEEP_SPACE_BG, linewidths=1.5,
                        zorder=14)
            # Endpoints
            ax.scatter([route_xs[0]], [route_ys[0]], s=240, marker="^",
                        facecolor=NP.ACCENT_GREEN,
                        edgecolor=NP.DEEP_SPACE_BG, linewidths=1.5,
                        zorder=15)
            ax.scatter([route_xs[-1]], [route_ys[-1]], s=320, marker="*",
                        facecolor=NP.ACCENT_GOLD,
                        edgecolor=NP.DEEP_SPACE_BG, linewidths=1.5,
                        zorder=15)

    ax.set_xlim(-max_au, max_au); ax.set_ylim(-max_au, max_au)
    ax.set_aspect("equal")
    NP.style_axes(ax, title="", grid=False,
                   xlabel="X (AU)  -  HELIOCENTRIC J2000",
                   ylabel="Y (AU)  -  HELIOCENTRIC J2000")
    # Thin grid only (no major)
    ax.grid(True, color=NP.GRID_LINE, alpha=0.15, linewidth=0.4)

    route_label = " -> ".join(atlas.optimal_route) if atlas.optimal_route else "no route selected"
    NP.mission_title(fig,
        title="Solar-system cost field",
        subtitle=f"Heatmap = Hohmann dv from Earth at "
                  f"{atlas.epoch_start_utc[:10]}. Cyan glow = selected route: {route_label}")
    NP.mission_footer(fig,
        mission_id="SOLAR_COST_FIELD",
        cert=atlas.certificate_hash,
        fidelity="hohmann approximation + DE ephemeris positions",
        extras=[("grid", f"{n_grid}x{n_grid}"),
                  ("clip", f"{dv_clip_kms:.0f} km/s")])

    out = Path(outpath)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight",
                 facecolor=NP.DEEP_SPACE_BG)
    plt.close(fig)
    NP.reset_style()
    return out


# ===========================================================================
# Trail atlas (every-body-to-every-body tube map; selected route glows)
# ===========================================================================

def render_solar_trail_atlas(atlas: SolarTransferAtlas,
                              outpath: str | Path,
                              *, max_au: float | None = None) -> Path:
    """Tube-style transit map of every body-to-body corridor.

    Each pair of bodies is connected by a curved tube. The tube's
    LINE WIDTH encodes corridor quality (low cost = thick), the COLOUR
    encodes total dv (cividis low-to-high), and the OPACITY tapers
    toward the planets so labels stay legible. The selected route is
    painted on top as a multi-pass glowing gold trail with directional
    arrows.

    Reads like a transit map -- you see every "trail" in the forest at
    once and the recommended one shines through.
    """
    _patch_matplotlib_deepcopy_bug()
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from . import nasa_plate as NP

    NP.apply_style()
    e0 = et(atlas.epoch_start_utc)
    pos3 = _body_positions(atlas)
    pos = {name: xyz[:2] for name, xyz in pos3.items()}
    body_by_name = {b.name: b for b in atlas.bodies}

    if max_au is None:
        max_au = max(b.a_au for b in atlas.bodies) * 1.05
    max_au = float(max_au)

    costs_kms = np.array([c.total_dv_ms / 1000.0 for c in atlas.corridors])
    cmin = float(np.percentile(costs_kms, 10)) if costs_kms.size else 0.0
    cmax = float(np.percentile(costs_kms, 90)) if costs_kms.size else 1.0
    if cmax <= cmin:
        cmax = float(costs_kms.max()) if costs_kms.size else 1.0
    norm = Normalize(vmin=cmin, vmax=cmax)
    cm = plt.get_cmap("cividis")
    route_edges = (list(zip(atlas.optimal_route[:-1], atlas.optimal_route[1:]))
                    if atlas.optimal_route and len(atlas.optimal_route) >= 2 else [])
    by_edge = _corridor_lookup(atlas)

    def draw_scene(ax, *, view_lim, inner=False):
        """Render the trail-atlas scene into `ax`. Skips bodies/corridors
        that fall outside the view (so the inner-system inset doesn't try
        to label Saturn/Neptune)."""
        ax.set_facecolor(NP.DEEP_SPACE_BG)

        # Orbit rings inside the view
        for body in atlas.bodies:
            if body.a_au > view_lim * 1.05:
                continue
            circ = plt.Circle((0, 0), body.a_au, fill=False,
                                color=NP.GRID_LINE, alpha=0.55,
                                linewidth=0.6, linestyle=(0, (4, 3)),
                                zorder=2)
            ax.add_patch(circ)

        # Tubes between body-pairs that fit inside the view.
        # Sort by cost descending so cheapest ones are drawn LAST (on top).
        for c in sorted(atlas.corridors, key=lambda x: -x.total_dv_ms):
            if c.origin not in pos or c.target not in pos:
                continue
            if (body_by_name[c.origin].a_au > view_lim
                    or body_by_name[c.target].a_au > view_lim):
                continue
            p0 = pos[c.origin]; p1 = pos[c.target]
            curve = _corridor_curve(p0, p1,
                                      bend=0.16 if c.origin < c.target else -0.16,
                                      n=160)
            cost_kms = c.total_dv_ms / 1000.0
            rgba = cm(norm(cost_kms))
            q = max(0.0, min(1.0, (cost_kms - cmin) / max(cmax - cmin, 1e-6)))
            lw_base = (2.8 if inner else 3.4) - 2.4 * q
            # Glow halo + main tube
            ax.plot(curve[:, 0], curve[:, 1], color=NP.DEEP_SPACE_BG,
                      lw=lw_base + 1.6, alpha=0.55,
                      solid_capstyle="round", zorder=3)
            ax.plot(curve[:, 0], curve[:, 1], color=rgba, lw=lw_base,
                      alpha=0.78, solid_capstyle="round", zorder=4)

        # Sun
        ax.scatter([0], [0], s=620 if not inner else 280,
                    color=NP.ACCENT_GOLD,
                    edgecolor=NP.DEEP_SPACE_BG, linewidths=1.5, zorder=10)

        # Bodies + labels
        for body in atlas.bodies:
            if body.a_au > view_lim * 1.05:
                continue
            x, y = pos[body.name]
            ax.scatter([x], [y],
                        s=(170 if not inner else 110) * body.radius_scale,
                        color=body.color, edgecolor=NP.TEXT_PRIMARY,
                        linewidths=1.2, zorder=11)
            r_body = math.hypot(x, y)
            if r_body > 0:
                # Label outward from Sun + scale offset to the view size
                dx, dy = x / r_body, y / r_body
                offset = view_lim * (0.05 if inner else 0.03)
                NP.halo_text(ax, x + dx * offset, y + dy * offset,
                              body.name.title(),
                              color=NP.TEXT_PRIMARY,
                              fontsize=(9 if inner else 10),
                              weight="bold",
                              ha="center", va="center", halo_width=2.5)

        # Selected route on top
        for a, b in route_edges:
            if a not in pos or b not in pos:
                continue
            if (body_by_name[a].a_au > view_lim
                    or body_by_name[b].a_au > view_lim):
                continue
            c = by_edge.get((a, b))
            curve = _corridor_curve(pos[a], pos[b],
                                      bend=0.10 if a < b else -0.10,
                                      n=240)
            for lw, alpha, color in (
                ((7.5 if inner else 9.0), 0.25, NP.ACCENT_GOLD),
                ((5.0 if inner else 6.0), 0.45, NP.ACCENT_GOLD),
                ((2.6 if inner else 3.0), 0.95, NP.ACCENT_GOLD),
            ):
                ax.plot(curve[:, 0], curve[:, 1], color=color, lw=lw,
                          alpha=alpha, solid_capstyle="round", zorder=12)
            if c is not None:
                k0, k1 = int(0.60 * len(curve)), int(0.66 * len(curve))
                ax.annotate("", xy=curve[k1], xytext=curve[k0],
                              arrowprops={"arrowstyle": "-|>",
                                            "color": NP.ACCENT_GOLD,
                                            "lw": 2.2, "mutation_scale": 22,
                                            "alpha": 0.95}, zorder=13)
                if not inner:
                    mid = curve[int(0.50 * len(curve))]
                    NP.halo_text(ax, mid[0], mid[1],
                                  f"{c.total_dv_ms/1000:.1f} km/s\n{c.tof_days:.0f} d",
                                  color=NP.ACCENT_GOLD, fontsize=8.5,
                                  weight="bold", ha="center", va="center",
                                  halo_width=2.5)

        ax.set_xlim(-view_lim, view_lim); ax.set_ylim(-view_lim, view_lim)
        ax.set_aspect("equal")

    fig = plt.figure(figsize=(14.4, 12.4), facecolor=NP.DEEP_SPACE_BG)
    fig.subplots_adjust(left=0.05, right=0.93, top=0.91, bottom=0.07)
    ax = fig.add_subplot(111)
    draw_scene(ax, view_lim=max_au, inner=False)
    NP.style_axes(ax, title="", grid=False,
                    xlabel="X (AU)  -  HELIOCENTRIC J2000",
                    ylabel="Y (AU)  -  HELIOCENTRIC J2000")

    # Inner-system inset so Mercury/Venus/Earth/Mars labels don't collide
    axins = inset_axes(ax, width="32%", height="32%", loc="lower left",
                         borderpad=2.0)
    draw_scene(axins, view_lim=2.1, inner=True)
    axins.set_xticks([]); axins.set_yticks([])
    axins.set_title("INNER SYSTEM (<= 2.1 AU)",
                      color=NP.TEXT_SECONDARY,
                      fontsize=8.5, weight="bold", pad=4)
    for spine in axins.spines.values():
        spine.set_edgecolor(NP.GRID_LINE)
        spine.set_linewidth(0.8)

    # Colorbar for the tube colour mapping
    sm = ScalarMappable(norm=norm, cmap=cm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.015, shrink=0.78)
    cb.set_label("CORRIDOR dv (km/s)  -  thicker = cheaper, gold = selected",
                  color=NP.TEXT_SECONDARY, fontsize=8.5)
    cb.ax.tick_params(colors=NP.TEXT_SECONDARY)
    cb.outline.set_edgecolor(NP.GRID_LINE)

    route_label = " -> ".join(atlas.optimal_route) if atlas.optimal_route else "no route selected"
    NP.mission_title(fig,
        title="Solar-system trail atlas",
        subtitle=f"Every body-to-body corridor (thickness = quality); "
                  f"selected route: {route_label}")
    NP.mission_footer(fig,
        mission_id="SOLAR_TRAIL_ATLAS",
        cert=atlas.certificate_hash,
        fidelity="lambert pairwise + DE ephemerides",
        extras=[("bodies", str(len(atlas.bodies))),
                  ("trails", str(len(atlas.corridors)))])

    out = Path(outpath)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight",
                 facecolor=NP.DEEP_SPACE_BG)
    plt.close(fig)
    NP.reset_style()
    return out


def render_solar_transfer_atlas(atlas: SolarTransferAtlas, outdir: str | Path) -> dict:
    """Write atlas JSON plus solar corridor PNG artifacts."""
    _patch_matplotlib_deepcopy_bug()
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    from . import nasa_plate as NP

    NP.apply_style()

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    e0 = et(atlas.epoch_start_utc)
    pos3 = _body_positions(atlas)
    pos = {name: xyz[:2] for name, xyz in pos3.items()}
    costs = np.array([c.score for c in atlas.corridors], float)
    cmin, cmax = float(np.min(costs)), float(np.percentile(costs, 88))
    if cmax <= cmin:
        cmax = float(np.max(costs))

    route_edges = set(zip(atlas.optimal_route[:-1], atlas.optimal_route[1:]))
    body_by_name = {b.name: b for b in atlas.bodies}
    by_edge = _corridor_lookup(atlas)
    route_label = " -> ".join(atlas.optimal_route) if atlas.optimal_route else "no route selected"
    lim = max(b.a_au for b in atlas.bodies) * 1.08

    def draw_orbits_and_bodies(ax, *, inner=False, dark=True):
        for body in atlas.bodies:
            if inner and body.a_au > 2.2:
                continue
            orbit = _orbit_xy(body, e0)
            ax.plot(orbit[:, 0], orbit[:, 1], color=body.color, lw=0.75, alpha=0.42, zorder=1)
        ax.scatter([0], [0], s=300 if not inner else 180, color="gold",
                   edgecolor="k", linewidth=0.8, zorder=9)
        for body in atlas.bodies:
            if inner and body.a_au > 2.2:
                continue
            x, y = pos[body.name]
            ax.scatter([x], [y], s=(86 if not inner else 62) * body.radius_scale,
                       color=body.color, edgecolor="white" if dark else "k",
                       linewidth=0.8, zorder=10)
            ax.annotate(body.name.title(), (x, y), textcoords="offset points", xytext=(6, 5),
                        fontsize=8.8 if not inner else 7.3,
                        color="white" if dark else "black", weight="bold",
                        zorder=11, clip_on=False)

    def draw_corridors(ax, *, inner=False):
        for c in sorted(atlas.corridors, key=lambda x: x.score):
            if inner and (body_by_name[c.origin].a_au > 2.2 or body_by_name[c.target].a_au > 2.2):
                continue
            curve = _corridor_curve(
                pos[c.origin], pos[c.target],
                bend=0.10 if c.origin < c.target else -0.10,
                n=130,
            )
            q = (min(c.score, cmax) - cmin) / (cmax - cmin + 1e-12)
            ax.plot(curve[:, 0], curve[:, 1], color="cyan", lw=0.25 + 1.5 * (1.0 - q),
                    alpha=0.04 + 0.22 * (1.0 - q), zorder=2)
        for a, b in route_edges:
            c = by_edge.get((a, b))
            if c is None:
                continue
            curve = _corridor_curve(
                pos[a], pos[b],
                bend=0.10 if a < b else -0.10,
                n=180,
            )
            if inner and np.nanmax(np.linalg.norm(curve, axis=1)) > 2.35:
                continue
            ax.plot(curve[:, 0], curve[:, 1], color="white", lw=5.3, alpha=0.94, zorder=6)
            ax.plot(curve[:, 0], curve[:, 1], color="deepskyblue", lw=2.5, alpha=1.0, zorder=7)
            k0, k1 = int(0.60 * len(curve)), int(0.66 * len(curve))
            ax.annotate("", xy=curve[k1], xytext=curve[k0],
                        arrowprops={"arrowstyle": "-|>", "color": "white", "lw": 2.1,
                                    "mutation_scale": 20, "alpha": 0.95},
                        zorder=8)
            mid = curve[int(0.48 * len(curve))]
            ax.annotate(f"{c.total_dv_ms/1000:.1f} km/s\n{c.tof_days:.0f} d",
                        mid, textcoords="offset points", xytext=(8, -12), fontsize=7.2,
                        color="white", bbox={"boxstyle": "round,pad=0.25", "fc": "#10151f",
                                             "ec": "white", "alpha": 0.78}, zorder=12)

    fig, ax = plt.subplots(figsize=(15.6, 11.2), facecolor=NP.DEEP_SPACE_BG)
    fig.subplots_adjust(left=0.06, right=0.96, top=0.91, bottom=0.07)
    ax.set_facecolor(NP.DEEP_SPACE_BG)
    draw_orbits_and_bodies(ax)
    draw_corridors(ax)
    NP.style_axes(ax, title="",
                   xlabel="X (AU)  -  HELIOCENTRIC J2000",
                   ylabel="Y (AU)  -  HELIOCENTRIC J2000")
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

    axins = inset_axes(ax, width="32%", height="32%", loc="lower left",
                         borderpad=2.0)
    axins.set_facecolor(NP.PANEL_BG)
    draw_orbits_and_bodies(axins, inner=True)
    draw_corridors(axins, inner=True)
    axins.set_xlim(-2.25, 2.25); axins.set_ylim(-2.25, 2.25)
    axins.set_aspect("equal")
    axins.set_xticks([]); axins.set_yticks([])
    axins.set_title("INNER SYSTEM DETAIL", color=NP.TEXT_SECONDARY,
                      fontsize=8.5, weight="bold", pad=4)
    for spine in axins.spines.values():
        spine.set_edgecolor(NP.GRID_LINE)
        spine.set_linewidth(0.8)
    if atlas.optimal_route_score is not None:
        NP.card_box(ax, 0.73, 0.07,
                      text=(f"epoch:             {atlas.epoch_start_utc[:10]}\n"
                            f"corridors scored:  {len(atlas.corridors)}\n"
                            f"route score:       {atlas.optimal_route_score:.2f}\n"
                            f"selected route:    {route_label[:35]}"),
                      header="MISSION ATLAS SUMMARY", width=0.26, fontsize=8.5,
                      border=NP.ACCENT_CYAN)

    NP.mission_title(fig,
        title="Solar-system corridor atlas",
        subtitle=f"Faint cyan = sampled pairwise Lambert corridors; "
                  f"bright blue/white = selected route: {route_label}")
    NP.mission_footer(fig,
        mission_id="SOLAR_CORRIDOR_ATLAS",
        cert=atlas.atlas_hash if hasattr(atlas, "atlas_hash") else None,
        fidelity="lambert + DE ephemerides",
        extras=[("bodies", str(len(atlas.bodies))),
                  ("corridors", str(len(atlas.corridors)))])

    network_path = out / "solar_transfer_corridor_atlas.png"
    fig.savefig(network_path, dpi=190, bbox_inches="tight",
                 facecolor=NP.DEEP_SPACE_BG)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.8, 10.2))
    ax.set_facecolor("#fafafa")
    relevant = set(atlas.optimal_route) or {b.name for b in atlas.bodies}
    focus_lim = max([body_by_name[n].a_au for n in relevant if n in body_by_name] + [1.6]) * 1.18
    for body in atlas.bodies:
        if body.a_au <= focus_lim:
            orbit = _orbit_xy(body, e0)
            ax.plot(orbit[:, 0], orbit[:, 1], color=body.color, lw=0.9, alpha=0.35)
    ax.scatter([0], [0], s=260, color="gold", edgecolor="k", linewidth=0.8, zorder=9)
    for body in atlas.bodies:
        if body.name not in relevant and body.name != "EARTH":
            continue
        x, y = pos[body.name]
        ax.scatter([x], [y], s=105 * body.radius_scale, color=body.color,
                   edgecolor="k", linewidth=0.8, zorder=10)
        ax.annotate(body.name.title(), (x, y), textcoords="offset points", xytext=(7, 6),
                    fontsize=9, weight="bold", zorder=11)
    for a, b in route_edges:
        c = by_edge.get((a, b))
        if c is None:
            continue
        arc = _propagated_lambert_arc(c, body_by_name)
        curve = arc[:, :2] if arc is not None else _corridor_curve(pos[a], pos[b], n=180)
        ax.plot(curve[:, 0], curve[:, 1], color="k", lw=4.5, alpha=0.72, zorder=6)
        ax.plot(curve[:, 0], curve[:, 1], color="tab:blue", lw=2.2, alpha=1.0, zorder=7)
        ax.scatter([curve[0, 0], curve[-1, 0]], [curve[0, 1], curve[-1, 1]],
                   s=[90, 110], color=["limegreen", "crimson"], edgecolor="k", zorder=12)
        ax.annotate(f"{a.title()} at departure\n{utc(c.et_dep)[:10]}",
                    curve[0], textcoords="offset points", xytext=(-78, -28),
                    fontsize=8.2, bbox={"boxstyle": "round,pad=0.28", "fc": "white",
                                         "ec": "0.55", "alpha": 0.92}, zorder=14)
        ax.annotate(f"{b.title()} at arrival\n{utc(c.et_arr)[:10]}",
                    curve[-1], textcoords="offset points", xytext=(8, 9),
                    fontsize=8.2, bbox={"boxstyle": "round,pad=0.28", "fc": "white",
                                         "ec": "0.55", "alpha": 0.92}, zorder=14)
        k0, k1 = int(0.60 * len(curve)), int(0.67 * len(curve))
        ax.annotate("", xy=curve[k1], xytext=curve[k0],
                    arrowprops={"arrowstyle": "-|>", "color": "tab:blue", "lw": 2.1,
                                "mutation_scale": 20}, zorder=13)
        mid = curve[int(0.50 * len(curve))]
        ax.annotate(f"depart {utc(c.et_dep)[:10]}\narrive {utc(c.et_arr)[:10]}\n"
                    f"{c.total_dv_ms/1000:.2f} km/s, {c.tof_days:.0f} d",
                    mid, textcoords="offset points", xytext=(10, 10), fontsize=8,
                    bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.55", "alpha": 0.92})
    ax.set_aspect("equal")
    ax.set_xlim(-focus_lim, focus_lim); ax.set_ylim(-focus_lim, focus_lim)
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")
    ax.grid(True, alpha=0.18)
    ax.set_title(f"Optimal route flight-path map: {route_label}\nactual two-body Lambert arc over real DE ephemeris geometry")
    fig.tight_layout()
    route_path = out / "solar_transfer_optimal_route_map.png"
    fig.savefig(route_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13.5, 8.8))
    fig.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.13)
    order = [b.name for b in atlas.bodies]
    mat = np.full((len(order), len(order)), np.nan)
    idx = {name: i for i, name in enumerate(order)}
    by_pair = {(c.origin, c.target): c for c in atlas.corridors}
    for c in atlas.corridors:
        mat[idx[c.origin], idx[c.target]] = c.score
    finite = mat[np.isfinite(mat)]
    vmax = float(np.percentile(finite, 90)) if finite.size else 1.0
    hm = ax.imshow(np.clip(mat, 0, vmax), cmap="cividis")
    cb = fig.colorbar(hm, ax=ax)
    cb.set_label("ROUTE SCORE  -  lower is better",
                  color=NP.TEXT_SECONDARY, fontsize=8.5)
    cb.ax.tick_params(colors=NP.TEXT_SECONDARY)
    cb.outline.set_edgecolor(NP.GRID_LINE)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([s.title() for s in order], rotation=35, ha="right",
                          color=NP.TEXT_PRIMARY)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([s.title() for s in order], color=NP.TEXT_PRIMARY)
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            if i == j:
                ax.text(j, i, "·",
                          ha="center", va="center", fontsize=18,
                          color=NP.GRID_LINE)
                continue
            c = by_pair.get((a, b))
            if c is None:
                continue
            # Halo'd text for readability on any cell colour
            color = NP.TEXT_PRIMARY if c.score > 0.5 * vmax else NP.DEEP_SPACE_BG
            NP.halo_text(ax, j, i,
                            f"{c.score:.1f}\n{c.total_dv_ms/1000:.1f}",
                            ha="center", va="center",
                            fontsize=7.5, color=color,
                            halo_color=NP.PANEL_BG if c.score > 0.5 * vmax
                                        else NP.TEXT_PRIMARY,
                            halo_width=0.8)
    for a, b in route_edges:
        ax.add_patch(plt.Rectangle((idx[b] - 0.48, idx[a] - 0.48), 0.96, 0.96,
                                       fill=False, edgecolor=NP.ACCENT_GOLD,
                                       linewidth=2.6, zorder=5))
        ax.scatter([idx[b] + 0.34], [idx[a] - 0.34], marker="*", s=140,
                      facecolor=NP.ACCENT_GOLD,
                      edgecolor=NP.DEEP_SPACE_BG, linewidth=1.0, zorder=6)
    NP.style_axes(ax, title="",
                    xlabel="ARRIVAL BODY",
                    ylabel="DEPARTURE BODY",
                    grid=False)
    NP.mission_title(fig,
        title="Pairwise transfer score table",
        subtitle="Every directed corridor; cell shows score / dv (km/s); "
                  "gold star + box = selected route leg")
    NP.mission_footer(fig,
        mission_id="SOLAR_PAIRWISE_HEATMAP",
        fidelity=f"{len(order)} bodies x {len(order)} = "
                   f"{len(atlas.corridors)} corridors scored")
    heatmap_path = out / "solar_transfer_pairwise_heatmap.png"
    fig.savefig(heatmap_path, dpi=180, bbox_inches="tight",
                 facecolor=NP.DEEP_SPACE_BG)
    plt.close(fig)
    NP.reset_style()

    # The two SPATIAL views the operator asked for:
    #   (a) cost-field topology over the whole solar system
    #   (b) trail-atlas tube map between every gravity body
    cost_field_path = render_solar_cost_field(
        atlas, out / "solar_cost_field.png")
    trail_atlas_path = render_solar_trail_atlas(
        atlas, out / "solar_trail_atlas.png")

    payload = {
        **asdict(atlas),
        "corridors": [asdict(c) | {"utc_dep": utc(c.et_dep), "utc_arr": utc(c.et_arr)}
                      for c in atlas.corridors],
        "artifacts": {
            "corridor_atlas": str(network_path),
            "optimal_route_map": str(route_path),
            "pairwise_heatmap": str(heatmap_path),
            "cost_field": str(cost_field_path),
            "trail_atlas": str(trail_atlas_path),
        },
    }
    json_path = out / "solar_transfer_atlas.json"
    json_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "corridor_atlas": str(network_path),
        "optimal_route_map": str(route_path),
        "pairwise_heatmap": str(heatmap_path),
        "cost_field": str(cost_field_path),
        "trail_atlas": str(trail_atlas_path),
        "atlas_json": str(json_path),
    }
