"""Transport graph over the libration-point orbit network (MASTER_PLAN.md - Stage 14, G11).

The Interplanetary Transport Network, made discrete. Nodes are periodic (Lyapunov)
orbits at a grid of Jacobi energies for L1 and L2; a directed edge A->B is a patch on
the common Poincare section x = 1-mu (through the secondary): ride A's UNSTABLE tube
out to the section and join B's STABLE tube, which then coasts into B.

The patch is a TRUE section crossing, found exactly the way heteroclinic connections
are found: intersect the two tube cuts as curves in the (y, v_y) plane. At such a
crossing the position (x = 1-mu fixed, y matched) and v_y are identical, so the only
velocity that can differ is v_x -- and the patch Delta-v is |v_x^A - v_x^B|. At MATCHED
energy this is ~0 (the known ballistic heteroclinic); changing energy forces a real burn
because v^2 = 2*Omega - C differs. No position gap is ever tolerated: if the cuts do not
cross, there is no edge.

Minimum-Delta-v routing over this graph is therefore a single-source shortest-path (SSSP)
problem -- the Dijkstra/A* setting that motivated the project. `fragility` (a Floquet-
stretching proxy) carries the coherence/robustness cost for the coherence-weighted variant.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..orbits.families import lyapunov_orbit_at_jacobi
from ..connections.poincare import tube_section_cut
from ..dynamics.cr3bp import pseudo_potential

_Y_MOON_GUARD = 0.02     # drop crossings within ~7700 km of the Moon (section singularity)


@dataclass
class Node:
    key: str
    point: str
    jacobi: float
    orbit: object = None
    lambda_u: float = 1.0
    unstable_curves: list = field(default_factory=list)   # list of (k,6) branch polylines
    stable_curves: list = field(default_factory=list)


@dataclass
class Edge:
    src: str
    dst: str
    dv: float                      # patch Delta-v (nondimensional velocity) = |dv_x| at the crossing
    fragility: float = 0.0         # coherence cost (log Floquet stretching of the tube)
    meta: dict = field(default_factory=dict)


class TransportGraph:
    """Directed weighted graph of orbit-to-orbit manifold patches."""

    def __init__(self, mu: float, v_star: float = 1.0):
        self.mu = mu
        self.v_star = v_star               # km/s, to convert nondim dv -> physical
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, list[Edge]] = {}

    def add_node(self, node: Node) -> None:
        self.nodes[node.key] = node
        self.edges.setdefault(node.key, [])

    def add_edge(self, edge: Edge) -> None:
        self.edges.setdefault(edge.src, []).append(edge)

    def add_manual_node(self, key: str, jacobi: float = 0.0, point: str = "") -> None:
        """Lightweight node (no orbit) -- used for synthetic test graphs."""
        self.add_node(Node(key=key, point=point, jacobi=jacobi))

    def add_manual_edge(self, src: str, dst: str, dv: float, fragility: float = 0.0) -> None:
        self.add_edge(Edge(src=src, dst=dst, dv=float(dv), fragility=float(fragility)))

    def weight(self, edge: Edge, w_robust: float = 0.0) -> float:
        return edge.dv + w_robust * edge.fragility

    def dv_ms(self, nondim_dv: float) -> float:
        """Convert a nondimensional Delta-v to m/s using the system's V*."""
        return nondim_dv * self.v_star * 1000.0

    def reversed(self) -> "TransportGraph":
        g = TransportGraph(self.mu, self.v_star)
        for n in self.nodes.values():
            g.add_node(n)
        for elist in self.edges.values():
            for e in elist:
                g.add_edge(Edge(src=e.dst, dst=e.src, dv=e.dv, fragility=e.fragility))
        return g


def _branch_curves(mu, orbit, x_sec, stable, n_seeds, displacement, t_max):
    """Section crossings for each manifold branch, kept as separate ordered polylines.

    Returns (list of (k,6) state arrays, lambda_u). Crossings within _Y_MOON_GUARD of the
    secondary are dropped (they sit on the section's singularity and carry huge v_y).
    """
    curves = []
    lam = 1.0
    for branch in (+1, -1):
        cut = tube_section_cut(mu, orbit, x_sec, stable=stable, branch=branch,
                               n_seeds=n_seeds, displacement=displacement, t_max=t_max)
        lam = cut["lambda_u"]
        st = cut["states"]
        if len(st):
            st = st[np.abs(st[:, 1]) > _Y_MOON_GUARD]
        if len(st) >= 2:
            curves.append(st)
    return curves, float(lam)


def _seg_intersection(p1, p2, p3, p4):
    """Intersection of segments p1p2 and p3p4 -> (t, u, point) or None.

    t, u are the fractional positions along the first/second segment, so caller can
    linearly interpolate any other quantity (here v_x) at the crossing.
    """
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-15:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / den
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return t, u, (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return None


def _patch_edge(a: Node, b: Node, mu: float) -> Edge | None:
    """Minimum-Delta-v patch from a's unstable tube to b's stable tube on the section.

    Intersects every unstable-branch cut of a with every stable-branch cut of b in the
    (y, v_y) plane. At a crossing the position (x = x_sec, y) and v_y are shared, so each
    manifold's v_x follows EXACTLY from its own energy: v_x^2 = 2*Omega(x,y) - C - v_y^2
    (with C the source/destination orbit's Jacobi). The patch Delta-v is |v_x^src - v_x^dst|
    -- energy-consistent to machine precision (no curve interpolation of v_x). The edge is
    the minimum over all crossings.
    """
    c_src, c_dst = a.jacobi, b.jacobi
    best_dv, best_meta = math.inf, None
    for ca in a.unstable_curves:
        Ay = ca[:, [1, 4]]          # (y, v_y)
        for cb in b.stable_curves:
            By = cb[:, [1, 4]]
            for i in range(len(Ay) - 1):
                for j in range(len(By) - 1):
                    r = _seg_intersection(Ay[i], Ay[i + 1], By[j], By[j + 1])
                    if r is None:
                        continue
                    _, _, (yint, vyint) = r
                    xsec = float(ca[i, 0])     # = x_section (constant along the cut)
                    om = pseudo_potential([xsec, yint, 0.0, 0.0, 0.0, 0.0], mu)
                    arg_src = 2.0 * om - c_src - vyint * vyint
                    arg_dst = 2.0 * om - c_dst - vyint * vyint
                    if arg_src < 0.0 or arg_dst < 0.0:
                        continue               # crossing not physical at that energy
                    vx_src = math.sqrt(arg_src)     # require_vx_positive -> +root
                    vx_dst = math.sqrt(arg_dst)
                    dv = abs(vx_src - vx_dst)
                    if dv < best_dv:
                        best_dv = dv
                        pre = [xsec, float(yint), 0.0, vx_src, float(vyint), 0.0]
                        post = [xsec, float(yint), 0.0, vx_dst, float(vyint), 0.0]
                        best_meta = {"y": float(yint), "vy": float(vyint),
                                     "pre": pre, "post": post}
    if best_meta is None:
        return None
    frag = math.log(max(a.lambda_u, 1.0 + 1e-9))     # tube stretching = decoherence proxy
    return Edge(src=a.key, dst=b.key, dv=best_dv, fragility=frag, meta=best_meta)


def build_transport_graph(system, energies, points=("L1", "L2"), x_sec=None,
                          n_seeds: int = 80, displacement: float = 1e-4,
                          t_max: float = 12.0) -> TransportGraph:
    """Build the L1/L2 Lyapunov transport graph for a CR3BP system.

    `energies` is the Jacobi-constant grid (each must lie in both families' range).
    Returns a TransportGraph whose edges are exact section-crossing patches; an ordered
    pair gets an edge only if its tubes actually cross on the section.
    """
    mu = system.mu
    if x_sec is None:
        x_sec = 1.0 - mu
    g = TransportGraph(mu, system.V_star)

    for pt in points:
        for C in energies:
            orb = lyapunov_orbit_at_jacobi(mu, pt, C)
            key = f"{pt}@{C:.3f}"
            node = Node(key=key, point=pt, jacobi=float(orb.jacobi), orbit=orb)
            node.unstable_curves, node.lambda_u = _branch_curves(
                mu, orb, x_sec, False, n_seeds, displacement, t_max)
            node.stable_curves, _ = _branch_curves(
                mu, orb, x_sec, True, n_seeds, displacement, t_max)
            g.add_node(node)

    keys = list(g.nodes)
    for ka in keys:
        for kb in keys:
            if ka == kb:
                continue
            e = _patch_edge(g.nodes[ka], g.nodes[kb], mu)
            if e is not None:
                g.add_edge(e)
    return g
