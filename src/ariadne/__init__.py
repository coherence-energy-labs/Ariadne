"""Ariadne -- Python-native cislunar mission design + TNO discovery toolkit.

The classical astrodynamics tools (GMAT, Monte, Copernicus, STK) are heavyweight and
GUI/script-driven; the popular Python libraries (poliastro, Tudat, Heyoka) are excellent
but stop short of CR3BP-aware cislunar mission design or end-to-end TNO discovery.
Ariadne fills that gap: a tutorial-driven, validated, open-source Python toolkit for

  - CR3BP periodic-orbit families (Lyapunov, halos, NRHO) via continuation;
  - Invariant-manifold tubes + heteroclinic / Poincare-section connections;
  - Transport-graph routing on the manifold network (planar + 3D + NRHO via y=0);
  - Gravity-assist (VEEGA / DSM) interplanetary chains;
  - Lambert porkchop + collocation + autodiff trajectory optimization;
  - TNO discovery from MPC astrometry (HelioLinC linker + IOD orbit fit);
  - Coherence-HJB sampled-graph Helmholtz value function for 6D OCP;
  - proof-carrying route certificates with CR3BP->BCR4BP->DE440 promotion.

See MASTER_PLAN.md for the project bible. See examples/ for runnable tutorials.

Quick start::

    import ariadne
    print(ariadne.system("EARTH_MOON"))         # CR3BP system constants
    family = ariadne.lyapunov_family("L1", n=20)
    nrho = ariadne.gateway_nrho()
    fit = ariadne.discover_tno("90377")         # Sedna
"""
__version__ = "1.0.0rc2"

# -- Re-exports: the user-facing API --
from .data.constants import (
    EARTH_MOON, SUN_EARTH, SUN_MARS,
    JUPITER_IO, JUPITER_EUROPA, JUPITER_GANYMEDE, JUPITER_CALLISTO,
    GM_SUN, GM_EARTH, GM_MOON, GM_JUPITER, GM_MARS, AU_KM,
)

# CR3BP systems registry
_SYSTEMS = {
    "EARTH_MOON": EARTH_MOON,
    "SUN_EARTH": SUN_EARTH,
    "SUN_MARS": SUN_MARS,
    "JUPITER_IO": JUPITER_IO,
    "JUPITER_EUROPA": JUPITER_EUROPA,
    "JUPITER_GANYMEDE": JUPITER_GANYMEDE,
    "JUPITER_CALLISTO": JUPITER_CALLISTO,
}


def system(name: str):
    """Return the named CR3BP `System` (e.g. "EARTH_MOON", "SUN_EARTH", "JUPITER_EUROPA")."""
    if name not in _SYSTEMS:
        raise KeyError(f"Unknown system {name!r}. Known: {list(_SYSTEMS)}")
    return _SYSTEMS[name]


def lyapunov_family(point: str = "L1", system_name: str = "EARTH_MOON",
                    n: int = 30, amplitude0: float = 1e-3, dx: float = 2e-3):
    """Build the planar Lyapunov orbit family around the named libration point.

    Parameters
    ----------
    point : "L1" or "L2"
    system_name : CR3BP system (default "EARTH_MOON")
    n : number of family members
    amplitude0, dx : starting amplitude and continuation step (nondimensional)
    """
    from .orbits.families import lyapunov_family as _lf
    return _lf(system(system_name).mu, point=point, amplitude0=amplitude0, dx=dx, n=n)


def halo_family(point: str = "L1", system_name: str = "EARTH_MOON", n: int = 20):
    """Build the 3D halo orbit family around the named libration point (z-continuation)."""
    from .orbits.halo import halo_family as _hf
    return _hf(system(system_name).mu, point=point, n=n,
               dz=4e-3, fam_n=40, lyap_amp0=2e-3, lyap_dx=4e-3)


def gateway_nrho():
    """Return the Gateway-class 9:2 L2 Near-Rectilinear Halo Orbit (Earth-Moon)."""
    from .orbits.nrho import nrho_family as _nf
    s = EARTH_MOON
    nrho, _ = _nf(s.mu, "L2", t_star_days=s.T_star / 86400.0, l_star=s.L_star,
                  target_period_d=6.56, ds=4e-3)
    return nrho


def discover_tno(designation: str, window_days: int = 720):
    """Fetch real MPC astrometry for a TNO and fit a heliocentric orbit.

    Returns a dict with x_fit (km), v_fit (km/s), rms_arcsec, iod hypothesis info; or None
    if fewer than 4 tracklets are available in the densest opposition window.
    """
    import numpy as np
    from .discovery import linkage as L, iod as IOD
    tracks, _ = L.tracklets_from_mpc(designation, window_days=window_days, min_per_night=2)
    if len(tracks) < 4:
        return None
    t_ref = float(np.median([t["t"] for t in tracks]))
    return IOD.fit_candidate(tracks, t_ref=t_ref)


def helmholtz_hjb(samples, source_idx, *, k: int = 12, gamma: float = 1.0,
                  D_coef: float = 1.0):
    """Sampled-graph Helmholtz value function -- the HJB curse-of-dimensionality bypass.

    Given N samples of the state space and an index of the goal sample, build a k-NN
    graph, solve the sparse Helmholtz PDE, and return the surprise-transformed value
    field W = -ln(V/V_max) usable as a pseudo-eikonal for greedy optimal-control policy.

    Returns dict with V (raw Helmholtz field), W (surprise transform), W_graph (adjacency),
    n_iter, cg_info, V_max, sigma_used.
    """
    from .optimize.coherence_hjb import hjb_solve
    return hjb_solve(samples, source_idx, k=k, gamma=gamma, D_coef=D_coef, normalized=True)


def certify_route(graph, path, system_obj=EARTH_MOON, **kwargs):
    """Build a proof-carrying certificate for a transport-graph route."""
    from .certification import certify_transport_route
    return certify_transport_route(graph, path, system_obj, **kwargs)


__all__ = [
    "__version__",
    # systems
    "EARTH_MOON", "SUN_EARTH", "SUN_MARS",
    "JUPITER_IO", "JUPITER_EUROPA", "JUPITER_GANYMEDE", "JUPITER_CALLISTO",
    "GM_SUN", "GM_EARTH", "GM_MOON", "GM_JUPITER", "GM_MARS", "AU_KM",
    # user-facing entry points
    "system", "lyapunov_family", "halo_family", "gateway_nrho",
    "discover_tno", "helmholtz_hjb", "certify_route",
]
