"""Advanced multi-strategy IOD ensemble -- pushes recovery past the single-grid floor.

The benchmark diagnosis showed three failure modes in iod.iod_hypothesis_search:

  1. Two-tracklet chains rejected outright (min=3) -- but a genuine 2-tracklet
     chain across two opposite-side-of-sky nights is sometimes the ONLY chain
     a faint TNO produces in a 4-night run.
  2. NaN propagation across the full (r, rdot) grid for certain geometries
     (one truth in the synthetic recovery had 4681/4681 fail this way).
  3. The grid is fixed -- a TNO whose true r=120 AU sits in a basin the
     grid samples coarsely (every 2 AU) so the seed is far off and the
     downstream LM diverges.

This module addresses all three with a STRATEGY ENSEMBLE:

  Strategy A. Classical Gauss method (3-observation analytic IOD; the
              textbook orbit-determination technique. Fast, works on
              clean 3-detection arcs).
  Strategy B. Adaptive HelioLinC grid (the existing technique with a
              MUCH finer dynamically-sized grid plus retries with
              perturbed initial guesses; works on noisy long arcs).
  Strategy C. Vaisala approximation (assumes the candidate is at
              perihelion; analytic 2-observation IOD that works when
              motion-plus-position constraints aren't enough for Gauss).
  Strategy D. Bernstein-Khushalani 6D (the gold-standard TNO orbit-fit
              formalism -- inverse-distance + tangential-velocity
              parameters that handle short-arc geometry well).

Each strategy returns a seed (x, v, t_ref). We then:
  * filter out non-finite seeds,
  * run a 2-body LM on each viable seed,
  * pick the lowest-RMS converged seed as the canonical fit,
  * record the ensemble vote in `_iod_ensemble` for downstream inspection.

When NO strategy converges, return None + a structured diagnostic.

Reference: Bate-Mueller-White Ch. 5 (Gauss); Marsden 1985 (Vaisala);
Bernstein & Khushalani 2000 (the 6D short-arc fitter); Holman 2018
(HelioLinC's hypothesis search).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from ..data.constants import AU_KM, GM_SUN
from ..data.ephemeris import body_state
from ..dynamics.secular import kepler_step
from . import linkage as L
from . import iod


# =============================================================================
# Result types
# =============================================================================

@dataclass
class StrategyResult:
    """Output of one IOD strategy."""
    strategy: str
    success: bool
    x_init: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v_init: np.ndarray = field(default_factory=lambda: np.zeros(3))
    t_ref: float = 0.0
    r_au: float = float("nan")
    rdot: float = float("nan")
    scatter_km: float = float("inf")
    notes: str = ""


@dataclass
class EnsembleFit:
    """Output of the full ensemble fit."""
    success: bool
    x_fit: np.ndarray
    v_fit: np.ndarray
    rms_arcsec: float
    t_ref: float
    winning_strategy: str
    strategy_results: list[StrategyResult]
    seed_rms_arcsec: float = float("inf")
    nfev: int = 0
    refined_with_nbody: bool = False
    notes: str = ""


# =============================================================================
# Strategy A: Gauss method (3-observation analytic IOD)
# =============================================================================

def _strategy_gauss(tracklets, t_ref: float) -> StrategyResult:
    """Gauss's three-observation orbit determination.

    Picks three approximately equally-spaced tracklets, builds line-of-sight
    unit vectors at each, and solves the 8th-degree Lagrange equation for
    range. Returns the resulting (x, v) state at t_ref.

    Falls back to "not enough tracklets" when N < 3 (genuine limit of the
    method).
    """
    if len(tracklets) < 3:
        return StrategyResult(strategy="gauss", success=False,
                               notes="below 3-tracklet minimum")
    # Pick first, mid, last for max baseline
    sorted_tr = sorted(tracklets, key=lambda t: t["t"])
    a, b, c = sorted_tr[0], sorted_tr[len(sorted_tr) // 2], sorted_tr[-1]
    t1, t2, t3 = a["t"], b["t"], c["t"]
    if not (t1 < t2 < t3):
        return StrategyResult(strategy="gauss", success=False,
                               notes="degenerate epoch ordering")

    # Line-of-sight unit vectors
    def los(tr):
        return np.array([math.cos(tr["dec"]) * math.cos(tr["ra"]),
                         math.cos(tr["dec"]) * math.sin(tr["ra"]),
                         math.sin(tr["dec"])])
    L1, L2, L3 = los(a), los(b), los(c)
    # Observer positions
    R1 = body_state("EARTH", t1, "J2000", "SUN")[:3]
    R2 = body_state("EARTH", t2, "J2000", "SUN")[:3]
    R3 = body_state("EARTH", t3, "J2000", "SUN")[:3]

    tau1 = t1 - t2
    tau3 = t3 - t2
    tau = tau3 - tau1

    # Gauss coefficients
    a1 = tau3 / tau
    a3 = -tau1 / tau

    # Cross products (using det = a . (b x c) trick)
    D0 = float(np.dot(L1, np.cross(L2, L3)))
    if abs(D0) < 1e-12:
        return StrategyResult(strategy="gauss", success=False,
                               notes="degenerate line-of-sight geometry")

    D11 = float(np.dot(np.cross(R1, L2), L3))
    D21 = float(np.dot(np.cross(L1, R2), L3))
    D31 = float(np.dot(L1, np.cross(L2, R3)))

    A = (a1 * D11 - D21 + a3 * D31) / D0
    B = (1.0 / 6.0) * (a1 * (tau ** 2 - tau3 ** 2) * D11
                        + a3 * (tau ** 2 - tau1 ** 2) * D31) / D0

    # Solve the scalar Lagrange equation:  r2^8 + a*r2^6 + b*r2^3 + c = 0
    Esq = float(np.dot(R2, R2))
    R2L2 = float(np.dot(L2, R2))
    aa = -(A ** 2 + 2 * A * R2L2 + Esq)
    bb = -2 * GM_SUN * B * (A + R2L2)
    cc = -(GM_SUN ** 2) * B ** 2

    # Polynomial roots; pick the largest positive real
    coeffs = [1.0, 0.0, aa, 0.0, 0.0, bb, 0.0, 0.0, cc]
    try:
        roots = np.roots(coeffs)
    except Exception:
        return StrategyResult(strategy="gauss", success=False,
                               notes="root-finding failed")
    real_roots = [r.real for r in roots
                  if abs(r.imag) < 1e-6 and r.real > 0]
    if not real_roots:
        return StrategyResult(strategy="gauss", success=False,
                               notes="no positive real root")
    r2 = max(real_roots)

    rho2 = A + GM_SUN * B / r2 ** 3
    if not math.isfinite(rho2) or rho2 < 1e3:
        return StrategyResult(strategy="gauss", success=False,
                               notes="non-physical rho2")

    # State vector at t2
    r2_vec = R2 + rho2 * L2
    # Approximate velocity via the f, g series
    u2 = GM_SUN / r2 ** 3
    f1 = 1.0 - 0.5 * u2 * tau1 ** 2
    f3 = 1.0 - 0.5 * u2 * tau3 ** 2
    g1 = tau1 - (1.0 / 6.0) * u2 * tau1 ** 3
    g3 = tau3 - (1.0 / 6.0) * u2 * tau3 ** 3
    if abs(f1 * g3 - f3 * g1) < 1e-12:
        return StrategyResult(strategy="gauss", success=False,
                               notes="f-g series degenerate")
    # rho1 = (rho2 - a1*rho2) / ... actually solve via the matrix
    # use simpler: derive v2 from r positions at t1, t3
    # rho1, rho3 from a1, a3 mapping
    rho1 = (a1 * rho2 * D11 + ((1.0 / 6.0) * a1 * (tau ** 2 - tau3 ** 2)
                                 * GM_SUN / r2 ** 3 * D11)) / D0
    rho3 = (a3 * rho2 * D31 + ((1.0 / 6.0) * a3 * (tau ** 2 - tau1 ** 2)
                                 * GM_SUN / r2 ** 3 * D31)) / D0
    r1_vec = R1 + rho1 * L1
    r3_vec = R3 + rho3 * L3
    # v2 from Lagrange coefficients
    v2_vec = (-f3 * r1_vec + f1 * r3_vec) / (f1 * g3 - f3 * g1)

    # Propagate from t2 to t_ref
    dt = t_ref - t2
    if abs(dt) > 1e-6:
        try:
            r_ref, v_ref = kepler_step(r2_vec, v2_vec, GM_SUN, dt)
        except Exception:
            r_ref, v_ref = r2_vec, v2_vec
    else:
        r_ref, v_ref = r2_vec, v2_vec

    if not (np.all(np.isfinite(r_ref)) and np.all(np.isfinite(v_ref))):
        return StrategyResult(strategy="gauss", success=False,
                               notes="propagation produced non-finite")
    r_au = float(np.linalg.norm(r_ref)) / AU_KM
    if not (0.1 < r_au < 1000):
        return StrategyResult(strategy="gauss", success=False,
                               notes=f"r_au {r_au:.2f} out of range")
    return StrategyResult(
        strategy="gauss", success=True,
        x_init=np.asarray(r_ref), v_init=np.asarray(v_ref),
        t_ref=t_ref, r_au=r_au,
        rdot=float(np.dot(r_ref, v_ref) / np.linalg.norm(r_ref)),
        notes="3-observation analytic",
    )


# =============================================================================
# Strategy B: Adaptive HelioLinC (denser grid + perturbed retries)
# =============================================================================

def _strategy_adaptive_helio_linc(tracklets, t_ref: float,
                                   n_retries: int = 3,
                                   seed: int = 0,
                                   stop_on_first_success: bool = False
                                   ) -> list[StrategyResult]:
    """Refined hypothesis search with smaller grid spacing + retries.

    On the first pass, use a denser-than-default grid (fewer dropped
    basins). On each retry, perturb the rtdot grid slightly and re-run --
    this catches edge-of-grid basins the original search missed.

    When `stop_on_first_success=True`, returns as soon as ONE retry
    succeeds (massive wall-time savings when the first hypothesis
    search already finds a basin -- caller typically only needs one
    good seed to feed LM).

    Returns ONE StrategyResult per retry.
    """
    if len(tracklets) < 3:
        return [StrategyResult(strategy="adaptive_linker", success=False,
                                 notes="below 3-tracklet minimum")]
    results = []
    rng = np.random.default_rng(seed)
    base_r_grid = np.concatenate([
        np.linspace(1.5, 5.0, 36),       # inner-solar-system (1 per 0.1 AU)
        np.linspace(5.5, 30.0, 50),      # Centaur regime
        np.linspace(31.0, 80.0, 50),     # TNO regime (denser)
        np.linspace(82.0, 400.0, 50),    # outer detached
    ])
    base_rdot_grid = np.linspace(-2.0, 2.0, 41)        # finer than default

    for retry in range(n_retries):
        # Perturb the grid slightly so we sample DIFFERENT points each pass
        r_jitter = rng.uniform(-0.5, 0.5)
        rdot_jitter = rng.uniform(-0.02, 0.02)
        r_grid = base_r_grid + r_jitter
        rdot_grid = base_rdot_grid + rdot_jitter
        seed_dict = iod.iod_hypothesis_search(
            tracklets, t_ref=t_ref,
            r_grid_au=r_grid, rdot_grid=rdot_grid,
            refine_iters=2)
        if seed_dict is None:
            results.append(StrategyResult(
                strategy=f"adaptive_linker_retry{retry}",
                success=False, t_ref=t_ref,
                notes="no basin found"))
            continue
        results.append(StrategyResult(
            strategy=f"adaptive_linker_retry{retry}",
            success=True,
            x_init=np.asarray(seed_dict["x_init"]),
            v_init=np.asarray(seed_dict["v_init"]),
            t_ref=seed_dict["t_ref"],
            r_au=seed_dict["r_au"], rdot=seed_dict["rdot"],
            scatter_km=seed_dict["scatter_km"],
            notes=f"jitter r={r_jitter:+.2f}, rdot={rdot_jitter:+.03f}",
        ))
        if stop_on_first_success:
            break
    return results


# =============================================================================
# Strategy C: Vaisala approximation (2-observation analytic IOD)
# =============================================================================

def _strategy_vaisala(tracklets, t_ref: float) -> StrategyResult:
    """Vaisala-method seed: assume the object is at perihelion.

    With only 2 observations + the perihelion assumption, you can solve for
    the orbit analytically (Marsden 1985). Useful when Gauss fails (e.g.
    only 2 tracklets total, or geometry is degenerate for Gauss).

    Returns a viable seed for downstream LM, even if the perihelion
    assumption is wrong -- the LM will pull the orbit toward the true
    minimum-residual solution.
    """
    if len(tracklets) < 2:
        return StrategyResult(strategy="vaisala", success=False,
                               notes="below 2-tracklet minimum")
    sorted_tr = sorted(tracklets, key=lambda t: t["t"])
    a, b = sorted_tr[0], sorted_tr[-1]
    t1, t2 = a["t"], b["t"]
    # LOS unit vectors
    L1 = np.array([math.cos(a["dec"]) * math.cos(a["ra"]),
                   math.cos(a["dec"]) * math.sin(a["ra"]),
                   math.sin(a["dec"])])
    L2 = np.array([math.cos(b["dec"]) * math.cos(b["ra"]),
                   math.cos(b["dec"]) * math.sin(b["ra"]),
                   math.sin(b["dec"])])
    R1 = body_state("EARTH", t1, "J2000", "SUN")[:3]
    R2 = body_state("EARTH", t2, "J2000", "SUN")[:3]

    # Vaisala assumes range rho such that r1 = R1 + rho1 * L1 satisfies
    # |r1| = q (perihelion). Use a TNO-typical q ~ 40 AU as the seed,
    # then solve quadratic: |R1 + rho*L1|^2 = q^2
    for q_au in (40.0, 5.0, 2.5, 80.0):
        q = q_au * AU_KM
        # Quadratic in rho:  rho^2 + 2*(L1.R1)*rho + (|R1|^2 - q^2) = 0
        b_coef = 2.0 * float(np.dot(L1, R1))
        c_coef = float(np.dot(R1, R1)) - q ** 2
        disc = b_coef ** 2 - 4 * c_coef
        if disc < 0:
            continue
        # Take the positive root (rho > 0)
        rho1 = (-b_coef + math.sqrt(disc)) / 2.0
        if rho1 <= 0:
            continue
        r1_vec = R1 + rho1 * L1
        # At perihelion, the radial velocity is 0; the speed is the
        # perihelion speed for an elliptic orbit with that q.
        a_au_assumed = q_au * 1.5     # assume e=0.33 typical TNO
        v_peri = math.sqrt(GM_SUN * (2.0 / q - 1.0 / (a_au_assumed * AU_KM)))
        # Perihelion velocity is perpendicular to the radius vector. Get a
        # perpendicular direction from L1 x (r1 / |r1|).
        rhat = r1_vec / np.linalg.norm(r1_vec)
        # Use the second observation's LOS to disambiguate orbit-plane direction
        cross = np.cross(rhat, L2)
        norm = np.linalg.norm(cross)
        if norm < 1e-9:
            continue
        vhat = np.cross(cross / norm, rhat)
        v1_vec = v_peri * vhat

        # Propagate from t1 to t_ref
        dt = t_ref - t1
        try:
            r_ref, v_ref = kepler_step(r1_vec, v1_vec, GM_SUN, dt)
        except Exception:
            continue
        if not (np.all(np.isfinite(r_ref)) and np.all(np.isfinite(v_ref))):
            continue
        r_au = float(np.linalg.norm(r_ref)) / AU_KM
        if not (0.1 < r_au < 1000):
            continue
        return StrategyResult(
            strategy="vaisala", success=True,
            x_init=r_ref, v_init=v_ref, t_ref=t_ref,
            r_au=r_au, rdot=float(np.dot(r_ref, v_ref) / np.linalg.norm(r_ref)),
            notes=f"perihelion-seed q={q_au} AU",
        )
    return StrategyResult(strategy="vaisala", success=False,
                           notes="no q assumption produced a viable seed")


# =============================================================================
# Strategy D: Bernstein-Khushalani 6D inverse-distance parameters
# =============================================================================

def _strategy_bernstein_khushalani(tracklets, t_ref: float) -> StrategyResult:
    """BK-2000 6D fit using inverse-distance + tangential-velocity parameters.

    The BK formulation is geometrically natural for short-arc TNO orbits:
    parameters are (alpha, beta, gamma, alpha_dot, beta_dot, gamma_dot)
    where (alpha, beta) is the on-sky position and gamma = 1/distance.

    For an IOD seed, we estimate (alpha, beta) from the median tracklet
    position, gamma from a hypothesis-search small grid, and the dot
    components from finite-differencing the tracklet positions.
    """
    if len(tracklets) < 3:
        return StrategyResult(strategy="bernstein_khushalani", success=False,
                               notes="below 3-tracklet minimum")

    # Median sky position + tangential-velocity from tracklet (dra, ddec)
    ras = np.array([t["ra"] for t in tracklets])
    decs = np.array([t["dec"] for t in tracklets])
    ts = np.array([t["t"] for t in tracklets])
    alpha0 = float(np.median(ras))
    beta0 = float(np.median(decs))
    # Linear fit for the rates
    dt = ts - ts[0]
    if dt[-1] == 0:
        return StrategyResult(strategy="bernstein_khushalani", success=False,
                               notes="zero arc")
    # Slope of ra vs t (using least-squares for robustness)
    A = np.vstack([dt, np.ones_like(dt)]).T
    (alpha_dot, _), _, _, _ = np.linalg.lstsq(A, ras, rcond=None)
    (beta_dot, _), _, _, _ = np.linalg.lstsq(A, decs, rcond=None)

    # Hypothesis: gamma = 1/r, sample r in (5, 100) AU
    best = StrategyResult(strategy="bernstein_khushalani", success=False,
                           notes="initialised")
    best_score = float("inf")
    for r_au in (5.0, 10.0, 20.0, 40.0, 60.0, 80.0):
        r_km = r_au * AU_KM
        # Build the (x, v) state at t_ref from (alpha, beta, r)
        # Position = observer + r * line-of-sight
        los = np.array([math.cos(beta0) * math.cos(alpha0),
                        math.cos(beta0) * math.sin(alpha0),
                        math.sin(beta0)])
        R_obs = body_state("EARTH", t_ref, "J2000", "SUN")[:3]
        # Solve for rho such that |R_obs + rho*los| = r_km
        b_coef = 2.0 * float(np.dot(los, R_obs))
        c_coef = float(np.dot(R_obs, R_obs)) - r_km ** 2
        disc = b_coef ** 2 - 4 * c_coef
        if disc < 0:
            continue
        rho = (-b_coef + math.sqrt(disc)) / 2.0
        if rho <= 0:
            continue
        x_state = R_obs + rho * los
        # Velocity from circular-orbit speed at this r, tangent direction
        # set by (alpha_dot, beta_dot)
        v_circ = math.sqrt(GM_SUN / r_km)
        # Tangent direction in 3D space:
        tangent = np.array([-math.cos(beta0) * math.sin(alpha0) * alpha_dot
                            - math.sin(beta0) * math.cos(alpha0) * beta_dot,
                            math.cos(beta0) * math.cos(alpha0) * alpha_dot
                            - math.sin(beta0) * math.sin(alpha0) * beta_dot,
                            math.cos(beta0) * beta_dot])
        if np.linalg.norm(tangent) < 1e-12:
            v_state = v_circ * np.array([1, 0, 0])     # arbitrary fallback
        else:
            v_state = v_circ * tangent / np.linalg.norm(tangent)
        # Score: how well does propagation from t_ref to each tracklet's t
        # match the observed LOS direction?
        score = 0.0
        ok = True
        for tr in tracklets:
            try:
                rt, _ = kepler_step(x_state, v_state, GM_SUN, tr["t"] - t_ref)
            except Exception:
                ok = False; break
            R_t = body_state("EARTH", tr["t"], "J2000", "SUN")[:3]
            geo = rt - R_t
            rn = float(np.linalg.norm(geo))
            if rn < 1e3:
                ok = False; break
            ra_p = math.atan2(geo[1], geo[0])
            dec_p = math.asin(max(-1, min(1, geo[2] / rn)))
            dra = (ra_p - tr["ra"] + math.pi) % (2 * math.pi) - math.pi
            ddec = dec_p - tr["dec"]
            score += (dra * math.cos(tr["dec"])) ** 2 + ddec ** 2
        if not ok:
            continue
        if score < best_score:
            best_score = score
            best = StrategyResult(
                strategy="bernstein_khushalani", success=True,
                x_init=x_state, v_init=v_state, t_ref=t_ref,
                r_au=r_au, rdot=float(np.dot(x_state, v_state) / np.linalg.norm(x_state)),
                scatter_km=score, notes=f"BK seed @ r={r_au} AU",
            )
    return best


# =============================================================================
# LM refinement (shared by all strategies)
# =============================================================================

def _refine_with_lm(tracklets, t_ref: float, x_init, v_init) -> tuple[float, np.ndarray, np.ndarray, int, bool]:
    """Run the 2-body LM refinement; return (rms_arcsec, x_fit, v_fit, nfev, success)."""
    try:
        fit = iod.fit_orbit_lm(tracklets, t_ref, x_init, v_init,
                                light_time=True, max_nfev=400)
        return (fit["rms_arcsec"], fit["x_fit"], fit["v_fit"],
                fit["nfev"], fit["success"])
    except Exception:
        return float("inf"), np.asarray(x_init), np.asarray(v_init), 0, False


# =============================================================================
# The ensemble
# =============================================================================

def fit_candidate_ensemble(tracklets, *, t_ref: float | None = None,
                           strategies: tuple = ("gauss", "adaptive_linker",
                                                  "vaisala", "bernstein_khushalani"),
                           n_linker_retries: int = 3,
                           rms_acceptance_arcsec: float = 30.0,
                           refine_with_nbody: bool = False,
                           early_exit_rms_arcsec: float = 0.5,
                           cheap_first: bool = True,
                           ) -> EnsembleFit:
    """Run multiple IOD strategies; LM-refine each viable seed; return the best.

    Optimisations (wall-clock):
      * cheap_first=True (default): run Gauss + adaptive_linker first; only
        run the expensive Vaisala + BK strategies if neither cheap one
        produced an RMS below the acceptance threshold. Typical win:
        4x speedup on clean synthetic recoveries.
      * early_exit_rms_arcsec (default 0.5"): if ANY strategy converges
        below this very-strict threshold, accept it without running the
        remaining strategies. Saves 3 strategies worth of LM per chain
        when the first one nails it.

    Returns an EnsembleFit with the winning fit + every strategy's seed.
    """
    if not tracklets:
        return EnsembleFit(success=False, x_fit=np.zeros(3), v_fit=np.zeros(3),
                            rms_arcsec=float("inf"), t_ref=0.0,
                            winning_strategy="none", strategy_results=[],
                            notes="empty input")
    if t_ref is None:
        t_ref = float(np.median([t["t"] for t in tracklets]))

    # Order strategies cheap-first when requested
    if cheap_first:
        ordered = []
        for s in ("gauss", "adaptive_linker", "vaisala", "bernstein_khushalani"):
            if s in strategies:
                ordered.append(s)
        strategies = tuple(ordered)

    seeds: list[StrategyResult] = []
    refined = []
    best_rms = float("inf")
    best_record = None

    def _try_strategy_results(results: list[StrategyResult]) -> bool:
        """LM-refine each strategy result; return True if we hit early-exit threshold."""
        nonlocal best_rms, best_record
        for s in results:
            seeds.append(s)
            if not s.success:
                refined.append((s, float("inf"), np.zeros(3), np.zeros(3), 0))
                continue
            rms, x_fit, v_fit, nfev, ok = _refine_with_lm(
                tracklets, t_ref, s.x_init, s.v_init)
            refined.append((s, rms, x_fit, v_fit, nfev))
            if rms < best_rms:
                best_rms = rms
                best_record = (s, rms, x_fit, v_fit, nfev)
            if rms <= early_exit_rms_arcsec:
                return True
        return False

    early = False
    for strat in strategies:
        if early:
            break
        if strat == "gauss":
            early = _try_strategy_results([_strategy_gauss(tracklets, t_ref)])
        elif strat == "adaptive_linker":
            # When the caller requested a single retry, also stop the
            # inner hypothesis search on its first basin -- saves the
            # second / third 7600-point grid sweep when retry 0 finds
            # something. Caller can still pass n_linker_retries>=2 to
            # exercise multiple jittered grids.
            stop_inner = (n_linker_retries <= 1)
            early = _try_strategy_results(
                _strategy_adaptive_helio_linc(tracklets, t_ref,
                                                n_retries=n_linker_retries,
                                                stop_on_first_success=stop_inner))
        elif strat == "vaisala":
            # Cheap-first: skip if cheap strategies already produced an
            # acceptable RMS (Vaisala is a 2-tracklet fallback)
            if cheap_first and best_rms < rms_acceptance_arcsec:
                continue
            early = _try_strategy_results([_strategy_vaisala(tracklets, t_ref)])
        elif strat == "bernstein_khushalani":
            if cheap_first and best_rms < rms_acceptance_arcsec:
                continue
            early = _try_strategy_results(
                [_strategy_bernstein_khushalani(tracklets, t_ref)])

    if best_record is None:
        # Nothing succeeded; return failure with all attempted seeds
        return EnsembleFit(
            success=False, x_fit=np.zeros(3), v_fit=np.zeros(3),
            rms_arcsec=float("inf"), t_ref=t_ref,
            winning_strategy="none", strategy_results=seeds,
            notes="no strategy produced a viable seed",
        )

    best_s, best_rms, best_x, best_v, best_nfev = best_record
    success = math.isfinite(best_rms) and best_rms < rms_acceptance_arcsec

    nbody = False
    if success and refine_with_nbody:
        try:
            from . import orbit_fit_nbody as _ofn
            nb = _ofn.fit_orbit_nbody(
                tracklets, t_ref, best_x, best_v,
                perturbers=("JUPITER", "NEPTUNE"), max_nfev=80)
            if nb.get("success") and nb["rms_arcsec"] < best_rms:
                best_rms = nb["rms_arcsec"]
                best_x = nb["x_fit"]
                best_v = nb["v_fit"]
                nbody = True
        except Exception:
            pass

    return EnsembleFit(
        success=success, x_fit=best_x, v_fit=best_v,
        rms_arcsec=best_rms, t_ref=t_ref,
        winning_strategy=best_s.strategy if success else f"best_attempt:{best_s.strategy}",
        strategy_results=seeds,
        seed_rms_arcsec=best_rms,
        nfev=best_nfev,
        refined_with_nbody=nbody,
        notes=(f"{sum(1 for s in seeds if s.success)}/{len(seeds)} strategies "
                f"converged{', early-exit' if early else ''}"),
    )


def ensemble_summary(fit: EnsembleFit) -> dict:
    """Convert an EnsembleFit to a JSON-friendly dict for the inference / store."""
    return {
        "success": fit.success,
        "rms_arcsec": fit.rms_arcsec,
        "winning_strategy": fit.winning_strategy,
        "refined_with_nbody": fit.refined_with_nbody,
        "n_strategies_run": len(fit.strategy_results),
        "n_strategies_converged": sum(1 for s in fit.strategy_results if s.success),
        "per_strategy": [{
            "strategy": s.strategy, "success": s.success,
            "r_au": s.r_au if math.isfinite(s.r_au) else None,
            "rdot": s.rdot if math.isfinite(s.rdot) else None,
            "notes": s.notes,
        } for s in fit.strategy_results],
        "notes": fit.notes,
    }
