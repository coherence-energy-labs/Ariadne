"""Coherence-field variable classifier — the Equation-of-ONE selector for typing.

Maps the variable-typing problem onto a coherence (tau) field: each class is a
basin defined by a prototype location + scale in feature space (log-period,
Fourier asymmetry R21, amplitude, and COLOR g-r). The verdict is the MOST
COHERENT class — the one that minimizes the incoherence energy

    E_c = (1/N) * sum_i  w_i * ((x_i - mu_ci) / sigma_ci)^2     over available dims

with coherence(c) = exp(-0.5 * E_c) and the posterior = normalized coherence.
This is exactly the gap22 pattern ("the hand-tuned rules ARE the discretization
of this continuous energy minimization"), now applied to classification.

KEY: color (g-r) is the axis that breaks the light-curve-degenerate classes.
RR Lyrae are hot/blue (g-r ~ 0.3); contact binaries (EW) are cool/red
(g-r ~ 0.7). With only the light curve those classes overlap; with color the
coherence basins separate. The classifier degrades gracefully when color is
absent (back to the genuinely-degenerate light-curve-only regime).
"""
from __future__ import annotations

import math

# Prototype basins in the feature tau-field. Periods are the RECOVERED
# (photometric) period: contact binaries fold at half the orbital period, so
# their basin sits at the short half-period. Centroids/scales from the
# astrophysics (OGLE/Gaia variable populations).
PROTOTYPES = {
    "RR Lyrae (RRab)": dict(
        mu=dict(logP=-0.26, R21=0.45, amp=0.65, g_r=0.28),
        sig=dict(logP=0.13, R21=0.14, amp=0.35, g_r=0.14)),
    "RRc / sinusoidal pulsator": dict(
        mu=dict(logP=-0.46, R21=0.10, amp=0.40, g_r=0.25),
        sig=dict(logP=0.12, R21=0.09, amp=0.22, g_r=0.14)),
    "contact eclipsing binary (EW)": dict(
        mu=dict(logP=-0.60, R21=0.18, amp=0.45, g_r=0.78),
        sig=dict(logP=0.22, R21=0.16, amp=0.30, g_r=0.22)),
    "Cepheid": dict(
        # g-r basin is broad: Cepheids span F-K AND suffer heavy Galactic-plane
        # reddening (observed g-r up to ~2), so color is weakly constraining here.
        mu=dict(logP=0.85, R21=0.45, amp=0.60, g_r=0.90),
        sig=dict(logP=0.42, R21=0.18, amp=0.40, g_r=0.70)),
    "Mira / long-period variable": dict(
        mu=dict(logP=2.30, R21=0.20, amp=3.00, g_r=1.30),
        sig=dict(logP=0.45, R21=0.25, amp=1.20, g_r=0.50)),
}
# per-axis weight. PERIOD is the strongest, extinction-independent discriminator
# (separates RR Lyrae <1d, EW <0.5d, Cepheid 1-50d, Mira >80d), so it leads;
# color breaks the within-short-period degeneracy (EW vs RR Lyrae) but is
# dust-contaminated near the Galactic plane, so it does not dominate.
_W = dict(logP=1.7, R21=1.0, amp=0.7, g_r=1.3)


def classify_variable(period: float, R21: float, amplitude: float,
                      g_r: float | None = None, eclipse: bool = False) -> dict:
    """Return {class: probability} by coherence-energy minimization over the
    feature tau-field. `eclipse` (flat + brief deep dips) routes to Algol-type
    eclipsing binaries, a shape that is NOT light-curve-degenerate."""
    if eclipse:
        return {"eclipsing binary (Algol-type)": 0.85,
                "transiting / occulting": 0.10, "other": 0.05}
    x = {"logP": math.log10(period) if period and period > 0 else 0.0,
         "R21": float(R21), "amp": float(amplitude)}
    if g_r is not None and g_r == g_r:
        x["g_r"] = float(g_r)
    # delegate to the shared coherence-field selector (the reusable primitive)
    from .coherence_field import coherence_posterior
    return {k: round(v, 3) for k, v in coherence_posterior(x, PROTOTYPES, _W).items()}


def most_coherent(post: dict):
    return max(post, key=post.get) if post else None


# Mover (orbit-class) basins in the coherence field, keyed on log10(heliocentric
# distance / AU) and eccentricity. The dynamical-class analog of the variable
# basins above -- same Equation-of-One selector, applied to moving objects.
# Basins over log10(mean distance), eccentricity, AND log10(perihelion q). The
# perihelion axis is the one that DEFINES the inner taxonomy (NEO q<1.3, Mars-
# crosser 1.3<=q<1.666); keying NEOs on mean distance alone misfiles eccentric
# ones (a~2, e~0.5 -> q<1.3 is a NEO but looks main-belt by distance). logq is
# only populated when eccentricity is known (the with-orbit regime); without it
# the classifier degrades gracefully to the distance+ecc basins. Real-data tuned
# on 60k MPC orbits (validate_mover_real.py).
MOVER_PROTOTYPES = {
    "NEO / Earth-approaching":     dict(mu=dict(logr=0.00, ecc=0.50, logq=-0.10), sig=dict(logr=0.30, ecc=0.30, logq=0.16)),
    "Mars-crosser / inner-belt":   dict(mu=dict(logr=0.28, ecc=0.20, logq=0.17),  sig=dict(logr=0.13, ecc=0.20, logq=0.06)),
    "main-belt":                   dict(mu=dict(logr=0.43, ecc=0.15, logq=0.37),  sig=dict(logr=0.13, ecc=0.15, logq=0.13)),
    "outer-belt / Hilda / Trojan": dict(mu=dict(logr=0.68, ecc=0.12, logq=0.56),  sig=dict(logr=0.13, ecc=0.15, logq=0.14)),
    "Centaur":                     dict(mu=dict(logr=1.15, ecc=0.40, logq=0.95),  sig=dict(logr=0.25, ecc=0.30, logq=0.35)),
    "TNO / distant":               dict(mu=dict(logr=1.60, ecc=0.15, logq=1.45),  sig=dict(logr=0.30, ecc=0.25, logq=0.40)),
}
_MOVER_W = dict(logr=1.2, ecc=0.5, logq=1.6)
# Calibration temperature: fit on real orbits to minimise ECE (0.41->0.09); only
# scales confidence (argmax/accuracy unchanged) so the reported probabilities are
# trustworthy. See scripts/calibrate_confidence.py.
_MOVER_T = 0.30

# bucket -> readable class name, to relabel calibrated (bucket-keyed) basins.
_BUCKET_TO_NAME = {
    "NEO": "NEO / Earth-approaching", "Mars-crosser": "Mars-crosser / inner-belt",
    "main-belt": "main-belt", "outer-belt": "outer-belt / Hilda / Trojan",
    "Centaur": "Centaur", "TNO": "TNO / distant"}
_MOVER_CALIB = None     # lazy cache: (basins, weights) or ({}, None)


def _mover_basins_weights():
    """Default: the principled hand-tuned basins (robust across BOTH the with-orbit
    and snapshot-distance-only regimes). OPT-IN (COH_MOVER_CALIBRATED=1): load the
    discriminatively-refined basins from data/mover_basins_calibrated.json -- those
    raise with-orbit balanced accuracy (NEO/Mars-crosser/outer-belt ~99%) but are
    tuned on the (a,e,logq) regime, so they are NOT the default to avoid regressing
    the snapshot-distance-only path."""
    global _MOVER_CALIB
    if _MOVER_CALIB is None:
        _MOVER_CALIB = (MOVER_PROTOTYPES, _MOVER_W)
        try:
            import os
            if os.environ.get("COH_MOVER_CALIBRATED", "0") == "1":
                import json
                from pathlib import Path
                p = Path(__file__).resolve().parents[4] / "data" / "mover_basins_calibrated.json"
                if p.exists():
                    d = json.loads(p.read_text())
                    basins = {_BUCKET_TO_NAME.get(k, k): v for k, v in d["basins"].items()}
                    _MOVER_CALIB = (basins, d.get("weights", _MOVER_W))
        except Exception:
            _MOVER_CALIB = (MOVER_PROTOTYPES, _MOVER_W)
    return _MOVER_CALIB


def classify_mover(helio_r_au: float, eccentricity: float | None = None,
                   a_star: float | None = None) -> dict:
    """Orbit-class posterior by coherence-energy minimisation over (log distance,
    eccentricity). Hyperbolic (e>=1) routes to interstellar. `a_star` (color)
    refines main-belt surface type (S vs C) but not the dynamical class.

    The coherence reformulation of the validated single-snapshot distance
    classifier (orbit_geometry, 79% class on 117 real asteroids); with a fitted
    eccentricity it also separates comets and interstellar objects."""
    if eccentricity is not None and eccentricity >= 1.0:
        return {"interstellar / hyperbolic": 0.9, "comet (active)": 0.1}
    if not (helio_r_au and helio_r_au > 0):
        return {}
    from .coherence_field import coherence_posterior
    x = {"logr": math.log10(helio_r_au)}
    if eccentricity is not None and eccentricity == eccentricity:
        x["ecc"] = float(eccentricity)
        # perihelion q = a(1-e); the defining axis for NEO / Mars-crosser. In the
        # with-orbit regime helio_r is the semi-major axis, so q is exact.
        q = helio_r_au * (1.0 - float(eccentricity))
        if q > 0:
            x["logq"] = math.log10(q)
    basins, weights = _mover_basins_weights()
    post = {k: round(v, 3) for k, v in
            coherence_posterior(x, basins, weights, temperature=_MOVER_T).items()}
    # cometary hint: eccentric + beyond the inner belt
    if eccentricity is not None and eccentricity > 0.5 and 2.0 < helio_r_au < 6.0:
        post.setdefault("comet-like / active candidate", 0.0)
        post["comet-like / active candidate"] = max(post["comet-like / active candidate"], 0.15)
    return post
