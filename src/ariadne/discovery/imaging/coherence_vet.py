"""Equation-of-ONE candidate VETTING selector for linked tracks.

This is the CORRECT application of the coherence framework to discovery, mirroring
the One Link forge_shootouts (gap22/gap29): the EoO energy is a fast SELECTOR over
a small candidate set, NOT a brute-force search. The fast pairwise linker generates
candidate chains; this scores each by one unified energy and accepts the coherent
ones -- replacing the hand-tuned AND-threshold vet (rate-CV AND heading AND
residual), which (per gap22) IS the discretization of this continuous energy.

    E_total = C_dynamic * (E_quantum + E_coherence + E_alignment + E_dark)

mapped from transport-selection to track-vetting:
  E_quantum   -> motion-fit cost: RMS residual of the single linear-motion
                 hypothesis (the "work" needed to explain the detections as one
                 object). A real object is ONE line in (x,y,t); chance is not.
  E_coherence -> alpha*|grad tau|^2 + entropy: rate/heading instability across
                 segments (coherent motion has near-constant velocity) plus an
                 information term that rewards more nights / points (less likely
                 by chance).
  E_alignment -> (dC/dS)*A(d,tau,L): cross-night "trust" via the Gaussian
                 alignment kernel A=exp(-(gap^2+bright^2)/L) -- regular cadence and
                 consistent brightness => aligned => low energy; ragged gaps and
                 photometric jumps => low trust => high energy.
  E_dark      -> irreducible base cost + a penalty for too-short arcs.
  C_dynamic   -> e^{-lambda D} per-track scale normalization (longer, brighter,
                 more-nights arcs sit at a different scale).

A track is accepted iff E_total <= tau. Lowering tau trades recall for precision
along a real ROC -- something the hard AND-rules cannot do.
"""
from __future__ import annotations

import math

import numpy as np

from .coherence_field import alignment_energy


def _project(ra, dec):
    ra = np.asarray(ra, float); dec = np.asarray(dec, float)
    dec0 = float(np.median(dec)); ra0 = float(np.median(ra))
    cd = math.cos(math.radians(dec0))
    return (ra - ra0) * cd * 3600.0, (dec - dec0) * 3600.0


def track_features(ra, dec, mjd, mag=None):
    """Extract the discriminating feature vector of a candidate track, for the ONE
    coherence engine (coherence_posterior) to score. Uses CURVATURE-TOLERANT
    motion features (rate spread + heading scatter, not a raw linear residual that
    penalises genuinely-curved multi-night arcs) plus the cadence/arc coverage that
    separates real chains from chance. Returns {} for < 2 points."""
    ra = np.asarray(ra, float); dec = np.asarray(dec, float); mjd = np.asarray(mjd, float)
    order = np.argsort(mjd); ra, dec, mjd = ra[order], dec[order], mjd[order]
    n = len(ra)
    if n < 2:
        return {}
    x, y = _project(ra, dec)
    th = (mjd - float(np.median(mjd))) * 24.0
    nights = sorted(set(int(round(m)) for m in mjd))
    n_nights = len(nights)
    arc_hours = float((mjd[-1] - mjd[0]) * 24.0)
    # segment rates + headings (curvature-tolerant: a smooth orbit has low SPREAD)
    dt = np.diff(th); good = np.abs(dt) > 1e-6
    rate_cv = head_scatter = 0.0
    resid = 0.0
    if good.sum() >= 1:
        vx = np.diff(x)[good] / dt[good]; vy = np.diff(y)[good] / dt[good]
        sp = np.hypot(vx, vy); msp = float(np.mean(sp)) if sp.size else 0.0
        if sp.size > 1 and msp > 1e-6:
            rate_cv = float(np.std(sp) / msp)
        if vx.size > 1:
            head_scatter = float(np.std(np.unwrap(np.arctan2(vy, vx))))
    # secondary: linear residual (kept but down-weightable)
    if float(np.ptp(th)) > 1e-9:
        A = np.column_stack([np.ones(n), th])
        cx, *_ = np.linalg.lstsq(A, x, rcond=None); cy, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = float(np.sqrt(np.mean((x - A @ cx) ** 2 + (y - A @ cy) ** 2)))
    mag_scatter = 0.0
    if mag is not None and len(mag) == len(order):
        m = np.asarray(mag, float)[order]
        if np.all(np.isfinite(m)) and m.size > 1:
            mag_scatter = float(np.std(m))
    # coverage as ONE-SIDED deficits (0 = ample), so "more epochs / longer arc"
    # is never penalised -- the correct shape for the symmetric incoherence energy.
    epoch_deficit = max(0.0, 1.0 - n_nights / 4.0)
    arc_deficit = max(0.0, 1.0 - arc_hours / 48.0)
    return {"rate_cv": rate_cv, "heading": head_scatter, "resid": resid,
            "mag_scatter": mag_scatter, "epoch_deficit": epoch_deficit,
            "arc_deficit": arc_deficit}


# The single "coherent track" basin: every feature is 0 when ideal (no rate
# spread, no heading scatter, no residual, constant brightness, ample coverage),
# so one incoherence_energy call scores any track through the SAME engine as
# classification. Principled widths + weights (not overfit to a generator);
# curvature-tolerant (resid down-weighted, rate SPREAD leads).
REAL_TRACK_BASIN = {
    "mu": {"rate_cv": 0.0, "heading": 0.0, "resid": 0.0, "mag_scatter": 0.0,
           "epoch_deficit": 0.0, "arc_deficit": 0.0},
    "sig": {"rate_cv": 0.35, "heading": 0.5, "resid": 2.5, "mag_scatter": 0.35,
            "epoch_deficit": 0.35, "arc_deficit": 0.5},
}
VET_W = {"rate_cv": 1.6, "heading": 0.6, "resid": 0.2, "mag_scatter": 1.0,
         "epoch_deficit": 1.6, "arc_deficit": 1.2}


def track_energy(ra, dec, mjd, mag=None, *, basin=None, weights=None,
                 return_components=False):
    """Incoherence energy of a candidate track under the ONE coherence engine:
    extract track_features, then score them with coherence_field.incoherence_energy
    against the single coherent-track basin -- the SAME primitive that scores
    classification basins. Lower = more coherent (more likely a real object).
    `basin`/`weights` override the defaults (e.g. a calibrated fit)."""
    from .coherence_field import incoherence_energy
    f = track_features(ra, dec, mjd, mag)
    if not f:
        return (float("inf"), {}) if return_components else float("inf")
    E = incoherence_energy(f, basin or REAL_TRACK_BASIN, weights or VET_W)
    if return_components:
        return E, f
    return E


def vet_coherence(tracks, *, tau=2.0, key=None, **energy_kw):
    """Select the coherent tracks: accept iff track_energy <= tau. `tracks` is an
    iterable of (ra, dec, mjd[, mag]) tuples, or objects via `key(t)->(ra,dec,mjd,mag)`.
    Returns the accepted subset, most-coherent first."""
    scored = []
    for t in tracks:
        ra, dec, mjd, mag = key(t) if key else (t[0], t[1], t[2], t[3] if len(t) > 3 else None)
        scored.append((track_energy(ra, dec, mjd, mag, **energy_kw), t))
    scored.sort(key=lambda z: z[0])
    return [t for E, t in scored if E <= tau]
