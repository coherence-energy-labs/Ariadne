"""Real-vs-bogus classifier: filter spurious detections from real candidates.

Most "detections" in any survey pipeline are NOT real moving objects. The
common spurious classes:

  * cosmic-ray hits         single-pixel spikes flagged as point sources
  * satellite trails        long elongated streaks misread as a chain of points
  * detector edge artefacts ghosting + scattered light near focal-plane bounds
  * bad columns / bleeds    saturation trails from bright stars
  * stellar variability     same-pixel residual on a stochastically variable star
  * AGN / SN at rest        bright transients that don't move (PM ~ 0)
  * subtraction residuals   imperfect ref-frame alignment leaving dipoles
  * stacking ghosts         same source appearing in 2+ stacked frames at small
                            offsets (NOT a moving object -- a pipeline artefact)

This module is a transparent rule-based filter that catches the dominant
categories WITHOUT depending on a trained ML model. Each rule contributes a
0-1 'bogus_score'; high scores -> spurious. A candidate is REAL only when
the cumulative bogus_score is below `threshold`.

Use upstream of fit_filter to drop blatant false positives BEFORE the expensive
IOD search. Use downstream of the morphology classifier (which catches the
single-image junk) to catch tracklet-level spuriouses (no-motion, same-pixel-
stacking, etc.).

ML real/bogus models (Duev 2019; Mahabal 2019; Smith 2020) require labeled
training data we don't have. This rule-based filter is the bridge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RealBogusVerdict:
    """Per-candidate bogus rating.

    Fields:
      is_real:           True if the cumulative bogus_score < threshold.
      bogus_score:       0..1, total weighted score from all rules that fired.
      rules_fired:       list of (rule_name, contribution) pairs.
      threshold:         the cutoff used to decide is_real.
      severity:          low/medium/high/critical risk bucket.
      action:            keep/review/discard operator recommendation.
      explanation:       compact human-readable reason.
    """
    is_real: bool
    bogus_score: float
    rules_fired: list
    threshold: float
    severity: str = "low"
    action: str = "keep"
    explanation: str = ""


# --------------- single-detection rules (operate on Source/morphology) ----

def rule_cosmic_ray(morphology) -> float:
    """Return bogus contribution if the morphology says COSMIC_RAY."""
    if hasattr(morphology, 'label') and morphology.label == "COSMIC_RAY":
        return 0.95 * morphology.confidence
    return 0.0


def rule_edge_artefact(morphology) -> float:
    if hasattr(morphology, 'label') and morphology.label == "EDGE_ARTEFACT":
        return 0.95 * morphology.confidence
    return 0.0


def rule_extended(morphology) -> float:
    """Extended sources are mostly galaxies, not asteroids."""
    if hasattr(morphology, 'label') and morphology.label == "EXTENDED":
        return 0.85 * morphology.confidence
    return 0.0


# --------------- tracklet-level rules (operate on tracklet/chain dicts) -

def rule_zero_motion(tracklet: dict, *, min_rate_arcsec_hr: float = 0.05) -> float:
    """A tracklet with no on-sky motion is a stellar residual, not an asteroid.

    Args:
      tracklet:             dict with 'rate_arcsec_hr' or 'dra'/'ddec' (rad/s).
      min_rate_arcsec_hr:   below this, the source isn't moving.
    """
    rate = tracklet.get("rate_arcsec_hr")
    if rate is None:
        dra = tracklet.get("dra", 0.0)
        ddec = tracklet.get("ddec", 0.0)
        rate = math.hypot(dra, ddec) * 206265 * 3600.0
    if rate < min_rate_arcsec_hr:
        return 0.90
    return 0.0


def rule_implausible_rate(tracklet: dict,
                          *, max_rate_arcsec_hr: float = 3600.0) -> float:
    """A tracklet faster than 1 deg/hr is almost always a satellite trail.

    For real solar-system objects: NEOs max ~30 arcmin/hr; MBAs ~5 arcsec/hr;
    TNOs <1 arcsec/hr. 3600 arcsec/hr = 1 deg/hr -> satellite territory.
    """
    rate = tracklet.get("rate_arcsec_hr")
    if rate is None:
        dra = tracklet.get("dra", 0.0)
        ddec = tracklet.get("ddec", 0.0)
        rate = math.hypot(dra, ddec) * 206265 * 3600.0
    if rate > max_rate_arcsec_hr:
        return min(1.0, 0.5 + 0.5 * (rate - max_rate_arcsec_hr) / max_rate_arcsec_hr)
    return 0.0


def rule_same_pixel_stacking(tracklet: dict,
                             *, repeat_position_tol_arcsec: float = 0.5) -> float:
    """Repeated detections at IDENTICAL position across stacked frames are
    stacking artefacts, not real motion.

    Args:
      tracklet:               dict; must have 'members' (list of Alert) or
                              'positions' (list of (mjd, ra, dec)).
      repeat_position_tol_arcsec: tolerance for "same pixel".
    """
    members = tracklet.get("members") or tracklet.get("positions") or []
    if len(members) < 3:
        return 0.0
    # Compute max RA/Dec spread; if it's smaller than tolerance, suspicious.
    if hasattr(members[0], "ra"):
        ras = [m.ra for m in members]
        decs = [m.dec for m in members]
    elif isinstance(members[0], (list, tuple)) and len(members[0]) >= 3:
        ras = [m[1] for m in members]
        decs = [m[2] for m in members]
    else:
        return 0.0
    if not ras:
        return 0.0
    ra_span = (max(ras) - min(ras)) * math.cos(math.radians(decs[0])) * 3600
    dec_span = (max(decs) - min(decs)) * 3600
    if math.hypot(ra_span, dec_span) < repeat_position_tol_arcsec:
        return 0.85
    return 0.0


def rule_collinear_but_unequal_spacing(tracklet: dict) -> float:
    """Real moving objects move with consistent rate (over short arcs).

    If we have 4+ detections, the time-spacing-vs-position-spacing should be
    linear. Strong deviation -> spurious chain (e.g. greedy linker glued
    unrelated detections).
    """
    members = tracklet.get("members") or []
    if len(members) < 4:
        return 0.0
    if not hasattr(members[0], "mjd"):
        return 0.0
    members = sorted(members, key=lambda m: m.mjd)
    mjds = [m.mjd for m in members]
    ras = [m.ra for m in members]
    decs = [m.dec for m in members]
    # Predicted positions assuming constant rate
    dt_total = mjds[-1] - mjds[0]
    if dt_total <= 0:
        return 0.0
    dra_rate = (ras[-1] - ras[0]) / dt_total
    ddec_rate = (decs[-1] - decs[0]) / dt_total
    residuals_arcsec = []
    for i in range(1, len(members) - 1):
        ra_pred = ras[0] + dra_rate * (mjds[i] - mjds[0])
        dec_pred = decs[0] + ddec_rate * (mjds[i] - mjds[0])
        dra = (ras[i] - ra_pred) * math.cos(math.radians(decs[i]))
        ddec = decs[i] - dec_pred
        residuals_arcsec.append(math.hypot(dra, ddec) * 3600.0)
    if not residuals_arcsec:
        return 0.0
    max_res = max(residuals_arcsec)
    # 30 arcsec misfit for a short arc is way beyond noise -> bogus chain
    if max_res > 30.0:
        return min(0.9, max_res / 120.0)
    return 0.0


def rule_high_rms_fit(tracklet: dict, *, max_rms_arcsec: float = 30.0) -> float:
    """A fitted orbit with huge RMS isn't a real orbit, even if the IOD converged."""
    rms = tracklet.get("rms_arcsec")
    if rms is None or not math.isfinite(rms):
        return 0.0
    if rms > max_rms_arcsec:
        return min(0.9, rms / 100.0)
    return 0.0


def rule_short_arc_artefact_risk(tracklet: dict) -> float:
    """Short arcs are underdetermined; weak artifact-looking chains need review."""
    members = tracklet.get("members") or []
    n = len(members) if members else int(tracklet.get("n_detections", 0) or 0)
    arc = tracklet.get("arc_days")
    if arc is None and members and hasattr(members[0], "mjd"):
        mjds = [m.mjd for m in members]
        arc = max(mjds) - min(mjds)
    if n and n <= 2 and arc is not None and arc <= 0.1:
        return 0.35
    return 0.0


def risk_bucket(score: float) -> str:
    if score >= 0.9:
        return "critical"
    if score >= 0.6:
        return "high"
    if score >= 0.3:
        return "medium"
    return "low"


def recommended_action(score: float, rules_fired: list) -> str:
    names = {name for name, _ in rules_fired}
    if score >= 0.9:
        return "discard"
    if "short_arc_artefact_risk" in names and score >= 0.3:
        return "review"
    if score >= 0.6:
        return "discard"
    if score >= 0.3:
        return "review"
    return "keep"


# --------------- composite scorer ----------------------------------------

DEFAULT_RULES = (
    ("zero_motion", rule_zero_motion, 1.0),
    ("implausible_rate", rule_implausible_rate, 1.0),
    ("same_pixel_stacking", rule_same_pixel_stacking, 1.0),
    ("collinear_unequal_spacing", rule_collinear_but_unequal_spacing, 1.0),
    ("high_rms_fit", rule_high_rms_fit, 1.0),
    ("short_arc_artefact_risk", rule_short_arc_artefact_risk, 1.0),
)


def score_realbogus(tracklet: dict,
                    morphology=None,
                    *, rules=DEFAULT_RULES,
                    morphology_rules=(("cosmic_ray", rule_cosmic_ray, 1.0),
                                       ("edge_artefact", rule_edge_artefact, 1.0),
                                       ("extended", rule_extended, 0.8)),
                    threshold: float = 0.6) -> RealBogusVerdict:
    """Compute the bogus score for one tracklet (optionally with a morphology).

    Args:
      tracklet:          dict with tracklet metrics (rate, rms, members).
      morphology:        optional MorphologyVerdict from imaging classifier.
      rules:             tuple of (name, fn, weight) tracklet-level rules.
      morphology_rules:  tuple of (name, fn, weight) morphology-level rules.
      threshold:         is_real = (score < threshold). Default 0.6.

    Returns:
      RealBogusVerdict with cumulative score + which rules fired.
    """
    fired = []
    total = 0.0
    for name, fn, weight in rules:
        s = fn(tracklet) * weight
        if s > 0:
            fired.append((name, s))
            total += s
    if morphology is not None:
        for name, fn, weight in morphology_rules:
            s = fn(morphology) * weight
            if s > 0:
                fired.append((name, s))
                total += s
    total = min(1.0, total)             # cap so multiple rules don't oversaturate
    action = recommended_action(total, fired)
    severity = risk_bucket(total)
    explanation = (
        "no bogus rules fired" if not fired
        else ", ".join(f"{name}:{score:.2f}" for name, score in fired[:4])
    )
    return RealBogusVerdict(
        is_real=(total < threshold and action != "discard"),
        bogus_score=total,
        rules_fired=fired,
        threshold=threshold,
        severity=severity,
        action=action,
        explanation=explanation,
    )


def filter_real(tracklets: list[dict], threshold: float = 0.6,
                **kwargs) -> list[dict]:
    """Drop tracklets whose bogus score exceeds threshold.

    Each surviving tracklet is annotated with `_realbogus` = the verdict.
    """
    out = []
    for tr in tracklets:
        v = score_realbogus(tr, threshold=threshold, **kwargs)
        if v.is_real:
            tr2 = dict(tr)
            tr2["_realbogus"] = v
            out.append(tr2)
    return out
