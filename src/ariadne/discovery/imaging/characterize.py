"""The characterization spine: spot -> run every applicable analyzer -> verdict.

Given a candidate (a MOVING object or a VARYING/stationary source), this runs
the analyzers that apply and returns a single structured ObjectDossier: a
probability distribution over the object taxonomy, the derived physical
properties, an honest confidence, and -- crucially -- what extra data would
sharpen the answer. It is the unifying layer over the analyzers that already
exist (rate, distance, orbit class, size, morphology) plus a first-pass
variability classifier; new analyzers (color, light-curve, host association)
plug in here without changing callers.

Design principle: never assert a hard label from thin data. Report what each
analysis CONSTRAINS, with uncertainty, and say how to resolve what's left.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .orbit_geometry import single_snapshot_estimate, implied_absolute_magnitude

# Distance -> class bins (heliocentric AU), mirroring orbit_geometry.classify_by_distance
_DIST_BINS = [(-math.inf, 1.3, "NEO/inner"),
              (1.3, 2.0, "Mars-crosser/inner-belt"),
              (2.0, 3.3, "main-belt"),
              (3.3, 6.0, "outer-belt/Hilda/Trojan"),
              (6.0, 30.0, "Centaur"),
              (30.0, math.inf, "TNO/distant")]


@dataclass
class ObjectDossier:
    kind: str                     # 'mover' | 'variable'
    headline: str                 # one-line plain-language verdict
    type_probabilities: dict      # {class: prob}, sums to ~1
    properties: dict              # distance_au, helio_r_au, H, size_km, rate, ...
    confidence: str               # 'low' | 'moderate' | 'high'
    flags: list = field(default_factory=list)
    known_id: str | None = None
    disambiguate: list = field(default_factory=list)


def size_km_from_H(H: float, albedo: float = 0.10) -> float:
    """Diameter (km) from absolute magnitude H and assumed albedo:
    D = 1329 / sqrt(p_V) * 10^(-H/5)."""
    if not (H == H):
        return float("nan")
    return 1329.0 / math.sqrt(albedo) * 10 ** (-0.2 * H)


def _mc_H_size(v_mag: float, r_helio: float, sigma: float, albedo: float,
               n: int = 240) -> dict:
    """Monte-Carlo-propagate the heliocentric-distance uncertainty into
    calibrated H and size confidence intervals (16-84th percentile). A point
    size estimate hides that distance error dominates it; this reports the
    honest range. Deterministic (fixed seed)."""
    if not (r_helio == r_helio) or r_helio <= 1.0 or sigma <= 0:
        return {}
    rng = np.random.default_rng(0)
    rs = rng.normal(r_helio, sigma, n)
    rs = rs[rs > 1.0]
    if len(rs) < 10:
        return {}
    Hs = np.array([implied_absolute_magnitude(v_mag, r, r - 1.0) for r in rs])
    Hs = Hs[np.isfinite(Hs)]
    if len(Hs) < 10:
        return {}
    sizes = np.array([size_km_from_H(h, albedo) for h in Hs])
    return {"H_lo": float(np.percentile(Hs, 16)), "H_hi": float(np.percentile(Hs, 84)),
            "size_lo_km": float(np.percentile(sizes, 16)),
            "size_hi_km": float(np.percentile(sizes, 84))}


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _class_probs_from_distance(r_helio: float, sigma: float) -> dict:
    """P(orbit class) by integrating a Gaussian belief in heliocentric distance
    over the class bins -- honest soft classification from a single snapshot."""
    if not (r_helio == r_helio) or sigma <= 0:
        return {}
    out = {}
    for lo, hi, name in _DIST_BINS:
        plo = _phi((lo - r_helio) / sigma) if lo > -math.inf else 0.0
        phi = _phi((hi - r_helio) / sigma) if hi < math.inf else 1.0
        out[name] = max(0.0, phi - plo)
    s = sum(out.values()) or 1.0
    return {k: v / s for k, v in out.items() if v / s > 0.005}


def classify_orbit(a_au: float, e: float, i_deg: float | None = None) -> str:
    """Definitive-ish class from fitted orbital elements (when an arc exists)."""
    if e >= 1.0:
        return "interstellar/hyperbolic"
    q = a_au * (1.0 - e)               # perihelion
    if q < 1.3:
        return "NEO"
    if e > 0.3 and 2.0 < a_au < 6.0:
        return "comet-like/active-candidate"
    if a_au < 2.0:
        return "Mars-crosser/inner-belt"
    if a_au < 3.3:
        return "main-belt"
    if a_au < 6.0:
        return "outer-belt/Hilda/Trojan"
    if a_au < 30.0:
        return "Centaur"
    return "TNO/distant"


def characterize_mover(rate_arcsec_hr: float, v_mag: float,
                       observer_helio_km, los_unit, *,
                       orbit_elements: tuple | None = None,
                       morphology: str | None = None,
                       n_nights: int = 1,
                       colors: dict | None = None,
                       known_id: str | None = None,
                       albedo: float = 0.10) -> ObjectDossier:
    """Characterize a moving object from its rate + brightness + geometry,
    using a fitted orbit when available (definitive) and the validated
    single-snapshot estimator otherwise (soft)."""
    est = single_snapshot_estimate(rate_arcsec_hr, v_mag, observer_helio_km, los_unit)
    H = est.implied_H
    size = size_km_from_H(H, albedo)
    flags, disamb = [], []

    if orbit_elements is not None:
        a, e = orbit_elements[0], orbit_elements[1]
        i = orbit_elements[2] if len(orbit_elements) > 2 else None
        cls = classify_orbit(a, e, i)
        probs = {cls: 0.9}
        confidence = "high" if n_nights >= 3 else "moderate"
        if e >= 1.0:
            flags.append("HYPERBOLIC -> interstellar-object candidate")
    else:
        probs = _class_probs_from_distance(est.helio_r_au,
                                           max(est.distance_sigma_au, 0.15 * (est.helio_r_au or 1)))
        if not (est.helio_r_au == est.helio_r_au):
            confidence = "low"          # distance undetermined (e.g. non-positive rate)
        else:
            confidence = "moderate" if est.near_opposition else "low"
        disamb.append("a second night yields a real orbit (a, e, i) -> definitive class")
        if not est.near_opposition:
            disamb.append("off opposition: full geometric inversion (not the rate->distance law)")

    if morphology == "extended":
        flags.append("EXTENDED -> possible coma/tail (comet / active asteroid)")
    if est.incomer_flag:
        flags.append("INCOMER candidate (bright + low-rate -> implausibly large if distant)")
    if known_id:
        flags.append(f"matches known object {known_id}")
    else:
        disamb.append("no catalog/SkyBoT match yet -> check at the exact epoch before claiming new")
    surface = None
    if colors is None:
        disamb.append("a second band (g/i) gives a color -> asteroid taxonomy (C/S/X) or red-TNO")
    else:
        from .color import classify_color
        cc = classify_color(colors, is_distant=(est.helio_r_au == est.helio_r_au
                                                 and est.helio_r_au > 6.0))
        surface = cc.get("taxonomy") or None
        if surface and cc.get("confidence") != "low":
            flags.append("surface type: " + ", ".join(
                f"{k} {v:.0%}" for k, v in sorted(surface.items(), key=lambda x: -x[1])[:2]))

    ci = _mc_H_size(v_mag, est.helio_r_au,
                    max(est.distance_sigma_au, 0.15 * (est.helio_r_au or 1)), albedo)
    best = max(probs, key=probs.get) if probs else (est.orbit_class or "unknown")
    if size == size and ci:
        sz = f"~{size:.1f} km [{ci['size_lo_km']:.1f}-{ci['size_hi_km']:.1f}]"
    elif size == size:
        sz = f"~{size:.1f} km"
    else:
        sz = "size unknown"
    activity = "active/cometary" if morphology == "extended" else "no coma (bare body)"
    ident = known_id if known_id else "NO known match (candidate)"
    surf = ""
    if surface:
        surf = " [" + max(surface, key=surface.get).split(" ")[0] + "]"
    headline = (f"{best}{surf}: ~{est.helio_r_au:.1f} AU, H~{H:.1f} ({sz}), "
                f"rate {rate_arcsec_hr:.1f}\"/hr, {activity} -> {ident}")
    props = {"rate_arcsec_hr": rate_arcsec_hr, "v_mag": v_mag,
             "helio_r_au": est.helio_r_au, "geo_distance_au": est.distance_au,
             "distance_sigma_au": est.distance_sigma_au, "implied_H": H,
             "size_km_est": size, "elongation_deg": est.elongation_deg,
             "near_opposition": est.near_opposition, "n_nights": n_nights,
             "surface_type": surface}
    props.update(ci)
    if est.note:
        flags.append(est.note)
    return ObjectDossier(kind="mover", headline=headline, type_probabilities=probs,
                         properties=props, confidence=confidence, flags=flags,
                         known_id=known_id, disambiguate=disamb)


def characterize_variable(times, mags, magerrs=None, g_r=None) -> ObjectDossier:
    """First-pass classification of a VARYING source from its light curve.
    Distinguishes a DIP (eclipse/transit/occultation), a symmetric BRIGHTENING
    (microlensing/nova), periodicity (variable star), or flat. A full typing
    (SN sub-class, period families) is Stage 3; this is the honest first cut."""
    t = np.asarray(times, float); m = np.asarray(mags, float)
    if len(m) < 4:
        return ObjectDossier(kind="variable", headline="too few points to classify",
                             type_probabilities={}, properties={}, confidence="low",
                             disambiguate=["need a denser light curve"])
    # With enough points, use the real Lomb-Scargle period analyzer (typed by
    # period + amplitude + folded shape). Falls through to the shape heuristic
    # below for sparse/aperiodic curves.
    if len(m) >= 8:
        from .light_curve import analyze_light_curve
        from .coherence_classifier import classify_variable, most_coherent
        lc = analyze_light_curve(t, m, magerrs)
        if lc.best_period is not None and lc.false_alarm_prob < 0.1:
            # type by the coherence (Equation-of-ONE) field over the feature
            # tau-space (period, R21 asymmetry, amplitude, color) -- validated on
            # real ZTF data (100% period recovery, ~89% type vs VSX labels).
            post = classify_variable(lc.best_period, lc.harmonic_r21, lc.amplitude_mag,
                                     g_r=g_r, eclipse=(lc.shape == "eclipse"))
            top = most_coherent(post) or "?"
            disamb = ["more cycles -> refine period + sub-type"]
            if g_r is None:
                disamb.insert(0, "a g-band color (g-r) breaks the EW/RR-Lyrae degeneracy")
            head = (f"periodic P={lc.best_period:.3g} d, amp {lc.amplitude_mag:.2f} mag, "
                    f"R21={lc.harmonic_r21:.2f} -> {top} (coherence field)")
            return ObjectDossier(kind="variable", headline=head,
                                 type_probabilities=post,
                                 properties={"period_days": lc.best_period,
                                             "amplitude_mag": lc.amplitude_mag,
                                             "false_alarm_prob": lc.false_alarm_prob,
                                             "harmonic_r21": lc.harmonic_r21,
                                             "g_r": g_r, "shape": lc.shape,
                                             "n_points": lc.n_points},
                                 confidence=lc.confidence, disambiguate=disamb)
    base = float(np.median(m))
    amp = float(np.percentile(m, 95) - np.percentile(m, 5))
    faint_excursion = float(np.max(m) - base)     # mag goes UP = fainter (a dip)
    bright_excursion = float(base - np.min(m))     # mag goes DOWN = brighter
    flags, disamb = [], []
    probs = {}
    if amp < 0.05:
        probs = {"non-variable/flat": 0.9}
        headline = f"flat to {amp:.2f} mag -- no significant variability"
        conf = "moderate"
    elif faint_excursion > 1.5 * bright_excursion and faint_excursion > 0.1:
        probs = {"eclipsing/transiting/occultation": 0.6, "irregular-dipper": 0.2,
                 "variable-star": 0.2}
        headline = (f"DIP of {faint_excursion:.2f} mag -> something crossed in front "
                    f"(eclipsing binary / transit / occultation)")
        conf = "moderate"
        disamb += ["denser/faster cadence -> dip shape (sharp=occultation, round=transit)",
                   "periodicity test -> eclipsing binary vs one-off occultation"]
    elif bright_excursion > 1.5 * faint_excursion and bright_excursion > 0.1:
        probs = {"microlensing": 0.4, "nova/CV-outburst": 0.3, "flare": 0.3}
        headline = (f"BRIGHTENING of {bright_excursion:.2f} mag then return -> "
                    f"microlensing / nova / flare")
        conf = "moderate"
        disamb += ["symmetry + achromatic test -> microlensing vs nova",
                   "color evolution -> nova/SN vs lensing (lensing is achromatic)"]
    else:
        probs = {"variable-star": 0.7, "irregular": 0.3}
        headline = f"symmetric variation {amp:.2f} mag -> variable star (periodicity TBD)"
        conf = "low"
        disamb.append("Lomb-Scargle period search -> RR Lyrae / Cepheid / eclipsing / Mira")
    props = {"baseline_mag": base, "amplitude_mag": amp,
             "dip_mag": faint_excursion, "brighten_mag": bright_excursion,
             "n_points": len(m), "timespan": float(t.max() - t.min())}
    return ObjectDossier(kind="variable", headline=headline, type_probabilities=probs,
                         properties=props, confidence=conf, flags=flags, disambiguate=disamb)


def characterize(candidate: dict) -> ObjectDossier:
    """Dispatch: candidate['kind'] == 'mover' or 'variable'. Other keys are the
    analyzer inputs (see characterize_mover / characterize_variable)."""
    kind = candidate.get("kind", "mover")
    if kind == "variable":
        return characterize_variable(candidate["times"], candidate["mags"],
                                     candidate.get("magerrs"), candidate.get("g_r"))
    args = {k: candidate[k] for k in ("rate_arcsec_hr", "v_mag", "observer_helio_km", "los_unit")}
    opt = {k: candidate[k] for k in ("orbit_elements", "morphology", "n_nights",
                                     "colors", "known_id", "albedo") if k in candidate}
    return characterize_mover(**args, **opt)
