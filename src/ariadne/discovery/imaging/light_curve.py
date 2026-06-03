"""Light-curve analyzer (Stage 2): periodicity + variable typing + rotation.

Given a time series of magnitudes, find a period (Lomb-Scargle), measure
amplitude + significance, classify the folded shape (eclipse-like dip vs
smooth/sawtooth), and map (period, amplitude, shape) onto variable-star
families. For a MOVING object the same machinery gives a rotation period
(asteroid light curve is typically double-peaked -> true period = 2x photometric).

Honest by construction: a high false-alarm probability or too-few points
yields an "aperiodic/uncertain" verdict, not a fabricated period.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LightCurveResult:
    n_points: int
    timespan: float
    amplitude_mag: float
    best_period: float | None
    power: float
    false_alarm_prob: float
    shape: str                       # 'eclipse' | 'smooth' | 'flat' | 'aperiodic'
    var_type: dict = field(default_factory=dict)   # {type: prob}
    harmonic_r21: float = 0.0        # 2nd/1st Fourier amplitude (sawtooth-ness)
    rotation_period: float | None = None
    confidence: str = "low"
    notes: str = ""


def analyze_light_curve(times, mags, magerrs=None, *, is_mover: bool = False,
                        min_period: float | None = None,
                        max_period: float | None = None) -> LightCurveResult:
    t = np.asarray(times, float); m = np.asarray(mags, float)
    good = np.isfinite(t) & np.isfinite(m)
    t, m = t[good], m[good]
    if magerrs is not None:
        dy = np.asarray(magerrs, float)[good]
        dy = np.where(np.isfinite(dy) & (dy > 0), dy, np.nan)
        if not np.isfinite(dy).all():
            dy = None
    else:
        dy = None
    n = len(m)
    span = float(t.max() - t.min()) if n else 0.0
    amp = float(np.percentile(m, 95) - np.percentile(m, 5)) if n else 0.0
    if n < 6 or span <= 0:
        return LightCurveResult(n, span, amp, None, 0.0, 1.0, "aperiodic",
                                {"uncertain (too few points)": 1.0}, 0.0, None, "low",
                                "need >=6 points over a real timespan")
    if amp < 0.04:
        return LightCurveResult(n, span, amp, None, 0.0, 1.0, "flat",
                                {"non-variable/flat": 0.9}, 0.0, None, "moderate",
                                "no significant variability")

    # Lomb-Scargle period search
    try:
        from astropy.timeseries import LombScargle
    except ImportError:
        return LightCurveResult(n, span, amp, None, 0.0, 1.0, "aperiodic",
                                {"periodicity untested (no astropy)": 1.0}, 0.0, None, "low")
    # NB: for irregularly-sampled survey data, Lomb-Scargle validly recovers
    # periods far SHORTER than the median cadence (super-Nyquist), so the floor
    # is set by the science (0.05 d catches RR Lyrae / eclipsing binaries /
    # asteroid rotation), not by 2x the sampling.
    pmin = float(min_period if min_period else 0.05)
    pmax = float(max_period if max_period else 0.45 * span)
    if pmax <= pmin:
        pmax = pmin * 5
    ls = LombScargle(t, m, dy=dy) if dy is not None else LombScargle(t, m)
    freq, power = ls.autopower(minimum_frequency=1.0 / pmax,
                               maximum_frequency=1.0 / pmin,
                               samples_per_peak=8)
    if len(power) == 0:
        return LightCurveResult(n, span, amp, None, 0.0, 1.0, "aperiodic",
                                {"aperiodic": 0.8}, 0.0, None, "low")
    k = int(np.argmax(power)); best_p = float(1.0 / freq[k]); pk = float(power[k])
    try:
        fap = float(ls.false_alarm_probability(pk, method="baluev"))
    except Exception:
        fap = float(np.exp(-pk * (n - 1) / 2.0))   # rough fallback

    # fold + classify shape
    base = float(np.median(m))
    depth = float(np.max(m) - base)
    faint_frac = float(np.mean(m > base + 0.15 * amp))   # time spent fainter

    # Fourier harmonic ratio R21 = a2/a1 at the best period: asymmetric
    # sawtooths (RR Lyrae ab, Cepheids) have high R21; near-sinusoidal curves
    # (contact binaries EW, RRc) have low R21. The standard OGLE/Gaia
    # discriminator that separates EW from RR Lyrae (validated on real ZTF data).
    ph2 = 2 * math.pi * ((t / best_p) % 1.0)
    A = np.column_stack([np.ones_like(ph2), np.cos(ph2), np.sin(ph2),
                         np.cos(2 * ph2), np.sin(2 * ph2)])
    try:
        coef, *_ = np.linalg.lstsq(A, m, rcond=None)
        a1 = math.hypot(coef[1], coef[2]); a2 = math.hypot(coef[3], coef[4])
        R21 = float(a2 / a1) if a1 > 1e-6 else 0.0
    except Exception:
        R21 = 0.0

    if fap > 0.1:
        shape = "aperiodic"
    elif depth > 0.6 * amp and faint_frac < 0.35:
        shape = "eclipse"        # mostly flat with brief deep dips (Algol-type)
    else:
        shape = "smooth"

    var = _type(best_p, amp, fap, shape, R21)
    rot = None
    if is_mover and fap < 0.1 and best_p < 1.0:
        rot = 2.0 * best_p * 24.0   # hours; double-peaked asteroid light curve
    conf = "high" if fap < 1e-3 else ("moderate" if fap < 0.05 else "low")
    note = f"P={best_p:.3g} d, power={pk:.2f}, FAP={fap:.1e}, shape={shape}, R21={R21:.2f}"
    return LightCurveResult(n, span, amp, (best_p if fap < 0.1 else None), pk, fap,
                            shape, var, R21, rot, conf, note)


def _type(period, amp, fap, shape, R21) -> dict:
    if fap > 0.1:
        return {"aperiodic/irregular": 0.6, "transient": 0.2, "low-S/N periodic": 0.2}
    if shape == "eclipse":
        return {"eclipsing binary": 0.75, "transiting/occulting": 0.15, "RR Lyrae (sharp)": 0.1}
    if period < 1.0:
        if R21 > 0.18:                      # asymmetric sawtooth
            return {"RR Lyrae": 0.6, "short-period Cepheid": 0.2, "rotational variable": 0.2}
        return {"contact eclipsing binary (EW)": 0.5, "RRc / sinusoidal pulsator": 0.3,
                "rotational variable": 0.2}
    if period < 70.0:
        if R21 > 0.12:                      # pulsator sawtooth
            return {"Cepheid": 0.55, "RR Lyrae (long-period)": 0.2, "eclipsing binary": 0.25}
        return {"eclipsing binary": 0.5, "rotational/spotted": 0.3, "Cepheid": 0.2}
    if amp > 1.5:
        return {"Mira / long-period variable": 0.8, "semiregular": 0.2}
    return {"semiregular / slow irregular": 0.7, "periodic variable": 0.3}
