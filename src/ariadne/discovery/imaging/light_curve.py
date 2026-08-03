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
    shape: str  # 'eclipse' | 'smooth' | 'flat' | 'aperiodic'
    var_type: dict = field(default_factory=dict)  # {type: prob}
    harmonic_r21: float = 0.0  # 2nd/1st Fourier amplitude (sawtooth-ness)
    rotation_period: float | None = None
    confidence: str = "low"
    notes: str = ""


def analyze_light_curve(
    times,
    mags,
    magerrs=None,
    *,
    is_mover: bool = False,
    min_period: float | None = None,
    max_period: float | None = None,
) -> LightCurveResult:
    t = np.asarray(times, float)
    m = np.asarray(mags, float)
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
        return LightCurveResult(
            n,
            span,
            amp,
            None,
            0.0,
            1.0,
            "aperiodic",
            {"uncertain (too few points)": 1.0},
            0.0,
            None,
            "low",
            "need >=6 points over a real timespan",
        )
    if amp < 0.04:
        return LightCurveResult(
            n,
            span,
            amp,
            None,
            0.0,
            1.0,
            "flat",
            {"non-variable/flat": 0.9},
            0.0,
            None,
            "moderate",
            "no significant variability",
        )

    # Lomb-Scargle period search
    try:
        from astropy.timeseries import LombScargle
    except ImportError:
        return LightCurveResult(
            n,
            span,
            amp,
            None,
            0.0,
            1.0,
            "aperiodic",
            {"periodicity untested (no astropy)": 1.0},
            0.0,
            None,
            "low",
        )
    # NB: for irregularly-sampled survey data, Lomb-Scargle validly recovers
    # periods far SHORTER than the median cadence (super-Nyquist), so the floor
    # is set by the science (0.05 d catches RR Lyrae / eclipsing binaries /
    # asteroid rotation), not by 2x the sampling.
    pmin = float(min_period if min_period else 0.05)
    pmax = float(max_period if max_period else 0.45 * span)
    if pmax <= pmin:
        pmax = pmin * 5
    ls = LombScargle(t, m, dy=dy) if dy is not None else LombScargle(t, m)
    freq, power = ls.autopower(
        minimum_frequency=1.0 / pmax, maximum_frequency=1.0 / pmin, samples_per_peak=8
    )
    if len(power) == 0:
        return LightCurveResult(
            n, span, amp, None, 0.0, 1.0, "aperiodic", {"aperiodic": 0.8}, 0.0, None, "low"
        )
    k = int(np.argmax(power))
    best_p = float(1.0 / freq[k])
    pk = float(power[k])

    # THE FALLBACK NEVER FIRED, because the failure it guards is not an exception.
    #
    # Baluev's analytic FAP overflows on high-power peaks: astropy emits
    # "overflow encountered in expm1/exp" and "invalid value in scalar power" as
    # RuntimeWarnings and RETURNS a value. Measured on the smooth-brightening case:
    # peak power 4.036, FAP = nan, no exception raised. So `except Exception` was
    # never reached and `fap` carried a non-probability into the decision.
    #
    # That mattered because the gate downstream is `false_alarm_prob < 0.1`, which
    # admits a PERIODIC hypothesis. `nan < 0.1` is False, so on one library version a
    # smooth single-peak brightening is correctly left aperiodic -- and on another,
    # where the same overflow lands on 0.0 instead of nan, the identical data is
    # classified "contact eclipsing binary (EW)". Ariadne's CI passed on Python 3.10
    # and failed on 3.11/3.12/3.13 for exactly this reason: the verdict depended on
    # which way an overflow rounded.
    #
    # "It raised" and "it returned a wrong number" are different failures, and only
    # the first was handled. So the result is now CHECKED, not merely caught: a false
    # alarm probability that is not finite and inside [0, 1] is not a probability.
    #
    # Fails CLOSED. When no usable value can be obtained, fap = 1.0 -- "certainly a
    # false alarm" -- so an unreliable periodogram can never be used as evidence FOR
    # periodicity. The opposite default would let a numerical failure manufacture a
    # detection, which is the failure mode worth being paranoid about here.
    def _usable(x: float) -> bool:
        return bool(np.isfinite(x)) and 0.0 <= x <= 1.0

    # THE OLD FALLBACK IS GONE, and removing it is the actual fix.
    #
    # It was `exp(-power * (n-1) / 2)`, a single-trial estimate that ignores how many
    # independent frequencies were searched. On the smooth-brightening case it returns
    # 7e-44 -- finite, inside [0, 1], and therefore "usable" -- which sails through the
    # `< 0.1` gate and manufactures a confident detection out of a numerical failure.
    # Measured: keeping it reproduced the exact CI misclassification locally.
    #
    # Swapping a nan for a confidently wrong number is not a fix. When Baluev cannot
    # produce a usable value there is no reliable significance for this peak, and the
    # honest encoding of that is fap = 1.0: no evidence of periodicity, so no periodic
    # hypothesis. A detection must be earned by a computation that worked.
    fap = 1.0
    try:
        with np.errstate(all="ignore"):
            cand = float(ls.false_alarm_probability(pk, method="baluev"))
        if _usable(cand):
            fap = cand
    except Exception:
        pass

    # THE PEAK MUST BE A PHYSICALLY VALID POWER. This is the guard that actually fixes
    # the bug, and it fires on the invariant rather than on a library's arithmetic.
    #
    # Sanitising the FAP above was necessary and not sufficient. Measured: for a smooth
    # single-peak brightening this periodogram returns
    #
    #     normalization = 'standard'   -> power is DEFINED on [0, 1]
    #     peak power    = 4.036        -> impossible
    #
    # A standard-normalised Lomb-Scargle power cannot exceed 1. The cause is visible in
    # the sampling: 50 points over 30 days is a Nyquist frequency near 0.82/d, and the
    # search runs to 1/0.05 = 20/d -- about 24x beyond it. The super-Nyquist floor is
    # deliberate and correct for IRREGULARLY sampled survey data (see the note above),
    # but on regularly-spaced input it resolves aliases so finely that the returned
    # power leaves its own domain.
    #
    # Baluev is then handed a number that is not a power. On this machine it overflows
    # to nan and the sanitiser catches it; on the CI images for Python 3.11/3.12/3.13
    # the same call returns a small FINITE value, which is a perfectly usable
    # probability and sails through to classify a transient as a contact eclipsing
    # binary. Same broken input, two different lies, and only one of them was visible
    # locally -- which is why chasing the FAP across library versions was chasing a
    # symptom.
    #
    # An out-of-domain power means the periodogram did not do arithmetic we can trust,
    # so there is no significance to report. Fails CLOSED, like the FAP guard: no
    # evidence of periodicity rather than confident evidence built on a broken number.
    if not (0.0 <= pk <= 1.0):
        fap = 1.0

    # fold + classify shape
    base = float(np.median(m))
    depth = float(np.max(m) - base)
    faint_frac = float(np.mean(m > base + 0.15 * amp))  # time spent fainter

    # Fourier harmonic ratio R21 = a2/a1 at the best period: asymmetric
    # sawtooths (RR Lyrae ab, Cepheids) have high R21; near-sinusoidal curves
    # (contact binaries EW, RRc) have low R21. The standard OGLE/Gaia
    # discriminator that separates EW from RR Lyrae (validated on real ZTF data).
    ph2 = 2 * math.pi * ((t / best_p) % 1.0)
    A = np.column_stack(
        [np.ones_like(ph2), np.cos(ph2), np.sin(ph2), np.cos(2 * ph2), np.sin(2 * ph2)]
    )
    try:
        coef, *_ = np.linalg.lstsq(A, m, rcond=None)
        a1 = math.hypot(coef[1], coef[2])
        a2 = math.hypot(coef[3], coef[4])
        R21 = float(a2 / a1) if a1 > 1e-6 else 0.0
    except Exception:
        R21 = 0.0

    if fap > 0.1:
        shape = "aperiodic"
    elif depth > 0.6 * amp and faint_frac < 0.35:
        shape = "eclipse"  # mostly flat with brief deep dips (Algol-type)
    else:
        shape = "smooth"

    var = _type(best_p, amp, fap, shape, R21)
    rot = None
    if is_mover and fap < 0.1 and best_p < 1.0:
        rot = 2.0 * best_p * 24.0  # hours; double-peaked asteroid light curve
    conf = "high" if fap < 1e-3 else ("moderate" if fap < 0.05 else "low")
    note = f"P={best_p:.3g} d, power={pk:.2f}, FAP={fap:.1e}, shape={shape}, R21={R21:.2f}"
    return LightCurveResult(
        n, span, amp, (best_p if fap < 0.1 else None), pk, fap, shape, var, R21, rot, conf, note
    )


def _type(period, amp, fap, shape, R21) -> dict:
    if fap > 0.1:
        return {"aperiodic/irregular": 0.6, "transient": 0.2, "low-S/N periodic": 0.2}
    if shape == "eclipse":
        return {"eclipsing binary": 0.75, "transiting/occulting": 0.15, "RR Lyrae (sharp)": 0.1}
    if period < 1.0:
        if R21 > 0.18:  # asymmetric sawtooth
            return {"RR Lyrae": 0.6, "short-period Cepheid": 0.2, "rotational variable": 0.2}
        return {
            "contact eclipsing binary (EW)": 0.5,
            "RRc / sinusoidal pulsator": 0.3,
            "rotational variable": 0.2,
        }
    if period < 70.0:
        if R21 > 0.12:  # pulsator sawtooth
            return {"Cepheid": 0.55, "RR Lyrae (long-period)": 0.2, "eclipsing binary": 0.25}
        return {"eclipsing binary": 0.5, "rotational/spotted": 0.3, "Cepheid": 0.2}
    if amp > 1.5:
        return {"Mira / long-period variable": 0.8, "semiregular": 0.2}
    return {"semiregular / slow irregular": 0.7, "periodic variable": 0.3}
