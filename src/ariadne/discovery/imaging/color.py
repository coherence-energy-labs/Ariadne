"""Color -> taxonomy analyzer (Stage 2 of the characterization engine).

Multi-band photometry constrains the surface composition / type of a moving
object: asteroids split into spectral complexes by their optical colors, and
TNOs by how red they are. References: Ivezic et al. 2001 (SDSS asteroid
colors, the a* principal component), DeMeo & Carry 2013, Tegler et al. (TNO
color bimodality).

Inputs accept either band magnitudes ({'g':..,'r':..,'i':..,'z':..}) or
pre-computed colors ({'g-r':..,'r-i':..,'i-z':..}). Missing bands degrade
gracefully -> a wider (less certain) classification, never a hard wrong label.
"""
from __future__ import annotations

import math


def _colors_from(d: dict) -> dict:
    """Normalise an input dict to available color indices."""
    c = {}
    for k in ("g-r", "r-i", "i-z", "g-i"):
        if k in d and d[k] == d[k]:
            c[k] = float(d[k])
    bands = {b: float(d[b]) for b in ("g", "r", "i", "z") if b in d and d[b] == d[b]}
    for a, b, name in (("g", "r", "g-r"), ("r", "i", "r-i"),
                       ("i", "z", "i-z"), ("g", "i", "g-i")):
        if name not in c and a in bands and b in bands:
            c[name] = bands[a] - bands[b]
    return c


def asteroid_a_star(colors: dict):
    """Ivezic a* principal color (a*>0 reddish S-complex, a*<0 neutral
    C-complex). Needs g-r and r-i; returns None if unavailable."""
    c = _colors_from(colors)
    if "g-r" in c and "r-i" in c:
        return 0.89 * c["g-r"] + 0.45 * c["r-i"] - 0.57
    return None


def classify_color(colors: dict, *, is_distant: bool = False) -> dict:
    """Return a taxonomy probability dict + diagnostics from colors.

    is_distant: if the object is already known to be in the TNO/Centaur
    regime, interpret colors on the red-vs-neutral TNO axis instead of the
    asteroid complexes.
    """
    c = _colors_from(colors)
    out = {"colors": c, "taxonomy": {}, "a_star": None, "notes": "", "confidence": "low"}
    if not c:
        out["notes"] = "no usable colors"
        return out

    if is_distant:
        # TNO/Centaur red-vs-neutral bimodality (g-r is the workhorse)
        gr = c.get("g-r")
        ri = c.get("r-i")
        red_score = 0.0
        if gr is not None:
            red_score += (gr - 0.65) / 0.35     # ~0 at g-r 0.65, +1 by ~1.0
        if ri is not None:
            red_score += (ri - 0.35) / 0.3
        p_red = 1.0 / (1.0 + math.exp(-2.0 * red_score))
        out["taxonomy"] = {"red TNO (ultra-red / cold-classical)": round(p_red, 2),
                           "neutral/blue TNO (dynamically excited)": round(1 - p_red, 2)}
        out["confidence"] = "moderate" if (gr is not None and ri is not None) else "low"
        out["notes"] = "TNO color axis (red vs neutral)"
        return out

    a = asteroid_a_star(c)
    out["a_star"] = a
    tax = {}
    iz = c.get("i-z")
    if iz is not None and iz < -0.2:
        # 1-micron pyroxene band -> z bright -> i-z blue: basaltic V-type
        tax["V-type (basaltic / Vesta-family)"] = 0.6
    if a is not None:
        # logistic on a* around 0 -> S-complex (red) vs C-complex (neutral)
        p_s = 1.0 / (1.0 + math.exp(-8.0 * a))
        tax["S-complex (silicaceous / stony)"] = round(p_s * (1 - tax.get("V-type (basaltic / Vesta-family)", 0)), 2)
        tax["C-complex (carbonaceous / dark)"] = round((1 - p_s) * (1 - tax.get("V-type (basaltic / Vesta-family)", 0)), 2)
        out["confidence"] = "moderate"
        out["notes"] = f"a*={a:+.2f} ({'red/S' if a > 0 else 'neutral/C'})"
    else:
        out["notes"] = "need g-r AND r-i for a* (S vs C); have " + ",".join(c)
    # normalise
    s = sum(tax.values()) or 1.0
    out["taxonomy"] = {k: round(v / s, 2) for k, v in tax.items() if v > 0}
    return out
