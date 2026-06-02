"""Orbital taxonomy: given a fitted orbit, what KIND of solar-system object is it?

The orbital elements alone partition the solar system into well-defined
dynamical classes. From (a, e, i) we derive perihelion q = a*(1-e) and aphelion
Q = a*(1+e), then walk a fixed decision tree of physically-motivated boundaries
(IAU MPC + Bowell + Granvik classifications).

Classes (mutually exclusive labels):

  ATIRA           a < 1.0 AU, Q < 0.983 AU             (entirely inside Earth's orbit)
  ATEN            a < 1.0 AU, Q > 0.983 AU             (Earth-crossing, semi-major < 1)
  APOLLO          a > 1.0 AU, q < 1.017 AU             (Earth-crossing, semi-major > 1)
  AMOR            1.017 AU < q < 1.3 AU                (Mars-crossing, near-Earth)
  HUNGARIA        1.78 < a < 2.0 AU, high inclination   (Hungaria group)
  IMB             2.0 < a < 2.5 AU                      (inner main belt)
  MBA             2.0 < a < 3.3 AU, not Hilda/Trojan   (main-belt asteroid)
  HILDA           3.7 < a < 4.2 AU, e<0.3, i<20         (3:2 Jupiter resonance)
  THULE           4.2 < a < 4.4 AU, low e/i             (4:3 Jupiter resonance)
  JTROJAN         5.0 < a < 5.4 AU                      (Jupiter Trojan, L4/L5)
  CENTAUR         5.4 < a < 30.1 AU, q > 5.2 AU         (between Jupiter & Neptune)
  CLASSICAL_KBO   42 < a < 48 AU, e<0.2, i<5            (cold classical Kuiper belt)
  HOT_CLASSICAL   42 < a < 48 AU, i>5                   (hot classical KBO)
  RESONANT_KBO    Neptune mean-motion resonance         (Plutinos 39.4 AU, twotinos 47.7, etc)
  SCATTERED_KBO   a > 30 AU, q > 30 AU, large e          (scattered disk)
  DETACHED        a > 50 AU, q > 38 AU                  (detached / inner Oort)
  SEDNOID         q > 38 AU, a > 150 AU                 (Sedna-like)
  COMET           any orbit + e > 1 (hyperbolic) OR active                (parabolic/hyperbolic)
  UNCLASSIFIED    no rule matches

Each classification returns a label + a confidence score in [0, 1]. Confidence
is 1.0 when the orbit is deep in the class's parameter box, and decays toward
0.5 near the boundary. Use this score to flag follow-up priority (e.g. don't
publish a "new Sedna" if it sits on the Sednoid/Scattered border).

Reference: Lykawka & Mukai 2007; Gladman et al. 2008 (Solar System Beyond
Neptune); IAU MPC dynamical class boundaries.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..data.constants import GM_SUN, AU_KM


class OrbitClass:
    """Mutually-exclusive dynamical class labels (strings, JSON-friendly)."""
    ATIRA = "ATIRA"
    ATEN = "ATEN"
    APOLLO = "APOLLO"
    AMOR = "AMOR"
    MARS_CROSSER = "MARS_CROSSER"
    HUNGARIA = "HUNGARIA"
    IMB = "IMB"
    MBA = "MBA"
    OMB = "OMB"
    HILDA = "HILDA"
    THULE = "THULE"
    JTROJAN = "JTROJAN"
    CENTAUR = "CENTAUR"
    CLASSICAL_KBO = "CLASSICAL_KBO"
    HOT_CLASSICAL = "HOT_CLASSICAL"
    RESONANT_KBO = "RESONANT_KBO"
    SCATTERED_KBO = "SCATTERED_KBO"
    DETACHED = "DETACHED"
    SEDNOID = "SEDNOID"
    COMET_HYPERBOLIC = "COMET_HYPERBOLIC"
    UNCLASSIFIED = "UNCLASSIFIED"


# Boundary constants (AU)
EARTH_Q = 0.983         # Earth perihelion (NEO crossing limit)
EARTH_a = 1.000         # Earth semi-major
EARTH_QQ = 1.017        # Earth aphelion (NEO outer crossing)
MARS_Q = 1.381
MARS_QQ = 1.666

JUPITER_a = 5.204
NEPTUNE_a = 30.07


@dataclass(frozen=True)
class OrbitTaxonomy:
    """Classification result for one orbit."""
    label: str
    confidence: float
    a_au: float
    e: float
    i_deg: float
    q_au: float
    Q_au: float
    tisserand_jupiter: float
    notes: str = ""


def elements_from_state(r_km: np.ndarray, v_km_s: np.ndarray,
                        mu: float = GM_SUN) -> tuple[float, float, float, float, float]:
    """Heliocentric (x, v) -> (a_au, e, i_deg, q_au, Q_au).

    Returns NaNs for hyperbolic orbits (e > 1). Caller should check.
    """
    r = float(np.linalg.norm(r_km))
    v2 = float(np.dot(v_km_s, v_km_s))
    energy = 0.5 * v2 - mu / r
    if energy >= 0:
        # hyperbolic / parabolic
        return float("nan"), float("inf"), float("nan"), float("nan"), float("nan")
    a_km = -mu / (2 * energy)
    h = np.cross(r_km, v_km_s)
    h_mag = float(np.linalg.norm(h))
    e_vec = np.cross(v_km_s, h) / mu - r_km / r
    e = float(np.linalg.norm(e_vec))
    inc_deg = math.degrees(math.acos(max(-1.0, min(1.0, h[2] / max(h_mag, 1e-9)))))
    a_au = a_km / AU_KM
    q_au = a_au * (1 - e)
    Q_au = a_au * (1 + e)
    return a_au, e, inc_deg, q_au, Q_au


def tisserand_parameter(a_au: float, e: float, i_deg: float,
                         a_p_au: float = JUPITER_a) -> float:
    """Tisserand parameter w.r.t. a perturber (Jupiter by default).

    T_J = a_p/a + 2 cos(i) sqrt(a/a_p * (1 - e^2))

    Classic discriminator: T_J < 3 -> comet-like; T_J > 3 -> asteroid-like.
    """
    if a_au <= 0 or not math.isfinite(a_au):
        return float("nan")
    cos_i = math.cos(math.radians(i_deg))
    return a_p_au / a_au + 2 * cos_i * math.sqrt(max(0.0, (a_au / a_p_au) * (1 - e ** 2)))


def _smoothstep(x: float, edge0: float, edge1: float) -> float:
    """Smooth interpolation 0 -> 1 across [edge0, edge1]. Saturates outside."""
    if edge1 == edge0:
        return 1.0 if x >= edge0 else 0.0
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3 - 2 * t)


def _box_confidence(x: float, lo: float, hi: float, soft_pct: float = 0.1) -> float:
    """Confidence that x sits in [lo, hi]. 1.0 deep inside, 0.5 at boundary."""
    if x < lo or x > hi:
        return 0.0
    soft = (hi - lo) * soft_pct
    return min(_smoothstep(x, lo, lo + soft),
               _smoothstep(hi, x, hi - soft) if hi > soft + lo else 1.0)


def classify_orbit(a_au: float, e: float, i_deg: float) -> OrbitTaxonomy:
    """Map (a, e, i) AU/deg to one OrbitTaxonomy label + confidence.

    Hyperbolic orbits (e >= 1) -> COMET_HYPERBOLIC.
    NaN / nonsensical orbits -> UNCLASSIFIED (confidence 0).
    """
    if not (math.isfinite(a_au) and math.isfinite(e) and math.isfinite(i_deg)):
        return OrbitTaxonomy(OrbitClass.UNCLASSIFIED, 0.0,
                             a_au, e, i_deg, float("nan"), float("nan"),
                             float("nan"), notes="non-finite element")
    if e >= 1.0:
        return OrbitTaxonomy(OrbitClass.COMET_HYPERBOLIC, 0.95,
                             a_au, e, i_deg, float("nan"), float("nan"),
                             float("nan"), notes="unbound (e>=1)")

    q = a_au * (1 - e)
    Q = a_au * (1 + e)
    T_J = tisserand_parameter(a_au, e, i_deg)

    # --- Sednoid / detached (do this BEFORE general scattered, to catch the rare cases) ---
    if q > 38.0 and a_au > 150.0:
        return OrbitTaxonomy(OrbitClass.SEDNOID,
                             0.9 if a_au > 250 else 0.7,
                             a_au, e, i_deg, q, Q, T_J,
                             notes="extreme detached -- Sedna-class")

    # --- NEO classes (inner -> outer) ---
    if a_au < EARTH_a and Q < EARTH_Q:
        return OrbitTaxonomy(OrbitClass.ATIRA, 0.95,
                             a_au, e, i_deg, q, Q, T_J,
                             notes="interior to Earth")
    if a_au < EARTH_a and Q >= EARTH_Q:
        return OrbitTaxonomy(OrbitClass.ATEN,
                             _box_confidence(a_au, 0.5, 1.0),
                             a_au, e, i_deg, q, Q, T_J, notes="NEA Aten")
    if a_au >= EARTH_a and q < EARTH_QQ:
        return OrbitTaxonomy(OrbitClass.APOLLO,
                             _box_confidence(q, 0.5, EARTH_QQ),
                             a_au, e, i_deg, q, Q, T_J, notes="NEA Apollo")
    if EARTH_QQ <= q < 1.3:
        return OrbitTaxonomy(OrbitClass.AMOR,
                             _box_confidence(q, 1.05, 1.3),
                             a_au, e, i_deg, q, Q, T_J, notes="NEA Amor")
    if MARS_Q <= q < MARS_QQ:
        return OrbitTaxonomy(OrbitClass.MARS_CROSSER, 0.8,
                             a_au, e, i_deg, q, Q, T_J, notes="Mars-crosser")
    if 1.78 <= a_au < 2.0 and e < 0.25 and i_deg >= 12.0:
        return OrbitTaxonomy(OrbitClass.HUNGARIA, 0.85,
                             a_au, e, i_deg, q, Q, T_J,
                             notes="Hungaria high-inclination inner-belt group")

    # --- Main belt ---
    if 2.0 <= a_au < 2.5:
        return OrbitTaxonomy(OrbitClass.IMB, _box_confidence(a_au, 2.0, 2.5),
                             a_au, e, i_deg, q, Q, T_J, notes="inner main belt")
    if 2.5 <= a_au < 3.3:
        return OrbitTaxonomy(OrbitClass.MBA, _box_confidence(a_au, 2.5, 3.3),
                             a_au, e, i_deg, q, Q, T_J, notes="main belt")
    if 3.3 <= a_au < 3.7:
        return OrbitTaxonomy(OrbitClass.OMB, _box_confidence(a_au, 3.3, 3.7),
                             a_au, e, i_deg, q, Q, T_J, notes="outer main belt (Cybele/Themis)")

    # --- Jupiter region ---
    if 3.7 <= a_au < 4.2 and e < 0.3 and i_deg < 20:
        return OrbitTaxonomy(OrbitClass.HILDA, 0.85,
                             a_au, e, i_deg, q, Q, T_J,
                             notes="3:2 Jupiter resonance (Hilda)")
    if 4.2 <= a_au < 4.4 and e < 0.2 and i_deg < 10:
        return OrbitTaxonomy(OrbitClass.THULE, 0.85,
                             a_au, e, i_deg, q, Q, T_J,
                             notes="4:3 Jupiter resonance (Thule group)")
    if 5.0 <= a_au < 5.4 and i_deg < 40:
        return OrbitTaxonomy(OrbitClass.JTROJAN, 0.9,
                             a_au, e, i_deg, q, Q, T_J,
                             notes="Jupiter Trojan (L4/L5 region)")

    # --- Centaur (between Jupiter and Neptune) ---
    if JUPITER_a < a_au < NEPTUNE_a and q > 5.2:
        return OrbitTaxonomy(OrbitClass.CENTAUR,
                             _box_confidence(a_au, JUPITER_a, NEPTUNE_a),
                             a_au, e, i_deg, q, Q, T_J, notes="Centaur")

    # --- Kuiper belt classes ---
    # Plutino: 2:3 Neptune resonance ~ a = 39.4 AU
    if 39.0 < a_au < 39.8:
        return OrbitTaxonomy(OrbitClass.RESONANT_KBO, 0.85,
                             a_au, e, i_deg, q, Q, T_J,
                             notes="Plutino (2:3 Neptune resonance)")
    # Twotino: 1:2 Neptune resonance ~ a = 47.7 AU
    if 47.4 < a_au < 48.1:
        return OrbitTaxonomy(OrbitClass.RESONANT_KBO, 0.80,
                             a_au, e, i_deg, q, Q, T_J,
                             notes="Twotino (1:2 Neptune resonance)")
    # Cold classical: 42-48 AU, low e, low i
    if 42.0 <= a_au <= 48.0 and e < 0.2 and i_deg < 5.0:
        return OrbitTaxonomy(OrbitClass.CLASSICAL_KBO, 0.9,
                             a_au, e, i_deg, q, Q, T_J,
                             notes="cold classical Kuiper belt")
    # Hot classical: same a, higher i
    if 42.0 <= a_au <= 48.0 and i_deg >= 5.0:
        return OrbitTaxonomy(OrbitClass.HOT_CLASSICAL, 0.85,
                             a_au, e, i_deg, q, Q, T_J,
                             notes="hot classical Kuiper belt")
    # Detached: large q, large a, modest e
    if 50.0 < a_au and q > 38.0 and e < 0.5:
        return OrbitTaxonomy(OrbitClass.DETACHED, 0.75,
                             a_au, e, i_deg, q, Q, T_J,
                             notes="detached object")
    # Scattered disk: large a + perihelion still near Neptune
    if NEPTUNE_a < a_au and 25.0 < q < 38.0:
        return OrbitTaxonomy(OrbitClass.SCATTERED_KBO,
                             0.7 + 0.2 * min(1.0, e),
                             a_au, e, i_deg, q, Q, T_J,
                             notes="scattered disk")

    # --- Comet by Tisserand parameter (asteroid-like vs comet-like dynamics) ---
    if math.isfinite(T_J) and T_J < 2.0:
        return OrbitTaxonomy(OrbitClass.COMET_HYPERBOLIC,
                             0.4 + 0.3 * (2.0 - T_J),
                             a_au, e, i_deg, q, Q, T_J,
                             notes="comet-like (T_J<2)")

    # Fall-through
    return OrbitTaxonomy(OrbitClass.UNCLASSIFIED, 0.0,
                         a_au, e, i_deg, q, Q, T_J,
                         notes="no rule matched")


def classify_state(r_km: np.ndarray, v_km_s: np.ndarray) -> OrbitTaxonomy:
    """Heliocentric (x, v) -> OrbitTaxonomy via element conversion."""
    a, e, i, q, Q = elements_from_state(r_km, v_km_s)
    return classify_orbit(a, e, i)
