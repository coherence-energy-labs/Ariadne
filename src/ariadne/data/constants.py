"""Physical constants and CR3BP system definitions.

Gravitational parameters are DE440-consistent (km^3/s^2). Characteristic scales
(L*, T*, V*) are derived self-consistently from GM and the mean primary
separation, so the nondimensional CR3BP is internally exact. See MASTER_PLAN.md
§5.3 for the units/frames/time policy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Gravitational parameters GM (km^3/s^2), DE440-consistent
GM_SUN = 1.32712440018e11
GM_EARTH = 398600.435436
GM_MOON = 4902.800066
GM_MARS = 42828.375214
GM_JUPITER = 1.26686534e8

AU_KM = 149597870.7  # astronomical unit (km)

# Mean body radii (km)
R_EARTH = 6378.137
R_MOON = 1737.4
R_SUN = 695700.0


@dataclass(frozen=True)
class System:
    """A two-primary CR3BP system, fully nondimensionalized.

    Attributes
    ----------
    mu : mass parameter m2 / (m1 + m2), the only CR3BP system parameter.
    L_star, T_star, V_star : characteristic length (km), time (s), velocity (km/s).
    gm_total : GM1 + GM2 (km^3/s^2).
    """

    name: str
    mu: float
    L_star: float
    T_star: float
    V_star: float
    gm_total: float
    primary: str
    secondary: str


def make_system(name: str, gm1: float, gm2: float, l_star: float,
                primary: str, secondary: str) -> System:
    """Build a System from the two GMs and the characteristic length."""
    gm_total = gm1 + gm2
    mu = gm2 / gm_total
    t_star = math.sqrt(l_star ** 3 / gm_total)  # = 1 / mean motion
    v_star = l_star / t_star
    return System(name, mu, l_star, t_star, v_star, gm_total, primary, secondary)


# Earth-Moon: secondary = Moon. Mean separation 384,400 km.
EARTH_MOON = make_system("Earth-Moon", GM_EARTH, GM_MOON, 384400.0, "Earth", "Moon")

# Sun-Earth: canonical mu ~ 3.0035e-6 uses Earth-only GM (literature convention).
SUN_EARTH = make_system("Sun-Earth", GM_SUN, GM_EARTH, AU_KM, "Sun", "Earth")

# Sun-(Earth+Moon barycenter): the physically consistent Sun-Earth system.
SUN_EMB = make_system("Sun-EMB", GM_SUN, GM_EARTH + GM_MOON, AU_KM, "Sun", "EMB")

# Jovian (Galilean) moon systems: GM (km^3/s^2) and mean orbital radius (km).
GM_IO, GM_EUROPA, GM_GANYMEDE, GM_CALLISTO = 5959.916, 3202.739, 9887.834, 7179.289
JUPITER_IO = make_system("Jupiter-Io", GM_JUPITER, GM_IO, 421800.0, "Jupiter", "Io")
JUPITER_EUROPA = make_system("Jupiter-Europa", GM_JUPITER, GM_EUROPA, 671100.0, "Jupiter", "Europa")
JUPITER_GANYMEDE = make_system("Jupiter-Ganymede", GM_JUPITER, GM_GANYMEDE, 1070400.0, "Jupiter", "Ganymede")
JUPITER_CALLISTO = make_system("Jupiter-Callisto", GM_JUPITER, GM_CALLISTO, 1882700.0, "Jupiter", "Callisto")
GALILEAN = [JUPITER_IO, JUPITER_EUROPA, JUPITER_GANYMEDE, JUPITER_CALLISTO]
