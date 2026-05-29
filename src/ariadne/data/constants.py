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
R_MARS = 3389.5
R_VENUS = 6051.8
GM_VENUS = 324858.592


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
# Galilean moon mean radii (km), for flyby-altitude geometry.
R_IO, R_EUROPA, R_GANYMEDE, R_CALLISTO = 1821.6, 1560.8, 2631.2, 2410.3
JUPITER_IO = make_system("Jupiter-Io", GM_JUPITER, GM_IO, 421800.0, "Jupiter", "Io")
JUPITER_EUROPA = make_system("Jupiter-Europa", GM_JUPITER, GM_EUROPA, 671100.0, "Jupiter", "Europa")
JUPITER_GANYMEDE = make_system("Jupiter-Ganymede", GM_JUPITER, GM_GANYMEDE, 1070400.0, "Jupiter", "Ganymede")
JUPITER_CALLISTO = make_system("Jupiter-Callisto", GM_JUPITER, GM_CALLISTO, 1882700.0, "Jupiter", "Callisto")
GALILEAN = [JUPITER_IO, JUPITER_EUROPA, JUPITER_GANYMEDE, JUPITER_CALLISTO]

# Stage 16 generalization systems: a deliberately diverse mass-ratio spectrum, all real
# mission targets. GM (km^3/s^2), mean separation (km).
GM_SATURN = 3.7931187e7
GM_TITAN, GM_RHEA, GM_ENCELADUS = 8978.1382, 153.9426, 7.2027
GM_PHOBOS = 7.087e-4
# Binary asteroid 65803 Didymos / Dimorphos (DART/Hera target); GM from estimated masses.
GM_DIDYMOS, GM_DIMORPHOS = 3.51e-8, 2.45e-10

SATURN_TITAN = make_system("Saturn-Titan", GM_SATURN, GM_TITAN, 1221870.0, "Saturn", "Titan")
SATURN_RHEA = make_system("Saturn-Rhea", GM_SATURN, GM_RHEA, 527108.0, "Saturn", "Rhea")
SATURN_ENCELADUS = make_system("Saturn-Enceladus", GM_SATURN, GM_ENCELADUS, 237948.0, "Saturn", "Enceladus")
MARS_PHOBOS = make_system("Mars-Phobos", GM_MARS, GM_PHOBOS, 9376.0, "Mars", "Phobos")
SUN_MARS = make_system("Sun-Mars", GM_SUN, GM_MARS, 227939200.0, "Sun", "Mars")
DIDYMOS_DIMORPHOS = make_system("Didymos-Dimorphos", GM_DIDYMOS, GM_DIMORPHOS, 1.19, "Didymos", "Dimorphos")

# Ordered low-mu -> high-mu for the atlas (spans ~1.6e-8 to ~7e-3).
ATLAS_SYSTEMS = [MARS_PHOBOS, SATURN_ENCELADUS, SUN_MARS, SATURN_RHEA,
                 SATURN_TITAN, DIDYMOS_DIMORPHOS]

# ---------------------------------------------------------------------------
# Stage 26: a solar-system-wide registry of CR3BP systems (the whole-system scale-up).
# GM (km^3/s^2) and mean separation / semi-major axis (km), all DE440-class values.
# ---------------------------------------------------------------------------
GM_MERCURY = 22031.868
GM_URANUS = 5793939.0
GM_NEPTUNE = 6836529.0
GM_PLUTO = 869.6
GM_CHARON = 105.88
GM_TRITON = 1427.6
GM_IAPETUS = 120.50
GM_DEIMOS = 9.615e-5
GM_TITANIA = 235.4
GM_OBERON = 201.1

# Sun-planet systems (planet as secondary about the Sun).
SUN_MERCURY = make_system("Sun-Mercury", GM_SUN, GM_MERCURY, 57909050.0, "Sun", "Mercury")
SUN_VENUS = make_system("Sun-Venus", GM_SUN, GM_VENUS, 108209500.0, "Sun", "Venus")
SUN_JUPITER = make_system("Sun-Jupiter", GM_SUN, GM_JUPITER, 778570000.0, "Sun", "Jupiter")
SUN_SATURN = make_system("Sun-Saturn", GM_SUN, GM_SATURN, 1433530000.0, "Sun", "Saturn")
SUN_URANUS = make_system("Sun-Uranus", GM_SUN, GM_URANUS, 2872460000.0, "Sun", "Uranus")
SUN_NEPTUNE = make_system("Sun-Neptune", GM_SUN, GM_NEPTUNE, 4495060000.0, "Sun", "Neptune")

# Additional moon systems.
MARS_DEIMOS = make_system("Mars-Deimos", GM_MARS, GM_DEIMOS, 23463.2, "Mars", "Deimos")
SATURN_IAPETUS = make_system("Saturn-Iapetus", GM_SATURN, GM_IAPETUS, 3560820.0, "Saturn", "Iapetus")
URANUS_TITANIA = make_system("Uranus-Titania", GM_URANUS, GM_TITANIA, 435910.0, "Uranus", "Titania")
URANUS_OBERON = make_system("Uranus-Oberon", GM_URANUS, GM_OBERON, 583520.0, "Uranus", "Oberon")
NEPTUNE_TRITON = make_system("Neptune-Triton", GM_NEPTUNE, GM_TRITON, 354759.0, "Neptune", "Triton")
# Pluto-Charon: the most extreme mass ratio in the solar system (mu ~ 0.11, a true binary).
PLUTO_CHARON = make_system("Pluto-Charon", GM_PLUTO, GM_CHARON, 19595.8, "Pluto", "Charon")

#: The whole-solar-system registry: Sun-planet + major moon systems + binaries, ordered
#: low-mu -> high-mu. Spans ~1.6e-8 (Mars-Phobos) to ~0.108 (Pluto-Charon) -- 7 orders of magnitude.
SOLAR_SYSTEM = [
    MARS_PHOBOS, SATURN_ENCELADUS, SUN_MERCURY, SUN_VENUS, SUN_EARTH, SUN_MARS,
    JUPITER_EUROPA, JUPITER_IO, SUN_JUPITER, SATURN_RHEA, JUPITER_GANYMEDE,
    JUPITER_CALLISTO, SUN_SATURN, SUN_URANUS, SUN_NEPTUNE, URANUS_TITANIA, URANUS_OBERON,
    SATURN_IAPETUS, SATURN_TITAN, NEPTUNE_TRITON, EARTH_MOON, DIDYMOS_DIMORPHOS, PLUTO_CHARON,
]

#: A spanning subset for the (expensive) coherence-skeleton generalization test.
COHERENCE_TEST_SYSTEMS = [MARS_PHOBOS, SUN_EARTH, JUPITER_EUROPA, SATURN_TITAN,
                          NEPTUNE_TRITON, PLUTO_CHARON]
