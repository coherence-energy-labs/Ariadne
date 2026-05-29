"""Tisserand-graph multi-moon tour design (MASTER_PLAN.md - Stage 20).

The Petit Grand Tour and real multi-moon missions (Galileo, JUICE, Europa Clipper) are
designed not with single-system L1<->L2 graphs but with the TISSERAND GRAPH: a spacecraft
on a Jupiter-centric orbit gets gravity assists at the moons, and a flyby conserves the
Tisserand parameter w.r.t. that moon (equivalently, |v_inf| at that moon is unchanged --
the flyby only ROTATES v_inf). So a tour pumps the Jupiter-centric orbit moon by moon for
free, and the only deterministic Delta-v is the v_inf MAGNITUDE mismatch between the
transfer orbits meeting at a shared moon (a flyby cannot change |v_inf|).

For a Jupiter-centric orbit (a, e) and a moon at radius a_m with circular speed v_m:
    T   = a_m/a + 2 sqrt( (a/a_m)(1-e^2) )          (Tisserand parameter, planar)
    v_inf = v_m sqrt(3 - T)                          (Tisserand <-> v_inf relation)
A flyby at altitude h (closest radius r_f = R_moon + h) can turn v_inf by at most
    delta_max = 2 arcsin( 1 / (1 + r_f v_inf^2 / GM_moon) ).

We compare a gravity-assist tour's deterministic Delta-v (sum of v_inf mismatches at the
intermediate moons) against the propulsive Hohmann baseline -- quantifying the assist saving.
HONEST: this is the ENERGY/Tisserand structure, not a phased trajectory; flyby phasing,
resonance timing, and plane changes are not modeled.
"""
from __future__ import annotations

import math

from ..data.constants import (GM_JUPITER, JUPITER_IO, JUPITER_EUROPA, JUPITER_GANYMEDE,
                              JUPITER_CALLISTO, R_IO, R_EUROPA, R_GANYMEDE, R_CALLISTO)

# Galilean moons inner->outer: (system, mean radius km, body radius km, GM)
GALILEAN_TOUR = [
    (JUPITER_IO, 421800.0, R_IO, JUPITER_IO.gm_total - GM_JUPITER),
    (JUPITER_EUROPA, 671100.0, R_EUROPA, JUPITER_EUROPA.gm_total - GM_JUPITER),
    (JUPITER_GANYMEDE, 1070400.0, R_GANYMEDE, JUPITER_GANYMEDE.gm_total - GM_JUPITER),
    (JUPITER_CALLISTO, 1882700.0, R_CALLISTO, JUPITER_CALLISTO.gm_total - GM_JUPITER),
]


def tisserand_parameter(a, e, a_moon):
    return a_moon / a + 2.0 * math.sqrt((a / a_moon) * (1.0 - e * e))


def vinf_at_moon(a, e, a_moon, gm_central=GM_JUPITER):
    """v_inf (km/s) of a Jupiter-centric orbit (a,e) relative to a moon at radius a_moon."""
    v_m = math.sqrt(gm_central / a_moon)
    T = tisserand_parameter(a, e, a_moon)
    return v_m * math.sqrt(max(3.0 - T, 0.0))


def connecting_transfer(r_inner, r_outer):
    """Periapsis-at-inner, apoapsis-at-outer transfer orbit. Returns (a, e)."""
    a = 0.5 * (r_inner + r_outer)
    e = (r_outer - r_inner) / (r_outer + r_inner)
    return a, e


def max_turn_angle(vinf, gm_moon, r_flyby):
    """Maximum single-flyby v_inf rotation (rad) at closest radius r_flyby."""
    return 2.0 * math.asin(1.0 / (1.0 + r_flyby * vinf * vinf / gm_moon))


def hohmann_leg_dv(r1, r2, gm_central=GM_JUPITER):
    """Two-impulse Hohmann Delta-v (km/s) to move between circular radii r1, r2."""
    at = 0.5 * (r1 + r2)
    v1 = math.sqrt(gm_central / r1)
    vp = math.sqrt(gm_central * (2.0 / r1 - 1.0 / at))
    va = math.sqrt(gm_central * (2.0 / r2 - 1.0 / at))
    v2 = math.sqrt(gm_central / r2)
    return abs(vp - v1) + abs(v2 - va)


def moon_tour(moons=None, flyby_alt_km=200.0, gm_central=GM_JUPITER):
    """Design a gravity-assist tour across adjacent moons (inner->outer).

    Returns per-leg v_inf + turn feasibility, the deterministic gravity-assist Delta-v
    (sum of v_inf mismatches at intermediate moons), and the Hohmann baseline for comparison.
    """
    moons = moons or GALILEAN_TOUR
    radii = [m[1] for m in moons]
    legs = []
    hohmann = 0.0
    for k in range(len(moons) - 1):
        r_in, r_out = radii[k], radii[k + 1]
        a, e = connecting_transfer(r_in, r_out)
        vi = vinf_at_moon(a, e, r_in, gm_central)        # at inner moon (periapsis)
        vo = vinf_at_moon(a, e, r_out, gm_central)       # at outer moon (apoapsis)
        rf_in = moons[k][2] + flyby_alt_km
        rf_out = moons[k + 1][2] + flyby_alt_km
        legs.append({
            "from": moons[k][0].secondary, "to": moons[k + 1][0].secondary,
            "vinf_inner_kms": vi, "vinf_outer_kms": vo,
            "turn_inner_deg": math.degrees(max_turn_angle(vi, moons[k][3], rf_in)),
            "turn_outer_deg": math.degrees(max_turn_angle(vo, moons[k + 1][3], rf_out)),
        })
        hohmann += hohmann_leg_dv(r_in, r_out, gm_central)

    # deterministic Delta-v: |v_inf| mismatch at each INTERMEDIATE moon (flyby can't change |v_inf|)
    mismatch = 0.0
    junctions = []
    for k in range(len(legs) - 1):
        v_arr = legs[k]["vinf_outer_kms"]      # arriving at moon k+1 as the outer of leg k
        v_dep = legs[k + 1]["vinf_inner_kms"]  # departing moon k+1 as the inner of leg k+1
        dv = abs(v_arr - v_dep)
        mismatch += dv
        junctions.append({"moon": legs[k]["to"], "vinf_in": v_arr,
                          "vinf_out": v_dep, "dv_ms": dv * 1000.0})

    return {"legs": legs, "junctions": junctions,
            "ga_deterministic_dv_ms": mismatch * 1000.0,
            "hohmann_dv_ms": hohmann * 1000.0,
            "saving_ms": (hohmann - mismatch) * 1000.0}
