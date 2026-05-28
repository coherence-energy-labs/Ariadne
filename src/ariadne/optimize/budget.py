"""Delta-v budgets for Earth-Moon transfers (MASTER_PLAN.md §3.13, Stage 4).

Patched-conic / vis-viva accounting, in SI (km, km/s). Used to (a) validate the
machinery against textbook Apollo-class numbers and (b) quantify the low-energy
(ballistic-capture) saving that the Coimbra optimization exploits.

Physics:
  - circular speed   v_c   = sqrt(GM / r)
  - vis-viva         v     = sqrt(GM (2/r - 1/a))
  - a direct transfer arrives at the Moon hyperbolically (v_inf > 0) and pays a
    large lunar-orbit-insertion (LOI) burn; a ballistic-capture transfer arrives
    with v_inf ~ 0 (near-parabolic), so its LOI is smaller. The difference is the
    core fuel saving of low-energy lunar transfers.
"""
from __future__ import annotations

import math

from ..data.constants import GM_EARTH, GM_MOON, R_EARTH, R_MOON


def circular_speed(gm: float, r: float) -> float:
    return math.sqrt(gm / r)


def vis_viva(gm: float, r: float, a: float) -> float:
    return math.sqrt(gm * (2.0 / r - 1.0 / a))


def earth_moon_budget(leo_alt: float = 200.0, llo_alt: float = 100.0,
                      r_moon_orbit: float = 384400.0) -> dict:
    """Δv budget LEO -> LLO for a direct Hohmann transfer and for a
    ballistic-capture (low-energy) lunar insertion.

    Returns a dict of all components (km/s). The TLI is shared; the difference is
    at lunar capture (hyperbolic vs near-parabolic arrival).
    """
    r_leo = R_EARTH + leo_alt
    r_llo = R_MOON + llo_alt

    # --- Trans-lunar injection (Hohmann from LEO to the Moon's orbital radius) ---
    a_t = 0.5 * (r_leo + r_moon_orbit)
    v_circ_leo = circular_speed(GM_EARTH, r_leo)
    v_peri_transfer = vis_viva(GM_EARTH, r_leo, a_t)
    dv_tli = v_peri_transfer - v_circ_leo

    # --- Arrival at the Moon's orbital radius ---
    v_apo_transfer = vis_viva(GM_EARTH, r_moon_orbit, a_t)     # spacecraft (tangential)
    v_moon = circular_speed(GM_EARTH, r_moon_orbit)            # Moon's orbital speed
    v_inf_direct = abs(v_moon - v_apo_transfer)               # hyperbolic excess at Moon

    # --- Lunar orbit insertion ---
    v_circ_llo = circular_speed(GM_MOON, r_llo)
    v_esc_llo = math.sqrt(2.0) * v_circ_llo
    v_hyp_direct = math.sqrt(v_inf_direct ** 2 + v_esc_llo ** 2)
    dv_loi_direct = v_hyp_direct - v_circ_llo                 # capture from hyperbola
    dv_loi_ballistic = v_esc_llo - v_circ_llo                 # capture from ~parabola

    total_direct = dv_tli + dv_loi_direct
    total_low_energy = dv_tli + dv_loi_ballistic
    capture_saving = dv_loi_direct - dv_loi_ballistic

    return {
        "r_leo": r_leo, "r_llo": r_llo,
        "v_circ_leo": v_circ_leo, "v_circ_llo": v_circ_llo,
        "dv_tli": dv_tli,
        "v_inf_direct": v_inf_direct,
        "dv_loi_direct": dv_loi_direct,
        "dv_loi_ballistic": dv_loi_ballistic,
        "total_direct": total_direct,
        "total_low_energy": total_low_energy,
        "capture_saving": capture_saving,
    }
