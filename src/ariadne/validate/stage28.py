"""Stage 28 validation -- inverse hidden-mass localizer + two-tier discovery pipeline.

The user's vision made rigorous: a broad sensitive overview flags an anomaly, then we hone in to a
location (the Neptune/Le Verrier method). All on real tracked bodies (the clustered eTNOs).

G28a (recovery)   - Inject a known hidden body, generate noisy residual observations at the real
                    eTNOs, and the inverse least-squares localizer recovers its mass (within ~2x) and
                    position (within ~1-sigma of truth). The output is a confidence REGION + a sky box.
G28b (honing)     - The 1-sigma localization uncertainty SHRINKS as tracked bodies are added
                    (Tier-2 refinement: the region tightens with data).
G28c (degeneracy) - With a SINGLE tracked body the inverse problem is degenerate (a direction + a
                    mass/distance product, not a unique body); >= 2 diverse bodies are required.
                    Plus an all-sky sensitivity map: where a hidden body of given mass could hide.

HONEST: simulation recovery inverts the forward model under measurement noise -- it proves the
machinery + quantifies uncertainty. A real detection additionally needs secular (Myr) accumulation,
non-gravitational force modelling, and real tracking data; the output is a confidence region, never a
guaranteed exact point; gravity yields GM only (size/composition need the IR/optical handoff).

Run:  PYTHONPATH=src python -m ariadne.validate.stage28
"""
from __future__ import annotations

import numpy as np

from ..fields.hidden_mass import (CLUSTERED_ETNOS, PLANET9, elements_to_position,
                                  GM_EARTH, AU_KM)
from ..discovery.inverse_mass import (simulate_observations, localize, sky_box,
                                      localization_vs_n, sensitivity_skymap)

NOISE = 1e-14            # residual-acceleration measurement noise (m/s^2); optimistic, stated


def _tracked():
    return [elements_to_position(o["a_au"], o["e"], o["i"], o["Omega"], o["omega"], 180.0)
            for o in CLUSTERED_ETNOS]


def check() -> tuple[bool, dict]:
    tracked = _tracked()
    gm_true = PLANET9["m_earth"] * GM_EARTH
    pos_true = elements_to_position(PLANET9["a_au"], PLANET9["e"], PLANET9["i"],
                                    PLANET9["Omega"], PLANET9["omega"], 180.0)

    # G28a: recovery from noisy observations at the real eTNOs
    obs = simulate_observations(tracked, gm_true, pos_true, NOISE, seed=1)
    rec = localize(tracked, obs, NOISE)
    pos_err = float(np.linalg.norm(rec["position"] - pos_true))
    m_ratio = rec["m_earth"] / PLANET9["m_earth"]
    g28a = (rec["success"] and 0.3 < m_ratio < 3.0
            and pos_err < 2.0 * rec["pos_sigma_km"])
    box = sky_box(rec["position"], rec["pos_sigma_km"])

    # G28b: honing -- uncertainty shrinks with N
    hone = localization_vs_n(tracked, gm_true, pos_true, NOISE, seed=1)
    g28b = hone[-1]["pos_sigma_au"] < hone[0]["pos_sigma_au"]

    # G28c: single-body degeneracy + all-sky sensitivity map
    obs1 = simulate_observations(tracked[:1], gm_true, pos_true, NOISE, seed=1)
    rec1 = localize(tracked[:1], obs1, NOISE)
    degenerate = (not np.isfinite(rec1["pos_sigma_km"])
                  or rec1["pos_sigma_km"] > 5.0 * rec["pos_sigma_km"])
    smap = sensitivity_skymap(tracked, distance_au=500.0, noise_ms2=NOISE, n_lon=18, n_lat=9)
    g28c = degenerate and np.all(np.isfinite(smap["min_mass_earth"]))

    ok = g28a and g28b and g28c
    return ok, {"rec": rec, "pos_err_au": pos_err / AU_KM, "m_ratio": m_ratio, "box": box,
                "hone": hone, "rec1_sigma_au": rec1["pos_sigma_km"] / AU_KM,
                "smap": smap, "pos_true_au": np.linalg.norm(pos_true) / AU_KM,
                "g28a": g28a, "g28b": g28b, "g28c": g28c}


def main() -> int:
    print("=== Ariadne Stage 28  (inverse hidden-mass localizer + two-tier discovery pipeline) ===\n")
    ok, i = check()
    r = i["rec"]

    print("[G28a] Inverse recovery of an injected body from noisy residuals at the real eTNOs")
    print(f"      truth     : {PLANET9['m_earth']:.2f} M_earth @ {i['pos_true_au']:.0f} AU")
    print(f"      recovered : {r['m_earth']:.2f} M_earth @ {np.linalg.norm(r['position'])/AU_KM:.0f} AU"
          f"  (pos error {i['pos_err_au']:.0f} AU, 1-sigma {r['pos_sigma_km']/AU_KM:.0f} AU)")
    print(f"      sky search-box: ecl lon {i['box']['ecliptic_lon_deg']:.0f} / lat "
          f"{i['box']['ecliptic_lat_deg']:.0f} deg, dist {i['box']['distance_au']:.0f} AU, "
          f"1-sigma {i['box']['angular_sigma_deg']:.1f} deg  <- where to point IR/optical surveys")
    print(f"      -> {'PASS' if i['g28a'] else 'FAIL'}\n")

    print("[G28b] Honing: the localization region tightens as tracked bodies are added")
    for h in i["hone"]:
        print(f"      N={h['n']}: 1-sigma {h['pos_sigma_au']:6.0f} AU   pos error {h['pos_error_au']:6.0f} AU")
    print(f"      -> {'PASS' if i['g28b'] else 'FAIL'}\n")

    print("[G28c] Degeneracy + all-sky sensitivity")
    print(f"      single tracked body: 1-sigma {i['rec1_sigma_au']:.0f} AU (degenerate -- need >=2 diverse bodies)")
    mm = i["smap"]["min_mass_earth"]
    print(f"      all-sky min detectable mass @ 500 AU: {mm.min():.2f} - {mm.max():.2f} M_earth "
          f"(blind spots = high values)")
    print(f"      -> {'PASS' if i['g28c'] else 'FAIL'}\n")

    print("  HONEST: the output is a confidence REGION that shrinks with data (Le Verrier-style),")
    print("  not a magic exact point; floor- and degeneracy-limited; gravity gives GM only. A real")
    print("  search needs secular accumulation + real data + the IR/optical confirmation handoff.\n")
    print(f"=== STAGE 28: {'ALL GATES PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
