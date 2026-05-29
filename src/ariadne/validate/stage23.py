"""Stage 23 validation gates (MASTER_PLAN.md - unified multi-objective grand optimizer).

G23a (trade)      - The (energy, time, robustness) trade space is real and shows no-free-lunch
                    structure: the fastest transfer is BOTH pricier and less launch-robust than the
                    cheapest one.
G23b (coherence)  - The coherence balancer picks a route that is a genuine MIDDLE GROUND (distinct
                    from both the pure-min-Delta-v and the pure-min-TOF extremes), and the weights
                    actually steer the choice (time-first picks a shorter TOF than energy-first).
G23c (low-thrust) - A low-thrust (Edelbaum heliocentric) estimate gives a sensible alternative-regime
                    point, with its honest scope caveat.

This is the synthesis: one optimizer balancing time, energy, and robustness, with coherence as the
dial -- "the most optimal route in time and energy, or anything else we want."

Run:  PYTHONPATH=src python -m ariadne.validate.stage23
"""
from __future__ import annotations

from ..data.ephemeris import et
from ..interplanetary.grand import (
    build_tradeoff, most_coherent_route, low_thrust_heliocentric,
)

START = "2026-01-01T00:00:00"
DEP, ARR = "EARTH", "MARS BARYCENTER"


def check() -> tuple[bool, dict]:
    e0 = et(START)
    tr = build_tradeoff(DEP, ARR, e0, dep_days=540, tof_range=(120, 400), n_dep=45, n_tof=35)
    cheapest = min(tr, key=lambda p: p["total_ms"])
    fastest = min(tr, key=lambda p: p["tof_days"])

    g23a = (fastest["total_ms"] > cheapest["total_ms"]
            and fastest["sensitivity_ms_per_day"] > cheapest["sensitivity_ms_per_day"])

    balanced = most_coherent_route(tr, (1.0, 1.0, 1.0))
    energy_first = most_coherent_route(tr, (3.0, 1.0, 0.5))
    time_first = most_coherent_route(tr, (0.5, 3.0, 0.5))
    g23b = (cheapest["tof_days"] > balanced["tof_days"] > fastest["tof_days"]
            and balanced["total_ms"] > cheapest["total_ms"]
            and time_first["tof_days"] <= energy_first["tof_days"])

    lt = low_thrust_heliocentric(DEP, ARR, e0, accel_mm_s2=0.2)
    g23c = 2000.0 < lt["dv_ms"] < 8000.0 and lt["tof_days"] > 100.0 and "excludes" in lt["note"]

    ok = g23a and g23b and g23c
    return ok, {"tradeoff": tr, "cheapest": cheapest, "fastest": fastest, "balanced": balanced,
                "energy_first": energy_first, "time_first": time_first, "low_thrust": lt,
                "g23a": g23a, "g23b": g23b, "g23c": g23c}


def main() -> int:
    print("=== Ariadne Stage 23 validation  (unified multi-objective grand optimizer) ===\n")
    ok, i = check()

    def row(p):
        return f"TOF {p['tof_days']:3.0f} d   {p['total_ms']:6.0f} m/s   sens {p['sensitivity_ms_per_day']:4.0f} m/s/day"

    print("[G23a] Energy x time x robustness trade space (no free lunch)")
    print(f"      cheapest : {row(i['cheapest'])}")
    print(f"      fastest  : {row(i['fastest'])}  (pricier AND less robust)")
    print(f"      -> {'PASS' if i['g23a'] else 'FAIL'}\n")

    print("[G23b] Coherence balancer (the dial across time / energy / robustness)")
    print(f"      balanced     w=(1,1,1)    : {row(i['balanced'])}  coherence {i['balanced']['coherence']:.3f}")
    print(f"      energy-first w=(3,1,.5)   : {row(i['energy_first'])}")
    print(f"      time-first   w=(.5,3,.5)  : {row(i['time_first'])}")
    print(f"      balanced is a genuine middle ground, weights steer the choice")
    print(f"      -> {'PASS' if i['g23b'] else 'FAIL'}\n")

    lt = i["low_thrust"]
    print("[G23c] Low-thrust regime (Edelbaum heliocentric estimate)")
    print(f"      Delta-v {lt['dv_ms']:.0f} m/s  TOF {lt['tof_days']:.0f} d  (a_T {lt['accel_mm_s2']} mm/s^2)")
    print(f"      scope: {lt['note']}")
    print(f"      -> {'PASS' if i['g23c'] else 'FAIL'}\n")

    print(f"=== STAGE 23: {'ALL GATES PASS' if ok else 'FAILURE'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
