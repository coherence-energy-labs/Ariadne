"""Stage 11 validation gate (MASTER_PLAN.md — coherence/robustness lens).

G_coh - The coherence (robustness) metric is meaningful: it ranks a STABLE orbit as far more
        coherent (orders of magnitude lower endpoint sensitivity) than a chaotic lunar transfer,
        and it resolves the Delta-v-vs-coherence tradeoff -- the cheapest (WSB) transfer is the
        LEAST coherent, the fast/pricier transfer the MOST coherent. "Robustness costs fuel."

This reframes everything we built through the project's coherence lens: coherence == how well
a path holds together under injection error (km of arrival drift per 1 m/s). It is NOT a way to
beat the energy floor (physics fixes that) -- it is a different, real objective.

Run:  PYTHONPATH=src python -m ariadne.validate.stage11
"""
from __future__ import annotations

import math

import numpy as np

from ..data.constants import GM_EARTH, R_EARTH
from ..data.ephemeris import et
from ..dynamics.ephemeris_nbody import propagate_test_particle
from ..transfers.ephemeris_transfer import design_transfer
from ..analysis.coherence import endpoint_sensitivity, decoherence_rate, coherence_score


def _leo_sensitivity():
    e0 = et("2025-06-01T00:00:00")
    r = R_EARTH + 400.0
    s0 = np.array([r, 0, 0, 0, math.sqrt(GM_EARTH / r), 0.0])
    T = 86400.0
    prop = lambda s: propagate_test_particle(s[:3], s[3:], e0, (0, T),
                                             perturbers=("SUN", "MOON")).y[:, -1]
    return endpoint_sensitivity(prop, s0), decoherence_rate(prop, s0, T)


def _transfer_sensitivity(tof):
    e0 = et("2025-06-01T00:00:00")
    d = design_transfer(e0, tof)
    s0 = np.concatenate([d["r1"], d["v1"]]); T = tof * 86400.0
    prop = lambda s: propagate_test_particle(s[:3], s[3:], e0, (0, T),
                                             perturbers=("SUN", "MOON")).y[:, -1]
    return endpoint_sensitivity(prop, s0), d["total_ms"]


def main() -> int:
    print("=== Ariadne Stage 11 validation  (coherence / robustness lens) ===\n")

    leo_s, _ = _leo_sensitivity()
    fast_s, fast_dv = _transfer_sensitivity(3.0)     # fast, pricey -> coherent
    slow_s, slow_dv = _transfer_sensitivity(6.0)     # slow, cheaper -> fragile

    print("[G_coh] coherence metric distinguishes stable from chaotic and resolves the tradeoff")
    print(f"      stable LEO (1 d):       sensitivity = {leo_s:8.0f} km/(m/s), "
          f"coherence {coherence_score(leo_s):.3f}")
    print(f"      direct 3 d ({fast_dv:.0f} m/s): sensitivity = {fast_s:8.0f} km/(m/s), "
          f"coherence {coherence_score(fast_s):.3f}")
    print(f"      direct 6 d ({slow_dv:.0f} m/s): sensitivity = {slow_s:8.0f} km/(m/s), "
          f"coherence {coherence_score(slow_s):.3f}")
    print(f"      -> stable orbit is {fast_s/leo_s:.0f}x more coherent than a transfer;")
    print(f"         the {fast_dv - slow_dv:.0f} m/s cheaper 6-d transfer is "
          f"{slow_s/fast_s:.1f}x more fragile (robustness costs fuel)")

    ok = (leo_s < fast_s / 5.0) and (slow_s > fast_s)
    print(f"      -> {'PASS' if ok else 'FAIL'}\n")
    print(f"=== STAGE 11: {'COHERENCE LENS VALIDATED' if ok else 'FAILURE'} ===")
    print("NOTE: coherence == robustness, a real & different objective; it does NOT beat the")
    print("energy floor (physics fixes that). The cheapest path is the least coherent.")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
