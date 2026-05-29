"""End-to-end low-energy Earth->Moon transfer assembly (MASTER_PLAN.md Stage 6).

Combines a standard trans-lunar injection (TLI) with a manifold-based ballistic
lunar capture, and sweeps the libration orbit / energy to minimize the total
LEO->LLO Delta-v. The TLI is held common between the direct and low-energy cases:
the manifold's value is the cheaper, ballistic lunar-orbit insertion, which is
computed from REAL CR3BP dynamics (see lunar_capture.py).

HONEST SCOPE (read MASTER_PLAN §13): the exact published optimum depends on the
departure targeting and the Sun's perturbation (BCR4BP), whose full optimization
is a later increment. Here the departure is the standard Hohmann TLI and the
capture is the rigorous manifold result; the total is therefore a defensible
CR3BP-class figure, not a claim of exact reproduction.
"""
from __future__ import annotations

from ..data.constants import EARTH_MOON
from ..optimize.budget import earth_moon_budget
from ..orbits.families import lyapunov_orbit_at_jacobi
from .lunar_capture import ballistic_capture

COIMBRA_TOTAL_MS = 3925.0     # the published optimized result (m/s)


def low_energy_lunar_transfer(system=EARTH_MOON, points=("L1", "L2"),
                              jacobi_values=(3.05, 3.10, 3.15),
                              leo_alt=200.0, llo_alt=100.0):
    """Sweep libration orbits/energies; return (best, all_records, baseline).

    `best`/records carry the TLI, ballistic LOI, total Delta-v (m/s), and coast TOF.
    `baseline` is the direct (hyperbolic-capture) budget for comparison.
    """
    mu = system.mu
    budget = earth_moon_budget(leo_alt=leo_alt, llo_alt=llo_alt)
    tli = budget["dv_tli"]                          # common departure (km/s)

    records = []
    for pt in points:
        for c in jacobi_values:
            orb = lyapunov_orbit_at_jacobi(mu, pt, c)
            cap = ballistic_capture(orb, llo_alt=llo_alt, system=system)
            if cap is None:
                continue
            total = tli + cap["dv_capture_kms"]
            records.append({
                "point": pt, "jacobi": c,
                "tli_ms": tli * 1000.0,
                "loi_ms": cap["dv_capture_kms"] * 1000.0,
                "total_ms": total * 1000.0,
                "tof_days": cap["tof_days"],
                "periapsis_alt_km": cap["periapsis_alt_km"],
            })

    if not records:
        return None, [], budget
    best = min(records, key=lambda r: r["total_ms"])
    baseline = {
        "direct_total_ms": budget["total_direct"] * 1000.0,
        "direct_loi_ms": budget["dv_loi_direct"] * 1000.0,
        "coimbra_ms": COIMBRA_TOTAL_MS,
    }
    return best, records, baseline
