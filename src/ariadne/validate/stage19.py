"""Stage 19 validation gates (MASTER_PLAN.md - 3D halos + the Gateway-class NRHO).

G19a (NRHO geometry)  - Pseudo-arclength continuation of the L2 halo family reaches a
                        Near-Rectilinear Halo Orbit matching NASA's Gateway 9:2 NRHO:
                        period ~6.5-6.6 d, perilune ~2,800-3,800 km (low pass over the pole),
                        apolune ~60,000-78,000 km, periodic to < 1e-9.
G19b (near-stability) - The NRHO is FAR more stable than a deep libration orbit: its maximum
                        Floquet multiplier is small (the reason Gateway flies an NRHO -- cheap
                        stationkeeping), vs the L1 Lyapunov orbit's multiplier in the thousands.

Honest scope: this makes 3D halos / NRHOs first-class (constructed + validated). Integrating them
as transport-graph NODES needs a 3D Poincare section (the planar (y,v_y) intersection does not
capture 3D crossings) -- noted as the remaining extension; the planar transport graph stands.

Run:  PYTHONPATH=src python -m ariadne.validate.stage19
"""
from __future__ import annotations

import numpy as np

from ..data.constants import EARTH_MOON, R_MOON
from ..dynamics.cr3bp import propagate
from ..orbits.differential_correction import monodromy
from ..orbits.families import lyapunov_orbit_at_jacobi
from ..orbits.nrho import nrho_family

T_DAYS = EARTH_MOON.T_star / 86400.0
LSTAR = EARTH_MOON.L_star
MU = EARTH_MOON.mu


def _peri_apo_km(s0, period):
    sol = propagate(s0, (0.0, period), MU, t_eval=np.linspace(0.0, period, 800))
    d = np.sqrt((sol.y[0] - (1 - MU)) ** 2 + sol.y[1] ** 2 + sol.y[2] ** 2) * LSTAR
    return float(d.min()), float(d.max())


def _max_floquet(orbit):
    return float(np.max(np.abs(np.linalg.eigvals(monodromy(MU, orbit)))))


def check() -> tuple[bool, dict]:
    nrho, fam = nrho_family(MU, "L2", t_star_days=T_DAYS, l_star=LSTAR,
                            target_period_d=6.56, ds=4e-3)
    info = {"n_family": len(fam),
            "period_range_d": (fam[0].period * T_DAYS, fam[-1].period * T_DAYS) if fam else None}
    if nrho is None:
        return False, {**info, "nrho": None, "g19a": False, "g19b": False}

    peri, apo = _peri_apo_km(nrho.s0, nrho.period)
    period_d = nrho.period * T_DAYS
    floq_nrho = _max_floquet(nrho)
    floq_lyap = _max_floquet(lyapunov_orbit_at_jacobi(MU, "L1", 3.16))

    g19a = (6.3 <= period_d <= 6.8 and 2800 <= peri <= 3800
            and 60000 <= apo <= 78000 and nrho.residual < 1e-9)
    g19b = floq_nrho < 100.0 and floq_nrho < 0.1 * floq_lyap
    ok = g19a and g19b
    info.update({"nrho": nrho, "period_d": period_d, "peri_km": peri, "apo_km": apo,
                 "peri_alt_km": peri - R_MOON, "floq_nrho": floq_nrho,
                 "floq_lyap": floq_lyap, "g19a": g19a, "g19b": g19b})
    return ok, info


def main() -> int:
    print("=== Ariadne Stage 19 validation  (3D halos + Gateway-class NRHO) ===\n")
    ok, i = check()
    if i["period_range_d"]:
        print(f"L2 halo->NRHO continuation: {i['n_family']} members, "
              f"period {i['period_range_d'][0]:.2f} d -> {i['period_range_d'][1]:.2f} d\n")

    print("[G19a] Near-Rectilinear Halo Orbit geometry (vs NASA Gateway 9:2 NRHO)")
    if i.get("nrho") is not None:
        print(f"      period   = {i['period_d']:.3f} d   (Gateway ~6.56 d)")
        print(f"      perilune = {i['peri_km']:.0f} km  (alt {i['peri_alt_km']:.0f} km over the pole)")
        print(f"      apolune  = {i['apo_km']:.0f} km   (Gateway ~70,000 km)")
        print(f"      periodic to {i['nrho'].residual:.1e}")
    print(f"      -> {'PASS' if i['g19a'] else 'FAIL'}\n")

    print("[G19b] Near-stability (why Gateway flies an NRHO)")
    if "floq_nrho" in i:
        print(f"      NRHO max Floquet multiplier   = {i['floq_nrho']:.2f}")
        print(f"      L1 Lyapunov (C=3.16) for scale = {i['floq_lyap']:.0f}")
    print(f"      -> {'PASS' if i['g19b'] else 'FAIL'}\n")

    print(f"=== STAGE 19: {'ALL GATES PASS' if ok else 'FAILURE'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
