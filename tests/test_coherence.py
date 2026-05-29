"""Coherence/robustness metric tests (Gate G_coh, Stage 11)."""
import math

import numpy as np

from ariadne.data.constants import GM_EARTH, R_EARTH
from ariadne.data.ephemeris import et
from ariadne.dynamics.ephemeris_nbody import propagate_test_particle
from ariadne.transfers.ephemeris_transfer import design_transfer
from ariadne.analysis.coherence import endpoint_sensitivity


def _prop(s0, e0, T):
    return lambda s: propagate_test_particle(s[:3], s[3:], e0, (0, T),
                                             perturbers=("SUN", "MOON")).y[:, -1]


def test_metric_ranks_stable_orbit_as_more_coherent():
    e0 = et("2025-06-01T00:00:00")
    # stable LEO: small endpoint sensitivity, ~no exponential divergence
    r = R_EARTH + 400.0
    leo = np.array([r, 0, 0, 0, math.sqrt(GM_EARTH / r), 0.0])
    leo_sens = endpoint_sensitivity(_prop(leo, e0, 86400.0), leo)

    # lunar transfer: large endpoint sensitivity (chaotic arrival)
    d = design_transfer(e0, 5.0)
    s0 = np.concatenate([d["r1"], d["v1"]])
    tr_sens = endpoint_sensitivity(_prop(s0, e0, 5 * 86400.0), s0)

    assert leo_sens > 0 and tr_sens > 0
    assert leo_sens < tr_sens / 5.0       # stable orbit is far more coherent (robust)


def test_robustness_costs_fuel():
    e0 = et("2025-06-01T00:00:00")
    fast = design_transfer(e0, 3.0)       # pricier, robust
    slow = design_transfer(e0, 6.0)       # cheaper, fragile
    s_fast = endpoint_sensitivity(_prop(np.concatenate([fast["r1"], fast["v1"]]), e0, 3 * 86400.0),
                                  np.concatenate([fast["r1"], fast["v1"]]))
    s_slow = endpoint_sensitivity(_prop(np.concatenate([slow["r1"], slow["v1"]]), e0, 6 * 86400.0),
                                  np.concatenate([slow["r1"], slow["v1"]]))
    assert slow["total_ms"] < fast["total_ms"]    # the 6-day transfer is cheaper
    assert s_slow > s_fast                         # ... and less coherent (more fragile)
