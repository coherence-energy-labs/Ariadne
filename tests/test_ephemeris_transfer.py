"""Full-ephemeris transfer tests (Gate G8e/G8b, Stage 8)."""
import numpy as np

from ariadne.data.ephemeris import et
from ariadne.transfers.ephemeris_transfer import design_transfer, optimize_transfer

BALLISTIC_LOI_MS = 625.0
COIMBRA_MS = 3925.0


def test_single_transfer_targets_real_moon():
    e0 = et("2025-06-01T00:00:00")
    d = design_transfer(e0, 5.0)
    assert d is not None
    assert d["miss_km"] < 1.0                       # hits the real Moon in DE440 gravity
    assert 3050.0 < d["dv_tli_ms"] < 3250.0         # Apollo-class TLI
    assert 0.7 < d["v_inf_kms"] < 1.2


def test_optimized_transfer_brackets_coimbra():
    e0 = et("2025-06-01T00:00:00")
    best, recs = optimize_transfer(e0, tof_grid=np.arange(4.0, 6.01, 0.5))
    assert best is not None and len(recs) >= 3
    # ephemeris DIRECT transfer is in the literature direct class, just above 3925
    assert 3900.0 < best["total_ms"] < 4000.0
    # ephemeris departure + ballistic capture is just below 3925 -> brackets it
    low = best["dv_tli_ms"] + BALLISTIC_LOI_MS
    assert low < COIMBRA_MS < best["total_ms"]
