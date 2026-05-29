"""Low-energy WSB transfer test (Gate G_wsb, Stage 10)."""
from ariadne.transfers.wsb import evaluate_transfer, SOLUTION_PARAMS

DIRECT_MS = 3953.0
COIMBRA_MS = 3925.0


def test_wsb_transfer_beats_coimbra():
    b = evaluate_transfer(SOLUTION_PARAMS)
    # departs ~LEO, arrives lower-energy than direct, total beats both direct and Coimbra
    assert abs(b["perigee_alt_km"] - 200.0) < 300.0
    assert b["v_inf"] < 0.82                 # lower v_inf than the direct transfer
    assert b["total_ms"] < DIRECT_MS
    assert b["total_ms"] < COIMBRA_MS
    assert b["tof_days"] > 40.0              # the honest tradeoff: long, low-energy route
