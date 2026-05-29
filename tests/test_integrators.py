"""Stage 32 tests: intelligent backend selector + GPU/parallel ensemble correctness."""
import numpy as np
import pytest

from ariadne.dynamics import secular as S
from ariadne.dynamics import secular_fast as SF
from ariadne.dynamics import secular_gpu as SG
from ariadne.dynamics import integrators as IN
from ariadne.dynamics.secular import YEAR_S


# ---- selector decision logic (pure, no hardware needed) ----
def test_selector_single_trajectory_picks_numba():
    b, _ = IN.recommend_backend(1, 100000, have_cuda=True, have_numba=True)
    assert b == "numba"


def test_selector_gyr_picks_secular_avg():
    b, _ = IN.recommend_backend(6, 0, gyr=True, crossing=False)
    assert b == "secular_avg"
    b2, _ = IN.recommend_backend(6, 0, gyr=True, crossing=True)
    assert b2 == "exact"


def test_selector_large_ensemble_prefers_gpu_when_available():
    b, _ = IN.recommend_backend(10_000_000, 5000, have_cuda=True, have_numba=True)
    assert b == "gpu"
    b2, _ = IN.recommend_backend(10_000_000, 5000, have_cuda=False, have_numba=True)
    assert b2 == "parallel"               # no GPU -> 24-core CPU


def test_selector_small_ensemble_uses_single_core():
    b, _ = IN.recommend_backend(50, 5000, have_cuda=True, have_numba=True)
    assert b == "numba"


def test_selector_falls_back_to_python_without_numba():
    b, _ = IN.recommend_backend(1000, 5000, have_cuda=False, have_numba=False)
    assert b == "python"


# ---- ensemble correctness (the parallel CPU path must match the serial map) ----
def _setup(N):
    sys = S.build_system("2026-01-01T00:00:00")
    g0 = np.hstack([sys.Q[:sys.n_massive], sys.V[:sys.n_massive]])
    tp = []
    for k in range(N):
        pos, vel = S.elements_to_state(450.0 + 5 * k, 0.85, 15.0, 80.0, 250.0, 180.0)
        tp.append(np.hstack([pos, vel + sys._v_sun0]))
    return g0, sys.gm.copy(), sys.gm0, np.array(tp)


@pytest.mark.skipif(not SF.HAVE_NUMBA, reason="numba required")
def test_parallel_ensemble_matches_serial():
    g0, gm, gm0, tp0 = _setup(16)
    par = SF.integrate_ensemble_parallel(g0, gm, gm0, tp0.copy(), 1.0 * YEAR_S, 2000)
    ser = SG.integrate_ensemble_cpu(g0, gm, gm0, tp0.copy(), 1.0 * YEAR_S, 2000)
    rel = np.max(np.abs(par[:, :3] - ser[:, :3])) / np.linalg.norm(ser[0, :3])
    assert rel < 1e-9


@pytest.mark.skipif(not SG.HAVE_CUDA, reason="CUDA GPU required")
def test_gpu_ensemble_matches_cpu():
    g0, gm, gm0, tp0 = _setup(16)
    gpu = SG.integrate_ensemble_gpu(g0, gm, gm0, tp0.copy(), 1.0 * YEAR_S, 2000)
    cpu = SG.integrate_ensemble_cpu(g0, gm, gm0, tp0.copy(), 1.0 * YEAR_S, 2000)
    rel = np.max(np.abs(gpu[:, :3] - cpu[:, :3])) / np.linalg.norm(cpu[0, :3])
    assert rel < 1e-9


@pytest.mark.skipif(not SF.HAVE_NUMBA, reason="numba required")
def test_integrate_ensemble_auto_runs():
    g0, gm, gm0, tp0 = _setup(8)
    out = IN.integrate_ensemble(g0, gm, gm0, tp0.copy(), 1.0 * YEAR_S, 1000, backend="auto")
    assert out.shape == (8, 6) and np.all(np.isfinite(out))
