"""Intelligent integrator selector -- use exactly the backend that wins for the workload.

There is no single best integrator; the right one depends on the SHAPE of the job, and the
crossovers were MEASURED on this machine (RTX 5080 float64 + 24 cores; see validate/stage32):

  * single / few trajectories  -> numba CPU (`secular_fast`, 427x). A GPU cannot help: the
                                  symplectic timestep loop is serial and the per-step work is tiny.
  * small ensemble  (N < few k)-> single-core numba: launch/occupancy overhead of GPU and the
                                  thread-spawn of the parallel CPU are not amortised yet.
  * moderate ensemble          -> 24-core parallel CPU (`secular_fast.integrate_ensemble_parallel`).
  * large ensemble  (N >= ~big)-> GPU (`secular_gpu`) IF a CUDA device is present. NOTE: on a GeForce
                                  GPU float64 is throttled ~1/64, so the GPU edge is modest (~3x) --
                                  the selector still prefers it only past the measured crossover.
  * Gyr timescale, non-crossing-> doubly-averaged secular (`secular_avg`): the only backend that
                                  reaches Gyr. Crossing orbits fall back to the exact integrator.

This is the "intelligent selector" pattern (ACE forge gap17/gap22) applied to Ariadne: measure the
crossovers once, then route every job to the backend that is actually fastest for that job.
"""
from __future__ import annotations

from .secular_fast import HAVE_NUMBA, integrate_ensemble_parallel
from .secular_gpu import HAVE_CUDA, integrate_ensemble_gpu, integrate_ensemble_cpu

#: Measured crossovers (this machine: 24-core CPU + RTX 5080 float64). From validate/stage32.
#: N<1000: single-core numba (thread-spawn/launch overhead not amortised).
#: 1000<=N<5000: 24-core parallel CPU wins (up to ~9x over 1-core, beats GPU launch overhead).
#: N>=5000: GPU wins (float64-throttled, so only ~1.5x over 24-core, but it wins).
PARALLEL_MIN_N = 1_000
GPU_MIN_N = 5_000


def recommend_backend(n_particles: int, n_steps: int = 0, gyr: bool = False,
                      crossing: bool = False, have_cuda: bool | None = None,
                      have_numba: bool | None = None) -> tuple[str, str]:
    """Return (backend, reason) for the given workload. Pure decision logic (testable)."""
    cuda = HAVE_CUDA if have_cuda is None else have_cuda
    numba = HAVE_NUMBA if have_numba is None else have_numba

    if gyr and not crossing:
        return "secular_avg", "Gyr, non-crossing -> doubly-averaged (only backend that reaches Gyr)"
    if gyr and crossing:
        return "exact", "Gyr but orbit-crossing -> secular invalid; exact integrator required"
    if not numba:
        return "python", "numba unavailable -> pure-Python fallback"
    if n_particles <= 1:
        return "numba", "single trajectory -> numba (GPU/parallel cannot help a serial loop)"
    if cuda and n_particles >= GPU_MIN_N:
        return "gpu", f"large ensemble (N>={GPU_MIN_N}) -> GPU"
    if n_particles >= PARALLEL_MIN_N:
        return "parallel", f"moderate ensemble (N>={PARALLEL_MIN_N}) -> 24-core CPU"
    return "numba", "small ensemble -> single-core numba (parallel/GPU overhead not amortised)"


def integrate_ensemble(giants0, gm, gm0, tp0, dt, n_steps, backend="auto"):
    """Integrate N test particles under shared giants with the chosen (or auto) backend.

    giants0 (nm,6), gm (nm,), gm0 scalar, tp0 (N,6) -> returns (N,6) final states.
    """
    N = tp0.shape[0]
    if backend == "auto":
        backend, _ = recommend_backend(N, n_steps)
    if backend == "gpu" and HAVE_CUDA:
        return integrate_ensemble_gpu(giants0, gm, gm0, tp0, dt, n_steps)
    if backend == "parallel" and HAVE_NUMBA:
        return integrate_ensemble_parallel(giants0, gm, gm0, tp0, dt, n_steps)
    return integrate_ensemble_cpu(giants0, gm, gm0, tp0, dt, n_steps)
