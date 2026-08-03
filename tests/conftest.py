"""Test-suite policy: make the suite fast, and make that speed SAFE.

MEASURED, because every intuition here was wrong at least once. Fast subset,
`pytest -m "not slow"`, 1054 tests, 24-core box:

    serial                              474 s   1054 pass
    -n auto (24 workers), loadgroup     453 s   3 FAILED   <- oversubscribed AND slower
    -n 8  --dist loadgroup (gpu only)   772 s   1054 pass  <- far worse than serial
    -n 8  --dist loadgroup (per-module) 662 s   1054 pass  <- still worse
    -n 8  --dist loadfile               139 s   1054 pass  <- 3.4x, and the winner

`loadfile` wins because IMPORT COST DOMINATES this suite -- astropy alone is 0.9 s, over
122 test modules. Keeping a module on one worker imports it once there. Every
`loadgroup` variant distributes more finely and pays that import repeatedly; grouping
per-module did NOT recover the difference, which is why the numbers above are recorded
rather than the reasoning that predicted them.

WHY A LOCK AND NOT A SCHEDULING TRICK

`-n 8 --dist loadfile` was measured twice: once at 179 s with **5 failures**, once at
139 s with none. The warnings named it -- `numba_cuda`, `test_gpu_ensemble_matches_cpu`,
"Grid size 1 will likely result in GPU under-utilization". Three GPU modules can land on
three workers and contend for one device. A suite that passes on the second try is not
passing; it is a coin flip wearing a green tick, and it teaches people to re-run instead
of read.

Pinning them to one worker via `xdist_group` requires `--dist loadgroup`, which costs
more than the contention does. So the GPU tests take a cross-process file lock instead.
That is independent of the distribution mode, survives `-n auto`, `-p no:xdist` and a
plain serial run, and needs no third-party dependency.

WHAT WAS TRIED AND REJECTED -- do not re-add without re-measuring.

Capping BLAS threads (the textbook cure for worker oversubscription) BREAKS TORCH on
this platform. Measured one variable at a time:

    OMP_NUM_THREADS=1        -> torch fails to import
    OPENBLAS_NUM_THREADS=1   -> torch fails to import
    MKL_NUM_THREADS=1        -> torch fails to import
    NUMEXPR_NUM_THREADS=1    -> fine
    VECLIB_MAXIMUM_THREADS=1 -> fine

Torch ships its own OpenMP runtime on Windows and collides with those settings, so
`test_gpu_shift_stack.py` stopped COLLECTING entirely -- `pytest.importorskip("torch")`
raises rather than skips, because the failure is not an ImportError. A suite that cannot
import torch is not a faster suite.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

#: Modules that touch the GPU. Importing any of them initialises CUDA, so the resource
#: is owned at module scope -- the module is the honest unit, and a new GPU module is
#: one line here rather than a marker on each of its tests.
GPU_MODULES = frozenset({
    "test_gpu_shift_stack.py",
    "test_integrators.py",
    "test_secular_avg.py",
})

_LOCK = Path(os.environ.get("TEMP", "/tmp")) / "ariadne-gpu-tests.lock"


def pytest_collection_modifyitems(items):
    """Mark GPU-touching tests so the lock fixture below can find them."""
    for item in items:
        if item.path.name in GPU_MODULES:
            item.add_marker(pytest.mark.gpu)


@pytest.fixture(autouse=True)
def _serialise_gpu_tests(request):
    """One GPU test at a time, across processes.

    Deliberately a file lock and not `xdist_group`: the group marker only works under
    `--dist loadgroup`, and that mode was measured at 4-5x the wall clock of
    `loadfile`. This costs nothing when there is no contention and is correct under
    every distribution mode, including none.

    Stale locks self-heal: a lock older than the timeout is assumed to belong to a
    worker that died, because the alternative -- a suite that wedges forever after one
    crash -- is worse than a rare double-run of a GPU test.
    """
    if "gpu" not in request.keywords:
        yield
        return

    deadline = time.monotonic() + 300.0
    fd = None
    while fd is None:
        try:
            fd = os.open(str(_LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                age = time.time() - _LOCK.stat().st_mtime
            except OSError:
                continue                       # vanished between the two calls; retry
            if age > 300.0:
                _LOCK.unlink(missing_ok=True)  # a worker died holding it
                continue
            if time.monotonic() > deadline:
                pytest.fail("timed out waiting for the GPU lock -- refusing to run "
                            "concurrently rather than reporting a contended result")
            time.sleep(0.05)
    try:
        yield
    finally:
        os.close(fd)
        _LOCK.unlink(missing_ok=True)
