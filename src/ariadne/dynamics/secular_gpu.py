"""GPU ensemble integrator (numba.cuda) -- MASTER_PLAN.md Stage 32.

For a SINGLE long trajectory the timestep loop is serial and tiny per step, so a GPU
cannot help -- `secular_fast.py` (numba, 427x) is the right tool there. The GPU's place
is the ENSEMBLE: thousands-to-millions of independent test particles (uncertainty clouds
of the real eTNOs, synthetic populations for clustering statistics, clone clouds for
Lyapunov/chaos). That is embarrassingly parallel.

The naive approach -- a torch/cupy vectorised step with the timestep loop in Python --
is launch-overhead bound (the universal-variable Newton iteration fires hundreds of tiny
kernels per step) and barely beats the CPU. The right design is a SINGLE fused kernel:
one GPU thread integrates one particle through the ENTIRE rollout, so the whole ensemble
is one launch with zero per-step overhead. That is what this module does.

Each thread carries its own copy of the (shared) giant planets + its one test particle and
runs the identical democratic-heliocentric Wisdom-Holman map as `secular.py`/`secular_fast.py`
-- so the GPU result is faithful to the CPU result (validated in Stage 32 / tests).

Falls back to the numba-CPU ensemble if CUDA is unavailable.
"""
from __future__ import annotations

import math

import numpy as np

from .secular import YEAR_S

try:
    from numba import cuda, float64, int32
    HAVE_CUDA = cuda.is_available()
except Exception:                                            # pragma: no cover
    HAVE_CUDA = False

_MAXB = 12          # compile-time cap on (giants + 1 test) bodies per thread


if HAVE_CUDA:
    @cuda.jit(device=True, inline=True)
    def _stumpff_dev(psi):
        if psi > 1e-8:
            sp = psi ** 0.5
            return (1.0 - math.cos(sp)) / psi, (sp - math.sin(sp)) / (sp * sp * sp)
        elif psi < -1e-8:
            sn = (-psi) ** 0.5
            return (1.0 - math.cosh(sn)) / psi, (math.sinh(sn) - sn) / (sn * sn * sn)
        return (0.5 - psi / 24.0 + psi * psi / 720.0,
                1.0 / 6.0 - psi / 120.0 + psi * psi / 5040.0)

    @cuda.jit(device=True)
    def _kepler_dev(rx, ry, rz, vx, vy, vz, mu, dt):
        sqrtmu = math.sqrt(mu)
        r0 = math.sqrt(rx * rx + ry * ry + rz * rz)
        v2 = vx * vx + vy * vy + vz * vz
        rdotv = rx * vx + ry * vy + rz * vz
        u = rdotv / sqrtmu
        alpha = 2.0 / r0 - v2 / mu
        chi = sqrtmu * dt * alpha
        if alpha < 0.0:
            a = 1.0 / alpha
            den = rdotv + math.copysign(1.0, dt) * math.sqrt(-mu * a) * (1.0 - r0 * alpha)
            if den == 0.0:
                den = 1e-300
            chi = math.copysign(1.0, dt) * math.sqrt(-a) * math.log((-2.0 * mu * alpha * dt) / den)
        for _ in range(80):
            psi = chi * chi * alpha
            c2, c3 = _stumpff_dev(psi)
            chi2 = chi * chi
            r = chi2 * c2 + u * chi * (1.0 - psi * c3) + r0 * (1.0 - psi * c2)
            f = chi2 * chi * c3 + u * chi2 * c2 + r0 * chi * (1.0 - psi * c3) - sqrtmu * dt
            dchi = -f / r
            chi += dchi
            if dchi < 1e-12 and dchi > -1e-12:
                break
        psi = chi * chi * alpha
        c2, c3 = _stumpff_dev(psi)
        chi2 = chi * chi
        ff = 1.0 - chi2 / r0 * c2
        gg = dt - chi2 * chi / sqrtmu * c3
        nrx = ff * rx + gg * vx
        nry = ff * ry + gg * vy
        nrz = ff * rz + gg * vz
        rn = math.sqrt(nrx * nrx + nry * nry + nrz * nrz)
        gd = 1.0 - chi2 / rn * c2
        fd = sqrtmu / (rn * r0) * chi * (psi * c3 - 1.0)
        nvx = fd * rx + gd * vx
        nvy = fd * ry + gd * vy
        nvz = fd * rz + gd * vz
        return nrx, nry, nrz, nvx, nvy, nvz

    @cuda.jit
    def _ensemble_kernel(g0, gm, gm0, nm, tp0, dt, n_steps, out):
        p = cuda.grid(1)
        N = tp0.shape[0]
        if p >= N:
            return
        nb = nm + 1                                  # giants + this test particle
        qx = cuda.local.array(_MAXB, float64); qy = cuda.local.array(_MAXB, float64)
        qz = cuda.local.array(_MAXB, float64); vx = cuda.local.array(_MAXB, float64)
        vy = cuda.local.array(_MAXB, float64); vz = cuda.local.array(_MAXB, float64)
        for j in range(nm):
            qx[j] = g0[j, 0]; qy[j] = g0[j, 1]; qz[j] = g0[j, 2]
            vx[j] = g0[j, 3]; vy[j] = g0[j, 4]; vz[j] = g0[j, 5]
        t = nm
        qx[t] = tp0[p, 0]; qy[t] = tp0[p, 1]; qz[t] = tp0[p, 2]
        vx[t] = tp0[p, 3]; vy[t] = tp0[p, 4]; vz[t] = tp0[p, 5]
        hdt = 0.5 * dt
        for _ in range(n_steps):
            # half kick (interaction: giants on all bodies; Sun handled in drift)
            for i in range(nb):
                ax = 0.0; ay = 0.0; az = 0.0
                for j in range(nm):
                    if j == i:
                        continue
                    dx = qx[j] - qx[i]; dy = qy[j] - qy[i]; dz = qz[j] - qz[i]
                    d2 = dx * dx + dy * dy + dz * dz
                    inv = gm[j] / (d2 * d2 ** 0.5)
                    ax += dx * inv; ay += dy * inv; az += dz * inv
                vx[i] += hdt * ax; vy[i] += hdt * ay; vz[i] += hdt * az
            # half jump
            sx = 0.0; sy = 0.0; sz = 0.0
            for j in range(nm):
                sx += gm[j] * vx[j]; sy += gm[j] * vy[j]; sz += gm[j] * vz[j]
            sx = hdt * sx / gm0; sy = hdt * sy / gm0; sz = hdt * sz / gm0
            for i in range(nb):
                qx[i] += sx; qy[i] += sy; qz[i] += sz
            # Kepler drift
            for i in range(nb):
                nrx, nry, nrz, nvx, nvy, nvz = _kepler_dev(
                    qx[i], qy[i], qz[i], vx[i], vy[i], vz[i], gm0, dt)
                qx[i] = nrx; qy[i] = nry; qz[i] = nrz
                vx[i] = nvx; vy[i] = nvy; vz[i] = nvz
            # half jump
            sx = 0.0; sy = 0.0; sz = 0.0
            for j in range(nm):
                sx += gm[j] * vx[j]; sy += gm[j] * vy[j]; sz += gm[j] * vz[j]
            sx = hdt * sx / gm0; sy = hdt * sy / gm0; sz = hdt * sz / gm0
            for i in range(nb):
                qx[i] += sx; qy[i] += sy; qz[i] += sz
            # half kick
            for i in range(nb):
                ax = 0.0; ay = 0.0; az = 0.0
                for j in range(nm):
                    if j == i:
                        continue
                    dx = qx[j] - qx[i]; dy = qy[j] - qy[i]; dz = qz[j] - qz[i]
                    d2 = dx * dx + dy * dy + dz * dz
                    inv = gm[j] / (d2 * d2 ** 0.5)
                    ax += dx * inv; ay += dy * inv; az += dz * inv
                vx[i] += hdt * ax; vy[i] += hdt * ay; vz[i] += hdt * az
        out[p, 0] = qx[t]; out[p, 1] = qy[t]; out[p, 2] = qz[t]
        out[p, 3] = vx[t]; out[p, 4] = vy[t]; out[p, 5] = vz[t]


def integrate_ensemble_gpu(giants0, gm, gm0, tp0, dt, n_steps, threads=128):
    """Integrate N test particles (rows of tp0) under shared giants, on the GPU.

    giants0 : (nm,6) heliocentric pos+barycentric vel of the giant planets
    gm      : (nm,) giant GMs ; gm0 : Sun GM
    tp0     : (N,6) initial test-particle states (heliocentric pos + barycentric vel)
    Returns (N,6) final test-particle states. Falls back to CPU if no CUDA.
    """
    giants0 = np.ascontiguousarray(giants0, np.float64)
    gm = np.ascontiguousarray(gm, np.float64)
    tp0 = np.ascontiguousarray(tp0, np.float64)
    N = tp0.shape[0]
    nm = giants0.shape[0]
    if not HAVE_CUDA:                                        # pragma: no cover
        return integrate_ensemble_cpu(giants0, gm, gm0, tp0, dt, n_steps)
    out = np.zeros((N, 6))
    d_g = cuda.to_device(giants0); d_gm = cuda.to_device(gm)
    d_tp = cuda.to_device(tp0); d_out = cuda.to_device(out)
    blocks = (N + threads - 1) // threads
    _ensemble_kernel[blocks, threads](d_g, d_gm, gm0, nm, d_tp, dt, n_steps, d_out)
    cuda.synchronize()
    return d_out.copy_to_host()


def integrate_ensemble_cpu(giants0, gm, gm0, tp0, dt, n_steps):
    """CPU reference (numba via secular_fast) -- one shared system with N test particles."""
    from .secular import System
    from .secular_fast import integrate_fast
    nm = giants0.shape[0]
    Q = np.vstack([giants0[:, :3], tp0[:, :3]])
    V = np.vstack([giants0[:, 3:], tp0[:, 3:]])
    sys = System(Q=Q, V=V, gm=np.asarray(gm, float), gm0=gm0, n_massive=nm)
    integrate_fast(sys, dt, n_steps)
    return np.hstack([sys.Q[nm:], sys.V[nm:]])
