"""numba-accelerated core for the symplectic Wisdom-Holman map (MASTER_PLAN.md Stage 31).

`secular.py` is correct and readable but its per-step Python/numpy overhead caps it
near ~1.5k steps/s -- a Myr integration would take ~15 min, a Gyr is hopeless. This
module re-expresses the SAME map with explicit scalar loops compiled by numba
(@njit), where the universal-variable Kepler solve, the mutual-gravity kick and the
solar jump run as native machine code. It is bit-for-bit faithful to `secular.py`
(validated by Stage 31 / test_secular_avg) -- pure speed, no physics change.

If numba is unavailable the public entry point transparently falls back to
`secular.integrate`, so nothing breaks.
"""
from __future__ import annotations

import numpy as np

from .secular import YEAR_S, energy as _energy_py, integrate as _integrate_py

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:                                            # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):                                       # no-op decorator
        def wrap(f):
            return f
        return wrap if (a and callable(a[0])) is False else a[0]


@njit(cache=True)
def _stumpff(psi):
    if psi > 1e-8:
        sp = np.sqrt(psi)
        return (1.0 - np.cos(sp)) / psi, (sp - np.sin(sp)) / (sp * sp * sp)
    elif psi < -1e-8:
        sn = np.sqrt(-psi)
        return (1.0 - np.cosh(sn)) / psi, (np.sinh(sn) - sn) / (sn * sn * sn)
    return 0.5 - psi / 24.0 + psi * psi / 720.0, 1.0 / 6.0 - psi / 120.0 + psi * psi / 5040.0


@njit(cache=True)
def _kepler1(r, v, mu, dt, out_r, out_v):
    sqrtmu = np.sqrt(mu)
    r0 = np.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
    v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
    rdotv = r[0] * v[0] + r[1] * v[1] + r[2] * v[2]
    u = rdotv / sqrtmu
    alpha = 2.0 / r0 - v2 / mu
    chi = sqrtmu * dt * alpha
    if alpha < 0.0:
        a = 1.0 / alpha
        den = rdotv + np.sign(dt) * np.sqrt(-mu * a) * (1.0 - r0 * alpha)
        if den == 0.0:
            den = 1e-300
        chi = np.sign(dt) * np.sqrt(-a) * np.log((-2.0 * mu * alpha * dt) / den)
    for _ in range(80):
        psi = chi * chi * alpha
        c2, c3 = _stumpff(psi)
        chi2 = chi * chi
        rr = chi2 * c2 + u * chi * (1.0 - psi * c3) + r0 * (1.0 - psi * c2)
        fchi = chi2 * chi * c3 + u * chi2 * c2 + r0 * chi * (1.0 - psi * c3) - sqrtmu * dt
        dchi = -fchi / rr
        chi += dchi
        if abs(dchi) < 1e-12:
            break
    psi = chi * chi * alpha
    c2, c3 = _stumpff(psi)
    chi2 = chi * chi
    f = 1.0 - chi2 / r0 * c2
    g = dt - chi2 * chi / sqrtmu * c3
    for d in range(3):
        out_r[d] = f * r[d] + g * v[d]
    rn = np.sqrt(out_r[0] ** 2 + out_r[1] ** 2 + out_r[2] ** 2)
    gdot = 1.0 - chi2 / rn * c2
    fdot = sqrtmu / (rn * r0) * chi * (psi * c3 - 1.0)
    for d in range(3):
        out_v[d] = fdot * r[d] + gdot * v[d]


@njit(cache=True)
def _kick(Q, V, gm, nm, fac):
    n = Q.shape[0]
    for i in range(n):
        ax = 0.0; ay = 0.0; az = 0.0
        for j in range(nm):
            if j == i:
                continue
            dx = Q[j, 0] - Q[i, 0]; dy = Q[j, 1] - Q[i, 1]; dz = Q[j, 2] - Q[i, 2]
            d2 = dx * dx + dy * dy + dz * dz
            inv = 1.0 / (d2 * np.sqrt(d2))
            ax += gm[j] * dx * inv; ay += gm[j] * dy * inv; az += gm[j] * dz * inv
        V[i, 0] += fac * ax; V[i, 1] += fac * ay; V[i, 2] += fac * az


@njit(cache=True)
def _jump(Q, V, gm, gm0, nm, fac):
    sx = 0.0; sy = 0.0; sz = 0.0
    for j in range(nm):
        sx += gm[j] * V[j, 0]; sy += gm[j] * V[j, 1]; sz += gm[j] * V[j, 2]
    sx = fac * sx / gm0; sy = fac * sy / gm0; sz = fac * sz / gm0
    for i in range(Q.shape[0]):
        Q[i, 0] += sx; Q[i, 1] += sy; Q[i, 2] += sz


@njit(cache=True)
def _energy(Q, V, gm, gm0, nm):
    ke = 0.0; pe = 0.0
    px = 0.0; py = 0.0; pz = 0.0
    for i in range(nm):
        v2 = V[i, 0] ** 2 + V[i, 1] ** 2 + V[i, 2] ** 2
        ke += 0.5 * gm[i] * v2
        ri = np.sqrt(Q[i, 0] ** 2 + Q[i, 1] ** 2 + Q[i, 2] ** 2)
        pe -= gm0 * gm[i] / ri
        px += gm[i] * V[i, 0]; py += gm[i] * V[i, 1]; pz += gm[i] * V[i, 2]
    h_sun = 0.5 * (px * px + py * py + pz * pz) / gm0
    h_int = 0.0
    for i in range(nm):
        for j in range(i + 1, nm):
            dx = Q[i, 0] - Q[j, 0]; dy = Q[i, 1] - Q[j, 1]; dz = Q[i, 2] - Q[j, 2]
            h_int -= gm[i] * gm[j] / np.sqrt(dx * dx + dy * dy + dz * dz)
    return ke + pe + h_sun + h_int


@njit(cache=True)
def _run(Q, V, gm, gm0, nm, dt, n_steps, rec_every, energies, qtest):
    tmp_r = np.empty(3); tmp_v = np.empty(3)
    ri = 0
    for k in range(n_steps):
        if rec_every > 0 and (k % rec_every == 0):
            energies[ri] = _energy(Q, V, gm, gm0, nm)
            for t in range(qtest.shape[1]):
                for d in range(3):
                    qtest[ri, t, d] = Q[nm + t, d]
            ri += 1
        _kick(Q, V, gm, nm, 0.5 * dt)
        _jump(Q, V, gm, gm0, nm, 0.5 * dt)
        for i in range(Q.shape[0]):
            _kepler1(Q[i], V[i], gm0, dt, tmp_r, tmp_v)
            for d in range(3):
                Q[i, d] = tmp_r[d]; V[i, d] = tmp_v[d]
        _jump(Q, V, gm, gm0, nm, 0.5 * dt)
        _kick(Q, V, gm, nm, 0.5 * dt)
    return ri


@njit(cache=True)
def _run_full(Q, V, gm, gm0, nm, dt, n_steps, rec_every, q_all, v_all):
    """Like _run but records the FULL state (all bodies) at each snapshot."""
    tmp_r = np.empty(3); tmp_v = np.empty(3)
    ri = 0
    for k in range(n_steps):
        if rec_every > 0 and (k % rec_every == 0):
            for b in range(Q.shape[0]):
                for d in range(3):
                    q_all[ri, b, d] = Q[b, d]; v_all[ri, b, d] = V[b, d]
            ri += 1
        _kick(Q, V, gm, nm, 0.5 * dt)
        _jump(Q, V, gm, gm0, nm, 0.5 * dt)
        for i in range(Q.shape[0]):
            _kepler1(Q[i], V[i], gm0, dt, tmp_r, tmp_v)
            for d in range(3):
                Q[i, d] = tmp_r[d]; V[i, d] = tmp_v[d]
        _jump(Q, V, gm, gm0, nm, 0.5 * dt)
        _kick(Q, V, gm, nm, 0.5 * dt)
    return ri


def integrate_fast(sys, dt, n_steps, record_every=0):
    """numba-accelerated integration of a `secular.System`. Falls back to pure Python.

    Returns the same dict shape as `secular.integrate` (final system + optional
    times_yr / q_test / energy). Does NOT record osculating elements (use the pure
    integrator's record_test_elements for that, on shorter spans).
    """
    if not HAVE_NUMBA:                                       # pragma: no cover
        return _integrate_py(sys, dt, n_steps, record_every=record_every)

    Q = np.ascontiguousarray(sys.Q, float)
    V = np.ascontiguousarray(sys.V, float)
    gm = np.ascontiguousarray(sys.gm, float)
    nm = sys.n_massive
    n_test = Q.shape[0] - nm
    n_rec = (n_steps + record_every - 1) // record_every if record_every else 0
    energies = np.zeros(max(n_rec, 1))
    qtest = np.zeros((max(n_rec, 1), max(n_test, 1), 3))

    ri = _run(Q, V, gm, sys.gm0, nm, dt, n_steps, record_every or 0, energies, qtest)

    sys.Q, sys.V = Q, V
    out = {"system": sys, "n_steps": n_steps, "dt_s": dt, "span_yr": n_steps * dt / YEAR_S}
    if record_every:
        out["times_yr"] = np.arange(ri) * record_every * dt / YEAR_S
        out["energy"] = energies[:ri]
        out["q_test"] = qtest[:ri]
    return out


def integrate_fast_elements(sys, dt, n_steps, n_snap=24):
    """numba-integrate and return osculating elements of each test particle vs time.

    Converts recorded full state to heliocentric eTNO elements (with the Sun's
    barycentric velocity recovered from the massive bodies). Used to cross-validate
    the secular-averaged model against the exact integrator. Returns
    {times_yr, elements: list[n_snap] of list[n_test] element dicts}.
    """
    from .secular import state_to_elements
    if not HAVE_NUMBA:                                       # pragma: no cover
        raise RuntimeError("numba required for integrate_fast_elements")
    Q = np.ascontiguousarray(sys.Q, float)
    V = np.ascontiguousarray(sys.V, float)
    gm = np.ascontiguousarray(sys.gm, float)
    nm = sys.n_massive
    rec_every = max(1, n_steps // n_snap)
    n_rec = (n_steps + rec_every - 1) // rec_every
    q_all = np.zeros((n_rec, Q.shape[0], 3))
    v_all = np.zeros((n_rec, Q.shape[0], 3))
    ri = _run_full(Q, V, gm, sys.gm0, nm, dt, n_steps, rec_every, q_all, v_all)
    sys.Q, sys.V = Q, V
    times = np.arange(ri) * rec_every * dt / YEAR_S
    elements = []
    for s in range(ri):
        v_sun = -(gm[:, None] * v_all[s, :nm]).sum(axis=0) / sys.gm0
        row = [state_to_elements(q_all[s, nm + t], v_all[s, nm + t] - v_sun)
               for t in range(Q.shape[0] - nm)]
        elements.append(row)
    return {"times_yr": times, "elements": elements}
