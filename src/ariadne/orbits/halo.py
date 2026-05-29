"""Three-dimensional halo orbits (MASTER_PLAN.md §3.5).

Halo orbits bifurcate from the planar Lyapunov family at the vertical (out-of-plane)
stability transition located in Stage 2 (Earth-Moon L1: C ~ 3.186). We seed a small
out-of-plane amplitude there and correct with the classic x-z symmetric single
shooting: start on the x-z plane with a perpendicular crossing
[x0, 0, z0, 0, vy0, 0], integrate to the next y=0 crossing, and Newton-correct
{x0, vy0} (z0 fixed) so the crossing is perpendicular in BOTH in-plane and
out-of-plane velocity (vx = vz = 0). The full period is twice the crossing time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from ..dynamics.cr3bp import eom, eom_stm, jacobi_constant
from .families import lyapunov_family, find_halo_bifurcation

_INT = dict(method="DOP853", rtol=1e-12, atol=1e-12)


@dataclass
class HaloOrbit:
    s0: np.ndarray
    period: float
    jacobi: float
    z_amplitude: float
    point: str = ""
    residual: float = 0.0
    meta: dict = field(default_factory=dict)


def _y_event():
    def ev(t, y, mu):
        return y[1]
    ev.terminal = False
    ev.direction = -1.0
    return ev


def correct_halo(mu: float, guess, period_guess: float,
                 tol: float = 1e-11, max_iter: int = 80) -> HaloOrbit:
    """Correct a halo guess [x0,0,z0,0,vy0,0]; vary {x0, vy0} with z0 fixed."""
    s = np.asarray(guess, dtype=float).copy()
    ev = _y_event()
    max_t = 1.4 * period_guess
    t_min = 0.25 * period_guess

    for _ in range(max_iter):
        y0 = np.concatenate([s, np.eye(6).ravel()])
        sol = solve_ivp(eom_stm, (0.0, max_t), y0, args=(mu,), events=ev, **_INT)
        tev, yev = sol.t_events[0], sol.y_events[0]
        mask = tev > t_min
        if not mask.any():
            max_t *= 1.6
            continue
        k = int(np.argmax(mask))
        te, ye = tev[k], yev[k]
        sc = ye[:6]
        Phi = ye[6:].reshape(6, 6)

        vx_f, vz_f = sc[3], sc[5]
        if max(abs(vx_f), abs(vz_f)) < tol:
            return HaloOrbit(s0=s.copy(), period=2.0 * te,
                             jacobi=float(jacobi_constant(s, mu)),
                             z_amplitude=float(abs(s[2])),
                             residual=float(max(abs(vx_f), abs(vz_f))))

        sd = eom(te, sc, mu)
        ax, az, ydot = sd[3], sd[5], sc[4]
        # free vars [x0, vy0] = cols [0,4]; constraints [vx_f, vz_f] = rows [3,5];
        # eliminate dt via the y=0 condition (row 1)
        M = np.array([
            [Phi[3, 0] - ax * Phi[1, 0] / ydot, Phi[3, 4] - ax * Phi[1, 4] / ydot],
            [Phi[5, 0] - az * Phi[1, 0] / ydot, Phi[5, 4] - az * Phi[1, 4] / ydot],
        ])
        dx0, dvy0 = np.linalg.solve(M, -np.array([vx_f, vz_f]))
        s[0] += dx0
        s[4] += dvy0

    raise RuntimeError("Halo corrector failed to converge")


def halo_family(mu: float, point: str = "L1", n: int = 20, dz: float = 2e-3,
                fam_n: int = 60, lyap_amp0: float = 1e-3,
                lyap_dx: float = 2e-3) -> list[HaloOrbit]:
    """Generate a halo family by continuation in z-amplitude from the Lyapunov
    bifurcation (where halos branch from the planar family).

    For tight systems (small mu, e.g. Sun-Earth) the libration point sits very
    close to the secondary, so lyap_amp0/lyap_dx/dz must be much smaller than the
    Earth-Moon defaults (otherwise the seed is in the nonlinear regime).
    """
    lyap = lyapunov_family(mu, point, amplitude0=lyap_amp0, dx=lyap_dx, n=fam_n)
    bif = find_halo_bifurcation(lyap)
    if bif is None:
        raise RuntimeError("no halo bifurcation found in the Lyapunov family")
    # nearest Lyapunov member to the bifurcation amplitude -> planar seed
    seed_member = min(lyap, key=lambda m: abs(m.amplitude - bif["amplitude"]))
    x0 = float(seed_member.orbit.s0[0])
    vy0 = float(seed_member.orbit.s0[4])
    period = seed_member.orbit.period

    halos: list[HaloOrbit] = []
    for i in range(1, n + 1):
        z0 = i * dz
        guess = np.array([x0, 0.0, z0, 0.0, vy0, 0.0])
        try:
            h = correct_halo(mu, guess, period)
        except RuntimeError:
            break
        h.point = point
        halos.append(h)
        x0, vy0, period = float(h.s0[0]), float(h.s0[4]), h.period
    return halos
