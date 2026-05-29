"""Synodic (CR3BP nondimensional) <-> inertial (Earth-centered J2000) frame transform
(MASTER_PLAN.md - Stage 18).

To re-converge a CR3BP route in the real ephemeris we must embed nondimensional rotating-
frame states as dimensional inertial states at an epoch. We use the INSTANTANEOUS pulsating
synodic frame defined by the real Earth->Moon ephemeris state at time `et`:

    d      = |r_moon|                      (instantaneous Earth-Moon distance, the length unit)
    e1     = r_moon / d                     (toward the Moon)
    n      = (r_moon x v_moon)/|...|        (orbit-plane normal)
    e2     = n x e1
    omega  = |r_moon x v_moon| / d^2        (instantaneous angular rate)
    V*     = d * omega                       (velocity unit)

A nondimensional CR3BP state s = (x,y,z,vx,vy,vz) has its origin at the barycenter, with the
Earth at x = -mu. The Earth-centered inertial state is

    R = d * ( (x+mu) e1 + y e2 + z n )
    V = V* ( vx e1 + vy e2 + vz n ) + omega_vec x R          (omega_vec = omega n)

This transform is EXACTLY invertible (pure linear algebra), so `inertial_to_synodic` is its
exact inverse. Physically it maps the Moon's nondim point (1-mu,0,0) to the real Moon position
exactly; the Moon's velocity is reproduced only up to the radial (pulsation) component, which
is the standard circular-frame approximation when seeding an ephemeris correction.
"""
from __future__ import annotations

import numpy as np

from ..data.ephemeris import body_state


def synodic_frame(epoch_et: float) -> dict:
    """Instantaneous pulsating-synodic frame from the real Earth->Moon state at `epoch_et`."""
    st = body_state("MOON", epoch_et, "J2000", "EARTH")
    rm, vm = st[:3], st[3:]
    d = float(np.linalg.norm(rm))
    h = np.cross(rm, vm)
    omega = float(np.linalg.norm(h)) / (d * d)
    e1 = rm / d
    n = h / np.linalg.norm(h)
    e2 = np.cross(n, e1)
    return {"d": d, "omega": omega, "v_star": d * omega,
            "e1": e1, "e2": e2, "n": n, "omega_vec": omega * n}


def synodic_to_inertial(state_nd, epoch_et: float, mu: float, frame: dict | None = None):
    """Nondimensional synodic state -> Earth-centered J2000 inertial state (km, km/s)."""
    f = frame or synodic_frame(epoch_et)
    s = np.asarray(state_nd, float)
    d, vstar = f["d"], f["v_star"]
    e1, e2, n, om = f["e1"], f["e2"], f["n"], f["omega_vec"]
    rnd = np.array([s[0] + mu, s[1], s[2]])            # Earth-centered, nondim
    R = d * (rnd[0] * e1 + rnd[1] * e2 + rnd[2] * n)
    Vsyn = vstar * (s[3] * e1 + s[4] * e2 + s[5] * n)
    V = Vsyn + np.cross(om, R)
    return np.concatenate([R, V])


def inertial_to_synodic(state_inertial, epoch_et: float, mu: float, frame: dict | None = None):
    """Earth-centered J2000 inertial state -> nondimensional synodic state (exact inverse)."""
    f = frame or synodic_frame(epoch_et)
    S = np.asarray(state_inertial, float)
    d, vstar = f["d"], f["v_star"]
    e1, e2, n, om = f["e1"], f["e2"], f["n"], f["omega_vec"]
    R, V = S[:3], S[3:]
    x = R @ e1 / d - mu
    y = R @ e2 / d
    z = R @ n / d
    Vsyn = V - np.cross(om, R)
    vx = Vsyn @ e1 / vstar
    vy = Vsyn @ e2 / vstar
    vz = Vsyn @ n / vstar
    return np.array([x, y, z, vx, vy, vz])
