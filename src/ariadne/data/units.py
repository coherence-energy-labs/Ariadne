"""Nondimensional <-> SI conversions. This module owns ALL unit conversions
(MASTER_PLAN.md §5.3). Do not scatter conversion factors elsewhere.

State vectors are length-6: [x, y, z, vx, vy, vz].
Nondimensional position is in units of L*, velocity in units of V*, time in T*.
"""
from __future__ import annotations

import numpy as np

from .constants import System


def state_to_si(state_nd, system: System) -> np.ndarray:
    """Nondimensional state -> SI [km, km/s]."""
    s = np.asarray(state_nd, dtype=float)
    out = np.empty_like(s)
    out[..., :3] = s[..., :3] * system.L_star
    out[..., 3:] = s[..., 3:] * system.V_star
    return out


def state_to_nd(state_si, system: System) -> np.ndarray:
    """SI state [km, km/s] -> nondimensional."""
    s = np.asarray(state_si, dtype=float)
    out = np.empty_like(s)
    out[..., :3] = s[..., :3] / system.L_star
    out[..., 3:] = s[..., 3:] / system.V_star
    return out


def time_to_si(t_nd, system: System):
    """Nondimensional time -> seconds."""
    return np.asarray(t_nd, dtype=float) * system.T_star


def time_to_nd(t_si, system: System):
    """Seconds -> nondimensional time."""
    return np.asarray(t_si, dtype=float) / system.T_star


def days_to_nd(days, system: System):
    """Days -> nondimensional time."""
    return np.asarray(days, dtype=float) * 86400.0 / system.T_star


def nd_to_days(t_nd, system: System):
    """Nondimensional time -> days."""
    return np.asarray(t_nd, dtype=float) * system.T_star / 86400.0
