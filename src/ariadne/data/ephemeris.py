"""Thin wrappers over SPICE for real ephemeris data (MASTER_PLAN.md §5).

All states are SI: km and km/s, in the J2000/ICRF inertial frame by default.
Time is ephemeris time (ET/TDB seconds past J2000).
"""
from __future__ import annotations

import numpy as np

from .kernels import furnish


def _sp():
    furnish()
    import spiceypy as sp
    return sp


def et(utc: str) -> float:
    """UTC string -> ephemeris time (seconds past J2000 TDB)."""
    return _sp().str2et(utc)


def utc(et_seconds: float, fmt: str = "ISOC", prec: int = 3) -> str:
    """Ephemeris time -> UTC calendar string."""
    return _sp().et2utc(et_seconds, fmt, prec)


def body_state(target: str, et_seconds: float, frame: str = "J2000",
               center: str = "EARTH", abcorr: str = "NONE") -> np.ndarray:
    """State [x,y,z,vx,vy,vz] (km, km/s) of target relative to center."""
    st, _ = _sp().spkezr(target, et_seconds, frame, abcorr, center)
    return np.asarray(st, float)


def body_pos(target: str, et_seconds: float, frame: str = "J2000",
             center: str = "EARTH", abcorr: str = "NONE") -> np.ndarray:
    """Position [x,y,z] (km) of target relative to center."""
    p, _ = _sp().spkpos(target, et_seconds, frame, abcorr, center)
    return np.asarray(p, float)


def body_gm(body: str) -> float:
    """Gravitational parameter GM (km^3/s^2) from the furnished GM kernel."""
    _, val = _sp().bodvrd(body, "GM", 1)
    return float(val[0])
