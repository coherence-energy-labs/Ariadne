"""Broker abstraction -- one Alert schema regardless of which survey/broker is upstream."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator
import math


@dataclass(frozen=True)
class Alert:
    """A single survey detection (single epoch, single source).

    Fields:
      survey:   "ZTF", "ATLAS", "LSST", ...
      alert_id: broker's unique ID for this single detection
      obj_id:   the broker's source-cluster ID (may aggregate multiple alerts of one object)
      mjd:      Modified Julian Date of the detection
      ra:       right ascension (degrees, J2000)
      dec:      declination (degrees, J2000)
      mag:      apparent magnitude (-99 if unavailable)
      band:     filter band string ("g", "r", "i", ...)
      meta:     broker-specific extras (e.g. flagged "stellar" / "moving", crossmatch hints)
    """
    survey: str
    alert_id: str
    obj_id: str
    mjd: float
    ra: float
    dec: float
    mag: float = -99.0
    band: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def jd(self) -> float:
        return self.mjd + 2400000.5

    @property
    def ra_rad(self) -> float:
        return math.radians(self.ra)

    @property
    def dec_rad(self) -> float:
        return math.radians(self.dec)


class BrokerError(RuntimeError):
    """Raised when a broker query fails (network, auth, throttling, malformed response)."""


class BrokerBase:
    """Base class -- implement `query_box` to wire a new broker."""

    name: str = "unknown"

    def query_box(self, ra_min: float, ra_max: float,
                  dec_min: float, dec_max: float,
                  mjd_start: float, mjd_end: float,
                  max_alerts: int = 10000) -> Iterator[Alert]:
        """Yield Alerts in the box [ra, dec] x [mjd_start, mjd_end].

        RA / Dec in degrees J2000, MJD as TT or whatever the broker reports natively.
        Implementations should respect `max_alerts` to avoid request-storms.
        """
        raise NotImplementedError("subclasses implement query_box")

    def query_cone(self, ra: float, dec: float, radius_deg: float,
                   mjd_start: float, mjd_end: float,
                   max_alerts: int = 10000) -> Iterator[Alert]:
        """Yield Alerts in a cone (RA, Dec, radius) x [mjd_start, mjd_end].

        Default implementation queries a bounding box and filters by angular distance --
        subclasses with native cone search should override.
        """
        d = radius_deg
        for a in self.query_box(ra - d / math.cos(math.radians(dec)),
                                ra + d / math.cos(math.radians(dec)),
                                dec - d, dec + d, mjd_start, mjd_end, max_alerts):
            # Haversine angular distance (small-angle adequate for d << 1 rad)
            dr = math.radians(a.ra - ra)
            dd = math.radians(a.dec - dec)
            sep = math.degrees(math.hypot(dr * math.cos(a.dec_rad), dd))
            if sep <= radius_deg:
                yield a


def collect(stream: Iterable[Alert], max_n: int | None = None) -> list[Alert]:
    """Drain an Alert iterator into a list, optionally capped at max_n."""
    out = []
    for i, a in enumerate(stream):
        if max_n is not None and i >= max_n:
            break
        out.append(a)
    return out


def synthesise_keplerian_alerts(orbits: list[dict],
                                 n_interlopers: int = 200,
                                 mjd_nights: list[float] | None = None,
                                 seed: int = 0) -> list[Alert]:
    """Generate alerts with REAL Keplerian dynamics -- the alerts of planted
    objects will pass the IOD+LM filter because they're consistent with a
    heliocentric orbit. Use this to validate the full pipeline acceptance path.

    `orbits` is a list of dicts with at least {a_au, e, i, Omega, omega}; the
    objects are propagated via the standard Ariadne integrator and converted to
    geocentric (RA, Dec) at each image epoch.
    """
    import random
    import numpy as np
    from ...data.ephemeris import body_state
    from ...dynamics.secular import kepler_step, elements_to_state
    from ...data.constants import GM_SUN

    rng = random.Random(seed)
    if mjd_nights is None:
        mjd_nights = [60000.0, 60003.0, 60006.0]
    SEC = 86400.0
    alerts: list[Alert] = []
    for k, o in enumerate(orbits):
        r0, v0 = elements_to_state(o["a_au"], o["e"], o["i"], o["Omega"],
                                    o["omega"], o.get("M", 0.0))
        r0 = np.asarray(r0); v0 = np.asarray(v0)
        for n_idx, mjd0 in enumerate(mjd_nights):
            for half in (0.0, 2.0):
                t = mjd0 + half / 24.0
                et = (t + 2400000.5 - 2451545.0) * SEC
                # propagate from epoch 0 (which we treat as MJD = mjd_nights[0])
                dt = (t - mjd_nights[0]) * SEC
                rt, _ = kepler_step(r0, v0, GM_SUN, dt)
                R_e = body_state("EARTH", et, "J2000", "SUN")[:3]
                geo = rt - R_e
                rho = float(np.linalg.norm(geo))
                ra = math.degrees(math.atan2(geo[1], geo[0])) % 360.0
                dec = math.degrees(math.asin(geo[2] / rho))
                alerts.append(Alert(
                    survey="SYNTH-KEPLER",
                    alert_id=f"obj{k}_n{n_idx}_d{half:.0f}",
                    obj_id=f"obj{k}",
                    mjd=t, ra=ra, dec=dec, mag=20.0, band="g",
                    meta={"synthetic": True, "kepler": True,
                          "planted_obj": f"obj{k}", "a_au": o["a_au"]},
                ))
    # Random interlopers in a wide sky box around the planted objects' centroid
    if alerts and n_interlopers > 0:
        ra_c = sum(a.ra for a in alerts) / len(alerts)
        dec_c = sum(a.dec for a in alerts) / len(alerts)
        for j in range(n_interlopers):
            mjd0 = rng.choice(mjd_nights)
            alerts.append(Alert(
                survey="SYNTH-KEPLER", alert_id=f"int{j}", obj_id=f"int{j}",
                mjd=mjd0 + rng.uniform(0, 0.1),
                ra=ra_c + (rng.random() - 0.5) * 5.0,
                dec=dec_c + (rng.random() - 0.5) * 5.0,
                mag=21.0, band="g",
                meta={"synthetic": True, "planted_obj": None},
            ))
    rng.shuffle(alerts)
    return alerts


def synthesise_alerts(n_real_objects: int = 3, n_interlopers: int = 200,
                      ra_center: float = 180.0, dec_center: float = 0.0,
                      width_deg: float = 5.0,
                      mjd_nights: list[float] | None = None,
                      rate_arcsec_hr: tuple[float, float] = (0.3, 3.0),
                      seed: int = 0) -> list[Alert]:
    """Synthetic broker alerts for end-to-end testing without a live broker.

    Plants `n_real_objects` moving sources (each detected twice per night, 2-hour
    pair spacing, across 3 nights with 3-day cadence by default), sprinkled in
    `n_interlopers` random one-off detections. Designed to drive the realtime
    pipeline through its full path (cluster -> tracklet -> IOD+LM -> SkyBoT) and
    verify the planted objects survive the filter and the interlopers don't.
    """
    import random
    rng = random.Random(seed)
    if mjd_nights is None:
        mjd_nights = [60000.0, 60003.0, 60006.0]
    alerts: list[Alert] = []
    for k in range(n_real_objects):
        ra0 = ra_center + (rng.random() - 0.5) * width_deg
        dec0 = dec_center + (rng.random() - 0.5) * width_deg
        rate = rng.uniform(*rate_arcsec_hr)
        theta = rng.uniform(0, 2 * math.pi)
        d_ra_per_day = (rate * math.cos(theta) * 24.0 / 3600.0) / math.cos(math.radians(dec0))
        d_dec_per_day = (rate * math.sin(theta) * 24.0 / 3600.0)
        for n_idx, mjd0 in enumerate(mjd_nights):
            for half in (0.0, 2.0):
                t = mjd0 + half / 24.0
                d_t = t - mjd_nights[0]
                alerts.append(Alert(
                    survey="SYNTH",
                    alert_id=f"obj{k}_n{n_idx}_d{half:.0f}",
                    obj_id=f"obj{k}",
                    mjd=t,
                    ra=ra0 + d_ra_per_day * d_t,
                    dec=dec0 + d_dec_per_day * d_t,
                    mag=20.0, band="g",
                    meta={"synthetic": True, "planted_obj": f"obj{k}"},
                ))
    for j in range(n_interlopers):
        mjd0 = rng.choice(mjd_nights)
        alerts.append(Alert(
            survey="SYNTH", alert_id=f"int{j}", obj_id=f"int{j}",
            mjd=mjd0 + rng.uniform(0, 0.1),
            ra=ra_center + (rng.random() - 0.5) * width_deg,
            dec=dec_center + (rng.random() - 0.5) * width_deg,
            mag=21.0, band="g",
            meta={"synthetic": True, "planted_obj": None},
        ))
    rng.shuffle(alerts)
    return alerts
