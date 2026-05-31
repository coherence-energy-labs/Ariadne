"""Broker subscribers for live transient alert streams.

Surveys publish single-epoch detections via brokers; only a tiny fraction are moving
objects, and only a tiny fraction of those are new. This subpackage handles the
broker-API plumbing so the rest of Ariadne (linkage + IOD + LM + SkyBoT cross-match)
can operate on the same Alert/Tracklet schema regardless of which survey provided the
data.

Supported brokers (and the source surveys they currently aggregate):
  - ALeRCE (Chile, public API, no auth): Zwicky Transient Facility (ZTF) alerts.
  - ANTARES (NOIRLab, requires token): ZTF + ATLAS, future LSST.
  - Lasair (Edinburgh, requires API key): ZTF, future LSST.

Honest scope:
  - Brokers carry transient alerts -- single-epoch source detections with position +
    magnitude + ID. They do NOT pre-link tracklets (that's our job).
  - Most alerts are real (sub-)transients (supernovae, AGN flares, dwarf novae,
    cataclysmic variables), not solar-system objects.
  - The moving-object yield is small (~0.1-1% of alerts depending on filter cuts).
  - Ariadne's role is the filter that turns 1M nightly alerts into a handful of
    real moving-object candidates worth following up.
"""
from __future__ import annotations

from .base import Alert, BrokerError, BrokerBase, collect, synthesise_alerts

__all__ = ["Alert", "BrokerError", "BrokerBase", "collect", "synthesise_alerts"]
