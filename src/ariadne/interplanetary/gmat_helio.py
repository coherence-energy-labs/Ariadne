"""Export an interplanetary transfer as a NASA GMAT script (Sun-centered flight path).

Takes an optimized heliocentric transfer (departure epoch + state from a Lambert solve) and
writes a GMAT mission script that propagates the spacecraft in a Sun-centered frame under solar
(and optional planetary) point-mass gravity, reporting the trajectory. Drop the .script into
GMAT (or run it headless via ariadne.io.gmat_export.run_with_gmat) to fly the route Ariadne found.
"""
from __future__ import annotations

import os

from ..data.ephemeris import utc

_TEMPLATE = """%% Ariadne interplanetary transfer -- Sun-centered (auto-generated)
Create Spacecraft Probe;
GMAT Probe.DateFormat = TAIModJulian;
GMAT Probe.CoordinateSystem = SunICRF;
GMAT Probe.DisplayStateType = Cartesian;
GMAT Probe.Epoch = '{epoch}';
GMAT Probe.X = {x};
GMAT Probe.Y = {y};
GMAT Probe.Z = {z};
GMAT Probe.VX = {vx};
GMAT Probe.VY = {vy};
GMAT Probe.VZ = {vz};

Create CoordinateSystem SunICRF;
GMAT SunICRF.Origin = Sun;
GMAT SunICRF.Axes = ICRF;

Create ForceModel Helio;
GMAT Helio.CentralBody = Sun;
GMAT Helio.PointMasses = {{Sun, Earth, Luna, Mars, Jupiter, Venus}};
GMAT Helio.Drag = None;
GMAT Helio.SRP = Off;

Create Propagator Prop;
GMAT Prop.FM = Helio;
GMAT Prop.Type = RungeKutta89;
GMAT Prop.InitialStepSize = 600;
GMAT Prop.Accuracy = 1e-11;

Create ReportFile Rpt;
GMAT Rpt.Filename = '{report}';
GMAT Rpt.WriteHeaders = true;
GMAT Rpt.Add = {{Probe.SunICRF.X, Probe.SunICRF.Y, Probe.SunICRF.Z, Probe.SunICRF.VX, Probe.SunICRF.VY, Probe.SunICRF.VZ}};

BeginMissionSequence;
Propagate Prop(Probe) {{Probe.ElapsedDays = {days}}};
"""


def export_transfer_gmat(path, transfer, name="ariadne_transfer", report_path=None):
    """Write a Sun-centered GMAT script for an optimized transfer dict (r1, v1, et_dep, tof_days)."""
    if report_path is None:
        report_path = os.path.splitext(path)[0] + ".report.txt"
    r1, v1 = transfer["r1"], transfer["v1"]
    # GMAT epoch as a calendar string is most robust; use UTC Gregorian via a comment + TAIModJulian
    # left as the ISO UTC for human reference; GMAT reads the numeric epoch we set below.
    epoch_iso = utc(transfer["et_dep"])
    # GMAT TAIModJulian: write the UTC Gregorian form GMAT accepts by switching DateFormat.
    text = _TEMPLATE.replace("TAIModJulian", "UTCGregorian").replace(
        "{epoch}", _gmat_gregorian(epoch_iso))
    text = text.format(x=r1[0], y=r1[1], z=r1[2], vx=v1[0], vy=v1[1], vz=v1[2],
                       report=report_path, days=transfer["tof_days"])
    with open(path, "w") as f:
        f.write(text)
    return path


def _gmat_gregorian(iso_utc: str) -> str:
    """Convert '2026-11-01T00:00:00.000' -> GMAT '01 Nov 2026 00:00:00.000'."""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    date, _, time = iso_utc.partition("T")
    y, m, d = date.split("-")
    time = (time or "00:00:00.000").rstrip("Z")
    if "." not in time:
        time += ".000"
    return f"{int(d):02d} {months[int(m) - 1]} {y} {time}"
