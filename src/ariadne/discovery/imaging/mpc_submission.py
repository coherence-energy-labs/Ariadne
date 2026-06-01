"""MPC submission format generators.

The Minor Planet Center accepts astrometric observations in two formats:

  ADES (Astrometry Data Exchange Standard, XML)
    https://minorplanetcenter.net/iau/info/ADES.html
    Modern, machine-readable, includes uncertainties + photometry +
    program metadata. THE format MPC asks for new submissions.

  80-column (legacy)
    https://minorplanetcenter.net/iau/info/OpticalObs.html
    Fixed-column ASCII, one observation per line, no uncertainties.
    Still accepted for backward compatibility.

This module generates BOTH formats from a chain's per-detection records,
plus the quality-A gate that decides whether a chain is submission-
ready.

Public API:
  build_ades_submission(chain, observatory_code, ...) -> str (XML)
  build_80col_submission(chain, observatory_code, ...) -> str
  evaluate_grade_a(chain) -> GradeAResult
    Check the chain against MPC grade-A criteria: arc days, n_obs,
    astrometric residual, photometric uncertainty.
  write_submission_packet(chain, out_dir, designation_hint='')
    Write both formats to disk in a packet ready to email to MPC.

Grade-A criteria (per IAU/MPC standards):
  - arc length >= 2 days
  - >= 3 distinct epochs
  - astrometric RMS residual <= 0.3 arcsec
  - photometric uncertainty <= 0.5 mag
  - >= 6 total observations
  - orbital fit converged

We DO NOT auto-submit. The packet is written to disk; a human reviewer
inspects it and submits via email or web form.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence


# IAU observatory codes for common DECam-using observatories
OBSERVATORY_CODES = {
    "ctio_decam":  "W84",   # CTIO 4-m + DECam
    "ctio":         "807",  # Cerro Tololo Inter-American Observatory
    "kitt_peak":    "695",  # Kitt Peak / Mayall 4-m
    "subaru":       "T09",
    "lsst":         "X05",  # Rubin / LSST commissioning code
    "ps1":          "F51",
    "atlas":        "T05",
}


@dataclass
class GradeAResult:
    """Outcome of the MPC grade-A QC gate."""
    passed: bool
    n_observations: int
    n_distinct_epochs: int
    arc_days: float
    astrometric_rms_arcsec: float
    photometric_rms_mag: float
    iod_converged: bool
    reasons: list[str] = field(default_factory=list)


def evaluate_grade_a(chain: dict,
                      detections: Sequence[dict],
                      *,
                      min_arc_days: float = 2.0,
                      min_distinct_epochs: int = 3,
                      max_astrom_rms_arcsec: float = 0.3,
                      max_photom_rms_mag: float = 0.5,
                      min_observations: int = 6,
                      ) -> GradeAResult:
    """Decide whether `chain` (with its `detections`) qualifies for MPC
    grade-A submission.

    `chain` is a dict from the detection_db chains row.
    `detections` is a list of detection dicts attached to the chain.
    """
    reasons: list[str] = []
    n_obs = len(detections)
    if n_obs < min_observations:
        reasons.append(f"n_obs {n_obs} < {min_observations}")
    distinct_epochs = len({int(d.get("mjd", 0) * 24 * 4)  # 15-min bins
                              for d in detections})
    if distinct_epochs < min_distinct_epochs:
        reasons.append(f"epochs {distinct_epochs} < {min_distinct_epochs}")
    arc_days = float(chain.get("arc_days") or 0.0)
    if arc_days < min_arc_days:
        reasons.append(f"arc {arc_days:.2f}d < {min_arc_days}d")
    # Per-detection astrometric sigma; report the median
    astrom_sigmas = [float(d.get("astrom_sigma_arcsec", 0.5))
                       for d in detections
                       if d.get("astrom_sigma_arcsec") is not None]
    astrom_rms = (float(sum(s ** 2 for s in astrom_sigmas) / len(astrom_sigmas))
                    ** 0.5 if astrom_sigmas else 999.0)
    if astrom_rms > max_astrom_rms_arcsec:
        reasons.append(f"astrom_rms {astrom_rms:.3f}\" > "
                          f"{max_astrom_rms_arcsec}\"")
    # Photometric scatter
    mags = [float(d.get("mag", -99.0)) for d in detections
              if d.get("mag", -99) > -50]
    if len(mags) >= 2:
        mean_mag = sum(mags) / len(mags)
        photom_rms = (sum((m - mean_mag) ** 2 for m in mags) / len(mags)) ** 0.5
    else:
        photom_rms = 999.0
    if photom_rms > max_photom_rms_mag:
        reasons.append(f"photom_rms {photom_rms:.3f} > {max_photom_rms_mag}")
    iod_converged = (chain.get("iod_strategy") and
                       chain.get("iod_rms_arcsec", float("inf")) < 1.0)
    if not iod_converged:
        reasons.append(f"iod not converged "
                          f"(rms={chain.get('iod_rms_arcsec')})")
    passed = (n_obs >= min_observations
                and distinct_epochs >= min_distinct_epochs
                and arc_days >= min_arc_days
                and astrom_rms <= max_astrom_rms_arcsec
                and photom_rms <= max_photom_rms_mag
                and iod_converged)
    return GradeAResult(
        passed=passed,
        n_observations=n_obs,
        n_distinct_epochs=distinct_epochs,
        arc_days=arc_days,
        astrometric_rms_arcsec=astrom_rms,
        photometric_rms_mag=photom_rms,
        iod_converged=bool(iod_converged),
        reasons=reasons,
    )


def _mjd_to_iso_datetime(mjd: float) -> str:
    """Convert MJD to ISO-8601 with sub-second precision."""
    try:
        from astropy.time import Time
        t = Time(mjd, format="mjd", scale="utc")
        return t.isot   # 'YYYY-MM-DDTHH:MM:SS.sss'
    except Exception:
        # Fallback: rough conversion
        from datetime import datetime, timedelta
        ref = datetime(1858, 11, 17)   # MJD 0
        d = ref + timedelta(days=mjd)
        return d.isoformat(timespec="seconds")


def _ra_deg_to_hms_string(ra_deg: float) -> str:
    """Convert decimal degrees to HH MM SS.sss for 80-col format."""
    ra_h = (ra_deg % 360.0) / 15.0
    h = int(ra_h)
    m_f = (ra_h - h) * 60.0
    m = int(m_f)
    s = (m_f - m) * 60.0
    return f"{h:02d} {m:02d} {s:06.3f}"


def _dec_deg_to_dms_string(dec_deg: float) -> str:
    """Convert decimal degrees to +DD MM SS.ss for 80-col format."""
    sign = "+" if dec_deg >= 0 else "-"
    d = abs(dec_deg)
    deg = int(d)
    m_f = (d - deg) * 60.0
    m = int(m_f)
    s = (m_f - m) * 60.0
    return f"{sign}{deg:02d} {m:02d} {s:05.2f}"


def build_80col_submission(chain: dict,
                              detections: Sequence[dict],
                              *, observatory_code: str = "W84",
                              designation_hint: str = "",
                              ) -> str:
    """Build a legacy 80-column MPC submission for one chain.

    Returns a multi-line string (one detection per line). One line is
    80 columns + newline per the spec.

    Designation: if `designation_hint` is empty, use a placeholder
    'ARIxxxx' (MPC assigns the real designation on receipt).
    """
    lines = []
    desig = (designation_hint.strip()[:7].ljust(7)
               if designation_hint else "ARI0001")
    for det in sorted(detections, key=lambda d: d["mjd"]):
        mjd = float(det["mjd"])
        # MPC 80-col date is fractional day in 'YYYY MM DD.dddddd'
        from astropy.time import Time
        t = Time(mjd, format="mjd", scale="utc")
        dt_str = t.iso        # 'YYYY-MM-DD HH:MM:SS.sss'
        ymd = dt_str.split()[0].split("-")
        frac_day = (mjd - int(mjd))   # 0..1
        date_str = f"{ymd[0]} {ymd[1]} {int(ymd[2]):02d}.{int(frac_day * 1e6):06d}"
        # Position
        ra_str = _ra_deg_to_hms_string(float(det["ra"]))
        dec_str = _dec_deg_to_dms_string(float(det["dec"]))
        # Magnitude + band
        mag = float(det.get("mag", -99.0))
        mag_str = f"{mag:5.2f} r" if mag > -50 else "       "
        # Build the 80-col line by position. Each part has a fixed width.
        parts = [
            "     ",                    # cols 1-5   packed minor planet number
            f"{desig:7s}",              # cols 6-12  provisional designation
            "  ",                       # cols 13-14 disc asterisk + note
            "C",                        # col 15     obs code (C = CCD)
            f"{date_str:>17s}",         # cols 16-32 date (17 chars)
            f"{ra_str:>12s}",           # cols 33-44 RA
            f"{dec_str:>12s}",          # cols 45-56 Dec
            "         ",                # cols 57-65 reserved
            f"{mag_str:7s}",            # cols 66-72 mag + band
            "     ",                    # cols 73-77 reserved
            f"{observatory_code:>3s}",  # cols 78-80 observatory code
        ]
        line = "".join(parts)
        # Pad to exactly 80 chars (or truncate if over)
        if len(line) < 80:
            line = line + " " * (80 - len(line))
        elif len(line) > 80:
            line = line[:80]
        lines.append(line)
    return "\n".join(lines) + "\n"


def build_ades_submission(chain: dict,
                            detections: Sequence[dict],
                            *, observatory_code: str = "W84",
                            designation_hint: str = "",
                            program_name: str = "Ariadne",
                            submitter_name: str = "Ariadne pipeline",
                            ) -> str:
    """Build an ADES-XML submission for one chain.

    Returns the XML as a string. Conforms to the ADES XSD published
    at https://minorplanetcenter.net/iau/info/ADES.html (subset of
    fields).
    """
    desig = designation_hint.strip() or "ARI0001"
    xml = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<ades version="2017">',
             '  <obsContext>',
             '    <observatory>',
             f'      <mpcCode>{observatory_code}</mpcCode>',
             '    </observatory>',
             '    <submitter>',
             f'      <name>{submitter_name}</name>',
             '    </submitter>',
             '    <observers>',
             f'      <name>{submitter_name}</name>',
             '    </observers>',
             '    <measurers>',
             f'      <name>{submitter_name}</name>',
             '    </measurers>',
             '    <telescope>',
             '      <design>Reflector</design>',
             '      <aperture>4.0</aperture>',
             '      <detector>CCD</detector>',
             '    </telescope>',
             '    <software>',
             f'      <name>{program_name}</name>',
             '    </software>',
             '  </obsContext>',
             '  <obsData>',
            ]
    for det in sorted(detections, key=lambda d: d["mjd"]):
        mjd = float(det["mjd"])
        iso_t = _mjd_to_iso_datetime(mjd)
        ra = float(det["ra"])
        dec = float(det["dec"])
        mag = float(det.get("mag", -99.0))
        astrom_sigma = float(det.get("astrom_sigma_arcsec", 0.1))
        xml.extend([
            '    <optical>',
            f'      <permID></permID>',
            f'      <provID>{desig}</provID>',
            f'      <obsTime>{iso_t}Z</obsTime>',
            f'      <ra>{ra:.6f}</ra>',
            f'      <dec>{dec:+.6f}</dec>',
            f'      <rmsRA>{astrom_sigma:.3f}</rmsRA>',
            f'      <rmsDec>{astrom_sigma:.3f}</rmsDec>',
        ])
        if mag > -50:
            xml.extend([
                f'      <mag>{mag:.2f}</mag>',
                f'      <band>r</band>',
                f'      <rmsMag>0.05</rmsMag>',
            ])
        xml.extend([
            f'      <stn>{observatory_code}</stn>',
            f'      <mode>CCD</mode>',
            '    </optical>',
        ])
    xml.append('  </obsData>')
    xml.append('</ades>')
    return "\n".join(xml) + "\n"


def write_submission_packet(chain: dict,
                              detections: Sequence[dict],
                              out_dir: str | Path,
                              *, observatory_code: str = "W84",
                              designation_hint: str = "",
                              ) -> dict:
    """Write a submission packet (ADES + 80-col + grade-A report) for
    one chain. Returns dict of {file_kind: path}.

    The packet is INTENDED FOR HUMAN REVIEW before submission to MPC.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    chain_id = chain.get("id", 0)
    desig = designation_hint or f"ARI{chain_id:04d}"
    grade = evaluate_grade_a(chain, detections)
    files = {}
    # ADES XML
    ades_xml = build_ades_submission(
        chain, detections, observatory_code=observatory_code,
        designation_hint=desig)
    ades_path = out_dir / f"{desig}.ades.xml"
    ades_path.write_text(ades_xml, encoding="utf-8")
    files["ades"] = ades_path
    # 80-col
    eightycol = build_80col_submission(
        chain, detections, observatory_code=observatory_code,
        designation_hint=desig)
    eightycol_path = out_dir / f"{desig}.80col.txt"
    eightycol_path.write_text(eightycol, encoding="utf-8")
    files["80col"] = eightycol_path
    # Grade-A report
    grade_lines = [
        f"Chain ID: {chain_id}",
        f"Designation hint: {desig}",
        f"Status: {'GRADE A (submit)' if grade.passed else 'NOT GRADE A (hold)'}",
        f"n_observations: {grade.n_observations}",
        f"n_distinct_epochs: {grade.n_distinct_epochs}",
        f"arc_days: {grade.arc_days:.2f}",
        f"astrometric_rms_arcsec: {grade.astrometric_rms_arcsec:.3f}",
        f"photometric_rms_mag: {grade.photometric_rms_mag:.3f}",
        f"iod_converged: {grade.iod_converged}",
    ]
    if grade.reasons:
        grade_lines.append("Failure reasons:")
        grade_lines.extend(f"  - {r}" for r in grade.reasons)
    grade_path = out_dir / f"{desig}.grade.txt"
    grade_path.write_text("\n".join(grade_lines) + "\n",
                             encoding="utf-8")
    files["grade"] = grade_path
    return files
