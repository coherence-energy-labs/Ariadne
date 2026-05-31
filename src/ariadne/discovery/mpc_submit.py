"""MPC 80-column astrometric submission format -- the discovery loop closer.

When a candidate is confirmed (multiple nights, low RMS, no SkyBoT match), we
need to publish it. The Minor Planet Center accepts batched astrometry in a
fixed 80-character record format (Oort 1947, formalised in the 1990s). This
module emits compliant records straight from our internal Alert/Candidate
representation, plus a header block + observatory record.

Reference: https://www.minorplanetcenter.net/iau/info/OpticalObs.html

Column layout (CCD optical, the only mode we emit):
   1- 5   Permanent number (blank for new/unnumbered)
   6-12   Provisional/temporary designation
  13      Discovery asterisk ("*" on the first record of a discovery)
  14      Note 1 (blank for our reports)
  15      Note 2 ("C" = CCD)
  16-32   Date YYYY MM DD.dddddd
  33-44   RA  HH MM SS.ddd
  45-56   Dec sDD MM SS.dd
  57-65   blank
  66-70   Magnitude (NN.N or blank)
  71      Band character (V, R, g, r, i, z, ...)
  72-77   Reference / catalogue code (blank for first report)
  78-80   Observatory code (3 alphanumerics)

We pack our internal canonical-key into a 7-character TEMPORARY designation
(`~AAAAA` where AAAAA is the first 5 chars of the canonical key hash). This is
NOT a real MPC provisional -- it is a *placeholder* that lets the MPC pipeline
group the night's astrometry; the MPC then assigns a real provisional like
`2026 AA1` on receipt. Use the field for internal traceability only.

This module is FORMAT ONLY -- it does not submit. Submission is by email to
obs@minorplanetcenter.net per MPC policy (which requires a credentialed
observer). The output here is what you paste into that email.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timezone

from .brokers.base import Alert
from .operations.candidate_store import Candidate


@dataclass
class MPCHeader:
    """The free-form header that precedes the astrometric block in an MPC email.

    Required keys (MPC will reject without them):
      COD: observatory code (3 chars). 500 = geocenter; I41 = Palomar; W84 = CTIO/DECam.
      CON: contact "Name, Institution, Address, EMAIL"
      OBS: observer name(s), comma-separated
      MEA: measurer name(s)
      TEL: telescope description "0.5-m f/8 reflector + CCD"
      ACK: acknowledgement keyword to be quoted in MPC's reply
      AC2: an email address to receive the acknowledgement
    """
    observatory_code: str
    contact: str
    observers: str
    measurers: str
    telescope: str
    ack_keyword: str
    ack_email: str
    network: str = "ATLAS"
    band_catalogue: str = "Gaia2"
    comment: str | None = None


def _pack_temp_designation(canonical_key: str) -> str:
    """Pack our 25-char internal canonical key into a 7-char temporary tag.

    Format: `~AAAAA` (tilde + 5 base-36 hash characters) -- 6 chars padded to 7.
    Tilde prefix is reserved by the MPC for non-conformant private designations.
    """
    h = hashlib.blake2b(canonical_key.encode(), digest_size=4).hexdigest().upper()
    return f"~{h[:5]:>5s} "        # 7 chars: tilde + 5 hex + 1 space


def _format_date(mjd: float) -> str:
    """MJD -> 'YYYY MM DD.dddddd' (17 chars exactly)."""
    jd = mjd + 2400000.5
    j = int(jd + 0.5)
    f = jd + 0.5 - j
    a = j + 32044
    b = (4 * a + 3) // 146097
    c = a - (146097 * b) // 4
    d = (4 * c + 3) // 1461
    e = c - (1461 * d) // 4
    m = (5 * e + 2) // 153
    day = e - (153 * m + 2) // 5 + 1
    month = m + 3 - 12 * (m // 10)
    year = 100 * b + d - 4800 + (m // 10)
    day_frac = day + f
    return f"{year:04d} {month:02d} {day_frac:09.6f}"            # 4+1+2+1+9 = 17


def _format_ra(ra_deg: float) -> str:
    """RA degrees -> 'HH MM SS.ddd' (12 chars exactly).

    Rolls over at second-and-minute boundaries to avoid invalid "60.000"
    output near tick boundaries (floating-point representation issue).
    """
    ra_h = (ra_deg % 360.0) / 15.0
    h = int(ra_h)
    rem = (ra_h - h) * 60.0
    m = int(rem)
    s = (rem - m) * 60.0
    if round(s, 3) >= 60.0:
        s = 0.0; m += 1
    if m >= 60:
        m = 0; h = (h + 1) % 24
    return f"{h:02d} {m:02d} {s:06.3f}"                           # 2+1+2+1+6 = 12


def _format_dec(dec_deg: float) -> str:
    """Dec degrees -> 'sDD MM SS.dd' (12 chars exactly).

    Rolls over at second-and-minute boundaries (see _format_ra).
    """
    sign = "-" if dec_deg < 0 else "+"
    dd = abs(dec_deg)
    d = int(dd)
    rem = (dd - d) * 60.0
    m = int(rem)
    s = (rem - m) * 60.0
    if round(s, 2) >= 60.0:
        s = 0.0; m += 1
    if m >= 60:
        m = 0; d += 1
    return f"{sign}{d:02d} {m:02d} {s:05.2f}"                     # 1+2+1+2+1+5 = 12


def format_record(*,
                  mjd: float, ra_deg: float, dec_deg: float,
                  mag: float | None = None, band: str = " ",
                  designation: str = "       ",
                  number: str = "     ",
                  discovery_asterisk: bool = False,
                  observatory_code: str = "500") -> str:
    """Emit one 80-column MPC optical-CCD astrometric record.

    Args:
      designation: 7-char temporary or provisional ID (cols 6-12).
      number:      5-char permanent number, "     " if unnumbered (cols 1-5).
      discovery_asterisk: True for the FIRST record of a new-discovery batch.

    Returns:
      Exactly 80 characters, no trailing newline.
    """
    if len(observatory_code) != 3:
        raise ValueError(f"observatory_code must be 3 chars, got {observatory_code!r}")
    if len(designation) > 7:
        raise ValueError(f"designation must be <=7 chars, got {designation!r}")
    if len(number) > 5:
        raise ValueError(f"number must be <=5 chars, got {number!r}")
    if band and len(band) > 1:
        band = band[0]

    num = f"{number:>5s}"
    des = f"{designation:<7s}"
    disc = "*" if discovery_asterisk else " "
    note1 = " "
    note2 = "C"                                            # CCD
    date_s = _format_date(mjd)                             # 17 chars
    ra_s = _format_ra(ra_deg)                              # 12 chars
    dec_s = _format_dec(dec_deg)                           # 12 chars
    blank9 = " " * 9                                       # cols 57-65
    if mag is not None and math.isfinite(mag) and -3 <= mag <= 30:
        mag_s = f"{mag:5.2f}"                              # cols 66-70 (5 chars)
        band_s = (band[0] if band else " ")                # col 71
    else:
        mag_s = "     "
        band_s = " "
    ref_s = "      "                                       # cols 72-77 (6 blanks)
    obs = observatory_code                                 # cols 78-80

    rec = (num + des + disc + note1 + note2 + date_s + ra_s + dec_s
           + blank9 + mag_s + band_s + ref_s + obs)
    if len(rec) != 80:
        # one-off pad/trim to the exact spec; surfaces formatting bugs in tests
        rec = (rec + " " * 80)[:80]
    return rec


def header_lines(header: MPCHeader) -> list[str]:
    """Emit the MPC header block (precedes the astrometric records)."""
    lines = [
        f"COD {header.observatory_code}",
        f"CON {header.contact}",
        f"OBS {header.observers}",
        f"MEA {header.measurers}",
        f"TEL {header.telescope}",
        f"NET {header.network}",
        f"BND {header.band_catalogue}",
        f"ACK {header.ack_keyword}",
        f"AC2 {header.ack_email}",
    ]
    if header.comment:
        lines.append(f"COM {header.comment}")
    return lines


def candidate_to_records(candidate: Candidate, alerts: list[Alert],
                         *, observatory_code: str = "500",
                         band: str = " ",
                         discovery: bool = True) -> list[str]:
    """Emit one 80-col record per Alert, all under the candidate's temp designation.

    Args:
      candidate: the stored Candidate (provides the canonical key -> temp designation).
      alerts:    chronologically-ordered detections to publish.
      observatory_code: 3-char MPC code (500=geocenter; I41=Palomar; W84=CTIO/DECam).
      band:      photometric band character ("g", "r", "i", "z", "V"...).
      discovery: True to add the discovery "*" on the FIRST record.

    Returns:
      List of 80-col strings, one per alert.
    """
    if not alerts:
        return []
    des = _pack_temp_designation(candidate.key)
    sorted_alerts = sorted(alerts, key=lambda a: a.mjd)
    recs = []
    for i, a in enumerate(sorted_alerts):
        rec = format_record(
            mjd=a.mjd, ra_deg=a.ra, dec_deg=a.dec,
            mag=a.mag if a.mag and -3 <= a.mag <= 30 else None,
            band=(a.band[0] if a.band else band),
            designation=des,
            discovery_asterisk=(discovery and i == 0),
            observatory_code=observatory_code,
        )
        recs.append(rec)
    return recs


def emit_submission(header: MPCHeader,
                    candidates_with_alerts: list[tuple[Candidate, list[Alert]]],
                    *, observatory_code: str | None = None,
                    band: str = " ") -> str:
    """Emit a complete MPC-format submission text (header + all records).

    Paste the returned text into an email body to obs@minorplanetcenter.net.

    Args:
      header: MPCHeader fully populated.
      candidates_with_alerts: list of (candidate, [alerts]) pairs, one per object.
      observatory_code: override; defaults to header.observatory_code.
      band: per-detection band character if Alert.band is missing.

    Returns:
      Multi-line string, header lines first, then 80-col astrometric block.
    """
    obs = observatory_code or header.observatory_code
    out = list(header_lines(header))
    for cand, alerts in candidates_with_alerts:
        recs = candidate_to_records(cand, alerts,
                                    observatory_code=obs,
                                    band=band,
                                    discovery=(cand.n_runs == 1))
        out.extend(recs)
    return "\n".join(out) + "\n"


def parse_record(line: str) -> dict:
    """Round-trip helper -- decode one 80-col record back into a dict.

    Used by tests and audit. Returns {number, designation, mjd, ra_deg, dec_deg,
    mag, band, observatory_code}. Robust to short lines (right-pads to 80).
    """
    if len(line) < 80:
        line = (line + " " * 80)[:80]

    number = line[0:5].strip()
    designation = line[5:12].strip()
    date_s = line[15:32]
    ra_s = line[32:44]
    dec_s = line[44:56]
    mag_s = line[65:70].strip()
    band = line[70:71].strip() or None
    obs = line[77:80]

    # Decode date
    y = int(date_s[0:4]); m = int(date_s[5:7]); d = float(date_s[8:].strip())
    day = int(d); frac = d - day
    a = (14 - m) // 12
    Y = y + 4800 - a
    M = m + 12 * a - 3
    jdn = day + (153 * M + 2) // 5 + 365 * Y + Y // 4 - Y // 100 + Y // 400 - 32045
    jd = jdn - 0.5 + frac
    mjd = jd - 2400000.5

    # Decode RA HH MM SS.ddd
    hh, mm, ss = ra_s.split()
    ra_deg = (int(hh) + int(mm) / 60.0 + float(ss) / 3600.0) * 15.0

    # Decode Dec sDD MM SS.dd
    sign = -1 if dec_s.lstrip().startswith("-") else 1
    parts = dec_s.replace("+", " ").replace("-", " ").split()
    dec_deg = sign * (int(parts[0]) + int(parts[1]) / 60.0 + float(parts[2]) / 3600.0)

    return {
        "number": number, "designation": designation,
        "mjd": mjd, "ra_deg": ra_deg, "dec_deg": dec_deg,
        "mag": float(mag_s) if mag_s else None, "band": band,
        "observatory_code": obs,
    }
