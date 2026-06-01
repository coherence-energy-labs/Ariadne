"""NOIRLab Astro Data Archive REST client.

The standard `astroquery.noirlab` plugin is not always available and
the SIA2 endpoints have inconsistent metadata. This module uses the
REST adv_search API directly which is what the NOIRLab web UI uses:

    POST https://astroarchive.noirlab.edu/api/adv_search/find/
    body: {"outfields": [...], "search": [["instrument", "decam"], ...]}

The response is JSON: row 0 is METADATA + HEADER, rows 1+ are data
records. We parse those into `ExposureRecord` dataclasses.

Files are downloaded via the `url` field which points at
    https://astroarchive.noirlab.edu/api/retrieve/<md5>/

Public API:
  query_decam_exposures(ra, dec, radius_deg, ...) -> list[ExposureRecord]
    Cone query for DECam single-epoch exposures.
  download_decam_exposure(record, out_dir) -> Path | None
    Download a FITS file.
  ExposureRecord
    Container with metadata + download URL.

Notes on DECam Community Pipeline (CP) products:
  proc_type='instcal' = instrumental calibration, single-CCD 60-extension
                        FITS with WCS + photometric ZP. What we want.
  proc_type='resampled' = warped to regular sky grid (not for detection).
  proc_type='stacked' = coadded (not for moving objects).

Filenames carry product-kind codes:
  c4d_*_ooi_*.fits.fz   science image (instcal)
  c4d_*_oow_*.fits.fz   weight (1/variance) map
  c4d_*_ood_*.fits.fz   data quality mask
The science image is what we want; we filter on filename to get only ooi.
"""
from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


NOIRLAB_FIND_URL = "https://astroarchive.noirlab.edu/api/adv_search/find/"


@dataclass(frozen=True)
class ExposureRecord:
    """Metadata for one DECam exposure or one product of one exposure."""
    archive_id: str        # md5sum; stable across re-queries
    instrument: str        # 'decam'
    proc_type: str         # 'instcal' / 'resampled' / 'stacked'
    obs_mjd: float         # MJD-OBS at exposure start
    exptime_s: float       # exposure time in seconds
    band: str              # filter band ('g'/'r'/'i'/'z'/'Y' parsed from ifilter)
    ra_center: float       # exposure pointing RA (deg)
    dec_center: float      # exposure pointing Dec (deg)
    data_url: str = ""     # download URL
    archive_filename: str = ""   # path on NOIRLab archive (for ooi/oow/ood split)
    product_kind: str = ""       # 'ooi' / 'oow' / 'ood'
    obs_id: str = ""             # numerical observation id (or filename stem)
    s_region: str = ""
    proposal_id: str = ""
    extras: dict = field(default_factory=dict)


def _parse_dateobs_to_mjd(iso_str: str) -> float:
    """Convert ISO-8601 'YYYY-MM-DDTHH:MM:SS.ssss[Z]' to MJD.

    Uses astropy.time for correctness. Returns 0.0 on failure.
    """
    if not iso_str:
        return 0.0
    try:
        from astropy.time import Time
        t = Time(iso_str.rstrip("Z"), format="isot", scale="utc")
        return float(t.mjd)
    except Exception:
        return 0.0


def _parse_filter_band(ifilter: str) -> str:
    """The NOIRLab `ifilter` field is verbose, e.g.:
       'r DECam SDSS c0002 6415.0 1480.0'
    Extract the single-letter band code (first token) and lowercase it.
    """
    if not ifilter:
        return ""
    first = ifilter.strip().split()[0]
    # Some bands are 'u', 'g', 'r', 'i', 'z', 'Y' -- preserve case 'Y'
    return first if first == "Y" else first.lower()


def _parse_product_kind(archive_filename: str) -> str:
    """DECam CP files have a 3-char kind code in the filename:
       'ooi' = science image
       'oow' = weight map
       'ood' = data quality mask
    Returns the kind or empty string if not recognized.
    """
    if not archive_filename:
        return ""
    name = archive_filename.split("/")[-1].lower()
    for kind in ("ooi", "oow", "ood"):
        if f"_{kind}_" in name:
            return kind
    return ""


def query_decam_exposures(ra: float, dec: float,
                            radius_deg: float = 0.5,
                            *,
                            mjd_min: float | None = None,
                            mjd_max: float | None = None,
                            band: str | None = "r",
                            proc_type: str = "instcal",
                            max_results: int = 20,
                            include_kinds: Sequence[str] = ("ooi",),
                            timeout_s: float = 30.0,
                            ) -> list[ExposureRecord]:
    """POST to NOIRLab adv_search/find and return parsed ExposureRecords.

    `include_kinds` filters by the DECam CP product code in the filename:
      'ooi' = science image (default)
      'oow' = weight map
      'ood' = data quality mask
    Caller usually wants ('ooi',) for detection; ('ooi','oow','ood') to
    download all three products of each exposure.

    Returns empty list on any failure -- caller can fall back to synth.
    """
    # Build the search payload. NOIRLab's syntax for cone search uses
    # bounding-box on ra_center/dec_center. We over-fetch then filter
    # to a cone in code.
    cos_dec = max(math.cos(math.radians(dec)), 1e-3)
    ra_min = ra - radius_deg / cos_dec
    ra_max = ra + radius_deg / cos_dec
    dec_min = dec - radius_deg
    dec_max = dec + radius_deg
    search = [
        ["instrument", "decam"],
        ["proc_type", proc_type],
        ["ra_center", ra_min, ra_max],
        ["dec_center", dec_min, dec_max],
    ]
    if mjd_min is not None or mjd_max is not None:
        # Convert MJD bounds to ISO-8601 for the dateobs_center filter
        from astropy.time import Time
        if mjd_min is not None:
            iso_min = Time(mjd_min, format="mjd").isot
        else:
            iso_min = "1900-01-01T00:00:00"
        if mjd_max is not None:
            iso_max = Time(mjd_max, format="mjd").isot
        else:
            iso_max = "2100-01-01T00:00:00"
        search.append(["dateobs_center", iso_min, iso_max])

    payload = {
        "outfields": [
            "md5sum", "url", "archive_filename",
            "instrument", "proc_type", "ifilter",
            "exposure", "ra_center", "dec_center",
            "dateobs_center", "proposal",
        ],
        "search": search,
    }
    try:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{NOIRLAB_FIND_URL}?limit={max_results * 5}",
            data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []
    if not isinstance(data, list) or len(data) < 2:
        return []
    # Row 0 is META + HEADER; rows 1+ are data
    rows = data[1:]
    out: list[ExposureRecord] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        kind = _parse_product_kind(str(row.get("archive_filename", "")))
        if include_kinds and kind and kind not in include_kinds:
            continue
        rec = ExposureRecord(
            archive_id=str(row.get("md5sum", "")),
            instrument=str(row.get("instrument", "decam")),
            proc_type=str(row.get("proc_type", proc_type)),
            obs_mjd=_parse_dateobs_to_mjd(
                str(row.get("dateobs_center", ""))),
            exptime_s=float(row.get("exposure", 0.0) or 0.0),
            band=_parse_filter_band(str(row.get("ifilter", ""))),
            ra_center=float(row.get("ra_center", ra) or ra),
            dec_center=float(row.get("dec_center", dec) or dec),
            data_url=str(row.get("url", "")),
            archive_filename=str(row.get("archive_filename", "")),
            product_kind=kind,
            obs_id=str(row.get("archive_filename", "")).split(
                "/")[-1].replace(".fits.fz", "").replace(".fits", ""),
            proposal_id=str(row.get("proposal", "")),
        )
        # Tight cone filter (the API does a bounding box, we want a circle)
        d_ra = (rec.ra_center - ra) * cos_dec
        d_dec = rec.dec_center - dec
        if math.hypot(d_ra, d_dec) > radius_deg:
            continue
        if band:
            if rec.band != band:
                continue
        out.append(rec)
        if len(out) >= max_results * 3:
            break
    # De-duplicate observations that appear multiple times under different
    # processing versions (CP marks files with _v1, _ls9, etc. suffixes
    # AFTER the timestamp portion of the filename). The base observation
    # is identified by the c4d_YYMMDD_HHMMSS prefix.
    def _obs_key(rec: ExposureRecord) -> str:
        fn = rec.archive_filename.split("/")[-1]
        # c4d_YYMMDD_HHMMSS_ooi_..., grab through HHMMSS
        parts = fn.split("_")
        if len(parts) >= 3:
            return "_".join(parts[:3])
        return fn
    seen_keys = set()
    deduped: list[ExposureRecord] = []
    for rec in out:
        k = _obs_key(rec)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        deduped.append(rec)
        if len(deduped) >= max_results:
            break
    deduped.sort(key=lambda r: r.obs_mjd)
    return deduped


def download_decam_exposure(record: ExposureRecord,
                              out_dir: str | Path,
                              *, force: bool = False,
                              timeout_s: float = 600.0,
                              chunk_size: int = 8192,
                              ) -> Path | None:
    """Stream-download a NOIRLab FITS file via its `url`. Returns the
    local Path, or None on failure.

    Streams to disk rather than buffering in memory so multi-hundred-MB
    DECam mosaics don't OOM.
    """
    if not record.data_url:
        return None
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fname = record.obs_id
    if not fname.endswith((".fits", ".fits.fz")):
        fname += ".fits.fz"
    fname = fname.replace("/", "_").replace(":", "_")
    local = out_dir / fname
    if local.exists() and local.stat().st_size > 0 and not force:
        return local
    try:
        with urllib.request.urlopen(record.data_url,
                                       timeout=timeout_s) as resp:
            with open(local, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
    except Exception:
        if local.exists():
            try:
                local.unlink()
            except Exception:
                pass
        return None
    if not local.exists() or local.stat().st_size == 0:
        return None
    return local


def _ping_noirlab(timeout_s: float = 10.0) -> bool:
    """Reachability test against the find/ endpoint with a trivial query."""
    try:
        payload = json.dumps({
            "outfields": ["md5sum"],
            "search": [["instrument", "decam"], ["proc_type", "instcal"]],
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{NOIRLAB_FIND_URL}?limit=1",
            data=payload,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return isinstance(data, list) and len(data) >= 1
    except Exception:
        return False
