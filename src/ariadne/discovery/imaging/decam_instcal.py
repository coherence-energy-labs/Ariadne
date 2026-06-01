"""Multi-extension DECam Community Pipeline (CP) instcal FITS handler.

A DECam instcal exposure has 60 CCDs, each in its own FITS extension.
Each extension has:
  * The science image (float32)
  * A weight/variance extension (next index)
  * A data quality mask (next next index)
  * Per-CCD WCS in the science extension header
  * Photometric zero-point (MAGZERO) and aperture corrections in primary

The standard processing flow is:
  1. Open the primary HDU + read global headers (proc date, MJDOBS, etc.)
  2. For each CCD extension, read science image + DQM + WCS + ZP
  3. Mask bad pixels via DQM (saturation, cosmic rays, bad columns)
  4. Run source extraction on the masked image
  5. Apply photometric calibration: mag = -2.5 log10(flux) + MAGZERO
  6. Run Gaia astrometric refinement on the catalog

This module provides the multi-extension iterator + per-CCD container.
Single-CCD calling code (existing photutils-based pipeline) works on
each CCD's image individually.

Public API:
  load_decam_instcal(path) -> DecamInstCalFile
    Container with primary header + list of CCDExtension records.

  iterate_ccds(file) -> Iterator[CCDExtension]
    Yield (science_image, wcs, dqm, header, magzero) per CCD.

  apply_dqm_mask(image, dqm) -> masked image
    Replace bad pixels with NaN so detection skips them.

  calibrate_magnitudes(flux, magzero) -> AB magnitude
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass
class CCDExtension:
    """One CCD of a DECam instcal mosaic."""
    ccdnum: int            # CCD number 1..62 (some have gaps)
    name: str              # 'N1', 'N2', ..., 'S31' DECam naming convention
    science: np.ndarray    # science image (float)
    dqm: np.ndarray | None # data quality mask, or None
    weight: np.ndarray | None  # weight map (1/variance), or None
    wcs: object            # astropy.wcs.WCS
    header: dict           # CCD-extension header (subset)
    magzero: float         # photometric zero point (AB)
    seeing_arcsec: float = 0.0
    band: str = ""
    mjd: float = 0.0


@dataclass
class DecamInstCalFile:
    """Multi-extension DECam instcal container."""
    path: Path
    primary_header: dict
    n_ccds: int
    ccds: list[CCDExtension] = field(default_factory=list)
    mjd: float = 0.0
    band: str = ""
    exptime_s: float = 0.0
    obs_id: str = ""


# DECam DQM bit definitions (Community Pipeline; from DECam Data Handbook)
# Bits in the data-quality mask. Any non-zero pixel is suspect.
DQM_BIT_NAMES = {
    0: "bad_column",
    1: "saturated",
    2: "interpolated",
    3: "cosmic_ray",
    4: "bleed_trail",
    5: "edge",
    6: "unused",
    7: "transient",
}


def load_decam_instcal(path: str | Path,
                          *, ccd_filter: list[int] | None = None,
                          read_dqm: bool = True,
                          read_weight: bool = False
                          ) -> DecamInstCalFile:
    """Open a DECam instcal FITS and return a structured container.

    `ccd_filter` (optional): restrict to a subset of CCD numbers (1..62).
    Default: load all available CCDs.

    Photometric ZP is read from each extension's MAGZERO header if
    present, otherwise from the primary HDU's MAGZERO. AB-magnitude zero
    is the natural target.
    """
    from astropy.io import fits as astrofits
    from astropy.wcs import WCS

    path = Path(path)
    with astrofits.open(path, memmap=True) as hdul:
        primary = hdul[0].header
        primary_dict = {k: primary[k] for k in primary.keys()
                          if isinstance(k, str) and k}
        mjd = float(primary_dict.get("MJD-OBS", 0.0))
        band = str(primary_dict.get("FILTER", "")).strip()
        exptime = float(primary_dict.get("EXPTIME", 0.0))
        obs_id = str(primary_dict.get("OBSID",
                                          primary_dict.get("EXPNUM", ""))).strip()
        # Global ZP (will be overridden per-CCD if available)
        global_zp = float(primary_dict.get("MAGZERO",
                                              primary_dict.get("MAGZPT", 0.0)))

        ccds: list[CCDExtension] = []
        for i, hdu in enumerate(hdul):
            if i == 0:
                continue
            if hdu.data is None or hdu.data.ndim != 2:
                continue
            ext_name = str(hdu.header.get("EXTNAME", f"EXT{i}")).strip()
            ccdnum_raw = hdu.header.get("CCDNUM", i)
            try:
                ccdnum = int(ccdnum_raw)
            except (TypeError, ValueError):
                ccdnum = i
            if ccd_filter and ccdnum not in ccd_filter:
                continue
            # The Community Pipeline uses 3-extension groups: SCI, WGT, DQM
            # We identify which kind this extension is from its EXTNAME / HDUNAME
            kind = _classify_extension(ext_name, hdu.header)
            if kind != "science":
                continue
            # Find the matching weight + dqm by looking for adjacent extensions
            weight_arr = None
            dqm_arr = None
            for j in range(i + 1, min(i + 4, len(hdul))):
                sub = hdul[j]
                if sub.data is None:
                    continue
                sub_kind = _classify_extension(
                    str(sub.header.get("EXTNAME", "")).strip(), sub.header)
                # The matching weight/dqm should have the same CCDNUM
                sub_ccd = sub.header.get("CCDNUM", -1)
                if sub_ccd != ccdnum_raw:
                    continue
                if sub_kind == "weight" and read_weight:
                    weight_arr = sub.data.astype(float)
                elif sub_kind == "dqm" and read_dqm:
                    dqm_arr = sub.data.astype(np.uint8)
            try:
                wcs = WCS(hdu.header)
            except Exception:
                wcs = None
            zp = float(hdu.header.get("MAGZERO",
                                          hdu.header.get("MAGZPT",
                                                            global_zp)))
            seeing = float(hdu.header.get("FWHM",
                                              hdu.header.get("SEEING", 0.0)))
            ccd = CCDExtension(
                ccdnum=ccdnum, name=ext_name,
                science=hdu.data.astype(np.float32),
                dqm=dqm_arr, weight=weight_arr,
                wcs=wcs,
                header={k: hdu.header[k] for k in hdu.header.keys()
                          if isinstance(k, str) and k},
                magzero=zp, seeing_arcsec=seeing,
                band=band, mjd=mjd,
            )
            ccds.append(ccd)
    return DecamInstCalFile(
        path=path, primary_header=primary_dict,
        n_ccds=len(ccds), ccds=ccds,
        mjd=mjd, band=band, exptime_s=exptime, obs_id=obs_id,
    )


def _classify_extension(extname: str, header: dict) -> str:
    """Heuristic classifier: science / weight / dqm / other.

    Community Pipeline naming:
      'SCI ' or '_SCI'   science
      'WGT ' or '_WGT'   weight (inverse variance)
      'DQM ' or '_DQM'   data quality mask
    """
    en = extname.upper()
    bunit = str(header.get("BUNIT", "")).upper()
    if "DQM" in en or "DQMASK" in en or "MASK" in en:
        return "dqm"
    if "WGT" in en or "WEIGHT" in en or "VAR" in en or "WT" in en[-3:]:
        return "weight"
    if "SCI" in en or "IMAGE" in en or bunit == "ELECTRONS":
        return "science"
    # Fallback: first extension after primary is usually science
    return "science"


def iterate_ccds(file: DecamInstCalFile,
                   *, mask_dqm: bool = True) -> Iterator[CCDExtension]:
    """Yield CCDExtension records; optionally apply DQM mask before yielding."""
    for ccd in file.ccds:
        if mask_dqm and ccd.dqm is not None:
            ccd.science = apply_dqm_mask(ccd.science, ccd.dqm)
        yield ccd


def apply_dqm_mask(image: np.ndarray, dqm: np.ndarray,
                     *, fill_value: float = float("nan")) -> np.ndarray:
    """Replace bad-pixel locations (DQM != 0) with `fill_value`.

    DECam Community Pipeline DQM uses bit-packed flags. A pixel with
    DQM == 0 is good; any nonzero bit indicates some defect.
    """
    out = image.astype(float, copy=True)
    bad = dqm != 0
    out[bad] = fill_value
    return out


def calibrate_magnitudes(flux: float | np.ndarray,
                            magzero: float) -> float | np.ndarray:
    """Convert raw electron flux to AB magnitude via the Community Pipeline ZP.

    DECam CP `MAGZERO` is defined such that:
      mag_AB = -2.5 * log10(flux_in_electrons / exptime) + MAGZERO
    For our purposes (where flux is already a single-sky-integrated number),
    we use:
      mag_AB = -2.5 * log10(flux) + MAGZERO

    Caller is responsible for ensuring `flux` is in the right units for
    the ZP convention being used.
    """
    f = np.asarray(flux, dtype=float)
    f = np.where(f > 0, f, np.nan)
    return -2.5 * np.log10(f) + magzero
