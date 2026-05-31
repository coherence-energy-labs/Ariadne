"""Source extraction from FITS imaging via photutils -- the front of the imaging pipeline.

For each FITS image, estimate a background, subtract it, and run a DAO-style star
finder to locate point sources. Convert pixel positions to (RA, Dec) via the image's
WCS. Output is a per-image catalogue of Source objects ready to feed into the
tracklet builder.

Photutils is the canonical Python astronomy library for source detection (the
sextractor-equivalent in pure Python). Requires `pip install photutils astropy`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import math


@dataclass(frozen=True)
class Source:
    """A single source detection in one image.

    ra, dec   : J2000 degrees from the image WCS
    flux      : raw photutils flux estimate (counts)
    mag       : derived apparent magnitude (-99 if no zeropoint available)
    fwhm_px   : measured FWHM in pixels (helps cut cosmic-ray hits / artefacts)
    mjd       : image's MJD-OBS (mid-exposure)
    image_id  : the image's identifier (filename, exposure_id, etc.)
    x, y      : pixel coordinates within the image (for diagnostics)
    """
    ra: float
    dec: float
    flux: float
    mag: float
    fwhm_px: float
    mjd: float
    image_id: str
    x: float
    y: float

    @property
    def ra_rad(self) -> float:
        return math.radians(self.ra)

    @property
    def dec_rad(self) -> float:
        return math.radians(self.dec)


def detect_sources_in_image(image_data, wcs, mjd: float, image_id: str,
                             fwhm_px: float = 3.0, threshold_sigma: float = 5.0,
                             zeropoint_mag: float | None = None,
                             min_fwhm_px: float = 1.0,
                             max_fwhm_px: float = 8.0) -> list[Source]:
    """Detect sources in an image array (2-D numpy) given its WCS + observation MJD.

    Returns a list of Source objects. Uses photutils DAOStarFinder with the supplied
    FWHM + threshold (default 5-sigma above background). Filters out detections with
    pathological FWHM (cosmic-ray hits, edge clusters).

    Parameters
    ----------
    image_data : 2-D numpy array of the image (counts after bias / dark / flat).
    wcs        : an astropy.wcs.WCS object for this image.
    mjd        : observation MJD (mid-exposure).
    image_id   : a string identifier (filename or exposure_id).
    fwhm_px    : DAO-style detection FWHM in pixels.
    threshold_sigma : detection threshold above the background sigma.
    zeropoint_mag : magnitude of a 1-count source. If None, mag is set to -99.
    """
    try:
        import numpy as np
        from photutils.background import Background2D, MedianBackground
        from photutils.detection import DAOStarFinder
        from astropy.stats import sigma_clipped_stats
    except ImportError as e:
        raise ImportError("photutils + astropy required for image source extraction; "
                          "`pip install photutils astropy`") from e

    data = np.asarray(image_data, dtype=float)
    # background estimate
    bkg = Background2D(data, box_size=(64, 64), bkg_estimator=MedianBackground())
    sub = data - bkg.background
    # global sigma after background subtraction
    _mean, _med, std = sigma_clipped_stats(sub, sigma=3.0)
    finder = DAOStarFinder(fwhm=fwhm_px, threshold=threshold_sigma * std)
    tbl = finder(sub)
    if tbl is None or len(tbl) == 0:
        return []

    sources = []
    for row in tbl:
        x = float(row["xcentroid"])
        y = float(row["ycentroid"])
        # FWHM filter to reject cosmic-ray hits / extended artefacts
        s_fwhm = float(row.get("sharpness", 1.0)) * fwhm_px  # rough proxy
        if not (min_fwhm_px <= s_fwhm <= max_fwhm_px):
            continue
        flux = float(row["flux"])
        if flux <= 0:
            continue
        # WCS pixel -> world (returns RA, Dec in degrees)
        ra, dec = wcs.pixel_to_world_values(x, y)
        ra = float(ra) % 360.0
        dec = float(dec)
        mag = (zeropoint_mag - 2.5 * math.log10(flux)) if zeropoint_mag else -99.0
        sources.append(Source(ra=ra, dec=dec, flux=flux, mag=mag, fwhm_px=s_fwhm,
                              mjd=mjd, image_id=image_id, x=x, y=y))
    return sources


def synthesise_sources(n: int, ra_center: float, dec_center: float,
                        width_deg: float = 1.0, mjd: float = 60000.0,
                        image_id: str = "synth", seed: int = 0) -> list[Source]:
    """Generate N synthetic Source detections for testing the tracklet builder.

    Random RA/Dec in a box, uniform flux, perfect WCS. Use to validate the
    imaging->tracklet->IOD pipeline without needing real DECam images.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        ra = ra_center + (rng.random() - 0.5) * width_deg / math.cos(math.radians(dec_center))
        dec = dec_center + (rng.random() - 0.5) * width_deg
        flux = float(10 ** rng.uniform(2, 5))
        out.append(Source(ra=ra % 360.0, dec=dec, flux=flux,
                          mag=20.0 + rng.normal(0, 0.5), fwhm_px=3.0,
                          mjd=mjd, image_id=f"{image_id}_{i:04d}",
                          x=rng.uniform(0, 4096), y=rng.uniform(0, 4096)))
    return out
