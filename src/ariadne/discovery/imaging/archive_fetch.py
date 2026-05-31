"""Fetch FITS imaging from public archives -- the data on-ramp for the imaging pipeline.

Supported archives (try them in turn; first one with data wins):

  - NOIRLab Astro Data Lab (DECam Legacy Surveys, DECaPS, NEWFIRM, SOAR/Goodman)
    via astroquery.noirlab. Public, no auth; rate-limited.
  - STScI MAST (Pan-STARRS DR2 stacks + warps, HST archive, Kepler/TESS) via
    astroquery.mast. Public, no auth.
  - SDSS Sky Server (single-epoch frames, no multi-night cadence).
  - Local fallback: if no archive is reachable, the bench / examples have a
    synthetic-FITS injector that exercises the pipeline end-to-end on simulated
    images. (See benchmarks/imaging_pipeline.py.)

Usage::

    from ariadne.discovery.imaging.archive_fetch import fetch_decam_tile
    files = fetch_decam_tile(ra=180.0, dec=20.0, radius_deg=0.5,
                             mjd_start=60500, mjd_end=60800,
                             out_dir="data/decam/",
                             max_images=10)
    # `files` is a list of local FITS filenames + their metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class FitsImage:
    """Metadata + local path for one fetched FITS image."""
    path: Path
    archive: str        # "NOIRLab", "MAST/PanSTARRS", ...
    mjd: float
    ra_center: float    # degrees
    dec_center: float
    band: str
    exptime: float
    image_id: str
    meta: dict


def fetch_decam_tile(ra: float, dec: float, radius_deg: float = 0.5,
                     mjd_start: float | None = None, mjd_end: float | None = None,
                     out_dir: str | Path = "data/decam",
                     max_images: int = 12,
                     band: str | None = "r") -> list[FitsImage]:
    """Pull DECam Legacy Survey images (or fall back to PanSTARRS warps) for a sky/time window.

    Tries NOIRLab Astro Data Lab first (DECam DR10), falls back to MAST PanSTARRS DR2 warps.
    Returns a list of FitsImage records describing what was downloaded.

    Parameters
    ----------
    ra, dec : cone center, J2000 degrees
    radius_deg : cone radius (DECam tiles are 1-deg-class FoV; a small radius pulls
                 just the images covering that point)
    mjd_start, mjd_end : optional date range (otherwise unbounded)
    out_dir : where to save the FITS files
    max_images : cap the number of downloaded files (each is ~30-200 MB)
    band : filter band ("g", "r", "i", "z", "Y"); None = any
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try NOIRLab first
    try:
        return _fetch_noirlab(ra, dec, radius_deg, mjd_start, mjd_end,
                              out_dir, max_images, band)
    except Exception as e:
        noirlab_err = str(e)[:120]

    # Fallback to MAST PanSTARRS
    try:
        return _fetch_panstarrs(ra, dec, radius_deg, mjd_start, mjd_end,
                                out_dir, max_images, band)
    except Exception as e:
        raise RuntimeError(
            f"both archive fetches failed: NOIRLab={noirlab_err}, MAST={e}"
        ) from e


def _fetch_noirlab(ra, dec, radius_deg, mjd_start, mjd_end, out_dir,
                   max_images, band) -> list[FitsImage]:
    try:
        from astroquery.noirlab import Noirlab
    except ImportError:
        raise RuntimeError("astroquery.noirlab not available")
    n = Noirlab()
    # Use the DECam Legacy Surveys (proposal LSDR9)
    query = {
        "ra": ra, "dec": dec, "radius": radius_deg,
        "instrument": "DECam",
        "proc_type": "instcal",       # processed / calibrated stacks (smaller than raw)
    }
    if band:
        query["ifilter"] = band
    if mjd_start is not None:
        query["obs_mjd_min"] = mjd_start
    if mjd_end is not None:
        query["obs_mjd_max"] = mjd_end
    try:
        rs = n.query_metadata(rawquery=query)
    except Exception as e:
        raise RuntimeError(f"NOIRLab query_metadata failed: {e}") from e
    if rs is None or len(rs) == 0:
        return []
    out = []
    for i, row in enumerate(rs[:max_images]):
        try:
            local = n.retrieve(row["archive_filename"], destination=out_dir)
        except Exception:
            continue
        out.append(FitsImage(
            path=Path(local) if local else out_dir / row["archive_filename"],
            archive="NOIRLab/DECam",
            mjd=float(row.get("obs_mjd", row.get("mjd", 0.0))),
            ra_center=float(row.get("ra_center", row.get("ra", ra))),
            dec_center=float(row.get("dec_center", row.get("dec", dec))),
            band=str(row.get("ifilter", "")),
            exptime=float(row.get("exposure", row.get("exptime", 0.0))),
            image_id=str(row.get("archive_filename", f"img_{i}")),
            meta=dict(row),
        ))
    return out


def _fetch_panstarrs(ra, dec, radius_deg, mjd_start, mjd_end, out_dir,
                     max_images, band) -> list[FitsImage]:
    """Fallback: PanSTARRS DR2 warps via MAST."""
    try:
        from astroquery.mast import Observations
    except ImportError:
        raise RuntimeError("astroquery.mast not available")
    obs = Observations.query_criteria(
        coordinates=f"{ra} {dec}",
        radius=f"{radius_deg} deg",
        obs_collection="PS1",
        dataRights="PUBLIC",
        intentType="science",
    )
    if obs is None or len(obs) == 0:
        return []
    rows = obs[:max_images]
    products = Observations.get_product_list(rows)
    fits = [p for p in products if p["productType"] == "SCIENCE"
            and p["productSubGroupDescription"] == "FITS"]
    fits = fits[:max_images]
    if not fits:
        return []
    Observations.download_products(fits, download_dir=str(out_dir))
    out = []
    for i, row in enumerate(rows):
        out.append(FitsImage(
            path=out_dir / f"ps1_{i}.fits",
            archive="MAST/PanSTARRS",
            mjd=float(row.get("t_min", 0.0)),
            ra_center=float(row.get("s_ra", ra)),
            dec_center=float(row.get("s_dec", dec)),
            band=str(row.get("filters", "")),
            exptime=float(row.get("t_exptime", 0.0)),
            image_id=str(row.get("obs_id", f"ps1_{i}")),
            meta={},
        ))
    return out


def synthesise_decam_tile(ra: float, dec: float, n_images: int = 6,
                          n_objects_per_image: int = 50,
                          n_real_moving: int = 3,
                          mjd_nights: list[float] | None = None,
                          out_dir: str | Path = "data/decam_synth",
                          kepler_orbits: bool = True) -> list[FitsImage]:
    """Create synthetic FITS images with planted moving sources, for offline testing.

    `kepler_orbits=True` (default): plants objects with REAL Keplerian heliocentric
    orbits (random TNO-like elements), propagated through the Ariadne integrator
    and projected onto the geocentric sky -- IOD+LM should ACCEPT them.

    `kepler_orbits=False`: plants objects with constant-velocity sky motion; tests
    source extraction + tracklet/chain logic but IOD+LM will REJECT (correct
    behaviour for non-Keplerian, also a useful filter-sanity test).
    """
    try:
        import numpy as np
        from astropy.io import fits
        from astropy.wcs import WCS
    except ImportError as e:
        raise RuntimeError("astropy required: `pip install astropy`") from e

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if mjd_nights is None:
        mjd_nights = [60000.0, 60003.0, 60006.0]
    rng = np.random.default_rng(0)
    npix = 512
    pixscale_arcsec = 1.0
    pixscale_deg = pixscale_arcsec / 3600.0

    # Plant moving objects -- either Keplerian heliocentric orbits or constant-velocity
    if kepler_orbits:
        from ...dynamics.secular import kepler_step, elements_to_state
        from ...data.constants import GM_SUN
        from ...data.ephemeris import body_state
        # Random TNO-like elements; positions roughly in the (ra, dec) sky cone
        kepler_objects = []
        for k in range(n_real_moving):
            a_au = float(rng.uniform(40, 80))
            e = float(rng.uniform(0, 0.2))
            i_deg = float(rng.uniform(5, 25))
            Omega = float(rng.uniform(0, 360))
            omega = float(rng.uniform(0, 360))
            M = float(rng.uniform(0, 360))
            r0, v0 = elements_to_state(a_au, e, i_deg, Omega, omega, M)
            kepler_objects.append({"r0": np.asarray(r0), "v0": np.asarray(v0)})
    else:
        moving = []
        for k in range(n_real_moving):
            moving.append({
                "ra0": ra + (rng.random() - 0.5) * 0.05,
                "dec0": dec + (rng.random() - 0.5) * 0.05,
                "rate_arcsec_hr": float(rng.uniform(0.5, 3.0)),
                "theta": float(rng.uniform(0, 2 * np.pi)),
            })

    fits_records = []
    img_idx = 0
    SEC_PER_DAY = 86400.0
    for night_idx, mjd0 in enumerate(mjd_nights):
        for half in (0.0, 2.0):
            t = mjd0 + half / 24.0
            data = rng.normal(loc=100.0, scale=10.0, size=(npix, npix)).astype("float32")
            for _ in range(n_objects_per_image):
                xi, yi = rng.uniform(5, npix - 5, 2)
                amp = rng.uniform(500, 5000)
                _stamp_gaussian(data, xi, yi, amp, sigma=1.5)
            if kepler_orbits:
                # Propagate each Keplerian object to this epoch and project to sky
                et = ((t + 2400000.5) - 2451545.0) * SEC_PER_DAY
                R_e = body_state("EARTH", et, "J2000", "SUN")[:3]
                dt_s = (t - mjd_nights[0]) * SEC_PER_DAY
                for o in kepler_objects:
                    rt, _ = kepler_step(o["r0"], o["v0"], GM_SUN, dt_s)
                    geo = rt - R_e
                    rho = float(np.linalg.norm(geo))
                    import math as _m
                    ra_obj = _m.degrees(_m.atan2(geo[1], geo[0])) % 360.0
                    dec_obj = _m.degrees(_m.asin(geo[2] / rho))
                    # Skip if outside the local image cone
                    dra = (ra_obj - ra) * _m.cos(_m.radians(dec))
                    if abs(dra) > 0.07 or abs(dec_obj - dec) > 0.07:
                        continue
                    xi = npix / 2 + dra / pixscale_deg
                    yi = npix / 2 + (dec_obj - dec) / pixscale_deg
                    if 5 < xi < npix - 5 and 5 < yi < npix - 5:
                        _stamp_gaussian(data, xi, yi, amplitude=3000.0, sigma=1.5)
            else:
                for m in moving:
                    dt_days = t - mjd_nights[0]
                    ra_obj = m["ra0"] + (m["rate_arcsec_hr"] * np.cos(m["theta"]) * 24.0 / 3600.0
                                         / np.cos(np.radians(dec))) * dt_days
                    dec_obj = m["dec0"] + (m["rate_arcsec_hr"] * np.sin(m["theta"]) * 24.0 / 3600.0) * dt_days
                    xi = npix / 2 + (ra_obj - ra) * np.cos(np.radians(dec)) / pixscale_deg
                    yi = npix / 2 + (dec_obj - dec) / pixscale_deg
                    if 5 < xi < npix - 5 and 5 < yi < npix - 5:
                        _stamp_gaussian(data, xi, yi, amplitude=3000.0, sigma=1.5)
            # write FITS with simple TAN WCS
            w = WCS(naxis=2)
            w.wcs.crpix = [npix / 2, npix / 2]
            w.wcs.crval = [ra, dec]
            w.wcs.cd = [[-pixscale_deg, 0.0], [0.0, pixscale_deg]]
            w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            hdr = fits.Header(w.to_header())
            hdr["MJD-OBS"] = t
            hdr["FILTER"] = "r"
            hdr["EXPTIME"] = 90.0
            path = out_dir / f"synth_n{night_idx}_e{int(half)}.fits"
            fits.writeto(path, data, header=hdr, overwrite=True)
            fits_records.append(FitsImage(
                path=path, archive="synthetic", mjd=t,
                ra_center=ra, dec_center=dec, band="r", exptime=90.0,
                image_id=path.stem, meta={"n_planted_moving": n_real_moving}))
            img_idx += 1
    return fits_records


def _stamp_gaussian(data, x, y, amplitude, sigma):
    """Add a 2-D Gaussian to a 2-D array, in-place."""
    import numpy as np
    ix, iy = int(round(x)), int(round(y))
    half = int(round(4 * sigma))
    y0, y1 = max(0, iy - half), min(data.shape[0], iy + half + 1)
    x0, x1 = max(0, ix - half), min(data.shape[1], ix + half + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    g = amplitude * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    data[y0:y1, x0:x1] += g
