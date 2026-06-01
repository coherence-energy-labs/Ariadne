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
    """Fallback: PanSTARRS DR2 stacks via MAST.

    PS1 metadata rows carry a `dataURL` pointing to the FITS file at
    ps1images.stsci.edu. We download those directly via urllib --
    cleaner than wrestling with Observations.download_products(), which
    drops files into a mastDownload/ subtree with hard-to-predict names.
    """
    try:
        from astroquery.mast import Observations
    except ImportError:
        raise RuntimeError("astroquery.mast not available")
    import urllib.request

    obs = Observations.query_criteria(
        coordinates=f"{ra} {dec}",
        radius=f"{radius_deg} deg",
        obs_collection="PS1",
        dataRights="PUBLIC",
        intentType="science",
    )
    if obs is None or len(obs) == 0:
        return []

    # Filter by band + mjd if requested
    rows = list(obs)
    if band:
        rows = [r for r in rows
                if band in str(r.get("filters", "")).split(",")
                or str(r.get("filters", "")) == band]
    if mjd_start is not None:
        rows = [r for r in rows
                if float(r.get("t_min", 0.0)) >= mjd_start]
    if mjd_end is not None:
        rows = [r for r in rows
                if float(r.get("t_min", 0.0)) <= mjd_end]
    rows = rows[:max_images]

    out = []
    for i, row in enumerate(rows):
        data_url = str(row.get("dataURL", "")).strip()
        if not data_url:
            continue
        # Save under a stable filename
        local_name = f"ps1_{i}_{str(row.get('obs_id', '')).replace('.', '_')}.fits"
        local_path = out_dir / local_name
        if not local_path.exists() or local_path.stat().st_size == 0:
            try:
                urllib.request.urlretrieve(data_url, str(local_path))
            except Exception:
                continue
        if not local_path.exists() or local_path.stat().st_size == 0:
            continue
        out.append(FitsImage(
            path=local_path,
            archive="MAST/PanSTARRS",
            mjd=float(row.get("t_min", 0.0)),
            ra_center=float(row.get("s_ra", ra)),
            dec_center=float(row.get("s_dec", dec)),
            band=str(row.get("filters", "")),
            exptime=float(row.get("t_exptime", 0.0)),
            image_id=str(row.get("obs_id", f"ps1_{i}")),
            meta={"data_url": data_url},
        ))
    return out


def synthesise_decam_tile(ra: float, dec: float, n_images: int = 6,
                          n_objects_per_image: int = 50,
                          n_real_moving: int = 3,
                          mjd_nights: list[float] | None = None,
                          out_dir: str | Path = "data/decam_synth",
                          kepler_orbits: bool = True,
                          emit_truth_catalog: bool = True,
                          seed: int = 0,
                          npix: int = 512,
                          pixscale_arcsec: float = 1.0,
                          family_mix: dict | None = None,
                          cone_radius_deg: float = 0.04) -> list[FitsImage]:
    """Create synthetic FITS images with planted moving sources, for offline testing.

    `kepler_orbits=True` (default): plants objects with REAL Keplerian heliocentric
    orbits (random TNO-like elements), propagated through the Ariadne integrator
    and projected onto the geocentric sky -- IOD+LM should ACCEPT them.

    `kepler_orbits=False`: plants objects with constant-velocity sky motion; tests
    source extraction + tracklet/chain logic but IOD+LM will REJECT (correct
    behaviour for non-Keplerian, also a useful filter-sanity test).

    When `emit_truth_catalog=True` (default), writes a `truth_catalog.json`
    sidecar next to the FITS images that records (truth_id, image_id, mjd,
    ra, dec, x_pix, y_pix, mag, family) for every planted moving object.
    Downstream code can load it via `TruthCatalog.load()` and use
    `measure_linker_quality()` to score chain precision/recall.
    """
    from .synthetic_truth import TruthCatalog, TruthEntry
    truth_entries: list[TruthEntry] = []
    try:
        import numpy as np
        from astropy.io import fits
        from astropy.wcs import WCS
    except ImportError as e:
        raise RuntimeError("astropy required: `pip install astropy`") from e

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if mjd_nights is None:
        mjd_nights = [60000.0, 60003.0, 60006.0]
    rng = np.random.default_rng(seed)
    pixscale_deg = pixscale_arcsec / 3600.0
    SEC_PER_DAY = 86400.0    # reused below in the per-epoch loop too
    # Default family mix: 60% TNO scattered, 25% Centaur, 10% MBA-outer, 5% NEO
    if family_mix is None:
        family_mix = {"tno": 0.60, "centaur": 0.25, "mba": 0.10, "neo": 0.05}

    # Plant moving objects -- either Keplerian heliocentric orbits or constant-velocity
    if kepler_orbits:
        import math
        from ...dynamics.secular import kepler_step, elements_to_state
        from ...data.constants import GM_SUN, AU_KM
        from ...data.ephemeris import body_state
        # Build initial state ON THE TARGET LINE OF SIGHT so the object
        # is GUARANTEED to be inside the image cone at t = mjd_nights[0].
        # Velocity is chosen to give a near-circular bound orbit at that
        # heliocentric distance.
        kepler_objects = []
        t0 = mjd_nights[0] if mjd_nights else 60000.0
        et0 = ((t0 + 2400000.5) - 2451545.0) * SEC_PER_DAY
        R_e_t0 = np.array(body_state("EARTH", et0, "J2000", "SUN")[:3],
                            dtype=float)
        ra_rad = math.radians(ra)
        dec_rad = math.radians(dec)
        d_sky = np.array([math.cos(dec_rad) * math.cos(ra_rad),
                           math.cos(dec_rad) * math.sin(ra_rad),
                           math.sin(dec_rad)])
        # Build a family lookup so we can sample heliocentric distances
        # from a realistic mixture. Distances in AU per orbital family.
        family_au_ranges = {
            "neo":     (0.8, 1.5),
            "mba":     (2.0, 3.5),
            "centaur": (8.0, 20.0),
            "tno":     (40.0, 80.0),
        }
        # Build a CDF for sampling
        fam_names = list(family_mix.keys())
        fam_weights = np.array([family_mix[n] for n in fam_names], dtype=float)
        fam_weights /= fam_weights.sum()
        fam_cdf = np.cumsum(fam_weights)

        for k in range(n_real_moving):
            # Sample family from the mixture
            u = float(rng.random())
            fam_idx = int(np.searchsorted(fam_cdf, u))
            fam = fam_names[min(fam_idx, len(fam_names) - 1)]
            lo, hi = family_au_ranges.get(fam, (40.0, 80.0))
            rho_km = float(rng.uniform(lo, hi)) * AU_KM
            # Spread the objects across the image cone
            ra_jit = float(rng.uniform(-cone_radius_deg, cone_radius_deg))
            dec_jit = float(rng.uniform(-cone_radius_deg, cone_radius_deg))
            ra_obj_rad = math.radians(ra + ra_jit / math.cos(dec_rad))
            dec_obj_rad = math.radians(dec + dec_jit)
            d_sky_obj = np.array([math.cos(dec_obj_rad) * math.cos(ra_obj_rad),
                                    math.cos(dec_obj_rad) * math.sin(ra_obj_rad),
                                    math.sin(dec_obj_rad)])
            r0 = R_e_t0 + rho_km * d_sky_obj
            r0_norm = float(np.linalg.norm(r0))
            # Near-circular velocity perpendicular to r0, in the ecliptic
            v_circ = math.sqrt(GM_SUN / r0_norm)
            # Build an arbitrary unit vector orthogonal to r0
            r_hat = r0 / r0_norm
            up = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(r_hat, up)) > 0.95:
                up = np.array([1.0, 0.0, 0.0])
            tangent = np.cross(r_hat, up)
            tangent = tangent / np.linalg.norm(tangent)
            # Slight random kick to give each object a different orbit shape
            tilt_deg = float(rng.uniform(-15, 15))
            tilt = math.radians(tilt_deg)
            v0_dir = (math.cos(tilt) * tangent
                       + math.sin(tilt) * np.cross(r_hat, tangent))
            v0 = v_circ * v0_dir
            kepler_objects.append({
                "truth_id": f"kepler_{fam}_{k:04d}",
                "family": f"kepler_{fam}",
                "r0": np.asarray(r0), "v0": np.asarray(v0),
                "a_au": r0_norm / AU_KM, "e": 0.0, "i_deg": tilt_deg,
                "Omega": 0.0, "omega": 0.0, "M": 0.0,
            })
    else:
        moving = []
        for k in range(n_real_moving):
            moving.append({
                "truth_id": f"linear_obj_{k:03d}",
                "family": "linear_sky_motion",
                "ra0": ra + (rng.random() - 0.5) * 0.05,
                "dec0": dec + (rng.random() - 0.5) * 0.05,
                "rate_arcsec_hr": float(rng.uniform(0.5, 3.0)),
                "theta": float(rng.uniform(0, 2 * np.pi)),
            })

    fits_records = []
    img_idx = 0
    for night_idx, mjd0 in enumerate(mjd_nights):
        for half in (0.0, 2.0):
            t = mjd0 + half / 24.0
            data = rng.normal(loc=100.0, scale=10.0, size=(npix, npix)).astype("float32")
            for _ in range(n_objects_per_image):
                xi, yi = rng.uniform(5, npix - 5, 2)
                amp = rng.uniform(500, 5000)
                _stamp_gaussian(data, xi, yi, amp, sigma=1.5)
            # Build the image_id we will record into the truth catalog.
            # IMPORTANT: must match what the source-extraction step passes
            # as `image_id` later (see scripts/run_decam_e2e.py [2]), which
            # is `str(fi.path)` -- the full FITS path. Use the same string
            # so per-source truth matching works downstream.
            this_image_id = str(out_dir / f"synth_n{night_idx}_e{int(half)}.fits")
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
                    # Use the WCS to compute pixel coords from world coords.
                    # The image WCS has CD11 = -pixscale_deg meaning RA
                    # decreases with increasing pixel-x; using a hand-rolled
                    # `xi = npix/2 + dra/pixscale_deg` (no sign flip) was a
                    # longstanding bug that placed objects at WCS-inconsistent
                    # positions, so source-extraction recovered different
                    # (ra, dec) than the synth's "ra_obj/dec_obj" and the
                    # truth catalog stored wrong RA/Dec relative to what
                    # the chain extraction would see.
                    xi = npix / 2 - dra / pixscale_deg
                    yi = npix / 2 + (dec_obj - dec) / pixscale_deg
                    if 5 < xi < npix - 5 and 5 < yi < npix - 5:
                        _stamp_gaussian(data, xi, yi, amplitude=3000.0, sigma=1.5)
                        truth_entries.append(TruthEntry(
                            truth_id=o["truth_id"], image_id=this_image_id,
                            mjd=t, ra=ra_obj, dec=dec_obj,
                            x_pix=float(xi), y_pix=float(yi),
                            mag=float(-2.5 * _m.log10(3000.0 / 100.0) + 25.0),
                            family=o["family"],
                            extras={"a_au": o["a_au"], "e": o["e"],
                                     "i_deg": o["i_deg"], "rho_au": rho}))
            else:
                for m in moving:
                    dt_days = t - mjd_nights[0]
                    ra_obj = m["ra0"] + (m["rate_arcsec_hr"] * np.cos(m["theta"]) * 24.0 / 3600.0
                                         / np.cos(np.radians(dec))) * dt_days
                    dec_obj = m["dec0"] + (m["rate_arcsec_hr"] * np.sin(m["theta"]) * 24.0 / 3600.0) * dt_days
                    # Same sign-flip fix as the kepler branch: RA decreases with pixel-x
                    xi = npix / 2 - (ra_obj - ra) * np.cos(np.radians(dec)) / pixscale_deg
                    yi = npix / 2 + (dec_obj - dec) / pixscale_deg
                    if 5 < xi < npix - 5 and 5 < yi < npix - 5:
                        _stamp_gaussian(data, xi, yi, amplitude=3000.0, sigma=1.5)
                        truth_entries.append(TruthEntry(
                            truth_id=m["truth_id"], image_id=this_image_id,
                            mjd=t, ra=float(ra_obj), dec=float(dec_obj),
                            x_pix=float(xi), y_pix=float(yi),
                            mag=float(-2.5 * np.log10(3000.0 / 100.0) + 25.0),
                            family=m["family"]))
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

    if emit_truth_catalog and truth_entries:
        catalog = TruthCatalog(truth_entries)
        catalog.save(out_dir / "truth_catalog.json")

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
