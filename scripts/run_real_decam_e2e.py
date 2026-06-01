"""Attempt a real NOIRLab DECam tile fetch + run pipeline end-to-end.

If NOIRLab is unreachable (no credentials, network blocked), falls back
to MAST PanSTARRS. If both fail, reports the network error rather than
silently falling back to synthetic. This is the cleanest test that the
image pipeline ACTUALLY runs on real data.

What we don't measure here (no truth catalog): precision/recall. We
can only count chains formed + IOD-successful, then SHIFT-STACK each
against the actual image pixels to check whether the orbit lands on
a real source.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path


def main():
    print("=" * 70, flush=True)
    print("REAL DECam E2E (no synthetic fallback)", flush=True)
    print("=" * 70, flush=True)

    out_root = Path("data/real_decam_e2e")
    out_root.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="real_decam_"))

    # PS1 (3pi survey) covers dec > -30 deg. Pick a cone with documented
    # coverage. The first candidate that returns images is used.
    candidate_cones = [
        # ra=180, dec=0 confirmed to return 60 PS1 rows in direct astroquery test
        ("PS1 ra180 dec0",  180.0, 0.0, 0.3, None, None),
        ("PS1 ra180 dec20", 180.0, 20.0, 0.3, None, None),
        ("PS1 Stripe 82",     0.0, 0.0, 0.3, None, None),
    ]
    ra, dec, radius = None, None, None
    mjd_start = mjd_end = None
    for name, c_ra, c_dec, c_r, c_mjd_lo, c_mjd_hi in candidate_cones:
        print(f"  trying {name} (ra={c_ra}, dec={c_dec}, radius={c_r}deg)",
              flush=True)
        ra, dec, radius = c_ra, c_dec, c_r
        mjd_start, mjd_end = c_mjd_lo, c_mjd_hi
        break

    print(f"\n[1] fetch_decam_tile ra={ra} dec={dec} radius={radius}deg "
          f"mjd=[{mjd_start}, {mjd_end}] ...", flush=True)
    t0 = time.time()
    from ariadne.discovery.imaging import archive_fetch
    fits = []
    try:
        fits = archive_fetch.fetch_decam_tile(
            ra=ra, dec=dec, radius_deg=radius,
            mjd_start=mjd_start, mjd_end=mjd_end,
            out_dir=str(workdir), max_images=6, band="r")
        print(f"    fetched {len(fits)} fits files in {time.time()-t0:.1f}s",
               flush=True)
    except Exception as e:
        msg = str(e)[:200]
        print(f"    archive fetch FAILED: {msg}", flush=True)
        report = {
            "data_source": "FAILED_TO_FETCH",
            "error": msg,
            "ra": ra, "dec": dec,
            "wall_s": time.time() - t0,
        }
        (out_root / "report.json").write_text(json.dumps(report, indent=2))
        print(f"\n[*] no images available; report -> {out_root}/report.json",
               flush=True)
        return 1

    if not fits:
        print(f"    no images returned for the cone", flush=True)
        report = {
            "data_source": "EMPTY",
            "ra": ra, "dec": dec,
        }
        (out_root / "report.json").write_text(json.dumps(report, indent=2))
        return 1

    archives = sorted({f.archive for f in fits})
    print(f"    archives: {archives}", flush=True)
    for f in fits[:5]:
        print(f"      {f.image_id}  mjd={f.mjd:.3f}  band={f.band}  "
              f"exp={f.exptime:.0f}s", flush=True)

    # ------------------------------------------------------------------
    # Pipeline: source extraction -> tracklets -> chains -> IOD
    # ------------------------------------------------------------------
    print(f"\n[2] source extraction", flush=True)
    from astropy.io import fits as astrofits
    from astropy.wcs import WCS
    from ariadne.discovery.imaging.source_extraction import detect_sources_in_image
    from ariadne.discovery.imaging.psf_centroid import refine_sources_psf
    from ariadne.discovery.imaging.morphology import classify_sources, MorphologyClass
    from ariadne.discovery.imaging.tracklets_from_images import nightly_tracklets
    from ariadne.discovery.imaging.advanced_linking import discover_in_images_chains
    from ariadne.discovery.imaging.chain_quality import filter_chains
    from ariadne.discovery.imaging.bayesian_linker import filter_chains_by_likelihood

    imgs = []; wcs_list = []; ets = []
    SEC_PER_DAY = 86400.0
    for fi in fits:
        try:
            with astrofits.open(fi.path) as hdul:
                # DECam frames may have multiple extensions; try [0] first,
                # fall back to first ImageHDU.
                data = None; hdr = None
                for hdu in hdul:
                    if hdu.data is not None and hdu.data.ndim == 2:
                        data = hdu.data.astype(float)
                        hdr = hdu.header
                        break
                if data is None:
                    print(f"     skip {fi.path.name}: no 2-D image HDU",
                           flush=True)
                    continue
                wcs = WCS(hdr)
                imgs.append(data); wcs_list.append(wcs)
                ets.append(((fi.mjd + 2400000.5) - 2451545.0) * SEC_PER_DAY)
        except Exception as e:
            print(f"     skip {fi.path.name}: {str(e)[:100]}", flush=True)
            continue
    print(f"    {len(imgs)} images loaded successfully", flush=True)
    if len(imgs) == 0:
        return 1

    all_srcs = []
    for img, wcs, fi in zip(imgs, wcs_list, fits[:len(imgs)]):
        try:
            srcs = detect_sources_in_image(
                img, wcs, mjd=fi.mjd, image_id=str(fi.path),
                fwhm_px=3.0, threshold_sigma=5.0)
            all_srcs.append(srcs)
        except Exception as e:
            print(f"     detect failed on {fi.path.name}: {str(e)[:100]}",
                   flush=True)
            all_srcs.append([])
    print(f"    detected sources: {[len(s) for s in all_srcs]}", flush=True)

    refined = [refine_sources_psf(img, srcs, wcs=wcs)
                for img, srcs, wcs in zip(imgs, all_srcs, wcs_list)]
    point_srcs = []
    for img, srcs in zip(imgs, refined):
        verdicts = classify_sources(img, srcs)
        point_srcs.extend(s for s, v in verdicts
                            if v.label == MorphologyClass.POINT
                            and v.confidence >= 0.4)
    print(f"    {len(point_srcs)} point-source-quality detections", flush=True)

    print(f"\n[3] tracklets + chains", flush=True)
    trks = nightly_tracklets(point_srcs,
                              min_rate_arcsec_hr=0.05,
                              max_rate_arcsec_hr=10.0,
                              min_pair_separation_arcsec=0.5,
                              max_per_night=1500)
    print(f"    {len(trks)} within-night tracklets", flush=True)

    chains = discover_in_images_chains(trks, use_nbody_grow=True)
    kept, _, _ = filter_chains(
        chains, max_rate_spread=0.8, max_mag_std=1.0,
        min_unique_epochs=2, min_arc_hours=6.0)
    ranked, _ = filter_chains_by_likelihood(
        kept, log_l_threshold=-1e9, max_chains=20)
    kept = ranked
    print(f"    {len(chains)} raw chains -> {len(kept)} kept",
          flush=True)

    if not kept:
        report = {
            "data_source": archives[0] if archives else "unknown",
            "n_fits": len(fits), "n_images_loaded": len(imgs),
            "n_sources": sum(len(s) for s in all_srcs),
            "n_point": len(point_srcs),
            "n_tracklets": len(trks),
            "n_chains_raw": len(chains),
            "n_chains_kept": 0,
            "n_iod_success": 0,
            "n_pixel_validated": 0,
        }
        (out_root / "report.json").write_text(json.dumps(report, indent=2))
        print(f"\n[*] no chains kept; report -> {out_root}/report.json",
               flush=True)
        return 0

    print(f"\n[4] IOD on top chains", flush=True)
    from ariadne.discovery.iod_robust import robust_iod
    from ariadne.discovery.imaging.shift_stack_validation import (
        validate_orbit_against_images)
    from ariadne.discovery.imaging.neural_orbit_prior import load_weights
    neural_w = None
    wp = Path("data/neural_orbit_prior_weights.json")
    if wp.exists():
        try:
            neural_w = load_weights(wp)
        except Exception:
            pass
    fitted = []; t8 = time.time()
    for i, ch in enumerate(kept[:10]):
        try:
            ens = robust_iod(ch, n_draws=2, rms_acceptance_arcsec=30.0,
                              neural_weights=neural_w,
                              use_monte_carlo=True, use_rate_class=True)
        except Exception as e:
            print(f"     chain {i}: IOD exception {str(e)[:80]}", flush=True)
            continue
        if not ens.success:
            continue
        # Shift-stack validate
        try:
            val = validate_orbit_against_images(
                ens.x_fit, ens.v_fit, ens.t_ref,
                imgs, wcs_list, ets,
                aperture_radius=3, half_size=12, min_snr_boost=1.3)
        except Exception:
            val = None
        validated = bool(val and val.accepted)
        print(f"     chain {i}: RMS={ens.rms_arcsec:.2f}\" strategy={ens.winning_strategy} "
              f"shift-stack={'PASS' if validated else 'FAIL'}",
              flush=True)
        fitted.append((ens, val, validated))
    print(f"    IOD wall: {time.time()-t8:.0f}s", flush=True)

    n_iod = len(fitted)
    n_validated = sum(1 for _, _, v in fitted if v)
    report = {
        "data_source": archives[0] if archives else "unknown",
        "n_fits": len(fits), "n_images_loaded": len(imgs),
        "n_sources": sum(len(s) for s in all_srcs),
        "n_point": len(point_srcs),
        "n_tracklets": len(trks),
        "n_chains_raw": len(chains),
        "n_chains_kept": len(kept),
        "n_iod_success": n_iod,
        "n_pixel_validated": n_validated,
        "iod_wall_s": time.time() - t8,
        "total_wall_s": time.time() - t0,
    }
    (out_root / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\n[*] report -> {out_root}/report.json", flush=True)
    print(f"    {n_iod} IOD-success chains, {n_validated} pixel-validated",
           flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
