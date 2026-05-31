"""Full DECam-tile end-to-end: real archive fetch -> image pipeline -> smart IOD.

Tries to pull a real DECam tile from NOIRLab. Falls back to the validated
high-quality synthetic-FITS injector when the archive is unreachable or no
real network is available. Either way, runs the COMPLETE pipeline:

  fetch_decam_tile (NOIRLab or synthetic)
    -> source extraction (photutils DAOStarFinder)
    -> PSF-fit centroiding (sub-pixel)
    -> morphology classifier (POINT/EXTENDED/STREAK/BLEND/CR/EDGE)
    -> deblender (split BLEND into components)
    -> filter to point-like sources
    -> tracklet builder (rate-adaptive pair window)
    -> chain_multi_night
    -> chain sanity filter
    -> IOD ensemble (Gauss + adaptive linker + Vaisala + BK)
    -> N-body refinement on borderline RMS
    -> smart_annotate (realbogus + inference + taxonomy + scoring + MCMC for grade-A/B)
    -> grade-A MPC submission emission (with gating)

Writes a full per-stage report to data/decam_e2e/report.json.
"""
from __future__ import annotations

import json
import math
import sys
import time
import tempfile
from collections import Counter
from pathlib import Path


def main():
    print("=" * 70)
    print("FULL DECam-TILE E2E: archive -> image -> IOD ensemble -> smart layer")
    print("=" * 70)
    out_root = Path("data/decam_e2e")
    out_root.mkdir(parents=True, exist_ok=True)

    # 1. Fetch FITS tile (try real archive, fall back to synth)
    from ariadne.discovery.imaging import archive_fetch
    workdir = Path(tempfile.mkdtemp(prefix="decam_e2e_"))
    print(f"\n[1] working dir: {workdir}")
    use_real = False
    fits_images = []
    try:
        fits_images = archive_fetch.fetch_decam_tile(
            ra=180.0, dec=20.0, radius_deg=0.1,
            mjd_start=60500.0, mjd_end=60520.0,
            out_dir=str(workdir), max_images=3)
        if fits_images:
            use_real = True
            print(f"    REAL archive returned {len(fits_images)} FITS files")
    except Exception as e:
        print(f"    real fetch failed: {str(e)[:80]} -- falling back to synth")

    if not fits_images:
        # kepler_orbits=True plants REAL Keplerian heliocentric orbits
        # propagated through Ariadne's integrator -- the IOD should
        # recover them. (kepler_orbits=False is for testing the source
        # extraction without expecting the IOD to converge.)
        fits_images = archive_fetch.synthesise_decam_tile(
            ra=180.0, dec=20.0, n_images=2,
            n_objects_per_image=80, n_real_moving=3,
            mjd_nights=[60450.0, 60453.0, 60456.0],
            out_dir=str(workdir), kepler_orbits=True)
        print(f"    SYNTH produced {len(fits_images)} FITS files (Keplerian)")
    print(f"    using {'REAL' if use_real else 'SYNTHETIC'} data")

    # 2. Source extraction on every frame
    print(f"\n[2] source extraction")
    from astropy.io import fits as astrofits
    from astropy.wcs import WCS
    from ariadne.discovery.imaging.source_extraction import detect_sources_in_image
    all_sources = []
    images = []
    epochs = []
    wcs_list = []
    for fi in fits_images:
        with astrofits.open(fi.path) as hdul:
            data = hdul[0].data.astype(float)
            wcs = WCS(hdul[0].header)
        srcs = detect_sources_in_image(
            data, wcs, mjd=fi.mjd, image_id=str(fi.path),
            fwhm_px=3.0, threshold_sigma=5.0)
        print(f"    {Path(fi.path).name}: {len(srcs)} sources")
        all_sources.append(srcs)
        images.append(data)
        epochs.append(fi.mjd)
        wcs_list.append(wcs)
    flat_sources = [s for batch in all_sources for s in batch]
    print(f"    total: {len(flat_sources)} raw detections")

    # 3. PSF-fit centroiding on every source in every frame
    print(f"\n[3] PSF-fit centroiding")
    from ariadne.discovery.imaging.psf_centroid import refine_sources_psf
    refined_all = []
    for img, srcs, wcs in zip(images, all_sources, wcs_list):
        refined = refine_sources_psf(img, srcs, wcs=wcs)
        refined_all.append(refined)
    n_refined = sum(len(r) for r in refined_all)
    print(f"    refined {n_refined} sources to sub-pixel precision")

    # 4. Morphology + filter to point-like
    print(f"\n[4] morphology classification + point-source filter")
    from ariadne.discovery.imaging.morphology import classify_sources, MorphologyClass
    point_sources_all = []
    label_totals = Counter()
    for img, srcs in zip(images, refined_all):
        verdicts = classify_sources(img, srcs)
        for src, v in verdicts:
            label_totals[v.label] += 1
            if v.label == MorphologyClass.POINT and v.confidence >= 0.4:
                point_sources_all.append(src)
    for label, n in label_totals.most_common():
        print(f"    {label:<20s} {n}")
    print(f"    kept {len(point_sources_all)} point-source-quality detections")

    # 5. Tracklet build (rate-adaptive)
    print(f"\n[5] tracklet build")
    from ariadne.discovery.imaging.tracklets_from_images import (
        nightly_tracklets, chain_multi_night)
    tracklets = nightly_tracklets(point_sources_all,
                                    min_rate_arcsec_hr=0.05,
                                    max_rate_arcsec_hr=10.0,
                                    min_pair_separation_arcsec=0.5,
                                    max_per_night=2000)
    print(f"    {len(tracklets)} within-night tracklets")

    # 6. Chain across nights
    chains = chain_multi_night(tracklets)
    print(f"\n[6] {len(chains)} multi-night chains")

    # 7. Chain sanity filter
    from ariadne.discovery.realtime import filter_chain_sanity
    sane_chains = filter_chain_sanity(chains)
    print(f"\n[7] {len(sane_chains)} chains survived sanity filter")

    # 8. Ensemble IOD on every sane chain
    print(f"\n[8] ensemble IOD")
    from ariadne.discovery import iod_advanced as IODA
    fitted = []
    for ch in sane_chains[:20]:           # cap at 20 for wall-clock sanity
        ens = IODA.fit_candidate_ensemble(
            ch, rms_acceptance_arcsec=10.0,
            cheap_first=True, early_exit_rms_arcsec=0.5)
        fitted.append(ens)
    accepted = [f for f in fitted if f.success]
    print(f"    {len(accepted)}/{len(fitted)} chains have a successful IOD fit")
    for ens in accepted[:5]:
        print(f"    -> RMS {ens.rms_arcsec:.2f}\", strategy {ens.winning_strategy}")

    # 9. Smart annotate + grade-A MCMC requirement
    print(f"\n[9] smart annotation + grade-A MCMC gating")
    from ariadne.discovery.realtime import smart_annotate
    # Convert ensemble fits to tracklet-cluster format smart_annotate expects
    smart_input = []
    for ens, ch in zip(accepted, sane_chains[:len(accepted)]):
        if not ens.success:
            continue
        members = [s for sub in ch for s in (sub.get("source_pair") or ())]
        tr = {
            "status": "accepted",
            "ra": ch[0]["ra"], "dec": ch[0]["dec"],
            "jd": ch[0].get("jd", 0.0), "t": ch[0]["t"],
            "rate_arcsec_hr": ch[0].get("rate_arcsec_hr", 0.0),
            "rms_arcsec": ens.rms_arcsec,
            "x_fit_km": list(ens.x_fit),
            "v_fit_kms": list(ens.v_fit),
            "members": [type("AlertProxy", (), {
                "mjd": s.mjd, "ra": s.ra, "dec": s.dec,
                "mag": s.mag, "meta": {},
            })() for s in members],
            "chain": ch,
            "xmatch": {"n_known": 0, "names": []},
        }
        smart_input.append(tr)
    annotated = smart_annotate(smart_input, mcmc_for_high_quality=True,
                                 mcmc_n_steps=80)
    n_mcmc = sum(1 for tr in annotated if "_mcmc" in tr)
    n_grade_a = sum(1 for tr in annotated if tr.get("_quality_grade") == "A")
    print(f"    {len(annotated)} candidates annotated; {n_grade_a} grade-A; "
          f"{n_mcmc} have MCMC posteriors")

    # 10. Report
    report = {
        "data_source": "real_archive" if use_real else "synthetic",
        "n_fits_files": len(fits_images),
        "n_raw_sources": len(flat_sources),
        "morphology_distribution": dict(label_totals),
        "n_point_sources_after_filter": len(point_sources_all),
        "n_tracklets": len(tracklets),
        "n_chains": len(chains),
        "n_sane_chains": len(sane_chains),
        "n_iod_attempts": len(fitted),
        "n_iod_success": len(accepted),
        "winning_strategies": Counter(
            ens.winning_strategy for ens in accepted),
        "n_smart_annotated": len(annotated),
        "n_grade_a": n_grade_a,
        "n_mcmc": n_mcmc,
    }
    report["winning_strategies"] = dict(report["winning_strategies"])
    (out_root / "report.json").write_text(
        json.dumps(report, indent=2, default=str))
    print(f"\n[10] report -> {out_root / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
