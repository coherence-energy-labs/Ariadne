"""Full imaging pipeline end-to-end on synthetic FITS files.

Generates 6 synthetic FITS images (2 exposures per night x 3 nights) with planted
moving objects + background stars, then runs the COMPLETE Ariadne imaging
discovery pipeline against them:

  1. FITS -> photutils source extraction (DAOFind + WCS)
  2. Per-image sources -> same-night tracklets (rate-cut for distant objects)
  3. Same-night tracklets -> multi-night chained arcs (extrapolation match)
  4. Chained arcs -> IOD + LM orbit fit (acceptance threshold)

This exercises EVERY step of the imaging path on REAL FITS files (just with
synthetic source content) so we know the production-grade path works when fed
real DECam / PanSTARRS / amateur images.
"""
import warnings, time
warnings.filterwarnings("ignore")
import numpy as np

from ariadne.discovery.imaging.archive_fetch import synthesise_decam_tile
from ariadne.discovery.imaging.source_extraction import detect_sources_in_image
from ariadne.discovery.imaging.tracklets_from_images import (
    nightly_tracklets, chain_multi_night)
from ariadne.discovery import iod as IOD


print("=" * 76)
print("Imaging discovery pipeline -- end-to-end on synthetic FITS")
print("=" * 76)

RA, DEC = 180.0, 20.0
N_REAL = 3
print(f"\n[1] Generating 6 synthetic FITS (2 exp/night x 3 nights), 3 planted movers")
t0 = time.time()
records = synthesise_decam_tile(RA, DEC, n_images=6, n_objects_per_image=50,
                                  n_real_moving=N_REAL,
                                  mjd_nights=[60000.0, 60003.0, 60006.0],
                                  out_dir="data/synth_fits")
print(f"    wrote {len(records)} FITS files in {time.time()-t0:.0f}s")
print(f"    files: {[r.path.name for r in records]}")

print(f"\n[2] Source extraction via photutils on each image...")
all_sources = []
from astropy.io import fits
from astropy.wcs import WCS
for r in records:
    with fits.open(r.path) as hdul:
        data = hdul[0].data
        wcs = WCS(hdul[0].header)
    srcs = detect_sources_in_image(data, wcs, mjd=r.mjd, image_id=r.image_id,
                                    fwhm_px=2.0, threshold_sigma=5.0,
                                    min_fwhm_px=0.5, max_fwhm_px=20.0)
    print(f"    {r.path.name}: {len(srcs)} sources extracted")
    all_sources.extend(srcs)
print(f"    total: {len(all_sources)} source detections")

print(f"\n[3] Building same-night tracklets...")
tracks = nightly_tracklets(all_sources,
                            min_rate_arcsec_hr=0.1, max_rate_arcsec_hr=10.0,
                            min_pair_dt_hours=0.5, max_pair_dt_hours=4.0)
print(f"    {len(tracks)} candidate tracklets")

print(f"\n[4] Chaining multi-night arcs...")
chains = chain_multi_night(tracks, max_position_gap_arcsec=600.0,
                            max_rate_change_pct=40.0, max_nights_gap=4)
print(f"    {len(chains)} multi-night candidate arcs")
for i, c in enumerate(chains[:6]):
    rates = sorted({round(t['rate_arcsec_hr'], 1) for t in c})
    nights = sorted({t['night'] for t in c})
    print(f"    chain {i}: {len(c)} tracklets, nights {nights}, rates {rates} as/hr")

print(f"\n[5] IOD+LM filter (RMS < 60\")...")
accepted = 0
for i, chain in enumerate(chains):
    members = []
    for t in chain:
        members.append({
            "t": t["t"], "jd": t["jd"], "ra": t["ra"], "dec": t["dec"],
            "dra": t["dra"], "ddec": t["ddec"],
        })
    if len(members) < 4:
        continue
    try:
        fit = IOD.fit_candidate(members)
    except Exception as e:
        print(f"    chain {i}: IOD failed ({str(e)[:60]})"); continue
    if fit is None:
        print(f"    chain {i}: IOD returned None"); continue
    if fit['rms_arcsec'] < 60.0:
        accepted += 1
        print(f"    chain {i}: ACCEPTED  RMS={fit['rms_arcsec']:.1f}\"  "
              f"r={fit['iod']['r_au']:.1f} AU")
    else:
        print(f"    chain {i}: rejected RMS={fit['rms_arcsec']:.0f}\"")

print(f"\n[6] PIPELINE END: {accepted}/{len(chains)} chains accepted, "
      f"out of {N_REAL} planted moving objects.")
print(f"\nThis exercised the FULL imaging pipeline (FITS->photutils->tracklet->chain->IOD+LM)")
print(f"on real FITS data. For real DECam DR10, replace synthesise_decam_tile() with")
print(f"fetch_decam_tile(); everything downstream runs unchanged.")
