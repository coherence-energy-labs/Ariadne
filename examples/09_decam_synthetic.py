"""DECam-style image-based discovery pipeline -- end-to-end on synthetic data.

Demonstrates the imaging-discovery path Ariadne provides for surveys without an
alert stream (DECam Legacy Surveys, Subaru HSC archive, amateur image stacks):

  1. Image -> source extraction (photutils, see imaging/source_extraction.py)
  2. Per-image sources -> same-night tracklets (rate-cut for distant objects)
  3. Same-night tracklets -> multi-night chained arcs
  4. Chained arcs -> Ariadne HelioLinC linker + IOD+LM orbit fit

The fundamental pipeline is identical to the MPC ITF path; only the FRONT END
differs (FITS-image source extraction instead of MPC 80-col records).

This example uses synthesised source catalogues so it runs without needing GB of
DECam data; a production run would replace the synthesis with real per-image
photutils calls (one-line swap).

Run:  PYTHONPATH=src python examples/09_decam_synthetic.py
"""
import math, warnings
warnings.filterwarnings("ignore")

from ariadne.discovery.imaging.source_extraction import synthesise_sources, Source
from ariadne.discovery.imaging.tracklets_from_images import (
    nightly_tracklets, chain_multi_night,
)
from ariadne.discovery import iod as IOD

print("=" * 76)
print("DECam-style imaging discovery pipeline (synthetic source catalogues)")
print("=" * 76)

# Plant 3 real moving objects + many random sources across 3 nights.
# Each "real" object appears in BOTH images on each of 3 nights at the right
# rate-of-motion offset.
RA_C, DEC_C = 180.0, 0.0
NIGHTS = [60000.0, 60003.0, 60006.0]
REAL_RATES_AS_HR = [0.5, 2.0, 3.0]   # 3 real objects, slow / medium / fast TNO
REAL_DIRS = [(1.0, 0.2), (-0.5, 0.8), (0.0, -1.0)]   # (dra_dir, ddec_dir)

# Initial RA/Dec for each real object
import random
rng = random.Random(0)
real_seeds = [(RA_C + (rng.random() - 0.5) * 0.5, DEC_C + (rng.random() - 0.5) * 0.5)
              for _ in REAL_RATES_AS_HR]

sources_all = []
for night_idx, mjd0 in enumerate(NIGHTS):
    for half in (0.0, 2.0):    # two exposures per night, 2-hour gap
        t = mjd0 + half / 24.0
        # 2 detections per real object per exposure-pair
        for k, ((ra0, dec0), rate, (dra_dir, ddec_dir)) in enumerate(
                zip(real_seeds, REAL_RATES_AS_HR, REAL_DIRS)):
            norm = math.hypot(dra_dir, ddec_dir)
            dra_per_day = rate * 24.0 / 3600.0 * dra_dir / norm \
                           / math.cos(math.radians(dec0))
            ddec_per_day = rate * 24.0 / 3600.0 * ddec_dir / norm
            dt = t - NIGHTS[0]
            sources_all.append(Source(
                ra=ra0 + dra_per_day * dt,
                dec=dec0 + ddec_per_day * dt,
                flux=10000.0, mag=20.0, fwhm_px=3.0, mjd=t,
                image_id=f"img_n{night_idx}_e{int(half)}",
                x=0.0, y=0.0,
            ))
    # Background interlopers (one-off): 80 per night
    sources_all.extend(synthesise_sources(80, RA_C, DEC_C, width_deg=1.0,
                                          mjd=mjd0, image_id=f"bg_n{night_idx}",
                                          seed=night_idx * 100))

print(f"\n[1] Synthesised {len(sources_all)} source detections across "
      f"{len(NIGHTS)} nights x 2 exposures")
print(f"    Planted {len(real_seeds)} real moving objects "
      f"(rates {REAL_RATES_AS_HR} arcsec/hr)")

print(f"\n[2] Building same-night tracklets...")
tracks = nightly_tracklets(sources_all,
                            min_rate_arcsec_hr=0.1, max_rate_arcsec_hr=10.0,
                            min_pair_dt_hours=0.5, max_pair_dt_hours=4.0)
print(f"    {len(tracks)} candidate tracklets formed")

print(f"\n[3] Chaining multi-night arcs (extrapolation match)...")
chains = chain_multi_night(tracks, max_position_gap_arcsec=300.0,
                            max_rate_change_pct=30.0, max_nights_gap=4)
print(f"    {len(chains)} multi-night candidate arcs")
for i, c in enumerate(chains[:6]):
    rates = sorted({round(t['rate_arcsec_hr'], 1) for t in c})
    nights = sorted({t['night'] for t in c})
    print(f"    chain {i}: {len(c)} tracklets across nights {nights}, "
          f"rates {rates} as/hr")

print(f"\n[4] Orbit-fit each candidate arc (IOD + LM) and report acceptance:")
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
        print(f"    chain {i}: IOD failed ({str(e)[:60]})")
        continue
    if fit is None:
        continue
    if fit['rms_arcsec'] < 15.0:
        accepted += 1
        print(f"    chain {i}: ACCEPTED  RMS={fit['rms_arcsec']:.1f}\"  "
              f"r={fit['iod']['r_au']:.1f} AU")
    else:
        print(f"    chain {i}: high-RMS ({fit['rms_arcsec']:.0f}\")  -- rejected")

print(f"\n[5] PIPELINE END: {accepted}/{len(chains)} chains pass the orbit-fit filter")
print(f"\nHonest scope: this uses SYNTHETIC source catalogues. For real DECam DR10")
print(f"the photutils source extraction (imaging/source_extraction.py) handles the")
print(f"raw FITS images; everything else in this pipeline runs unchanged.")
