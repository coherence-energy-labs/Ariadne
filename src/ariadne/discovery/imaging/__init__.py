"""Imaging-data discovery pipeline -- source extraction + multi-night tracklets.

For surveys that DON'T publish an alert stream (DECam Legacy Surveys, Subaru HSC
archive, amateur image stacks), the input is FITS images rather than ALeRCE-style
alerts. This subpackage extracts sources from images and builds tracklets that the
rest of Ariadne (linkage + IOD+LM + SkyBoT cross-match) operates on.

Pipeline:
  1. EXTRACT (source_extraction.py): detect sources in each FITS image via
     photutils background subtraction + DAOFIND / IRAF starfinder. Output:
     per-image source catalogues with (RA, Dec, mag, MJD, image_id).
  2. CHAIN (tracklets_from_images.py): pair detections across same-night exposures
     into single-night tracklets; chain multi-night tracklets into candidate
     orbital arcs based on rate consistency.
  3. ARIADNE: feed the candidate arcs to discovery.linkage + IOD + LM (same path
     as MPC ITF).

The advantage over the MPC archive: DECam / HSC / Subaru go ~2 magnitudes DEEPER
than the MPC's public bar. A genuine new TNO at mag 24 will be missing from MPC
but visible in these surveys.

Honest scope:
  - This is the source-finding + tracklet-building scaffolding. It works on real
    FITS images via photutils + WCS plate solving.
  - Bulk-downloading hundreds of GB of DECam DR10 images and reprocessing them
    is a production-grade operation that lives outside this library.
  - Validation here uses SYNTHETIC source catalogues so the pipeline is testable
    without a multi-GB image dataset.
"""
from .source_extraction import (
    detect_sources_in_image,
    Source,
)
from .tracklets_from_images import (
    nightly_tracklets,
    chain_multi_night,
)

__all__ = ["detect_sources_in_image", "Source",
           "nightly_tracklets", "chain_multi_night"]
