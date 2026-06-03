"""Variability/transient detection (Stage 3): the Engine-2 spotter.

Difference each epoch of a same-field stack against a static reference (median,
which rejects both movers and one-off transients), detect the POSITIVE residuals
(things brighter than baseline), then cluster the residual detections across
epochs by sky position:

  * a cluster at a FIXED position present in several epochs with varying
    brightness  -> VARIABLE star
  * a fixed-position source present in only a subset of epochs (appears/fades)
    -> TRANSIENT (nova / SN / CV outburst)
  * a cluster whose position MOVES epoch-to-epoch -> a MOVER (hand to Engine 1)

Each stationary candidate carries its per-epoch light curve, ready for
light_curve.analyze_light_curve + characterize. Built on the validated
PSF-matched difference imaging.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .difference import psf_matched_difference
from .source_extraction import detect_sources_in_image


@dataclass
class VariabilityCandidate:
    ra: float
    dec: float
    kind: str                  # 'variable' | 'transient' | 'mover'
    n_epochs: int
    times: list = field(default_factory=list)     # mjd per detection
    mags: list = field(default_factory=list)
    pos_scatter_arcsec: float = 0.0
    note: str = ""


def find_variability_candidates(images, wcs_list, mjds, *, fwhm_px: float = 4.0,
                                threshold_sigma: float = 5.0,
                                match_arcsec: float = 2.5,
                                min_epochs: int = 2,
                                min_frac_change: float = 0.2) -> list[VariabilityCandidate]:
    """Detect variables/transients across same-field epochs via difference
    imaging + cross-epoch clustering. `images` are co-registered 2-D arrays."""
    stack = np.stack([np.asarray(im, float) for im in images], axis=0)
    reference = np.median(stack, axis=0)        # static sky (rejects movers + 1-epoch transients)

    dets = []   # (ra, dec, mjd, mag, epoch, resid_flux)
    for e, (im, wcs, mjd) in enumerate(zip(images, wcs_list, mjds)):
        try:
            res = psf_matched_difference(np.asarray(im, float), reference, fwhm_px=fwhm_px)
            srcs = detect_sources_in_image(res.residual, wcs, mjd=mjd, image_id=f"diff{e}",
                                           fwhm_px=fwhm_px, threshold_sigma=threshold_sigma,
                                           auto_fwhm=False)
        except Exception:
            continue
        for s in srcs:
            dets.append((s.ra, s.dec, mjd, s.mag, e, max(s.flux, 0.0)))
    if not dets:
        return []

    # baseline (reference) flux sampler -> reject subtraction artifacts: a real
    # variable/transient is a LARGE fractional brightness change, while a residual
    # on a constant bright star is a tiny fraction of that star's flux.
    ref = reference
    ref_bg = float(np.nanmedian(ref))
    rw = wcs_list[0]

    def _ref_flux(ra, dec):
        try:
            x, y = rw.world_to_pixel_values(ra, dec)
            xi, yi = int(round(float(x))), int(round(float(y)))
            H, W = ref.shape
            if 3 <= xi < W - 3 and 3 <= yi < H - 3:
                return max(0.0, float(np.max(ref[yi - 3:yi + 4, xi - 3:xi + 4])) - ref_bg)
        except Exception:
            pass
        return 0.0

    # cluster across epochs by sky position (tangent plane about the field centre)
    ra0 = float(np.median([d[0] for d in dets]))
    dec0 = float(np.median([d[1] for d in dets]))
    cd = math.cos(math.radians(dec0))
    xy = np.array([[(d[0] - ra0) * cd * 3600.0, (d[1] - dec0) * 3600.0] for d in dets])
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(xy)
        pairs = tree.query_ball_tree(tree, match_arcsec)
    except ImportError:
        pairs = [[j for j in range(len(xy))
                  if math.hypot(*(xy[i] - xy[j])) <= match_arcsec] for i in range(len(xy))]
    # union-find clustering
    parent = list(range(len(xy)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i, nb in enumerate(pairs):
        for j in nb:
            parent[find(i)] = find(j)
    clusters = {}
    for i in range(len(xy)):
        clusters.setdefault(find(i), []).append(i)

    out = []
    for idxs in clusters.values():
        epochs = sorted({dets[i][4] for i in idxs})
        if len(epochs) < 1:
            continue
        ras = np.array([dets[i][0] for i in idxs]); decs = np.array([dets[i][1] for i in idxs])
        times = [dets[i][2] for i in idxs]; mags = [dets[i][3] for i in idxs]
        scat = float(np.hypot((ras - ras.mean()) * cd, decs - decs.mean()).std() * 3600.0)
        n_ep = len(epochs)
        # REAL/BOGUS: reject subtraction artifacts on constant bright stars.
        # Keep only sources whose residual is a significant FRACTION of the
        # baseline flux (a true brightening), or that sit on ~empty reference
        # sky (a transient). A 1% residual on a bright constant star is dropped.
        res_flux = float(np.median([dets[i][5] for i in idxs]))
        rflux = _ref_flux(float(ras.mean()), float(decs.mean()))
        frac = res_flux / (rflux + res_flux + 1e-9)
        if frac < min_frac_change:
            continue
        # moving? position correlates with time across distinct epochs
        kind = "transient"
        note = ""
        if n_ep >= 3:
            tt = np.array([dets[i][2] for i in idxs])
            drift = float(np.hypot(np.polyfit(tt, ras * cd, 1)[0], np.polyfit(tt, decs, 1)[0])
                          * 3600.0 * (max(times) - min(times)))   # arcsec over the arc
            if drift > 3 * match_arcsec and scat > match_arcsec:
                kind = "mover"
                note = "position drifts with time -> moving object (Engine 1)"
        if kind != "mover":
            kind = "variable" if n_ep >= max(min_epochs, 2) else "transient"
        out.append(VariabilityCandidate(
            ra=float(ras.mean()), dec=float(decs.mean()), kind=kind, n_epochs=n_ep,
            times=times, mags=mags, pos_scatter_arcsec=scat, note=note))
    # only keep real candidates: stationary sources seen in >=min_epochs, or
    # bright single-epoch transients
    return [c for c in out if (c.kind == "variable" and c.n_epochs >= min_epochs)
            or c.kind == "transient" or c.kind == "mover"]
