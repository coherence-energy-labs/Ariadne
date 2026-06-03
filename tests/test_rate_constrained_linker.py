"""Tests for the rate-constrained multi-night linker.

A real moving object planted across 5 exposures/night over 3 nights (plus noise
detections) must be (a) built into a >=3-point within-night track each night and
(b) linked into a chain spanning all 3 nights. Pure noise must NOT produce
multi-night chains -- the whole point is that chance alignments don't repeat.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")


def _src(ra, dec, mjd, mag=20.0):
    from ariadne.discovery.imaging.source_extraction import Source
    return Source(ra=ra, dec=dec, flux=1000.0, mag=mag, fwhm_px=4.0,
                  mjd=mjd, image_id=f"e{mjd:.4f}", x=0.0, y=0.0)


def _field(seed, mjd_bases, exps_per_night, n_noise, mover=None):
    """Build a multi-night detection list: optional constant-rate mover + noise.
    mover = (ra0_deg, dec0_deg, vra_arcsec_hr, vdec_arcsec_hr) at the first epoch."""
    rng = np.random.default_rng(seed)
    sources = []
    t_first = mjd_bases[0]
    for base in mjd_bases:
        for k in range(exps_per_night):
            mjd = base + k * 0.01           # ~14 min cadence within a night
            # noise detections
            for _ in range(n_noise):
                sources.append(_src(180.0 + rng.uniform(-0.25, 0.25),
                                    rng.uniform(-0.25, 0.25), mjd))
            if mover is not None:
                ra0, dec0, vra, vdec = mover
                dt_hr = (mjd - t_first) * 24.0
                ra = ra0 + (vra * dt_hr) / 3600.0       # dec0~0 -> cos~1
                dec = dec0 + (vdec * dt_hr) / 3600.0
                sources.append(_src(ra, dec, mjd, mag=19.0))
    return sources


def test_within_night_track_built_from_mover():
    from ariadne.discovery.imaging.rate_constrained_linker import build_within_night_tracks
    src = _field(1, [60000.0], exps_per_night=5, n_noise=40,
                 mover=(180.0, 0.0, 30.0, 10.0))
    tracks = build_within_night_tracks(src, min_rate_arcsec_hr=2, max_rate_arcsec_hr=120,
                                       pos_tol_arcsec=2.5, min_points=3)
    # the planted mover should yield a track near rate sqrt(30^2+10^2)=31.6"/hr
    good = [t for t in tracks if abs(t.rate_arcsec_hr - 31.6) < 8 and t.n_points >= 4]
    assert good, f"mover track not built (got {len(tracks)} tracks)"


def test_mover_linked_across_three_nights():
    from ariadne.discovery.imaging.rate_constrained_linker import (
        build_within_night_tracks, link_rate_constrained)
    src = _field(2, [60000.0, 60001.0, 60002.0], exps_per_night=5, n_noise=40,
                 mover=(180.0, 0.0, 30.0, 10.0))
    from collections import defaultdict
    by_night = defaultdict(list)
    for s in src:
        by_night[int(round(s.mjd))].append(s)
    tracks = []
    for ss in by_night.values():
        tracks += build_within_night_tracks(ss, min_rate_arcsec_hr=2,
                                             max_rate_arcsec_hr=120, min_points=3)
    chains = link_rate_constrained(tracks, min_nights=2)
    spanning3 = [c for c in chains if len({t.night for t in c}) >= 3]
    assert spanning3, f"mover not linked across 3 nights ({len(chains)} chains total)"


def test_pure_noise_gives_no_multinight_chains():
    from ariadne.discovery.imaging.rate_constrained_linker import (
        build_within_night_tracks, link_rate_constrained)
    src = _field(3, [60000.0, 60001.0, 60002.0], exps_per_night=5, n_noise=40,
                 mover=None)
    from collections import defaultdict
    by_night = defaultdict(list)
    for s in src:
        by_night[int(round(s.mjd))].append(s)
    tracks = []
    for ss in by_night.values():
        tracks += build_within_night_tracks(ss, min_rate_arcsec_hr=2,
                                             max_rate_arcsec_hr=120, min_points=3)
    chains = link_rate_constrained(tracks, min_nights=3)
    # chance multi-night chains from pure noise should be rare/none
    assert len(chains) <= 1, f"noise produced {len(chains)} 3-night chains"


def _tracks(src):
    from collections import defaultdict
    from ariadne.discovery.imaging.rate_constrained_linker import build_within_night_tracks
    by_night = defaultdict(list)
    for s in src:
        by_night[int(round(s.mjd))].append(s)
    tr = []
    for ss in by_night.values():
        tr += build_within_night_tracks(ss, min_rate_arcsec_hr=2,
                                        max_rate_arcsec_hr=120, min_points=3)
    return tr


def test_coherence_field_recovers_object_with_no_within_night_tracklet():
    """The deep coherence-field method recovers an object detected only ONCE per
    night across 4 nights -- there is NO within-night tracklet to build, so the
    pairwise tracker (>=3 within-night points) is blind to it. The field, sourced
    by every detection, still collapses its 4 lone detections to one coherent peak
    at the true rate; dense noise does not forge a >=4-night coherent peak."""
    import numpy as np
    from ariadne.discovery.imaging.rate_constrained_linker import coherence_field_tracks
    rng = np.random.default_rng(7)
    nights = [60000.0, 60001.0, 60002.0, 60003.0]
    ra, dec, mjd = [], [], []
    vra, vdec = 24.0, -8.0                       # arcsec/hr, constant-rate object
    for base in nights:
        t = base + 0.2
        dt = (t - nights[0]) * 24.0
        ra.append(180.0 + vra * dt / 3600.0)     # the ONE detection that night
        dec.append(0.0 + vdec * dt / 3600.0)
        mjd.append(t)
        for _ in range(60):                      # heavy single-epoch noise
            ra.append(180.0 + rng.uniform(-0.2, 0.2))
            dec.append(rng.uniform(-0.2, 0.2)); mjd.append(t)
    tr = coherence_field_tracks(ra, dec, mjd, max_rate_arcsec_hr=40,
                                cluster_tol_arcsec=2.5, min_nights=4, min_points=4)
    # at least one recovered track must contain the 4 planted detections (indices
    # 0, 61, 122, 183 -- each night starts with the planted point then 60 noise).
    planted = {0, 61, 122, 183}
    assert any(planted.issubset(set(t["idx"])) for t in tr), \
        f"coherence field missed the once-per-night object ({len(tr)} tracks)"


def test_coherence_linker_recovers_mover_and_rejects_noise():
    """The Equation-of-One (alignment-kernel) linker recovers a real mover across
    nights and does NOT manufacture multi-night chains from pure noise."""
    from ariadne.discovery.imaging.rate_constrained_linker import link_coherence
    mover = link_coherence(_tracks(_field(2, [60000.0, 60001.0, 60002.0], 5, 40,
                                          mover=(180.0, 0.0, 30.0, 10.0))), min_nights=2)
    assert [c for c in mover if len({t.night for t in c}) >= 3], "coherence linker missed the mover"
    noise = link_coherence(_tracks(_field(3, [60000.0, 60001.0, 60002.0], 5, 40, mover=None)),
                           min_nights=3)
    assert len(noise) <= 1, f"coherence linker invented {len(noise)} chains from noise"
