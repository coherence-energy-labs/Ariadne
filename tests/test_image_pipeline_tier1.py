"""Tier-1 image-pipeline upgrades: truth tracking + chain quality + robust IOD.

Exercises the three new modules added on top of advanced_linking.py:

  synthetic_truth          truth catalogue / propagation
  chain_quality            rate / photometric / epoch / arc filters
  iod_robust               Monte Carlo + rate-class-aware ensemble
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ===========================================================================
# Phase 0: synthetic_truth
# ===========================================================================

def test_truth_catalog_save_load_roundtrip(tmp_path):
    from ariadne.discovery.imaging.synthetic_truth import (
        TruthEntry, TruthCatalog)

    entries = [
        TruthEntry(truth_id="obj_a", image_id="img_001", mjd=60450.0,
                    ra=180.0, dec=20.0, x_pix=256.0, y_pix=300.0,
                    mag=21.5, family="kepler_tno"),
        TruthEntry(truth_id="obj_b", image_id="img_001", mjd=60450.0,
                    ra=180.01, dec=20.01, x_pix=300.0, y_pix=350.0,
                    mag=22.0, family="kepler_tno"),
    ]
    cat = TruthCatalog(entries)
    p = tmp_path / "truth_catalog.json"
    cat.save(p)

    loaded = TruthCatalog.load(p)
    assert loaded.truth_ids == {"obj_a", "obj_b"}
    assert len(loaded.entries) == 2
    assert loaded.entries[0].ra == 180.0


def test_truth_catalog_source_matching():
    from ariadne.discovery.imaging.synthetic_truth import (
        TruthEntry, TruthCatalog)
    from ariadne.discovery.imaging.source_extraction import Source

    catalog = TruthCatalog([
        TruthEntry("obj_a", "img_001", 60450.0, 180.0, 20.0,
                    256.0, 300.0, 21.5, "kepler_tno"),
        TruthEntry("obj_b", "img_001", 60450.0, 181.0, 20.0,
                    400.0, 200.0, 22.0, "kepler_tno"),
    ])
    # Detection at (257, 301) is 1.4 pix from obj_a -> matches
    s1 = Source(ra=180.001, dec=20.001, flux=3000, mag=21.5, fwhm_px=3.0,
                 mjd=60450.0, image_id="img_001", x=257.0, y=301.0)
    assert catalog.match_source(s1, match_radius_pix=2.5) == "obj_a"
    # Detection at (260, 320) is 20+ pix from obj_a; no match
    s2 = Source(ra=180.005, dec=20.005, flux=3000, mag=21.5, fwhm_px=3.0,
                 mjd=60450.0, image_id="img_001", x=260.0, y=320.0)
    assert catalog.match_source(s2, match_radius_pix=2.5) is None


def test_assign_truth_to_chain_returns_dominant_truth():
    from ariadne.discovery.imaging.synthetic_truth import (
        TruthEntry, TruthCatalog, assign_truth_to_chain)

    cat = TruthCatalog([
        TruthEntry("a", "i1", 60450.0, 180.0, 20.0, 256, 300, 21, "x"),
        TruthEntry("a", "i2", 60453.0, 180.05, 20.05, 256, 300, 21, "x"),
        TruthEntry("a", "i3", 60456.0, 180.10, 20.10, 256, 300, 21, "x"),
    ])
    # Convert MJD->ET so the time-window matcher can find them.
    # ET_sec = (MJD - 51544.5) * 86400
    et_60450 = (60450.0 - 51544.5) * 86400
    et_60453 = (60453.0 - 51544.5) * 86400
    et_60456 = (60456.0 - 51544.5) * 86400
    chain_pure = [
        {"ra": math.radians(180.0), "dec": math.radians(20.0), "t": et_60450,
         "jd": 60450.0 + 2400000.5, "rate_arcsec_hr": 2.0},
        {"ra": math.radians(180.05), "dec": math.radians(20.05), "t": et_60453,
         "jd": 60453.0 + 2400000.5, "rate_arcsec_hr": 2.0},
        {"ra": math.radians(180.10), "dec": math.radians(20.10), "t": et_60456,
         "jd": 60456.0 + 2400000.5, "rate_arcsec_hr": 2.0},
    ]
    tid, purity = assign_truth_to_chain(chain_pure, cat,
                                          match_radius_arcsec=10.0)
    assert tid == "a"
    assert purity == 1.0

    # Off-sky chain: nothing matches anywhere
    chain_offsky = [
        {"ra": math.radians(0.0), "dec": math.radians(0.0), "t": et_60450,
         "jd": 60450.0 + 2400000.5, "rate_arcsec_hr": 2.0},
        {"ra": math.radians(0.01), "dec": math.radians(0.01), "t": et_60453,
         "jd": 60453.0 + 2400000.5, "rate_arcsec_hr": 2.0},
    ]
    tid, _ = assign_truth_to_chain(chain_offsky, cat,
                                     match_radius_arcsec=10.0)
    assert tid is None


def test_measure_linker_quality_precision_recall():
    from ariadne.discovery.imaging.synthetic_truth import (
        TruthEntry, TruthCatalog, measure_linker_quality)

    cat = TruthCatalog([
        TruthEntry("a", "i1", 60450.0, 180.0, 20.0, 256, 300, 21, "x"),
        TruthEntry("a", "i2", 60453.0, 180.05, 20.05, 256, 300, 21, "x"),
        TruthEntry("b", "i1", 60450.0, 181.0, 20.0, 400, 300, 21, "x"),
    ])
    et_60450 = (60450.0 - 51544.5) * 86400
    et_60453 = (60453.0 - 51544.5) * 86400
    chains = [
        # Pure 'a' chain
        [{"ra": math.radians(180.0), "dec": math.radians(20.0), "t": et_60450,
          "jd": 60450.0 + 2400000.5, "rate_arcsec_hr": 2.0},
         {"ra": math.radians(180.05), "dec": math.radians(20.05), "t": et_60453,
          "jd": 60453.0 + 2400000.5, "rate_arcsec_hr": 2.0}],
        # Spurious: matches nothing
        [{"ra": math.radians(5.0), "dec": math.radians(0.0), "t": et_60450,
          "jd": 60450.0 + 2400000.5, "rate_arcsec_hr": 2.0}],
    ]
    q = measure_linker_quality(chains, cat, match_radius_arcsec=10.0)
    assert q["n_chains"] == 2
    assert q["n_pure_chains"] == 1
    assert q["n_spurious_chains"] == 1
    assert q["precision"] == 0.5
    # 'a' covered, 'b' not -> recall = 1/2
    assert q["recall"] == 0.5


# ===========================================================================
# Phase 1: chain_quality
# ===========================================================================

def _make_chain(rates, mags=None, t_starts=None,
                 ra=math.radians(180.0), dec=math.radians(20.0)):
    """Build a synthetic chain with the given rates / mags."""
    n = len(rates)
    if mags is None:
        mags = [21.5] * n
    if t_starts is None:
        t_starts = [k * 86400.0 for k in range(n)]
    return [{"t": t_starts[k], "jd": 0.0,
             "ra": ra + k * 1e-5, "dec": dec + k * 1e-5,
             "rate_arcsec_hr": rates[k], "mag": mags[k],
             "source_pair": ()}
            for k in range(n)]


def test_rate_coherence_score_constant_rate_is_zero_spread():
    from ariadne.discovery.imaging.chain_quality import rate_coherence_score
    ch = _make_chain(rates=[5.0, 5.0, 5.0])
    med, spread = rate_coherence_score(ch)
    assert med == 5.0
    assert spread == 0.0


def test_rate_coherence_score_huge_spread_flagged():
    from ariadne.discovery.imaging.chain_quality import rate_coherence_score
    ch = _make_chain(rates=[1.0, 5.0, 10.0])
    _, spread = rate_coherence_score(ch)
    # spread = (10 - 1) / 5 = 1.8
    assert spread > 1.0


def test_photometric_coherence_constant_mag():
    from ariadne.discovery.imaging.chain_quality import (
        photometric_coherence_score)
    ch = _make_chain(rates=[3]*3, mags=[21.5, 21.5, 21.5])
    med, std = photometric_coherence_score(ch)
    assert med == 21.5
    assert std == 0.0


def test_photometric_coherence_variable_mag():
    from ariadne.discovery.imaging.chain_quality import (
        photometric_coherence_score)
    ch = _make_chain(rates=[3]*3, mags=[20.0, 22.0, 24.0])
    _, std = photometric_coherence_score(ch)
    assert std > 1.5


def test_epoch_coverage_three_nights():
    from ariadne.discovery.imaging.chain_quality import epoch_coverage
    ch = _make_chain(rates=[3]*3,
                      t_starts=[0, 3 * 86400, 6 * 86400])
    n_epochs, arc_hours = epoch_coverage(ch)
    assert n_epochs == 3
    assert arc_hours == pytest.approx(6 * 24.0)


def test_filter_chains_keeps_clean_drops_dirty():
    from ariadne.discovery.imaging.chain_quality import filter_chains
    clean = _make_chain(rates=[3, 3, 3], mags=[21.5]*3,
                          t_starts=[0, 3*86400, 6*86400])
    dirty_rates = _make_chain(rates=[1, 5, 12], mags=[21.5]*3,
                                t_starts=[0, 3*86400, 6*86400])
    dirty_short = _make_chain(rates=[3, 3], mags=[21.5]*2,
                                t_starts=[0, 86400 * 2])

    kept, dropped, verdicts = filter_chains(
        [clean, dirty_rates, dirty_short],
        max_rate_spread=0.5, max_mag_std=0.6,
        min_unique_epochs=3, min_arc_hours=12.0)
    assert len(kept) == 1
    assert len(dropped) == 2
    # First (clean) should pass; second (rates) failed rate; third failed epoch+arc
    assert verdicts[0].passes_all is True
    assert verdicts[1].passes_rate is False
    assert verdicts[2].passes_epoch is False


def test_chain_purity_score_in_unit_interval():
    from ariadne.discovery.imaging.chain_quality import chain_purity_score
    clean = _make_chain(rates=[3]*4, mags=[21.5]*4,
                         t_starts=[0, 86400, 2*86400, 3*86400])
    s = chain_purity_score(clean)
    assert 0.0 <= s <= 1.0
    assert s > 0.5   # clean chain should score in the upper half


# ===========================================================================
# Phase 2: iod_robust
# ===========================================================================

def test_rate_class_order_branches_correctly():
    from ariadne.discovery.iod_robust import rate_class_strategy_order
    # Slow movers -> BK / Vaisala first
    slow = rate_class_strategy_order(2.0)
    assert slow[0] == "bernstein_khushalani"
    assert slow[1] == "vaisala"
    # Medium -> linker / Gauss first
    med = rate_class_strategy_order(20.0)
    assert med[0] == "adaptive_linker"
    assert med[1] == "gauss"
    # Fast (NEO) -> Gauss first
    fast = rate_class_strategy_order(100.0)
    assert fast[0] == "gauss"


def test_chain_median_rate_handles_missing_rate():
    from ariadne.discovery.iod_robust import chain_median_rate
    ch = [{"rate_arcsec_hr": 5.0}, {"rate_arcsec_hr": 7.0}, {}]
    # Median of [5, 7] = 6
    assert chain_median_rate(ch) == 6.0
    assert chain_median_rate([]) == 0.0


def test_estimate_chain_sigma_returns_default_when_no_psf():
    from ariadne.discovery.iod_robust import estimate_chain_sigma_arcsec
    ch = _make_chain(rates=[3]*3)
    sigma = estimate_chain_sigma_arcsec(ch, default_sigma=0.4)
    assert sigma == 0.4


def test_perturb_chain_preserves_times_and_rates():
    from ariadne.discovery.iod_robust import _perturb_chain
    ch = _make_chain(rates=[3, 3, 3])
    rng = np.random.default_rng(0)
    p = _perturb_chain(ch, sigma_arcsec=0.5, rng=rng)
    for orig, new in zip(ch, p):
        assert orig["t"] == new["t"]
        assert orig["rate_arcsec_hr"] == new["rate_arcsec_hr"]
        # ra/dec perturbed: should differ from original (Gaussian with sigma>0)
        assert new["ra"] != orig["ra"]
        assert new["dec"] != orig["dec"]


def test_monte_carlo_iod_handles_empty_chain():
    """Empty chain must return success=False, not raise."""
    from ariadne.discovery.iod_robust import monte_carlo_iod
    result = monte_carlo_iod([], n_draws=2, sigma_arcsec=0.3,
                              rms_acceptance_arcsec=5.0)
    assert result.success is False


def test_robust_iod_returns_ensemble_fit_type():
    """Must return an EnsembleFit so callers don't have to fork. Even on a
    chain with insufficient/garbage data, the wrapper should not raise."""
    from ariadne.discovery.iod_robust import robust_iod
    from ariadne.discovery import iod_advanced as IODA
    # Include dra/ddec since the adaptive_helio_linc strategy needs them
    ch = [{"t": k * 86400.0, "jd": 0.0,
            "ra": math.radians(180.0) + k * 1e-5,
            "dec": math.radians(20.0) + k * 1e-5,
            "dra": 1e-9, "ddec": 1e-9,
            "rate_arcsec_hr": 3.0, "mag": 21.5,
            "source_pair": ()}
           for k in range(3)]
    # Force only one cheap strategy so the test doesn't depend on
    # whether SPICE/dynamics actually converge on the synthetic chain
    fit = robust_iod(ch, n_draws=2, sigma_arcsec=0.3,
                      use_monte_carlo=False, use_rate_class=False,
                      strategies=("gauss",))
    assert isinstance(fit, IODA.EnsembleFit)
