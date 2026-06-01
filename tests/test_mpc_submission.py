"""Tests for the MPC submission pipeline."""
from __future__ import annotations

import math
import pytest


def _good_chain_with_detections():
    """Build a chain + 6 detections meeting grade-A criteria."""
    chain = {
        "id": 1, "arc_days": 5.0,
        "iod_strategy": "bayesian_tno_class",
        "iod_rms_arcsec": 0.4,
    }
    detections = []
    for k in range(6):
        # 3 distinct nights, 2 obs/night, 5 days arc
        night = k // 2
        mjd = 60450.0 + night * 2.5 + (k % 2) * 0.1
        detections.append({
            "id": k, "mjd": mjd, "ra": 180.0 + k * 0.01,
            "dec": -10.0, "mag": 21.5 + 0.02 * (k % 2),
            "astrom_sigma_arcsec": 0.10,
        })
    return chain, detections


def test_evaluate_grade_a_passes_on_good_chain():
    from ariadne.discovery.imaging.mpc_submission import evaluate_grade_a
    chain, dets = _good_chain_with_detections()
    g = evaluate_grade_a(chain, dets)
    assert g.passed
    assert g.n_observations == 6
    assert g.n_distinct_epochs >= 3
    assert g.arc_days == pytest.approx(5.0)
    assert g.iod_converged
    assert g.reasons == []


def test_evaluate_grade_a_fails_short_arc():
    from ariadne.discovery.imaging.mpc_submission import evaluate_grade_a
    chain, dets = _good_chain_with_detections()
    chain["arc_days"] = 0.5
    g = evaluate_grade_a(chain, dets)
    assert not g.passed
    assert any("arc" in r for r in g.reasons)


def test_evaluate_grade_a_fails_few_observations():
    from ariadne.discovery.imaging.mpc_submission import evaluate_grade_a
    chain, dets = _good_chain_with_detections()
    g = evaluate_grade_a(chain, dets[:3])  # only 3 obs
    assert not g.passed
    assert any("n_obs" in r for r in g.reasons)


def test_evaluate_grade_a_fails_high_astrom_rms():
    from ariadne.discovery.imaging.mpc_submission import evaluate_grade_a
    chain, dets = _good_chain_with_detections()
    for d in dets:
        d["astrom_sigma_arcsec"] = 0.8   # exceeds 0.3" gate
    g = evaluate_grade_a(chain, dets)
    assert not g.passed
    assert any("astrom_rms" in r for r in g.reasons)


def test_evaluate_grade_a_fails_iod_not_converged():
    from ariadne.discovery.imaging.mpc_submission import evaluate_grade_a
    chain, dets = _good_chain_with_detections()
    chain["iod_rms_arcsec"] = 10.0   # large RMS
    g = evaluate_grade_a(chain, dets)
    assert not g.passed
    assert any("iod" in r for r in g.reasons)


def test_ra_deg_to_hms_string():
    from ariadne.discovery.imaging.mpc_submission import (
        _ra_deg_to_hms_string)
    # 180 deg = 12h 00m 00s
    assert _ra_deg_to_hms_string(180.0).strip() == "12 00 00.000"
    # 0 deg = 00h 00m 00s
    assert _ra_deg_to_hms_string(0.0).strip() == "00 00 00.000"


def test_dec_deg_to_dms_string():
    from ariadne.discovery.imaging.mpc_submission import (
        _dec_deg_to_dms_string)
    assert _dec_deg_to_dms_string(45.0).strip() == "+45 00 00.00"
    assert _dec_deg_to_dms_string(-10.5).strip() == "-10 30 00.00"


def test_build_80col_submission_one_line_per_detection():
    from ariadne.discovery.imaging.mpc_submission import (
        build_80col_submission)
    chain, dets = _good_chain_with_detections()
    text = build_80col_submission(chain, dets, observatory_code="W84")
    # Split on newlines WITHOUT stripping (the 80-col spec puts leading
    # spaces in cols 1-5 for unnumbered objects; stripping the line
    # would corrupt that).
    lines = [L for L in text.split("\n") if L]
    assert len(lines) == len(dets)
    for L in lines:
        assert len(L) == 80
    # Observatory code in last 3 cols
    for L in lines:
        assert L[77:80] == "W84"


def test_build_ades_submission_is_valid_xml():
    """Verify ADES output parses as valid XML."""
    from ariadne.discovery.imaging.mpc_submission import (
        build_ades_submission)
    import xml.etree.ElementTree as ET
    chain, dets = _good_chain_with_detections()
    xml = build_ades_submission(chain, dets, observatory_code="W84")
    # Should parse without error
    root = ET.fromstring(xml)
    assert root.tag == "ades"
    # Should have obsContext + obsData
    contexts = root.findall("obsContext")
    assert len(contexts) == 1
    data = root.findall("obsData")
    assert len(data) == 1
    # One <optical> per detection
    optical = data[0].findall("optical")
    assert len(optical) == len(dets)


def test_build_ades_includes_observatory_code():
    from ariadne.discovery.imaging.mpc_submission import (
        build_ades_submission)
    chain, dets = _good_chain_with_detections()
    xml = build_ades_submission(chain, dets, observatory_code="F51")
    assert "<mpcCode>F51</mpcCode>" in xml
    assert "<stn>F51</stn>" in xml


def test_write_submission_packet_creates_all_files(tmp_path):
    from ariadne.discovery.imaging.mpc_submission import (
        write_submission_packet)
    chain, dets = _good_chain_with_detections()
    files = write_submission_packet(
        chain, dets, tmp_path,
        observatory_code="W84", designation_hint="ARI0042")
    assert "ades" in files
    assert "80col" in files
    assert "grade" in files
    # ADES XML and 80col text exist + nonempty
    assert files["ades"].exists()
    assert files["ades"].read_text().startswith("<?xml")
    assert files["80col"].exists()
    assert len(files["80col"].read_text().strip().split("\n")) == len(dets)
    assert files["grade"].read_text().startswith("Chain ID:")


def test_grade_a_report_marks_pass_or_hold(tmp_path):
    from ariadne.discovery.imaging.mpc_submission import (
        write_submission_packet)
    chain, dets = _good_chain_with_detections()
    files = write_submission_packet(chain, dets, tmp_path,
                                       designation_hint="GOOD")
    assert "GRADE A (submit)" in files["grade"].read_text()
    # Now mess it up
    chain["arc_days"] = 0.5
    files = write_submission_packet(chain, dets, tmp_path,
                                       designation_hint="BAD")
    assert "NOT GRADE A (hold)" in files["grade"].read_text()
