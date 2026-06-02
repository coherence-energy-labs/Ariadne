import json


def _mpcorb_line(*, packed="K20A00A", H=8.0, e=0.85, a=500.0, inc=11.0,
                 rms=0.4, n_obs=100, name="Sedna proxy"):
    chars = [" "] * 200

    def put(start, text):
        s = str(text)
        chars[start:start + len(s)] = list(s)

    put(0, f"{packed:<7}")
    put(8, f"{H:5.2f}")
    put(59, f"{inc:9.4f}")
    put(70, f"{e:9.7f}")
    put(92, f"{a:11.7f}")
    put(117, f"{n_obs:5d}")
    put(137, f"{rms:4.2f}")
    put(166, f"{name:<28}")
    return "".join(chars)


def test_mpcorb_rows_become_labelled_external_cases():
    from ariadne.discovery.external_corpora import labelled_cases_from_mpcorb_lines

    cases = labelled_cases_from_mpcorb_lines([
        "header line ignored",
        _mpcorb_line(),
        _mpcorb_line(packed="K21B00B", H=15.0, e=0.1, a=2.7, inc=4.0,
                     name="MBA proxy"),
    ])
    assert len(cases) == 2
    assert cases[0].truth_label == "SEDNOID"
    assert cases[0].source == "mpc_mpcorb_live"
    assert cases[0].evidence.n_detections >= 4
    assert cases[1].truth_label == "MBA"


def test_ztf_jsonl_records_become_labelled_cases(tmp_path):
    from ariadne.discovery.external_corpora import labelled_cases_from_ztf_file

    path = tmp_path / "ztf.jsonl"
    rows = [
        {
            "objectId": "ZTF1",
            "jd": 2460000.5,
            "ra": 180.0,
            "dec": 10.0,
            "magpsf": 21.0,
            "fid": 2,
            "rb": 0.92,
            "truth_label": "MBA",
        },
        {
            "objectId": "ZTF1",
            "jd": 2460000.55,
            "ra": 180.01,
            "dec": 10.0,
            "magpsf": 21.1,
            "fid": 2,
            "rb": 0.90,
            "truth_label": "MBA",
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    cases = labelled_cases_from_ztf_file(path)
    assert len(cases) == 1
    assert cases[0].case_id == "ztf_ZTF1"
    assert cases[0].truth_label == "MBA"
    assert cases[0].evidence.rate_arcsec_hr is not None
    assert cases[0].evidence.band == "r"


def test_rubin_json_records_use_mpc_orbit_truth(tmp_path):
    from ariadne.discovery.external_corpora import labelled_cases_from_rubin_file

    path = tmp_path / "rubin.json"
    path.write_text(json.dumps([
        {
            "diaSource": {
                "diaSourceId": 123,
                "midPointTai": 60450.0,
                "ra": 10.0,
                "dec": -3.0,
                "psFluxMag": 22.3,
                "band": "r",
                "skyVelocity": 1.1,
                "nDiaSources": 5,
            },
            "mpcOrbit": {"a": 44.0, "e": 0.05, "i": 2.0},
        }
    ]), encoding="utf-8")
    cases = labelled_cases_from_rubin_file(path)
    assert len(cases) == 1
    assert cases[0].case_id == "rubin_123"
    assert cases[0].truth_label == "CLASSICAL_KBO"
    assert cases[0].source == "rubin_lsst_external_alerts"


def test_mpcorb_orbital_context_drives_known_object_recovery():
    from ariadne.discovery.external_corpora import labelled_cases_from_mpcorb_lines
    from ariadne.discovery.inference import infer

    cases = labelled_cases_from_mpcorb_lines([
        _mpcorb_line(packed="K21I00A", H=15.0, e=0.08, a=2.2, inc=3.0,
                     name="inner belt proxy"),
    ])
    assert cases[0].truth_label == "IMB"
    result = infer(cases[0].evidence)
    assert result.best.orbital_class == "IMB"
    assert "orbital_elements_context" in result.best.evidence_terms


def test_mpcorb_extended_taxonomy_covers_hungaria_and_thule():
    from ariadne.discovery.external_corpora import labelled_cases_from_mpcorb_lines
    from ariadne.discovery.inference import infer

    cases = labelled_cases_from_mpcorb_lines([
        _mpcorb_line(packed="00434", H=11.2, e=0.0739382, a=1.9443034,
                     inc=22.50584, name="Hungaria"),
        _mpcorb_line(packed="00279", H=8.58, e=0.044028, a=4.268363,
                     inc=2.33477, name="Thule"),
    ])
    assert [c.truth_label for c in cases] == ["HUNGARIA", "THULE"]
    assert infer(cases[0].evidence).best.orbital_class == "HUNGARIA"
    assert infer(cases[1].evidence).best.orbital_class == "THULE"


def test_labelled_case_jsonl_roundtrip(tmp_path):
    from ariadne.discovery.external_corpora import (
        labelled_cases_from_mpcorb_lines,
        read_labelled_cases_jsonl,
        write_labelled_cases_jsonl,
    )

    cases = labelled_cases_from_mpcorb_lines([
        _mpcorb_line(packed="K24T00A", H=17.0, e=0.12, a=2.4, inc=4.0,
                     name="roundtrip proxy"),
    ])
    path = tmp_path / "cases.jsonl"
    assert write_labelled_cases_jsonl(cases, path) == 1
    loaded = read_labelled_cases_jsonl(path)
    assert loaded[0].case_id == cases[0].case_id
    assert loaded[0].truth_label == "IMB"
    assert loaded[0].evidence.sky_context["a_au"] == 2.4


def test_alert_file_conversion_and_roundtrip(tmp_path):
    from ariadne.discovery.external_corpora import (
        alerts_from_ztf_file,
        read_alerts_jsonl,
        write_alerts_jsonl,
    )

    ztf_path = tmp_path / "ztf.jsonl"
    ztf_path.write_text(json.dumps({
        "objectId": "ZTF2",
        "candid": "cand1",
        "jd": 2460001.5,
        "ra": 181.0,
        "dec": 11.0,
        "magpsf": 20.0,
        "fid": 1,
        "rb": 0.88,
        "truth_label": "MBA",
    }) + "\n", encoding="utf-8")
    alerts = alerts_from_ztf_file(ztf_path)
    assert len(alerts) == 1
    assert alerts[0].survey == "ZTF"
    assert alerts[0].alert_id == "cand1"
    assert alerts[0].band == "g"

    out = tmp_path / "alerts.jsonl"
    assert write_alerts_jsonl(alerts, out) == 1
    loaded = read_alerts_jsonl(out)
    assert loaded == alerts
