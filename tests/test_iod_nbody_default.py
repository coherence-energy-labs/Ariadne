import numpy as np

from ariadne.discovery import iod


def _tracklets(arc_days):
    return [
        {"t": 0.0, "ra": 0.0, "dec": 0.0},
        {"t": 0.5 * arc_days * 86400.0, "ra": 0.01, "dec": 0.01},
        {"t": arc_days * 86400.0, "ra": 0.02, "dec": 0.02},
    ]


def _seed(*args, **kwargs):
    return {
        "x_init": np.array([1.0, 2.0, 3.0]),
        "v_init": np.array([0.1, 0.2, 0.3]),
        "r_au": 40.0,
        "rdot": 0.0,
        "scatter_km": 1.0,
        "n_valid": 3,
    }


def _fit2(*args, **kwargs):
    return {
        "x_fit": np.array([1.0, 2.0, 3.0]),
        "v_fit": np.array([0.1, 0.2, 0.3]),
        "rms_arcsec": 1.5,
        "nfev": 4,
        "success": True,
    }


def test_fit_candidate_promotes_long_arc_to_nbody(monkeypatch):
    import ariadne.discovery.orbit_fit_nbody as nbody

    def fit_nbody(tracklets, t_ref, x, v):
        return {
            "x_fit": np.asarray(x) + 1.0,
            "v_fit": np.asarray(v),
            "rms_arcsec": 0.5,
            "nfev": 8,
            "success": True,
            "perturbers_used": ["JUPITER"],
        }

    monkeypatch.setattr(iod, "iod_hypothesis_search", _seed)
    monkeypatch.setattr(iod, "fit_orbit_lm", _fit2)
    monkeypatch.setattr(nbody, "fit_orbit_nbody", fit_nbody)

    fit = iod.fit_candidate(_tracklets(500.0), t_ref=0.0)

    assert fit["dynamics_model"] == "nbody_lm"
    assert fit["rms_2body_arcsec"] == 1.5
    assert fit["arc_days"] == 500.0
    assert fit["perturbers_used"] == ["JUPITER"]


def test_fit_candidate_short_arc_keeps_2body(monkeypatch):
    monkeypatch.setattr(iod, "iod_hypothesis_search", _seed)
    monkeypatch.setattr(iod, "fit_orbit_lm", _fit2)

    fit = iod.fit_candidate(_tracklets(30.0), t_ref=0.0)

    assert fit["dynamics_model"] == "2body_kepler_lm"
    assert fit["nbody_promotion_status"] == "not_required_short_arc"


def test_fit_candidate_long_arc_marks_nbody_failure(monkeypatch):
    import ariadne.discovery.orbit_fit_nbody as nbody

    def fit_nbody(tracklets, t_ref, x, v):
        return {"success": False, "rms_arcsec": float("inf"), "err": "bad fit"}

    monkeypatch.setattr(iod, "iod_hypothesis_search", _seed)
    monkeypatch.setattr(iod, "fit_orbit_lm", _fit2)
    monkeypatch.setattr(nbody, "fit_orbit_nbody", fit_nbody)

    fit = iod.fit_candidate(_tracklets(500.0), t_ref=0.0)

    assert fit["dynamics_model"] == "2body_kepler_lm"
    assert fit["nbody_promotion_status"] == "failed"
    assert fit["nbody_promotion_error"] == "bad fit"
