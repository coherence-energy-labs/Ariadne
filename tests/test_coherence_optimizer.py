"""Coherence-weighted optimizer tests (Gate G_opt, Stage 12) — pure functions, fast."""
from ariadne.transfers.coherence_optimizer import pareto_front, knee, weighted_optimum

# synthetic Delta-v vs sensitivity points (mirrors the real frontier shape) + one dominated
PTS = [
    {"label": "a", "dv_ms": 3900, "sensitivity": 46000},   # cheapest, most fragile
    {"label": "b", "dv_ms": 3950, "sensitivity": 25000},
    {"label": "c", "dv_ms": 3980, "sensitivity": 12000},
    {"label": "d", "dv_ms": 4090, "sensitivity": 5900},     # priciest, most robust
    {"label": "dom", "dv_ms": 4000, "sensitivity": 40000},  # dominated by b
]


def test_pareto_excludes_dominated():
    labels = {p["label"] for p in pareto_front(PTS)}
    assert "dom" not in labels
    assert labels == {"a", "b", "c", "d"}


def test_knee_is_interior():
    k = knee(pareto_front(PTS))
    assert k["label"] in ("b", "c")          # a compromise, not an extreme


def test_weighted_optimum_extremes():
    assert weighted_optimum(PTS, 0.0)["label"] == "a"     # pure Delta-v -> cheapest
    assert weighted_optimum(PTS, 50.0)["label"] == "d"    # heavy robustness -> most robust
