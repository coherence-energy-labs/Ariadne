"""ONE_FRONTIER_INTEGRATION.md cites only numbers produced by its replay script."""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _replay():
    spec = importlib.util.spec_from_file_location(
        "one_frontier_replay", ROOT / "scripts" / "one_frontier_replay.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _normalise(text):
    return text.replace("\r\n", "\n")


def test_integration_doc_evidence_block_is_rendered_from_the_recorded_replay():
    replay = _replay()
    recorded = json.loads(replay.JSON_PATH.read_text(encoding="utf-8"))
    doc = _normalise(replay.DOC_PATH.read_text(encoding="utf-8"))
    block = doc[doc.index(replay.BEGIN) : doc.index(replay.END) + len(replay.END)]
    assert block == _normalise(replay.render(recorded))


def test_integration_doc_has_no_numbers_outside_generated_or_declared_text():
    replay = _replay()
    doc = _normalise(replay.DOC_PATH.read_text(encoding="utf-8"))
    outside = doc[: doc.index(replay.BEGIN)] + doc[doc.index(replay.END) :]
    # Measurements (4+ decimal places, timings, test counts, standard errors) live only
    # in the generated block; the prose may carry API constants and the worked witness.
    measured = r"\d+\.\d{4,}|\d ?ms\b|\d+ (passed|failed|tests)|standard errors"
    assert not re.search(measured, outside), re.search(measured, outside)


def test_recorded_replay_matches_the_current_focused_suite_and_kills_every_mutant():
    replay = _replay()
    recorded = json.loads(replay.JSON_PATH.read_text(encoding="utf-8"))
    assert recorded["focused_tests"]["failed"] == 0
    assert all(v["killed"] for v in recorded["mutants"].values())
    source = (ROOT / "src/ariadne/optimize/success_contract.py").read_text(encoding="utf-8")
    for name, (anchor, _) in replay.MUTANTS.items():
        assert source.count(anchor) == 1, f"mutant {name} no longer anchors in the source"


@pytest.mark.slow
def test_deterministic_replay_sections_reproduce_the_recorded_evidence():
    replay = _replay()
    recorded = json.loads(replay.JSON_PATH.read_text(encoding="utf-8"))
    assert replay.check(recorded) == []
