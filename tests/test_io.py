"""GMAT export tests."""
from ariadne.io.gmat_export import export_gmat_script


def test_gmat_script_contains_required_blocks(tmp_path):
    state = [6778.137, 0.0, 0.0, 0.0, 7.668, 0.0]
    path = export_gmat_script(str(tmp_path / "sat.script"), state,
                              "01 Jun 2025 00:00:00.000", days=3.0, name="TestSat")
    text = open(path).read()
    for token in ("Create Spacecraft TestSat", "FM.PointMasses = {Earth, Sun, Luna}",
                  "Create Propagator Prop", "BeginMissionSequence",
                  "Propagate Prop(TestSat)", "TestSat.X  = 6778.137"):
        assert token in text
