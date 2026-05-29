"""Stage 24 tests: packaging metadata consistency + open-source artifacts."""
import os
import re

import ariadne

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _read(rel):
    p = os.path.join(ROOT, rel)
    return open(p, encoding="utf-8").read() if os.path.exists(p) else None


def test_pyproject_version_matches_package():
    pp = _read("pyproject.toml")
    assert pp is not None
    m = re.search(r'version\s*=\s*"([^"]+)"', pp)
    assert m and m.group(1) == ariadne.__version__


def test_pyproject_declares_core_deps_and_entry_points():
    pp = _read("pyproject.toml")
    for dep in ("numpy", "scipy", "matplotlib", "spiceypy", "h5py"):
        assert dep in pp
    assert "build-backend" in pp
    assert "[project.scripts]" in pp
    assert "ariadne.atlas.release:main" in pp


def test_license_and_ci_present():
    assert "MIT License" in (_read("LICENSE") or "")
    ci = _read(os.path.join(".github", "workflows", "ci.yml"))
    assert ci is not None and "pytest" in ci


def test_key_subsystems_import():
    import importlib
    for mod in ["ariadne.dynamics.cr3bp", "ariadne.transport_graph.search",
                "ariadne.discovery.mining", "ariadne.atlas.store",
                "ariadne.interplanetary.porkchop", "ariadne.interplanetary.grand",
                "ariadne.orbits.nrho"]:
        importlib.import_module(mod)
