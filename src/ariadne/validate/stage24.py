"""Stage 24 validation gates (MASTER_PLAN.md - packaging + open-source).

G_pkg - The project is a real, installable, open-source package:
  (a) pyproject.toml declares the build system, name/version, runtime deps, and entry points;
  (b) the declared version matches ariadne.__version__;
  (c) an OSI license (LICENSE) and a CI workflow (.github/workflows/ci.yml) are present;
  (d) the package and its key subsystems import cleanly.

Run:  PYTHONPATH=src python -m ariadne.validate.stage24
"""
from __future__ import annotations

import os
import re

import ariadne

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _read(rel):
    p = os.path.join(REPO_ROOT, rel)
    return open(p, encoding="utf-8").read() if os.path.exists(p) else None


def check() -> tuple[bool, dict]:
    pyproject = _read("pyproject.toml") or ""
    license_txt = _read("LICENSE") or ""
    ci = _read(os.path.join(".github", "workflows", "ci.yml")) or ""

    # (a) pyproject declares the essentials
    m = re.search(r'version\s*=\s*"([^"]+)"', pyproject)
    pp_version = m.group(1) if m else None
    deps_ok = all(d in pyproject for d in ("numpy", "scipy", "matplotlib", "spiceypy", "h5py"))
    has_build = "build-backend" in pyproject
    has_scripts = "[project.scripts]" in pyproject
    a = bool(pp_version) and deps_ok and has_build and has_scripts

    # (b) version consistency
    b = pp_version == ariadne.__version__

    # (c) license + CI present
    c = "MIT License" in license_txt and "pytest" in ci

    # (d) key subsystems import
    import importlib
    mods = ["ariadne.dynamics.cr3bp", "ariadne.transport_graph.search",
            "ariadne.discovery.mining", "ariadne.atlas.store",
            "ariadne.interplanetary.porkchop", "ariadne.interplanetary.flyby",
            "ariadne.interplanetary.grand", "ariadne.orbits.nrho"]
    import_errs = []
    for mod in mods:
        try:
            importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001
            import_errs.append(f"{mod}: {e}")
    d = not import_errs

    ok = a and b and c and d
    return ok, {"pp_version": pp_version, "ariadne_version": ariadne.__version__,
                "deps_ok": deps_ok, "has_build": has_build, "has_scripts": has_scripts,
                "license_ok": "MIT License" in license_txt, "ci_ok": "pytest" in ci,
                "import_errs": import_errs, "n_mods": len(mods),
                "a": a, "b": b, "c": c, "d": d}


def main() -> int:
    print("=== Ariadne Stage 24 validation  (packaging + open-source) ===\n")
    ok, i = check()
    print("[G_pkg a] pyproject.toml")
    print(f"      version {i['pp_version']}  deps_ok={i['deps_ok']}  build_backend={i['has_build']}  "
          f"entry_points={i['has_scripts']}  -> {'ok' if i['a'] else 'MISSING'}")
    print(f"[G_pkg b] version matches ariadne.__version__ ({i['ariadne_version']}): {'ok' if i['b'] else 'MISMATCH'}")
    print(f"[G_pkg c] LICENSE (MIT)={i['license_ok']}  CI workflow={i['ci_ok']}: {'ok' if i['c'] else 'MISSING'}")
    print(f"[G_pkg d] {i['n_mods']} key subsystems import: {'ok' if i['d'] else 'FAIL'}")
    for e in i["import_errs"]:
        print(f"        ERROR {e}")
    print()
    print(f"=== STAGE 24: {'ALL GATES PASS' if ok else 'FAILURE'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
