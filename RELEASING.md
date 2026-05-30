# Releasing Ariadne to PyPI

Steps for publishing a new release. Run from a clean checkout on `master`.

## 0. Pre-flight

```bash
# Make sure tests pass on the fast suite
pytest -m "not slow" -q

# Reference benchmarks all green
PYTHONPATH=src python benchmarks/reference_targets.py

# All five tutorial examples run cleanly
for ex in examples/0*.py; do PYTHONPATH=src python "$ex" || break; done

# Ariadne imports cleanly + top-level API works
python -c "import ariadne; print(ariadne.__version__); ariadne.gateway_nrho()"
```

## 1. Bump version

Bump the version in both files (must match):

- `pyproject.toml` → `version = "..."`
- `src/ariadne/__init__.py` → `__version__ = "..."`

Versioning follows [PEP 440](https://peps.python.org/pep-0440/):
- `1.0.0rc2` → release candidate
- `1.0.0` → stable
- `1.0.1` → patch
- `1.1.0` → minor
- `2.0.0` → major (breaking)

## 2. Build the distribution

```bash
# Clean prior build artifacts
rm -rf dist/ build/ src/*.egg-info

# Build sdist + wheel
python -m build --sdist --wheel --outdir dist/

# Verify metadata
python -m twine check dist/*
```

You should see:
```
Checking dist/ariadne_astro-X.Y.Z-py3-none-any.whl: PASSED
Checking dist/ariadne_astro-X.Y.Z.tar.gz: PASSED
```

## 3. Upload to TestPyPI first

```bash
python -m twine upload --repository testpypi dist/*
```

Then verify by installing from TestPyPI in a fresh venv:

```bash
python -m venv /tmp/ariadne-test
source /tmp/ariadne-test/bin/activate                # Windows: /tmp/ariadne-test/Scripts/activate
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            ariadne-astro
python -c "import ariadne; print(ariadne.__version__); fam = ariadne.lyapunov_family('L1', n=3); print(f'{len(fam)} orbits')"
deactivate
```

## 4. Upload to real PyPI

```bash
python -m twine upload dist/*
```

## 5. Tag the release in git

```bash
git tag -a v1.0.0rc2 -m "Release v1.0.0rc2"
git push origin v1.0.0rc2
```

## 6. Cut a GitHub release

1. Go to https://github.com/Jphilbrick10/Ariadne/releases
2. Click *Draft a new release*, pick the tag.
3. Title: `v1.0.0rc2 — <one-line headline>`.
4. Description: paste from CHANGELOG or this release's narrative.
5. Attach `dist/ariadne_astro-X.Y.Z-py3-none-any.whl` and `.tar.gz` as binaries.
6. Publish.

## 7. Post-release sanity

```bash
# Fresh venv, install from PyPI proper
python -m venv /tmp/ariadne-prod
source /tmp/ariadne-prod/bin/activate
pip install ariadne-astro
python -c "import ariadne; print(ariadne.__version__)"
deactivate
```

If anything is wrong, you cannot delete a version from PyPI — only yank it. Bump the
version and re-release.

## 8. Update docs

If using readthedocs:
1. Make sure the new tag triggers a docs build (readthedocs auto-detects tags).
2. Set the new version as the default (if a stable release) in the readthedocs admin.

## Credentials

The PyPI / TestPyPI tokens live in `~/.pypirc`:

```ini
[pypi]
username = __token__
password = pypi-...

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-...
```

Never commit credentials. Use scoped tokens (project-specific) for least privilege.
