# Contributing to Ariadne

Thanks for considering a contribution. Ariadne is an open, validated astrodynamics
toolkit; we want it to stay open, validated, and pleasant to work on.

## Quick start for contributors

```bash
git clone https://github.com/Jphilbrick10/Ariadne.git
cd Ariadne
python -m venv .venv && . .venv/Scripts/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,docs,notebooks]"
pytest -m "not slow" -q                                    # fast tests should all pass
PYTHONPATH=src python benchmarks/reference_targets.py     # reference checks should be 16/16
```

If those three commands work cleanly, your environment is ready.

## What kinds of contributions are welcome

In order of "most welcome":

1. **Bug reports with reproducers** — a 5-line snippet that demonstrates the bug.
2. **New validation gates** — a `validate/stage##.py` that exercises something we don't
   yet exercise (a new orbit family, a different mass-ratio system, a published transfer
   we haven't reproduced).
3. **Tutorials** — a new `examples/##_*.py` script that demonstrates a capability with
   a runnable example + PNG output. Keep them under ~80 lines.
4. **Documentation** — anything that makes the docs clearer, especially honest-scope
   notes about what does and doesn't work.
5. **New CR3BP systems** — adding a new (μ, L\*, T\*) constant tuple is a 1-liner in
   `data/constants.py`; PRs welcome.
6. **Performance** — numba/Cython speedups for hot paths, with before/after benchmarks.
7. **Cross-validations against other tools** — port a published GMAT / Tudat / poliastro
   example into Ariadne and check we agree.

## What we ask in return

- **One commit, one purpose.** Don't bundle unrelated changes.
- **Tests for new behaviour.** Either a unit test in `tests/test_*.py` or a validation
  gate in `validate/stage##.py`. A claim without a test isn't a claim.
- **Honest scope.** If a change has limitations (only works in CR3BP, fails for very
  small mass ratios, requires a specific dependency), say so in the docstring AND in
  the PR description. Ariadne's credibility rests on its honesty firewall.
- **Match the existing style.** Numpy-style docstrings, type hints where they help,
  no emoji in docstrings/comments.
- **Run the test suite locally.** `pytest -m "not slow" -q` before pushing.

## The honesty firewall

Every numerical claim in Ariadne is cross-checked against an independent tool (GMAT,
REBOUND, spiceypy↔jplephem, published reference values). New code that produces a number
must say *what it agrees with and to what tolerance*. PRs that publish numbers without
an independent cross-check will be asked to add one before merging.

This is the most important rule. We'd rather be honestly wrong than
plausibly-right-but-unverified.

## PR checklist

When opening a PR, please confirm:

- [ ] `pytest -m "not slow" -q` passes locally
- [ ] `python benchmarks/reference_targets.py` is still 16/16 (or you've updated the
      benchmark suite to reflect a deliberate change)
- [ ] New behaviour has a test or a validation gate
- [ ] Docstrings written in numpy style
- [ ] No `print` statements outside of `__main__` blocks / examples / benchmarks
- [ ] Honest-scope notes added where any limitation applies
- [ ] CHANGELOG.md updated under `## Unreleased`

## Discussion / questions

Open an issue with the `question` label. We respond to issues; we may take longer to
respond to PRs because they need a careful review.

## License

By contributing you agree your contribution is licensed under the same MIT license as
the rest of Ariadne. No CLA required.
