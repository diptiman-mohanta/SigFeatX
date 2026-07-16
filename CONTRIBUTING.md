# Contributing to SigFeatX

Thanks for your interest in improving SigFeatX! Bug reports, feature
requests, documentation fixes, and pull requests are all welcome.

## Development setup

```bash
git clone https://github.com/diptiman-mohanta/SigFeatX.git
cd SigFeatX
pip install -e ".[dev,all]"
```

## Before opening a pull request

All three of these must pass — CI runs the exact same commands:

```bash
pytest              # full test suite (includes Hypothesis property tests)
ruff check .        # lint
mypy SigFeatX       # type check
```

If you touch the docs, also confirm they build without warnings:

```bash
pip install -e ".[docs]"
sphinx-build -b html -W docs docs/_build/html
```

## Guidelines

- **Correctness over speed.** This library implements published
  algorithms; every implementation should cite its reference paper in the
  module docstring, and any performance rewrite must be validated
  numerically against the version it replaces (see the MODWT/RQA
  vectorisation commits for the expected standard).
- **Math changes need tests that would have caught the bug.** Prefer
  invariant checks (perfect reconstruction, value bounds, known
  theoretical values on white noise) over fixed toy-signal snapshots —
  see `tests/test_properties.py`.
- **Keep the core dependency footprint small** (numpy, scipy, PyWavelets,
  pandas). Anything else belongs behind an optional extra.
- New features should update `CHANGELOG.md` under `[Unreleased]`.

## Reporting bugs

Please include: SigFeatX version (`SigFeatX.__version__`), Python and
numpy/scipy versions, a minimal signal + code snippet that reproduces the
problem, and what you expected instead. For suspected *mathematical*
errors, a reference (paper equation or an established implementation that
disagrees) is enormously helpful.
