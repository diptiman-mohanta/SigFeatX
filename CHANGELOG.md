# Changelog

All notable changes to SigFeatX are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-05-16

### Added
- **`SigFeatXTransformer`** — sklearn `TransformerMixin` for use in `Pipeline`,
  `GridSearchCV`, etc. Accepts 2D arrays, lists of 1D arrays, and 3D multichannel
  arrays. Exposes `get_feature_names_out()` for sklearn 1.0+ introspection.
- **`Pipeline`** — fluent builder for chaining
  `detrend → denoise → normalize → decompose → extract`. Lazy execution, clone
  support, full JSON serialisation via `to_dict()`.
- **`BatchIO`** — Parquet, HDF5, and Feather writers/readers for large batch
  feature tables. Auto-dispatch on file extension. Optional metadata storage.
- **`ProgressBar` / `progress_iter`** — tqdm-aware progress reporting in
  `extract_batch`. Silent no-op when tqdm is not installed; falls back to
  carriage-return counter.
- Optional dependency groups: `sklearn`, `progress`, `parquet`, `hdf5`, `viz`,
  `dev`, and `all`.
- PEP 561 typed-package marker (`py.typed`).

### Changed
- `pyproject.toml` is now the single source of truth for build metadata;
  `setup.py` removed.
- Version bumped to `0.2.0` and project status moved from Alpha to Beta.

### Deprecated
- `setup.py`. Install with `pip install -e .` continues to work via PEP 517.

### Internal
- New helper module `SigFeatX/_progress.py`.
- New module `SigFeatX/io_extensions.py`.
- New module `SigFeatX/pipeline.py`.
- New module `SigFeatX/sklearn_wrapper.py`.

## [0.1.0] — 2025-XX-XX

Initial release. EMD, VMD, SVMD, EFD, DWT, WPD, STFT, LMD, JMD. Time, frequency,
entropy, nonlinear, and decomposition-based features. Batch, windowed, and
multichannel extraction.
