# Changelog

All notable changes to SigFeatX are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Performance**: `MODWT` decompose/reconstruct now use FFT-based circular
  convolution instead of an O(N x taps) direct double loop -- validated
  bit-for-bit equivalent to the old implementation across 336 combinations
  of wavelet/length/level (max discrepancy ~1e-15), ~45-70x faster
  (0.21s -> 0.003s at N=10,000, 6 levels).
- **Performance**: `RQAFeatures._runs_of_ones` (diagonal/vertical line-length
  scanning) is now vectorised via diff-based edge detection instead of a
  per-element Python loop -- validated identical across 2000+ random
  trials including edge cases.
- **Performance**: `CEEMDAN` gained an opt-in `n_jobs` parameter (same
  convention as `FeatureAggregator.extract_batch`: 1=sequential/default,
  -1=all cores, N=N processes, falls back to threads if process-based
  parallelism is unavailable). Each trial's EMD call is independent, so
  results are bit-identical to `n_jobs=1` for the same `rng` seed
  (`Executor.map` preserves order; EMD has no internal randomness).
  ~1.5x speedup measured at N=5000/50 trials; default behaviour is
  unchanged.
- **Accuracy**: added `hypothesis`-based property tests
  (`tests/test_properties.py`) checking perfect-reconstruction identities
  (MODWT, EMD) and value bounds (RQA, MFDFA, advanced entropy) across
  hundreds of randomly generated signals, rather than a handful of fixed
  toy examples.
- **Discoverability**: added a Sphinx documentation scaffold (`docs/`,
  autodoc + napoleon + intersphinx) with a quickstart and full API
  reference, a `docs` optional-dependency group, a `.readthedocs.yaml`,
  and a CI job that builds the docs with warnings treated as errors.
  Hosting on Read the Docs requires connecting the repo via their site
  (not something this session can do).

### Fixed
- **MODWT**: biorthogonal wavelets (`bior*`, `rbio*`) now raise a clear
  `ValueError` at construction instead of silently producing a badly
  wrong reconstruction (~0.6-0.75 absolute error found during testing).
  MODWT's pyramid algorithm reuses the decomposition filters for
  reconstruction, which is only valid when they are the time-reversal of
  each other -- true for orthogonal wavelets (db*, sym*, coif*, haar),
  not for biorthogonal ones. Supporting biorthogonal wavelets correctly
  would need a different filter pair and is left for future work rather
  than guessed at.
- Two docstring rendering bugs found while building the new docs:
  `LMD`'s docstring used `|1-a|`/`|s-t|` for absolute value, which RST
  parses as an (undefined) substitution reference; and two docstrings in
  `preprocess.py` had list/example blocks missing the blank line RST
  requires before them, breaking their rendering.
- **RQA `DET`** (`SigFeatX.features.RQAFeatures`) was silently deflated by
  exactly 2x: `_diagonal_line_lengths` only scanned the upper triangle of
  the (symmetric) recurrence matrix, while the `DET` denominator counted
  points from both triangles. Now scans both triangles. `L`, `L_max`,
  `TT`, `ENTR` were unaffected (symmetric distributions).
- **CEEMDAN** (`SigFeatX.decompose.CEEMDAN`) was missing the "Adaptive
  Noise" the algorithm is named for: noise amplitude was fixed to the
  *original* signal's std for every stage instead of being rescaled to
  the *current residue's* std (Torres et al. 2011). Deep IMFs on signals
  with fast-decaying residues were degraded as a result.
- **Bubble Entropy** (`SigFeatX.features.AdvancedEntropyFeatures.bubble_entropy`)
  used a plain Shannon-entropy difference instead of the paper's Renyi
  entropy of order 2, and was missing the `log((m+1)/(m-1))` normalisation
  (Manis et al. 2017, matches EntropyHub's reference implementation).
- **`SigFeatX.__version__`** reported `"0.2.0"` even though the package
  had shipped `0.3.0` (per `pyproject.toml`/git tags) since the Phase 2
  release. Now kept in sync.
- Test `test_noise_spectral_flatness_high` asserted `spectral_flatness >
  0.7` for white noise; the mathematically correct expectation for a
  single (non-Welch-averaged) periodogram is `exp(-euler_gamma) ≈ 0.56`,
  not close to 1. Lowered to `> 0.5`, matching the threshold already used
  for the same computation in `test_bug_fixes.py`.

### Internal
- Added CI (`.github/workflows/ci.yml`): test matrix across Python
  3.11–3.13 (+ Windows), `ruff check`, `mypy`, and a package-build check
  on every push/PR. Added `.github/workflows/release.yml` to build and
  publish to PyPI via trusted publishing on `v*.*.*` tags. Added
  Dependabot for pip and GitHub Actions.
- Modernised all `typing.List/Dict/Tuple/Optional/Union` to PEP
  585/604 (`list`, `dict`, `X | None`, ...) and brought the codebase to a
  clean `ruff check` / `mypy` under the existing `pyproject.toml` config.
- Removed `requirements.txt` (redundant with `pyproject.toml`, which has
  been the single source of truth for dependencies since 0.2.0). Bumped
  the dependency floors to the oldest versions that actually ship
  `py311` wheels (`numpy>=1.24`, `scipy>=1.10`, `PyWavelets>=1.4`,
  `pandas>=1.5.3`), since the old floors predated `requires-python
  >=3.11` and were never actually installable.

## [0.3.0] — 2026-05-23

### Added

#### Decompositions
- **MODWT** (`SigFeatX.decompose.MODWT`) — Maximal Overlap Discrete Wavelet
  Transform (Percival & Walden 2000). Shift-invariant, works on any signal
  length, perfect reconstruction to machine precision, all coefficient arrays
  match the input length. Far better than DWT for windowed feature extraction.
- **CEEMDAN** (`SigFeatX.decompose.CEEMDAN`) — Complete Ensemble EMD with
  Adaptive Noise (Torres et al. 2011). Fixes EMD mode mixing. Reproducible via
  the `rng` parameter; configurable `trials`, `noise_amp`, `max_imf`.
- **HHT** (`SigFeatX.decompose.HHT`) — Hilbert-Huang Transform (Huang et al.
  1998). Per-IMF instantaneous amplitude/frequency, full and marginal Hilbert
  spectra, and a feature extractor. Pluggable decomposer (EMD or CEEMDAN).
- **SST** (`SigFeatX.decompose.SST`) — Fourier-based Synchrosqueezing Transform
  (Daubechies, Lu, Wu 2011) using the Auger-Flandrin instantaneous-frequency
  operator. Robust on tonal signals.

#### Feature classes
- **RQAFeatures** (`SigFeatX.features.RQAFeatures`) — Recurrence Quantification
  Analysis (Marwan et al. 2007). 10 features: RR, DET, LAM, average/max
  diagonal line length, trapping time, max vertical line, line-length entropy,
  divergence, and the recurrence threshold. Auto-eps targets RR ≈ 10%.
- **MFDFAFeatures** (`SigFeatX.features.MFDFAFeatures`) — Multifractal
  Detrended Fluctuation Analysis (Kantelhardt et al. 2002). Generalised Hurst
  exponents h(q), singularity-spectrum width, peak position, and asymmetry.
- **AdvancedEntropyFeatures** (`SigFeatX.features.AdvancedEntropyFeatures`) —
  Dispersion Entropy (Rostaghi & Azami 2016), Fuzzy Entropy (Chen et al. 2007),
  Lempel-Ziv Complexity (Lempel & Ziv 1976), and Bubble Entropy (Manis et al.
  2017). Available individually and as a bundle.

### Changed
- `FeatureAggregator` and `Pipeline` now accept `'modwt'`, `'ceemdan'`,
  `'hht'`, and `'sst'` as decomposition method names.
- Raw-feature extraction now includes RQA, MFDFA, and the four advanced
  entropies (~20 additional columns). Pass `extract_raw=False` to opt out.
- Version bumped to `0.3.0`. Added decomposition and entropy keywords for
  PyPI discoverability.

### Tests
- `tests/test_phase2.py` adds 43 tests covering all seven new modules,
  reference-validated where closed-form values exist (MODWT energy
  preservation, HHT carrier frequency, white-noise monofractal width).

### Internal
- New modules: `decompose/{modwt,ceemdan,hht,sst}.py`,
  `features/{rqa,mfdfa,advanced_entropy}.py`.

## [0.2.0] — 2026-05-16

### Added
- **`SigFeatXTransformer`** — sklearn `TransformerMixin` for use in `Pipeline`,
  `GridSearchCV`, etc. Accepts 2D arrays, lists of 1D arrays, and 3D multichannel
  arrays. Exposes `get_feature_names_out()` for sklearn 1.0+ introspection.
- **`Pipeline`** — fluent builder for chaining
  `detrend → denoise → normalize → decompose → extract`. Lazy execution, clone
  support, full JSON serialisation via `to_dict()`. Forwards `fs` to
  bandpass/notch/lowpass filters automatically.
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
- Pinned `scikit-learn>=1.3` to avoid wheel-less ancient versions on
  Python 3.12+.

### Fixed
- `feature_consistency` no longer flags `spectral_skewness` against the
  time-domain `skewness` bounds. The prefix-stripping lookup now matches on
  exact names first and only strips whitelisted method prefixes, so semantic
  prefixes like `spectral_` are preserved.

### Deprecated
- `setup.py`. Install with `pip install -e .` continues to work via PEP 517.

### Internal
- New modules: `_progress.py`, `io_extensions.py`, `pipeline.py`,
  `sklearn_wrapper.py`.

## [0.1.0] — 2025-XX-XX

Initial release. EMD, VMD, SVMD, EFD, DWT, WPD, STFT, LMD, JMD. Time, frequency,
entropy, nonlinear, and decomposition-based features. Batch, windowed, and
multichannel extraction.