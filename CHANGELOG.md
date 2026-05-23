# Changelog

All notable changes to SigFeatX are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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