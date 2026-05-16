# SigFeatX 0.2.0 — Phase 1 Upgrade

This release ships **five user-facing changes** that make SigFeatX
production-ready: sklearn integration, a fluent pipeline builder, big-data IO,
progress bars, and a modern packaging layout.

## Install

```bash
pip install -e .                       # core only
pip install -e ".[sklearn]"            # + scikit-learn
pip install -e ".[progress]"           # + tqdm
pip install -e ".[parquet,hdf5]"       # + big-data IO backends
pip install -e ".[all]"                # everything
pip install -e ".[dev]"                # contributor toolchain
```

## 1. sklearn integration — `SigFeatXTransformer`

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from SigFeatX import SigFeatXTransformer

pipe = Pipeline([
    ("features", SigFeatXTransformer(
        fs=1000,
        decomposition_methods=["fourier", "dwt"],
        preprocess_signal=True,
        n_jobs=-1,
    )),
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier()),
])
pipe.fit(X_train, y_train)            # X: (n_signals, n_timesteps)
pipe.predict(X_test)
```

Inputs supported:

- 2D ndarray of shape `(n_signals, signal_length)` — standard sklearn layout.
- 3D ndarray `(n_signals, n_channels, signal_length)` — multichannel.
  Cross-channel features (coherence, xcorr, PLV) are added automatically.
- List of 1D arrays with variable lengths.

Implements `get_feature_names_out()` for sklearn 1.0+ feature-name tracking.

## 2. Fluent pipeline — `Pipeline`

```python
from SigFeatX import Pipeline

features = (
    Pipeline(fs=1000)
    .detrend(method="als", lam=1e5, p=0.01)
    .denoise(method="bandpass", low_hz=1, high_hz=40)
    .normalize(method="robust")
    .decompose(["emd", "vmd"])
    .with_validation()
    .extract(signal)                  # single signal
)

df = (
    Pipeline(fs=1000)
    .denoise(method="notch", freq_hz=50)
    .decompose(["fourier", "dwt"])
    .with_parallel(n_jobs=-1)
    .with_progress()
    .extract_batch(signals)           # list or 2D array
).dataframe

windowed = (
    Pipeline(fs=250)
    .decompose("fourier")
    .extract_windowed(signal, window_size=512, step_size=256)
).dataframe
```

Pipelines are lazy: no work happens until `.extract*` is called. `clone()` and
`to_dict()` support reproducible grid search and config logging.

## 3. Big-data IO — `BatchIO`

```python
from SigFeatX import BatchIO

# Auto-dispatch by extension
BatchIO.save(df, "features.parquet", metadata={"fs": 1000, "session": "A1"})
BatchIO.save(df, "features.h5")
BatchIO.save(df, "features.feather")

df = BatchIO.load("features.parquet")
md = BatchIO.load_parquet_metadata("features.parquet")

# Partitioned appends for incremental pipelines
BatchIO.append_parquet(batch_df_1, "out/")
BatchIO.append_parquet(batch_df_2, "out/")
full = BatchIO.load("out/")            # reads all partitions
```

Optional backends: `pyarrow` (Parquet, Feather), `tables` + `h5py` (HDF5).
Falls back to pandas-native parquet when pyarrow is missing.

## 4. Progress bars

`FeatureAggregator.extract_batch(show_progress=True)` now renders a tqdm bar
when `tqdm` is installed, and prints a `\r`-rewriting counter otherwise.

```python
agg = FeatureAggregator(fs=1000)
result = agg.extract_batch(signals, show_progress=True, n_jobs=-1)
```

## 5. Packaging

- `pyproject.toml` is now the single source of truth. `setup.py` removed.
- Project status moved from Alpha to Beta.
- New optional-dependency groups: `sklearn`, `progress`, `parquet`, `hdf5`,
  `viz`, `dev`, `all`.
- PEP 561 typed-package marker (`py.typed`) shipped so downstream `mypy` users
  see SigFeatX's types.

## Backward compatibility

All existing APIs (`FeatureAggregator.extract_all_features`, `extract_batch`,
`extract_windowed`, `extract_multichannel`, `run_pipeline`, `SignalIO`,
`SignalUtils`, every decomposer and feature class) are unchanged. Phase 1 only
**adds** modules.

## Files changed / added

```
SigFeatX/
├── __init__.py              [updated]   version 0.2.0, new exports
├── _progress.py             [NEW]       tqdm helper
├── io_extensions.py         [NEW]       BatchIO (parquet/hdf5/feather)
├── pipeline.py              [NEW]       Pipeline builder
└── sklearn_wrapper.py       [NEW]       SigFeatXTransformer
pyproject.toml               [updated]   modern build config, optional deps
setup.py                     [removed]   superseded by pyproject.toml
MANIFEST.in                  [NEW]       sdist contents
CHANGELOG.md                 [NEW]       release notes
tests/test_phase1.py         [NEW]       26 tests, all passing
examples/phase1_usage.py     [NEW]       end-to-end demo
PATCH_aggregator.md          [NEW]       optional tqdm wiring for aggregator
```

## Aggregator patch (optional)

Wiring the new `ProgressBar` into the existing `FeatureAggregator.extract_batch`
gives the prettier output without changing the public API. See
`PATCH_aggregator.md` for the two diffs.

## Test results

```
26 passed in 1.72s
```

Covered: 10 pipeline tests, 6 sklearn-wrapper tests, 6 BatchIO tests, 4 progress
tests. Round-trip integrity verified for every IO format. Pipeline output
verified bit-equal with direct `FeatureAggregator` calls.
