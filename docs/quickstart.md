# Quickstart

## One-liner feature extraction

```python
import numpy as np
from SigFeatX import FeatureAggregator

# A synthetic two-tone signal
t = np.linspace(0, 1, 1000)
signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)

extractor = FeatureAggregator(fs=1000)
features = extractor.extract_all_features(
    signal,
    decomposition_methods=['fourier', 'dwt', 'emd'],
    preprocess_signal=True,
    denoise=True,
    normalize=True,
    detrend=True,
)

print(f"Extracted {len(features)} features")
print(f"Dominant Frequency: {features['raw_dominant_frequency']:.2f} Hz")
```

This single call runs preprocessing (denoise/normalize/detrend), one or more
decompositions, and every feature family (time-domain, frequency-domain,
entropy, nonlinear-dynamics, and per-component decomposition features),
returning a flat `dict[str, float]`.

## Batch processing

For many signals at once, `extract_batch` returns a pandas DataFrame ready
for a model:

```python
result = extractor.extract_batch(
    signals,          # list of 1D arrays, or a 2D array (n_signals, N)
    n_jobs=-1,        # use all CPU cores
)
result.dataframe.to_csv("features.csv")
```

## sklearn integration

```python
from SigFeatX import SigFeatXTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([
    ("features", SigFeatXTransformer(fs=1000)),
    ("clf", RandomForestClassifier()),
])
pipe.fit(X_train, y_train)
```

## A fluent pipeline

The pipeline is lazy: nothing runs until `.extract*` is called, and every
configuration method returns `self` so calls chain together.

```python
from SigFeatX import Pipeline

features = (
    Pipeline(fs=1000)
    .detrend(method="linear")
    .denoise(method="wavelet")
    .normalize(method="zscore")
    .decompose(["fourier", "dwt"])
    .extract(signal)
)

# Batch, parallel, with a progress bar
df = (
    Pipeline(fs=1000)
    .detrend(method="als", lam=1e5)
    .denoise(method="bandpass", low_hz=1, high_hz=40)
    .normalize(method="robust")
    .decompose(["emd", "vmd"])
    .with_parallel(n_jobs=-1)
    .with_progress()
    .extract_batch(signals)
)
```

## Next steps

- Browse the full {doc}`api` reference for every decomposition and feature
  class.
- See `examples/` in the repository for end-to-end EEG/ECG walkthroughs.
