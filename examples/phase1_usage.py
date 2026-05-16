"""
SigFeatX — Phase 1 Upgrade Examples
====================================
Demonstrates the four new capabilities shipped in 0.2.0:
  1. Pipeline           — fluent chaining
  2. SigFeatXTransformer— sklearn integration
  3. BatchIO            — Parquet / HDF5 / Feather IO
  4. tqdm progress      — opt-in progress bars

Run:
    python examples/phase1_usage.py
"""

import numpy as np

from SigFeatX import (
    FeatureAggregator,
    Pipeline,
    BatchIO,
)

# Generate synthetic signals
np.random.seed(42)
fs = 1000
n_signals = 30


def make_signal(freq: float, n: int = 2000) -> np.ndarray:
    t = np.arange(n) / fs
    return (
        np.sin(2 * np.pi * freq * t)
        + 0.5 * np.sin(2 * np.pi * (freq * 3) * t)
        + 0.1 * np.random.randn(n)
        + 0.05 * t                # mild trend
    )


signals = [make_signal(5 + i) for i in range(n_signals)]
single_signal = make_signal(15.0)


print("=" * 64)
print("  1. PIPELINE BUILDER")
print("=" * 64)

# Single-signal pipeline
pipe = (
    Pipeline(fs=fs)
    .detrend(method='linear')
    .denoise(method='bandpass', low_hz=1, high_hz=40)
    .normalize(method='zscore')
    .decompose(['fourier', 'dwt'])
)
print(f"\n{pipe}\n")

features = pipe.extract(single_signal)
print(f"  Single signal: {len(features)} features extracted")
print(f"  Sample: raw_rms = {features['raw_rms']:.4f}")
print(f"  Sample: fourier_dominant_freq = {features['fourier_dominant_freq']:.2f} Hz")

# Batch pipeline with progress
print("\n  Batch pipeline (sequential):")
batch_result = (
    Pipeline(fs=fs)
    .detrend('linear')
    .normalize('zscore')
    .decompose('fourier')
    .with_progress()
    .extract_batch(signals)
)
print(f"  -> {batch_result.dataframe.shape[0]} signals, "
      f"{batch_result.dataframe.shape[1]} features")

# Windowed pipeline
print("\n  Windowed pipeline:")
long_signal = make_signal(20.0, n=8000)
windowed = (
    Pipeline(fs=fs)
    .decompose('fourier')
    .extract_windowed(long_signal, window_size=1024, step_size=512)
)
print(f"  -> {windowed.dataframe.shape[0]} windows")
print(f"  -> Columns include: {[c for c in windowed.dataframe.columns[:5]]}")

# Configuration round-trip
print("\n  Pipeline configuration:")
print(f"  {pipe.to_dict()}")


# ------------------------------------------------------------------------
print("\n" + "=" * 64)
print("  2. SKLEARN INTEGRATION")
print("=" * 64)

try:
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from SigFeatX import SigFeatXTransformer

    # Fake binary labels for demonstration
    y = np.array([i % 2 for i in range(n_signals)])
    X = np.array([s[:1500] for s in signals])   # uniform length for 2D array

    pipe_sk = SkPipeline([
        ('features', SigFeatXTransformer(
            fs=fs,
            decomposition_methods=['fourier'],
            preprocess_signal=True,
            n_jobs=1,
        )),
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=10, random_state=0)),
    ])

    pipe_sk.fit(X, y)
    score = pipe_sk.score(X, y)
    n_features = pipe_sk.named_steps['features'].n_features_out_
    print(f"\n  sklearn Pipeline fitted")
    print(f"  -> {n_features} features extracted per signal")
    print(f"  -> Train accuracy: {score:.3f}  (overfit on tiny demo data)")

    feat_names = pipe_sk.named_steps['features'].get_feature_names_out()
    print(f"  -> First 3 feature names: {list(feat_names[:3])}")
except ImportError:
    print("\n  sklearn not installed — skipping demo.")


# ------------------------------------------------------------------------
print("\n" + "=" * 64)
print("  3. BATCH IO  (Parquet / HDF5 / Feather)")
print("=" * 64)

import tempfile
import os

with tempfile.TemporaryDirectory() as tmpdir:
    parquet_path = os.path.join(tmpdir, 'features.parquet')
    h5_path = os.path.join(tmpdir, 'features.h5')
    feather_path = os.path.join(tmpdir, 'features.feather')

    df = batch_result.dataframe

    # Parquet
    BatchIO.save_parquet(df, parquet_path, metadata={
        'fs': fs,
        'methods': 'fourier',
        'version': '0.2.0',
    })
    df_back = BatchIO.load_parquet(parquet_path)
    md = BatchIO.load_parquet_metadata(parquet_path)
    print(f"\n  Parquet: wrote {os.path.getsize(parquet_path)} bytes, "
          f"shape={df_back.shape}")
    print(f"  Parquet metadata: {md}")

    # HDF5 (skip if tables not installed)
    try:
        BatchIO.save_hdf5(df, h5_path, metadata={'fs': fs})
        df_h5 = BatchIO.load_hdf5(h5_path)
        print(f"  HDF5:    wrote {os.path.getsize(h5_path)} bytes, "
              f"shape={df_h5.shape}")
    except ImportError as exc:
        print(f"  HDF5:    skipped ({exc})")

    # Auto-dispatch
    BatchIO.save(df, parquet_path)
    print(f"  Auto-dispatch by extension works.")


# ------------------------------------------------------------------------
print("\n" + "=" * 64)
print("  4. PROGRESS BAR  (tqdm if installed, fallback otherwise)")
print("=" * 64)

print("\n  Running batch with show_progress=True:")
agg = FeatureAggregator(fs=fs)
result = agg.extract_batch(
    signals[:10],
    decomposition_methods=['fourier'],
    preprocess_signal=False,
    validate=False,
    show_progress=True,
    n_jobs=1,
)
print(f"  -> done ({result.n_success}/{result.n_success + result.n_failed})")


print("\n" + "=" * 64)
print("  All phase 1 features OK.")
print("=" * 64)