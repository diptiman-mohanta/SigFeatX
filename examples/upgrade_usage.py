"""
SigFeatX — Upgrade Usage Examples
===================================
Demonstrates the three new capabilities:
  1. Preprocessing: bandpass, notch, ALS baseline
  2. Architecture:  batch processing, parallel execution, multi-channel

Run with:  python examples/upgrade_usage.py
"""

import numpy as np

np.random.seed(42)
fs  = 1000
t   = np.linspace(0, 2, fs * 2, endpoint=False)

# ── Synthetic signals ──────────────────────────────────────────────────────
# Multi-sine: 10 + 50 + 100 Hz + 50 Hz power-line interference + curved baseline
signal_eeg = (
    np.sin(2 * np.pi * 10  * t)
    + 0.5 * np.sin(2 * np.pi * 50  * t)   # power-line component
    + 0.3 * np.sin(2 * np.pi * 100 * t)
    + 0.1 * np.random.randn(len(t))
    + 0.5 * np.sin(2 * np.pi * 0.3 * t)   # slow curved baseline
)

# Multi-channel: 3 channels with different phase relationships
eeg_3ch = np.array([
    signal_eeg,
    np.roll(signal_eeg, 50) + 0.05 * np.random.randn(len(t)),   # 50-sample lag
    np.roll(signal_eeg, 25) + 0.10 * np.random.randn(len(t)),   # 25-sample lag
])

# Batch: 20 signals of varying lengths
batch_signals = [
    np.sin(2 * np.pi * (5 + i) * np.linspace(0, 1, 500 + i * 50))
    + 0.1 * np.random.randn(500 + i * 50)
    for i in range(20)
]


# ==========================================================================
# 1. PREPROCESSING UPGRADES
# ==========================================================================

from SigFeatX.preprocess import SignalPreprocessor

pp = SignalPreprocessor()

print("=" * 60)
print("  1. PREPROCESSING UPGRADES")
print("=" * 60)

# ── Bandpass filter ────────────────────────────────────────────────────────
# Keep only 8–40 Hz (EEG alpha/beta band)
sig_bp = pp.bandpass(signal_eeg, low_hz=8, high_hz=40, fs=fs)
print(f"\nBandpass 8-40 Hz:")
print(f"  Input  RMS: {np.sqrt(np.mean(signal_eeg**2)):.4f}")
print(f"  Output RMS: {np.sqrt(np.mean(sig_bp**2)):.4f}")

# Also available as a denoise() method option
sig_bp2 = pp.denoise(signal_eeg, method='bandpass', low_hz=8, high_hz=40, fs=fs)
assert np.allclose(sig_bp, sig_bp2), "bandpass denoise should match direct call"
print("  bandpass via denoise() matches direct call: OK")

# ── Notch filter ──────────────────────────────────────────────────────────
# Remove 50 Hz power-line interference
sig_notch = pp.notch(signal_eeg, freq_hz=50, fs=fs, quality_factor=30)

# Verify: compute power in 48-52 Hz band before and after
from scipy.signal import welch
freqs, psd_before = welch(signal_eeg, fs=fs, nperseg=256)
freqs, psd_after  = welch(sig_notch,  fs=fs, nperseg=256)
notch_band = (freqs >= 48) & (freqs <= 52)
power_before = np.sum(psd_before[notch_band])
power_after  = np.sum(psd_after[notch_band])
print(f"\nNotch 50 Hz (Q=30):")
print(f"  Power in 48-52 Hz before: {power_before:.4f}")
print(f"  Power in 48-52 Hz after : {power_after:.6f}")
print(f"  Reduction: {(1 - power_after/power_before) * 100:.1f}%")

# Chain notch + bandpass: remove line noise, then keep 1-40 Hz
sig_clean = pp.notch(signal_eeg, freq_hz=50, fs=fs)
sig_clean = pp.bandpass(sig_clean, low_hz=1, high_hz=40, fs=fs)
print(f"\nChained notch + bandpass (1-40 Hz, 50 Hz removed):")
print(f"  Output RMS: {np.sqrt(np.mean(sig_clean**2)):.4f}")

# Also available as a denoise() option
sig_notch2 = pp.denoise(signal_eeg, method='notch', freq_hz=50, fs=fs)
print("  notch via denoise() works: OK")

# ── ALS baseline correction ────────────────────────────────────────────────
sig_als = pp.detrend(signal_eeg, method='als')

# Verify: the slow 0.3 Hz component should be removed
# After ALS, correlation with the baseline sinusoid should drop
baseline_pure = 0.5 * np.sin(2 * np.pi * 0.3 * t)
corr_before = np.corrcoef(signal_eeg, baseline_pure)[0, 1]
corr_after  = np.corrcoef(sig_als,    baseline_pure)[0, 1]
print(f"\nALS baseline correction (lam=1e4, p=0.01):")
print(f"  Correlation with 0.3 Hz baseline before: {corr_before:.4f}")
print(f"  Correlation with 0.3 Hz baseline after : {corr_after:.4f}")
print(f"  Baseline removed: {'YES' if abs(corr_after) < abs(corr_before) else 'NO'}")

# Direct ALS baseline retrieval
baseline_est = pp.als_baseline(signal_eeg, lam=1e4, p=0.01)
print(f"  Estimated baseline range: [{baseline_est.min():.3f}, {baseline_est.max():.3f}]")


# ==========================================================================
# 2. ARCHITECTURE UPGRADES
# ==========================================================================

from SigFeatX.aggregator import FeatureAggregator

agg = FeatureAggregator(fs=fs)

print("\n" + "=" * 60)
print("  2. ARCHITECTURE UPGRADES")
print("=" * 60)

# ── Batch processing (sequential) ─────────────────────────────────────────
print(f"\nBatch processing ({len(batch_signals)} signals, sequential):")
result = agg.extract_batch(
    batch_signals,
    decomposition_methods=['fourier'],
    preprocess_signal=True,
    validate=False,
    n_jobs=1,
    show_progress=True,
)
print(f"  {result}")
print(f"  DataFrame shape: {result.dataframe.shape}")
print(f"  Any NaN rows: {result.dataframe.isnull().any(axis=1).sum()}")
print(f"  Sample features: {result.feature_names[:5]}")

# ── Batch processing with intentional error isolation ─────────────────────
print("\nBatch with error isolation (one bad signal):")
bad_signals = batch_signals[:5] + [np.array([np.nan, np.nan])] + batch_signals[5:10]
result_err = agg.extract_batch(
    bad_signals,
    decomposition_methods=['fourier'],
    preprocess_signal=False,
    validate=False,
    n_jobs=1,
    on_error='warn',
)
print(f"  {result_err}")
print(f"  Failed signal indices: {list(result_err.errors.keys())}")
nan_rows = result_err.dataframe.isnull().all(axis=1).sum()
print(f"  NaN rows in DataFrame: {nan_rows}")

# ── Batch processing as 2D array ──────────────────────────────────────────
print("\nBatch from 2D numpy array (10 signals, fixed length 1000):")
signals_2d_batch = np.random.randn(10, 1000)
result_2d = agg.extract_batch(
    signals_2d_batch,
    decomposition_methods=['fourier'],
    preprocess_signal=False,
    validate=False,
    n_jobs=1,
)
print(f"  Input shape: {signals_2d_batch.shape}")
print(f"  {result_2d}")

# ── Multi-channel extraction ───────────────────────────────────────────────
print("\nMulti-channel extraction (3 channels, include_cross=True):")
ch_features = agg.extract_multichannel(
    eeg_3ch,
    channel_names=['Fz', 'Cz', 'Pz'],
    decomposition_methods=['fourier'],
    preprocess_signal=True,
    validate=False,
    include_cross=True,
    n_jobs=1,
)

# Summarise output
per_ch_keys   = [k for k in ch_features if not k.startswith('cross_')]
cross_keys    = [k for k in ch_features if k.startswith('cross_')]
print(f"  Total features:         {len(ch_features)}")
print(f"  Per-channel features:   {len(per_ch_keys)}")
print(f"  Cross-channel features: {len(cross_keys)}")
print(f"  Cross features per pair: {len(cross_keys) // 3}")   # 3 pairs from 3 channels

# Show PLV values between channel pairs
plv_keys = [k for k in cross_keys if 'plv' in k]
print(f"\n  Phase-Locking Values:")
for k in plv_keys:
    print(f"    {k}: {ch_features[k]:.4f}")

# Show cross-correlation lags (should reflect the 50 and 25-sample offsets)
lag_keys = [k for k in cross_keys if 'xcorr_lag' in k]
print(f"\n  Cross-correlation lags (expected ~50 and ~25 samples):")
for k in lag_keys:
    print(f"    {k}: {ch_features[k]:.0f} samples")

# ── Multi-channel without cross features (large datasets) ──────────────────
print("\nMulti-channel, cross=False (faster for many channels):")
ch_fast = agg.extract_multichannel(
    eeg_3ch,
    channel_names=['Fz', 'Cz', 'Pz'],
    decomposition_methods=['fourier'],
    preprocess_signal=False,
    include_cross=False,
    n_jobs=1,
)
print(f"  Features without cross: {len(ch_fast)}")
print(f"  Features with cross:    {len(ch_features)}")
print(f"  Cross overhead:         {len(ch_features) - len(ch_fast)} features")

print("\n" + "=" * 60)
print("  All examples completed successfully.")
print("=" * 60)