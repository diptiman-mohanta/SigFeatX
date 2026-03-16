import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from SigFeatX.preprocess import SignalPreprocessor


def test_denoise():
    """Test denoising methods."""
    print("Testing denoising...")
    
    # Create clean signal
    t = np.linspace(0, 1, 1000)
    clean = np.sin(2 * np.pi * 5 * t)
    noisy = clean + 0.5 * np.random.randn(1000)
    
    preprocessor = SignalPreprocessor()
    
    # Test wavelet denoising
    denoised = preprocessor.denoise(noisy, method='wavelet')
    assert len(denoised) == len(noisy), "Length mismatch after denoising"
    
    # Test median filtering
    denoised = preprocessor.denoise(noisy, method='median', kernel_size=5)
    assert len(denoised) == len(noisy), "Length mismatch after median filtering"
    
    print("✓ Denoising tests passed")


def test_normalize():
    """Test normalization methods."""
    print("Testing normalization...")
    
    sig = np.random.randn(1000) * 10 + 5
    preprocessor = SignalPreprocessor()
    
    # Z-score normalization
    normalized = preprocessor.normalize(sig, method='zscore')
    assert np.abs(np.mean(normalized)) < 1e-10, "Mean should be ~0"
    assert np.abs(np.std(normalized) - 1) < 1e-10, "Std should be ~1"
    
    # Min-max normalization
    normalized = preprocessor.normalize(sig, method='minmax')
    assert np.min(normalized) >= 0, "Min should be >= 0"
    assert np.max(normalized) <= 1, "Max should be <= 1"
    
    print("✓ Normalization tests passed")


def test_detrend():
    """Test detrending."""
    print("Testing detrending...")
    
    t = np.linspace(0, 1, 1000)
    trend = 2 * t + 1
    oscillation = np.sin(2 * np.pi * 5 * t)
    sig = trend + oscillation
    
    preprocessor = SignalPreprocessor()
    detrended = preprocessor.detrend(sig, method='linear')
    
    # Detrended signal should have mean close to 0
    assert np.abs(np.mean(detrended)) < 0.1, "Mean should be close to 0"
    
    print("✓ Detrending tests passed")


def test_bandpass_keeps_in_band_frequency():
    """Bandpass filtering should strongly favor the retained component."""
    fs = 200
    t = np.arange(fs * 2) / fs
    sig = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 60 * t)

    preprocessor = SignalPreprocessor()
    filtered = preprocessor.bandpass(sig, low_hz=8, high_hz=12, fs=fs, order=4)

    freqs = np.fft.rfftfreq(len(filtered), d=1 / fs)
    magnitude = np.abs(np.fft.rfft(filtered))
    mag_10 = magnitude[np.argmin(np.abs(freqs - 10))]
    mag_60 = magnitude[np.argmin(np.abs(freqs - 60))]

    assert mag_10 > 10 * mag_60


def test_notch_suppresses_target_frequency():
    """Notch filtering should attenuate the requested interference tone."""
    fs = 500
    t = np.arange(fs * 2) / fs
    sig = np.sin(2 * np.pi * 10 * t) + 0.8 * np.sin(2 * np.pi * 50 * t)

    preprocessor = SignalPreprocessor()
    filtered = preprocessor.notch(sig, freq_hz=50, fs=fs, quality_factor=30.0)

    freqs = np.fft.rfftfreq(len(filtered), d=1 / fs)
    magnitude = np.abs(np.fft.rfft(filtered))
    mag_10 = magnitude[np.argmin(np.abs(freqs - 10))]
    mag_50 = magnitude[np.argmin(np.abs(freqs - 50))]

    assert mag_10 > 10 * mag_50


def test_als_baseline_tracks_positive_peak_signal():
    """ALS baseline estimation should track slow drift beneath positive peaks."""
    fs = 100
    t = np.arange(fs * 10) / fs

    clean = (
        1.2 * np.exp(-((t - 2.0) ** 2) / (2 * 0.08 ** 2))
        + 0.9 * np.exp(-((t - 5.0) ** 2) / (2 * 0.10 ** 2))
        + 1.1 * np.exp(-((t - 8.0) ** 2) / (2 * 0.07 ** 2))
    )
    baseline = 0.8 * np.sin(2 * np.pi * 0.3 * t) + 0.1 * t
    sig = clean + baseline

    preprocessor = SignalPreprocessor()
    estimated_baseline = preprocessor.als_baseline(sig, lam=1e6, p=0.01, n_iter=20)
    corrected = sig - estimated_baseline

    raw_rmse = np.sqrt(np.mean((sig - clean) ** 2))
    corrected_rmse = np.sqrt(np.mean((corrected - clean) ** 2))

    assert corrected_rmse < raw_rmse


if __name__ == "__main__":
    print("Running Preprocessing Tests")
    print("=" * 60)
    test_denoise()
    test_normalize()
    test_detrend()
    print("\nAll preprocessing tests passed! ✓")
