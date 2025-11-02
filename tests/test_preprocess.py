import numpy as np
import sys
sys.path.insert(0, '..')

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


if __name__ == "__main__":
    print("Running Preprocessing Tests")
    print("=" * 60)
    test_denoise()
    test_normalize()
    test_detrend()
    print("\nAll preprocessing tests passed! ✓")
