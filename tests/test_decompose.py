import numpy as np
import sys
sys.path.insert(0, '..')

from SigFeatX.decompose import (
    FourierTransform, WaveletDecomposer, EMD, VMD
)


def test_fourier():
    """Test Fourier Transform."""
    print("Testing Fourier Transform...")
    
    t = np.linspace(0, 1, 1000)
    sig = np.sin(2 * np.pi * 10 * t)
    
    ft = FourierTransform(fs=1000)
    freqs, magnitude = ft.transform(sig)
    
    # Find dominant frequency
    dominant_freq = freqs[np.argmax(magnitude)]
    
    assert len(freqs) == len(magnitude), "Length mismatch"
    assert 9 < dominant_freq < 11, f"Dominant freq should be ~10 Hz, got {dominant_freq}"
    
    print("✓ Fourier Transform tests passed")


def test_wavelet():
    """Test Wavelet decomposition."""
    print("Testing Wavelet decomposition...")
    
    sig = np.random.randn(1000)
    
    wavelet = WaveletDecomposer(wavelet='db4')
    
    # Test DWT
    coeffs = wavelet.dwt(sig, level=3)
    assert len(coeffs) == 4, "Should have 4 coefficient arrays (3 levels + approx)"
    
    # Test reconstruction
    reconstructed = wavelet.idwt(coeffs)
    assert len(reconstructed) >= len(sig) - 10, "Reconstruction length check"
    
    print("✓ Wavelet decomposition tests passed")


def test_emd():
    """Test EMD."""
    print("Testing EMD...")
    
    t = np.linspace(0, 1, 500)
    sig = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 15 * t)
    
    emd = EMD(max_imf=5)
    imfs = emd.decompose(sig)
    
    assert len(imfs) > 0, "Should extract at least one IMF"
    
    # Reconstruction
    reconstructed = emd.reconstruct(imfs)
    assert len(reconstructed) == len(sig), "Reconstruction length should match"
    
    print("✓ EMD tests passed")


def test_vmd():
    """Test VMD."""
    print("Testing VMD...")
    
    sig = np.random.randn(500)
    
    vmd = VMD(K=3)
    modes = vmd.decompose(sig)
    
    assert modes.shape[0] == 3, "Should have 3 modes"
    assert modes.shape[1] == len(sig), "Mode length should match signal"
    
    print("✓ VMD tests passed")


if __name__ == "__main__":
    print("Running Decomposition Tests")
    print("=" * 60)
    test_fourier()
    test_wavelet()
    test_emd()
    test_vmd()
    print("\nAll decomposition tests passed! ✓")