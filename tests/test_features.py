import numpy as np
import sys
sys.path.insert(0, '..')

from SigFeatX.features.features import (
    TimeDomainFeatures, FrequencyDomainFeatures,
    EntropyFeatures, NonlinearFeatures
)


def test_time_domain():
    """Test time domain features."""
    print("Testing time domain features...")
    
    sig = np.random.randn(1000)
    features = TimeDomainFeatures.extract(sig)
    
    # Check that all expected features are present
    expected_features = ['mean', 'std', 'rms', 'energy', 'skewness', 
                        'kurtosis', 'crest_factor']
    
    for feat in expected_features:
        assert feat in features, f"Missing feature: {feat}"
    
    # Sanity checks
    assert features['std'] > 0, "Std should be positive"
    assert features['energy'] > 0, "Energy should be positive"
    
    print("✓ Time domain features tests passed")


def test_frequency_domain():
    """Test frequency domain features."""
    print("Testing frequency domain features...")
    
    t = np.linspace(0, 1, 1000)
    sig = np.sin(2 * np.pi * 10 * t)
    
    features = FrequencyDomainFeatures.extract(sig, fs=1000)
    
    # Check expected features
    expected_features = ['spectral_centroid', 'spectral_bandwidth',
                        'dominant_frequency', 'spectral_entropy']
    
    for feat in expected_features:
        assert feat in features, f"Missing feature: {feat}"
    
    # Check dominant frequency is close to 10 Hz
    assert 8 < features['dominant_frequency'] < 12, "Dominant freq should be ~10 Hz"
    
    print("✓ Frequency domain features tests passed")


def test_entropy():
    """Test entropy features."""
    print("Testing entropy features...")
    
    sig = np.random.randn(500)
    features = EntropyFeatures.extract(sig)
    
    expected_features = ['shannon_entropy', 'sample_entropy', 
                        'permutation_entropy', 'approximate_entropy']
    
    for feat in expected_features:
        assert feat in features, f"Missing feature: {feat}"
        assert not np.isnan(features[feat]), f"{feat} is NaN"
    
    print("✓ Entropy features tests passed")


def test_nonlinear():
    """Test nonlinear features."""
    print("Testing nonlinear features...")
    
    sig = np.random.randn(500)
    features = NonlinearFeatures.extract(sig)
    
    expected_features = ['hjorth_activity', 'hjorth_mobility',
                        'hurst_exponent', 'higuchi_fractal_dimension']
    
    for feat in expected_features:
        assert feat in features, f"Missing feature: {feat}"
        assert not np.isnan(features[feat]), f"{feat} is NaN"
    
    print("✓ Nonlinear features tests passed")


if __name__ == "__main__":
    print("Running Feature Extraction Tests")
    print("=" * 60)
    test_time_domain()
    test_frequency_domain()
    test_entropy()
    test_nonlinear()
    print("\nAll feature extraction tests passed! ✓")