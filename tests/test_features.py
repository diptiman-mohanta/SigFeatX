import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scipy import signal as scipy_signal

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
                        'kurtosis', 'crest_factor', 'tkeo_mean',
                        'tkeo_std', 'tkeo_max', 'line_length',
                        'autocorrelation_peak_lag', 'autocorrelation_peak_value']
    
    for feat in expected_features:
        assert feat in features, f"Missing feature: {feat}"
    
    # Sanity checks
    assert features['std'] > 0, "Std should be positive"
    assert features['energy'] > 0, "Energy should be positive"
    assert features['tkeo_max'] >= features['tkeo_mean'], "TKEO max should bound TKEO mean"
    assert features['line_length'] >= 0, "Line length should be non-negative"
    assert features['autocorrelation_peak_lag'] >= 0, "Autocorrelation peak lag should be non-negative"
    assert -1 <= features['autocorrelation_peak_value'] <= 1, "Autocorrelation peak value should be in [-1,1]"
    
    print("✓ Time domain features tests passed")


def test_frequency_domain():
    """Test frequency domain features."""
    print("Testing frequency domain features...")
    
    t = np.linspace(0, 1, 1000)
    sig = np.sin(2 * np.pi * 10 * t)
    
    features = FrequencyDomainFeatures.extract(sig, fs=1000)
    
    # Check expected features
    expected_features = ['spectral_centroid', 'spectral_bandwidth',
                        'spectral_bandwidth_90', 'dominant_frequency', 'spectral_entropy',
                        'instantaneous_freq_mean', 'instantaneous_freq_std',
                        'spectral_slope', 'bandpower_delta', 'bandpower_theta',
                        'bandpower_alpha', 'bandpower_beta', 'bandpower_gamma',
                        'bandpower_delta_rel', 'bandpower_theta_rel',
                        'bandpower_alpha_rel', 'bandpower_beta_rel', 'bandpower_gamma_rel']
    
    for feat in expected_features:
        assert feat in features, f"Missing feature: {feat}"
    
    # Check dominant frequency is close to 10 Hz
    assert 8 < features['dominant_frequency'] < 12, "Dominant freq should be ~10 Hz"
    assert features['spectral_bandwidth_90'] >= 0, "Fractional bandwidth should be non-negative"
    assert np.isfinite(features['instantaneous_freq_mean']), "Instantaneous frequency mean should be finite"
    assert features['instantaneous_freq_std'] >= 0, "Instantaneous frequency std should be non-negative"
    assert np.isfinite(features['spectral_slope']), "Spectral slope should be finite"

    rel_sum = (
        features['bandpower_delta_rel']
        + features['bandpower_theta_rel']
        + features['bandpower_alpha_rel']
        + features['bandpower_beta_rel']
        + features['bandpower_gamma_rel']
    )
    assert 0 <= rel_sum <= 1.0 + 1e-8, "Bandpower relative sum should be <= 1"
    
    print("✓ Frequency domain features tests passed")


def test_frequency_domain_instantaneous_frequency_edge_cases():
    """Instantaneous frequency summaries should stay finite on edge cases."""
    constant_sig = np.ones(256)
    constant_features = FrequencyDomainFeatures.extract(constant_sig, fs=1000)
    assert constant_features['instantaneous_freq_mean'] == 0.0
    assert constant_features['instantaneous_freq_std'] == 0.0

    short_sig = np.array([0.0, 1.0, 0.0])
    short_features = FrequencyDomainFeatures.extract(short_sig, fs=1000)
    assert short_features['instantaneous_freq_mean'] == 0.0
    assert short_features['instantaneous_freq_std'] == 0.0

    t = np.arange(2000) / 1000.0
    chirp_sig = scipy_signal.chirp(t, f0=10.0, f1=90.0, t1=t[-1] + 1 / 1000.0, method='linear')
    chirp_features = FrequencyDomainFeatures.extract(chirp_sig, fs=1000)
    expected_mean = 50.0
    expected_std = (90.0 - 10.0) / np.sqrt(12.0)
    assert np.isclose(chirp_features['instantaneous_freq_mean'], expected_mean, atol=3.0)
    assert np.isclose(chirp_features['instantaneous_freq_std'], expected_std, atol=3.0)


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
