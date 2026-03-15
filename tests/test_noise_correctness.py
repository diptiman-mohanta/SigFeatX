import numpy as np

from SigFeatX.features.features import (
    TimeDomainFeatures,
    FrequencyDomainFeatures,
)


def _make_white_noise(n: int = 5000, seed: int = 42) -> np.ndarray:
    """Generate reproducible white Gaussian noise."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)


def test_noise_generator_reproducible():
    """Fixed seed should produce identical noise samples."""
    sig_a = _make_white_noise(seed=42)
    sig_b = _make_white_noise(seed=42)
    sig_c = _make_white_noise(seed=7)

    assert np.array_equal(sig_a, sig_b)
    assert not np.array_equal(sig_a, sig_c)


def test_noise_mean_near_zero():
    """Sample mean of white noise converges to 0 by the law of large numbers."""
    sig = _make_white_noise()
    td = TimeDomainFeatures.extract(sig)

    assert "mean" in td, "Time-domain key missing: mean"
    assert np.abs(td["mean"]) < 0.05


def test_noise_std_near_unity():
    """Unit-variance noise should have std approximately 1."""
    sig = _make_white_noise()
    td = TimeDomainFeatures.extract(sig)

    assert "std" in td, "Time-domain key missing: std"
    assert np.isclose(td["std"], 1.0, atol=0.05)


def test_noise_spectral_flatness_high():
    """White noise has a flat spectrum, so spectral flatness should be high."""
    sig = _make_white_noise()
    fd = FrequencyDomainFeatures.extract(sig, fs=1000)

    assert "spectral_flatness" in fd, "Frequency-domain key missing: spectral_flatness"
    assert fd["spectral_flatness"] > 0.7


def test_noise_spectral_entropy_high():
    """White noise spreads energy widely across frequencies, so entropy is high."""
    sig = _make_white_noise()
    fd = FrequencyDomainFeatures.extract(sig, fs=1000)

    assert "spectral_entropy" in fd, "Frequency-domain key missing: spectral_entropy"

    # Robust check: use entropy as a fraction of the theoretical maximum.
    max_entropy = np.log2(len(sig) // 2 + 1)
    entropy_ratio = fd["spectral_entropy"] / max_entropy
    assert entropy_ratio > 0.8


def test_noise_zero_crossing_rate_high():
    """White noise crosses zero often, with expected rate near 0.5 per sample."""
    sig = _make_white_noise()
    td = TimeDomainFeatures.extract(sig)

    assert "zero_crossing_rate" in td, "Time-domain key missing: zero_crossing_rate"
    assert td["zero_crossing_rate"] > 0.4


def test_noise_skewness_near_zero():
    """Gaussian noise is symmetric, so skewness should be close to 0."""
    sig = _make_white_noise()
    td = TimeDomainFeatures.extract(sig)

    assert "skewness" in td, "Time-domain key missing: skewness"
    assert np.abs(td["skewness"]) < 0.15


def test_noise_kurtosis_near_gaussian_excess_zero():
    """Excess kurtosis for Gaussian noise should be close to 0."""
    sig = _make_white_noise()
    td = TimeDomainFeatures.extract(sig)

    assert "kurtosis" in td, "Time-domain key missing: kurtosis"
    assert np.abs(td["kurtosis"]) < 0.3


def test_noise_spectral_centroid_near_half_nyquist():
    """For white noise, one-sided spectral centroid should be near fs/4."""
    fs = 1000
    sig = _make_white_noise()
    fd = FrequencyDomainFeatures.extract(sig, fs=fs)

    assert "spectral_centroid" in fd, "Frequency-domain key missing: spectral_centroid"
    assert np.isclose(fd["spectral_centroid"], fs / 4.0, atol=40.0)

