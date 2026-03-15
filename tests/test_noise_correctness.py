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


def test_noise_spectral_bandwidth_90_near_expected_fractional_width():
    """For white noise, 90% cumulative-power bandwidth should be near 90% of Nyquist."""
    fs = 1000
    sig = _make_white_noise()
    fd = FrequencyDomainFeatures.extract(sig, fs=fs)

    assert "spectral_bandwidth_90" in fd, "Frequency-domain key missing: spectral_bandwidth_90"
    assert np.isclose(fd["spectral_bandwidth_90"], 0.45 * fs, atol=40.0)


def test_noise_spectral_slope_near_flat():
    """White noise is spectrally flat on average, so log-log slope should be near 0."""
    fs = 1000
    sig = _make_white_noise(n=8000, seed=123)
    fd = FrequencyDomainFeatures.extract(sig, fs=fs)

    assert "spectral_slope" in fd, "Frequency-domain key missing: spectral_slope"
    assert np.isclose(fd["spectral_slope"], 0.0, atol=0.35)


def test_noise_bandpower_relative_sum_near_one_when_nyquist_is_100hz():
    """At fs=200 (Nyquist=100), delta..gamma cover the full positive spectrum."""
    fs = 200
    sig = _make_white_noise(n=8000, seed=99)
    fd = FrequencyDomainFeatures.extract(sig, fs=fs)

    rel_sum = (
        fd["bandpower_delta_rel"]
        + fd["bandpower_theta_rel"]
        + fd["bandpower_alpha_rel"]
        + fd["bandpower_beta_rel"]
        + fd["bandpower_gamma_rel"]
    )
    assert np.isclose(rel_sum, 1.0, atol=0.06)


def test_noise_line_length_exceeds_sine_at_same_rms():
    """Line length should be larger for noise than a smooth sine with matched RMS."""
    fs = 1000
    noise = _make_white_noise(n=5000, seed=7)
    t = np.arange(len(noise)) / fs
    sine = np.sqrt(2.0) * np.sin(2.0 * np.pi * 10.0 * t)

    td_noise = TimeDomainFeatures.extract(noise)
    td_sine = TimeDomainFeatures.extract(sine)

    assert td_noise["line_length"] > td_sine["line_length"]

