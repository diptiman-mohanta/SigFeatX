import numpy as np

from SigFeatX.features.features import TimeDomainFeatures, FrequencyDomainFeatures


def _make_sine(amplitude: float = 3.0, frequency_hz: float = 50.0, fs: int = 1000, duration_s: float = 1.0) -> np.ndarray:
    """Generate a full-cycle sine wave with exact FFT-bin alignment."""
    n = int(fs * duration_s)
    t = np.arange(n) / fs
    return amplitude * np.sin(2.0 * np.pi * frequency_hz * t)


def test_sine_mathematical_correctness():
    """Validate features with known closed-form values for A*sin(2*pi*f*t)."""
    amplitude = 3.0
    frequency_hz = 50.0
    fs = 1000
    sig = _make_sine(amplitude=amplitude, frequency_hz=frequency_hz, fs=fs)

    td = TimeDomainFeatures.extract(sig)
    fd = FrequencyDomainFeatures.extract(sig, fs=fs)

    # Defensive checks so key renames fail with clear messages.
    assert "mean" in td, "Time-domain key missing: mean"
    assert "rms" in td, "Time-domain key missing: rms"
    assert "crest_factor" in td, "Time-domain key missing: crest_factor"
    assert "dominant_frequency" in fd, "Frequency-domain key missing: dominant_frequency"

    expected_rms = amplitude / np.sqrt(2.0)
    expected_mean = 0.0
    expected_crest_factor = np.sqrt(2.0)
    expected_dominant_frequency = frequency_hz

    # Full-cycle sine has zero mean.
    assert np.isclose(td["mean"], expected_mean, atol=1e-12)

    # RMS(A*sin(...)) = A/sqrt(2).
    assert np.isclose(td["rms"], expected_rms, rtol=1e-12, atol=1e-12)

    # Crest factor = peak / RMS = sqrt(2).
    assert np.isclose(td["crest_factor"], expected_crest_factor, rtol=1e-9, atol=1e-9)

    # Dominant frequency should match injected frequency (exact FFT bin here).
    assert np.isclose(fd["dominant_frequency"], expected_dominant_frequency, atol=1e-12)
