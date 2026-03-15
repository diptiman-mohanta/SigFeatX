import numpy as np
from scipy import signal as scipy_signal

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
    assert "tkeo_mean" in td, "Time-domain key missing: tkeo_mean"
    assert "tkeo_std" in td, "Time-domain key missing: tkeo_std"
    assert "tkeo_max" in td, "Time-domain key missing: tkeo_max"
    assert "line_length" in td, "Time-domain key missing: line_length"
    assert "autocorrelation_peak_lag" in td, "Time-domain key missing: autocorrelation_peak_lag"
    assert "autocorrelation_peak_value" in td, "Time-domain key missing: autocorrelation_peak_value"
    assert "dominant_frequency" in fd, "Frequency-domain key missing: dominant_frequency"
    assert "instantaneous_freq_mean" in fd, "Frequency-domain key missing: instantaneous_freq_mean"
    assert "instantaneous_freq_std" in fd, "Frequency-domain key missing: instantaneous_freq_std"
    assert "spectral_bandwidth_90" in fd, "Frequency-domain key missing: spectral_bandwidth_90"
    assert "spectral_slope" in fd, "Frequency-domain key missing: spectral_slope"
    assert "bandpower_gamma_rel" in fd, "Frequency-domain key missing: bandpower_gamma_rel"

    expected_rms = amplitude / np.sqrt(2.0)
    expected_mean = 0.0
    expected_crest_factor = np.sqrt(2.0)
    expected_dominant_frequency = frequency_hz
    omega = 2.0 * np.pi * frequency_hz / fs
    expected_tkeo = amplitude ** 2 * np.sin(omega) ** 2

    # Full-cycle sine has zero mean.
    assert np.isclose(td["mean"], expected_mean, atol=1e-12)

    # RMS(A*sin(...)) = A/sqrt(2).
    assert np.isclose(td["rms"], expected_rms, rtol=1e-12, atol=1e-12)

    # Crest factor = peak / RMS = sqrt(2).
    assert np.isclose(td["crest_factor"], expected_crest_factor, rtol=1e-9, atol=1e-9)

    # TKEO(A*sin(Omega*n)) = A^2 * sin(Omega)^2, which is constant over n.
    assert np.isclose(td["tkeo_mean"], expected_tkeo, rtol=1e-9, atol=1e-9)
    assert np.isclose(td["tkeo_std"], 0.0, atol=1e-12)
    assert np.isclose(td["tkeo_max"], expected_tkeo, rtol=1e-9, atol=1e-9)

    expected_period_samples = fs / frequency_hz
    assert np.isclose(td["autocorrelation_peak_lag"], expected_period_samples, atol=1.0)
    assert td["autocorrelation_peak_value"] > 0.95
    assert td["line_length"] > 0.0

    # Dominant frequency should match injected frequency (exact FFT bin here).
    assert np.isclose(fd["dominant_frequency"], expected_dominant_frequency, atol=1e-12)
    assert np.isclose(fd["spectral_bandwidth_90"], 0.0, atol=1e-12)

    # Instantaneous frequency of a pure stationary sine should be constant.
    assert np.isclose(fd["instantaneous_freq_mean"], expected_dominant_frequency, atol=0.5)
    assert fd["instantaneous_freq_std"] < 1e-6
    assert np.isfinite(fd["spectral_slope"])
    assert fd["bandpower_gamma_rel"] > 0.95


def test_instantaneous_frequency_rejects_edge_artifacts_on_am_sine():
    """Low-envelope samples should not destabilize the carrier-frequency summary."""
    fs = 1000
    duration_s = 2.0
    t = np.arange(int(fs * duration_s)) / fs
    carrier_hz = 40.0
    modulator_hz = 1.0
    envelope = 1.0 + 0.98 * np.sin(2.0 * np.pi * modulator_hz * t)
    sig = envelope * np.sin(2.0 * np.pi * carrier_hz * t)

    fd = FrequencyDomainFeatures.extract(sig, fs=fs)

    assert np.isclose(fd["instantaneous_freq_mean"], carrier_hz, atol=1.0)
    assert fd["instantaneous_freq_std"] < 1.0
