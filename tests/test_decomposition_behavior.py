import numpy as np
from scipy.signal import chirp

from SigFeatX.decompose import WaveletDecomposer, EMD


def _make_sine(
    amplitude: float = 2.0,
    frequency_hz: float = 50.0,
    fs: int = 1000,
    duration_s: float = 2.0,
) -> np.ndarray:
    """Pure sine signal for concentrated-band decomposition checks."""
    n = int(fs * duration_s)
    t = np.arange(n) / fs
    return amplitude * np.sin(2.0 * np.pi * frequency_hz * t)


def _make_chirp(fs: int = 1000, duration_s: float = 2.0) -> np.ndarray:
    """Linear chirp sweeping 10-200 Hz for spread-band decomposition checks."""
    n = int(fs * duration_s)
    t = np.arange(n) / fs
    return chirp(t, f0=10, f1=200, t1=duration_s, method="linear")


# --- DWT energy distribution ---

def _dwt_level_energies(signal: np.ndarray, wavelet: str = "db4", level: int = 5) -> list:
    """Return normalised energy per DWT coefficient level."""
    wd = WaveletDecomposer(wavelet=wavelet)
    coeffs = wd.dwt(signal, level=level)
    energies = [float(np.sum(c ** 2)) for c in coeffs]
    total = float(sum(energies))
    return [e / total for e in energies]


def test_dwt_sine_energy_concentrated():
    """Pure sine energy should dominate a single DWT level (>70% in one band)."""
    ratios = _dwt_level_energies(_make_sine())
    assert max(ratios) > 0.70, (
        f"Expected sine energy to concentrate in one DWT level, "
        f"got max ratio {max(ratios):.3f}"
    )


def test_dwt_chirp_energy_spread():
    """Chirp energy should be distributed; no single DWT level above 70%."""
    ratios = _dwt_level_energies(_make_chirp())
    assert max(ratios) < 0.70, (
        f"Expected chirp energy to spread across DWT levels, "
        f"got max ratio {max(ratios):.3f}"
    )


def test_dwt_sine_vs_chirp_concentration():
    """Sine should be more concentrated than chirp (contrast assertion)."""
    sine_ratios = _dwt_level_energies(_make_sine())
    chirp_ratios = _dwt_level_energies(_make_chirp())
    assert max(sine_ratios) > max(chirp_ratios), (
        "Sine should always be more spectrally concentrated than a chirp"
    )


# --- EMD reconstruction ---

def test_emd_reconstruction_sine():
    """Summed IMFs must reconstruct the original sine within numerical tolerance."""
    sig = _make_sine()
    emd = EMD(max_imf=10)
    imfs = emd.decompose(sig)
    reconstructed = emd.reconstruct(imfs)

    assert reconstructed is not None
    assert len(reconstructed) == len(sig)
    rmse = float(np.sqrt(np.mean((sig - reconstructed) ** 2)))
    assert rmse < 1e-6, f"EMD reconstruction RMSE too high: {rmse:.2e}"


def test_emd_sine_fewer_imfs_than_noise():
    """Pure sine should produce fewer IMFs than white noise."""
    sine = _make_sine()
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(len(sine))

    emd = EMD(max_imf=10)
    sine_imfs = emd.decompose(sine)
    noise_imfs = emd.decompose(noise)

    assert len(sine_imfs) < len(noise_imfs), (
        f"Sine IMFs ({len(sine_imfs)}) should be fewer than noise IMFs ({len(noise_imfs)})"
    )
