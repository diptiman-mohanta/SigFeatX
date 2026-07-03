"""
SST — Synchrosqueezing Transform (Fourier-based)
=================================================
Reference: Daubechies, Lu, Wu (2011), "Synchrosqueezed wavelet transforms:
           An empirical mode decomposition-like tool", Applied and
           Computational Harmonic Analysis 30(2), pp. 243-261.

Synchrosqueezing sharpens a time-frequency representation by reassigning
each TF point's energy to the instantaneous frequency estimated from the
local phase derivative. This produces a much sharper spectrum than STFT
while remaining invertible.

Implementation notes
--------------------
This is the Fourier-based SST (the wavelet-based form requires a CWT
backend). We use a sliding STFT and reassign each (t, omega) coefficient
to the frequency bin closest to:

    omega_hat(t, omega) = (1 / (2*pi)) * d_t arg(STFT(t, omega))

Numerically, we differentiate the phase along time with central differences
and skip points where |STFT| is too small (gives unstable phase).
"""


import numpy as np

from .._validation import validate_sampling_rate, validate_signal_1d


class SST:
    """
    Fourier-based Synchrosqueezing Transform.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    nperseg : int
        STFT window length. Larger = better frequency resolution, worse
        time resolution. Default 256.
    noverlap : int or None
        Overlap between successive STFT windows. Default = nperseg // 2.
    window : str
        scipy.signal window name. Default 'hann'.
    gamma : float
        Threshold for skipping low-amplitude STFT bins. Default 1e-6.
    """

    def __init__(
        self,
        fs: float = 1.0,
        nperseg: int = 256,
        noverlap: int | None = None,
        window: str = 'hann',
        gamma: float = 1e-6,
    ):
        self.fs = validate_sampling_rate(fs)
        if nperseg < 4:
            raise ValueError(f"nperseg must be >= 4; got {nperseg}.")
        self.nperseg = nperseg
        self.noverlap = noverlap if noverlap is not None else nperseg // 2
        if self.noverlap < 0 or self.noverlap >= self.nperseg:
            raise ValueError(
                f"noverlap must be in [0, nperseg); got {self.noverlap}."
            )
        self.window = window
        self.gamma = gamma

    # ------------------------------------------------------------------
    # Forward transform
    # ------------------------------------------------------------------

    def transform(self, sig: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the synchrosqueezed spectrogram via the Auger-Flandrin
        reassignment operator:

            omega_hat(t, omega) = omega - Im( STFT_{dh}(t, omega) / STFT_h(t, omega) ) / (2*pi)

        where dh = dh/dt is the time-derivative of the analysis window.

        Returns
        -------
        t : np.ndarray, length T (number of frames)
        f : np.ndarray, length F (number of frequency bins, one-sided)
        Tx : np.ndarray, shape (F, T)
        """
        sig = validate_signal_1d(sig, name='sig')

        from scipy.signal import get_window
        nperseg = min(self.nperseg, len(sig))
        min_overlap = max(self.noverlap, nperseg - nperseg // 8)
        noverlap = min(min_overlap, nperseg - 1)
        hop = nperseg - noverlap

        N = len(sig)
        # Number of frames
        n_frames = max(1, 1 + (N - nperseg) // hop)
        # One-sided frequency bins
        n_freq = nperseg // 2 + 1
        f = np.fft.rfftfreq(nperseg, d=1.0 / self.fs)
        t = np.arange(n_frames) * hop / self.fs + (nperseg / 2) / self.fs

        h = get_window(self.window, nperseg)
        # Time-derivative of the window. For a discrete window, use np.gradient
        # then multiply by fs to convert dx/d(index) -> dx/dt.
        dh = np.gradient(h) * self.fs

        Zh = np.zeros((n_freq, n_frames), dtype=complex)
        Zdh = np.zeros((n_freq, n_frames), dtype=complex)

        for k in range(n_frames):
            start = k * hop
            seg = sig[start : start + nperseg]
            if len(seg) < nperseg:
                seg = np.pad(seg, (0, nperseg - len(seg)))
            Zh[:, k] = np.fft.rfft(seg * h)
            Zdh[:, k] = np.fft.rfft(seg * dh)

        magnitude = np.abs(Zh)
        df = f[1] - f[0] if f.size > 1 else 1.0

        # Adaptive threshold
        gamma_eff = max(self.gamma, 1e-2 * float(np.max(magnitude)))

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = Zdh / Zh

        freq_grid = f[:, None]
        omega_hat = freq_grid - np.imag(ratio) / (2.0 * np.pi)

        # Reassignment
        Tx = np.zeros_like(magnitude, dtype=float)
        mask = (magnitude >= gamma_eff) & np.isfinite(omega_hat)
        valid = mask & (omega_hat >= f[0]) & (omega_hat <= f[-1])

        # Vectorised accumulation via np.add.at
        _ii, jj = np.where(valid)
        target = np.round((omega_hat[valid] - f[0]) / df).astype(int)
        target = np.clip(target, 0, len(f) - 1)
        np.add.at(Tx, (target, jj), magnitude[valid])

        return t, f, Tx

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(self, sig: np.ndarray) -> dict:
        """
        Quick feature summary from the synchrosqueezed spectrogram.
        """
        _t, f, Tx = self.transform(sig)

        marginal = np.sum(Tx, axis=1)
        total = float(np.sum(marginal))
        if total < 1e-12:
            return {
                'sst_peak_freq': 0.0,
                'sst_centroid': 0.0,
                'sst_bandwidth': 0.0,
                'sst_entropy': 0.0,
                'sst_concentration': 0.0,
            }
        p = marginal / total
        centroid = float(np.sum(f * p))
        return {
            'sst_peak_freq': float(f[int(np.argmax(marginal))]),
            'sst_centroid': centroid,
            'sst_bandwidth': float(np.sqrt(np.sum(((f - centroid) ** 2) * p))),
            'sst_entropy': float(-np.sum(p * np.log2(p + 1e-12))),
            # Concentration: how peaked the synchrosqueezed map is in time
            # at the peak frequency. Higher = more tonal.
            'sst_concentration': float(np.max(Tx) / (np.mean(Tx) + 1e-12)),
        }
