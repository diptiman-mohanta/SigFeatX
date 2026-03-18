"""
SigFeatX - preprocess.py 
"""

import warnings
import numpy as np
from scipy import signal as scipy_signal
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from typing import Optional

from ._validation import validate_sampling_rate, validate_signal_1d


class SignalPreprocessor:
    """Preprocessing utilities for 1D signals."""

    # ------------------------------------------------------------------
    # Denoising
    # ------------------------------------------------------------------

    def denoise(self, sig: np.ndarray, method: str = 'wavelet',
                **kwargs) -> np.ndarray:
        """
        Denoise a signal.

        Parameters
        ----------
        sig    : input signal
        method : 'wavelet'   — wavelet soft-thresholding (original)
                 'median'    — median filter (original)
                 'lowpass'   — Butterworth low-pass (original)
                 'bandpass'  — zero-phase bandpass  [NEW]
                 'notch'     — zero-phase notch      [NEW]
        kwargs : passed to the specific filter; see individual methods.
        """
        sig = validate_signal_1d(sig, name='sig')
        if method == 'wavelet':
            return self._denoise_wavelet(sig, **kwargs)
        elif method == 'median':
            return self._denoise_median(sig, **kwargs)
        elif method == 'lowpass':
            return self._denoise_lowpass(sig, **kwargs)
        elif method == 'bandpass':
            return self.bandpass(sig, **kwargs)
        elif method == 'notch':
            return self.notch(sig, **kwargs)
        else:
            raise ValueError(
                f"Unknown denoise method '{method}'. "
                "Choose from: 'wavelet', 'median', 'lowpass', 'bandpass', 'notch'."
            )

    def _denoise_wavelet(self, sig: np.ndarray, wavelet: str = 'db4',
                         level: int = 1) -> np.ndarray:
        """Wavelet soft-thresholding denoising (original, unchanged)."""
        sig = validate_signal_1d(sig, name='sig')
        try:
            import pywt
        except ImportError:
            warnings.warn("pywt not installed; returning signal unchanged.", RuntimeWarning)
            return sig.copy()

        coeffs = pywt.wavedec(sig, wavelet, level=level)
        sigma  = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(sig)))
        denoised_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
        return pywt.waverec(denoised_coeffs, wavelet)[:len(sig)]

    def _denoise_median(self, sig: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Median filter denoising (original, unchanged)."""
        sig = validate_signal_1d(sig, name='sig')
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer; got {kernel_size}."
            )
        return scipy_signal.medfilt(sig, kernel_size=kernel_size)

    def _denoise_lowpass(self, sig: np.ndarray, cutoff: float = 0.1,
                         order: int = 4, fs: float = 1.0) -> np.ndarray:
        """Butterworth low-pass filter (original, unchanged)."""
        sig = validate_signal_1d(sig, name='sig')
        fs = validate_sampling_rate(fs)
        if cutoff <= 0:
            raise ValueError(f"cutoff must be > 0; got {cutoff}.")
        if order < 1:
            raise ValueError(f"order must be >= 1; got {order}.")
        nyq = fs / 2
        norm_cutoff = cutoff / nyq
        norm_cutoff = np.clip(norm_cutoff, 1e-4, 0.9999)
        b, a = scipy_signal.butter(order, norm_cutoff, btype='low')
        _validate_filtfilt_length(sig, b, a)
        return scipy_signal.filtfilt(b, a, sig)

    # ------------------------------------------------------------------
    # NEW: Bandpass filter
    # ------------------------------------------------------------------

    def bandpass(self, sig: np.ndarray, low_hz: float, high_hz: float,
                 fs: float = 1.0, order: int = 4) -> np.ndarray:
        """
        Zero-phase Butterworth bandpass filter.

        Retains only the frequency content in [low_hz, high_hz].
        Uses filtfilt for zero phase distortion.

        Parameters
        ----------
        sig     : input signal
        low_hz  : lower cutoff frequency in Hz
        high_hz : upper cutoff frequency in Hz
        fs      : sampling frequency in Hz (default 1.0 for normalised)
        order   : filter order (default 4); higher = sharper rolloff

        Returns
        -------
        Bandpass-filtered signal, same length as input.

        Examples
        --------
        # Keep only 8-30 Hz (beta/alpha EEG)
        filtered = preprocessor.bandpass(sig, low_hz=8, high_hz=30, fs=250)

        # As a denoise method
        filtered = preprocessor.denoise(sig, method='bandpass',
                                        low_hz=8, high_hz=30, fs=250)
        """
        sig = validate_signal_1d(sig, name='sig')
        fs = validate_sampling_rate(fs)
        if order < 1:
            raise ValueError(f"order must be >= 1; got {order}.")
        nyq = fs / 2.0

        if low_hz <= 0:
            raise ValueError(f"low_hz must be > 0; got {low_hz}")
        if high_hz >= nyq:
            raise ValueError(
                f"high_hz ({high_hz} Hz) must be < Nyquist frequency ({nyq} Hz)."
            )
        if low_hz >= high_hz:
            raise ValueError(
                f"low_hz ({low_hz}) must be less than high_hz ({high_hz})."
            )

        low  = low_hz  / nyq
        high = high_hz / nyq
        low  = np.clip(low,  1e-4, 0.9999)
        high = np.clip(high, 1e-4, 0.9999)

        b, a = scipy_signal.butter(order, [low, high], btype='band')
        _validate_filtfilt_length(sig, b, a)
        return scipy_signal.filtfilt(b, a, sig)

    # ------------------------------------------------------------------
    # NEW: Notch filter
    # ------------------------------------------------------------------

    def notch(self, sig: np.ndarray, freq_hz: float, fs: float = 1.0,
              quality_factor: float = 30.0) -> np.ndarray:
        """
        Zero-phase IIR notch filter.

        Removes a single frequency component (e.g. 50 or 60 Hz power-line
        interference) while leaving the rest of the spectrum intact.

        Parameters
        ----------
        sig            : input signal
        freq_hz        : frequency to remove in Hz
        fs             : sampling frequency in Hz (default 1.0 for normalised)
        quality_factor : Q factor. Higher Q = narrower notch.
                         Typical values: 10–50. Default 30.

        Returns
        -------
        Notch-filtered signal, same length as input.

        Examples
        --------
        # Remove 50 Hz power-line noise from EEG sampled at 256 Hz
        cleaned = preprocessor.notch(sig, freq_hz=50, fs=256)

        # Remove both 50 Hz and 100 Hz harmonics
        cleaned = preprocessor.notch(sig, freq_hz=50,  fs=256)
        cleaned = preprocessor.notch(cleaned, freq_hz=100, fs=256)

        # As a denoise method
        cleaned = preprocessor.denoise(sig, method='notch', freq_hz=50, fs=256)
        """
        sig = validate_signal_1d(sig, name='sig')
        fs = validate_sampling_rate(fs)
        if quality_factor <= 0:
            raise ValueError(
                f"quality_factor must be > 0; got {quality_factor}."
            )
        nyq = fs / 2.0

        if freq_hz <= 0 or freq_hz >= nyq:
            raise ValueError(
                f"freq_hz ({freq_hz} Hz) must be in (0, {nyq}) Hz (Nyquist)."
            )

        norm_freq = freq_hz / nyq
        b, a      = scipy_signal.iirnotch(norm_freq, Q=quality_factor)
        _validate_filtfilt_length(sig, b, a)
        return scipy_signal.filtfilt(b, a, sig)

    # ------------------------------------------------------------------
    # Detrending (extended with ALS)
    # ------------------------------------------------------------------

    def detrend(self, sig: np.ndarray, method: str = 'linear',
                **kwargs) -> np.ndarray:
        """
        Remove trend from signal.

        Parameters
        ----------
        sig    : input signal
        method : 'linear'   — subtract least-squares linear fit (original)
                 'constant' — subtract mean (original)
                 'als'      — Asymmetric Least Squares baseline (NEW)
        kwargs : forwarded to the selected detrend method. For `als`, this
                 includes `lam`, `p`, and `n_iter`.
        """
        sig = validate_signal_1d(sig, name='sig')
        if method == 'linear':
            return scipy_signal.detrend(sig, type='linear')
        elif method == 'constant':
            return scipy_signal.detrend(sig, type='constant')
        elif method == 'als':
            baseline = self.als_baseline(sig, **kwargs)
            return sig - baseline
        else:
            raise ValueError(
                f"Unknown detrend method '{method}'. "
                "Choose from: 'linear', 'constant', 'als'."
            )

    # ------------------------------------------------------------------
    # NEW: ALS baseline estimation
    # ------------------------------------------------------------------

    def als_baseline(self, sig: np.ndarray, lam: float = 1e4,
                     p: float = 0.01, n_iter: int = 10) -> np.ndarray:
        """
        Asymmetric Least Squares (ALS) baseline estimation.

        Reference: Eilers & Boelens (2005), "Baseline Correction with
        Asymmetric Least Squares Smoothing."

        Fits a smooth baseline to the lower envelope of the signal by
        iteratively reweighting a penalised least squares problem:
          - Points above the current baseline get weight p  (small)
          - Points below the current baseline get weight 1-p (large)
        This asymmetry means the baseline is attracted to the signal
        floor rather than its mean, making it suitable for:
          - EEG baseline wander
          - Spectroscopic baselines
          - Vibration signals with slowly varying DC offset
          - Any signal where linear detrend leaves a curved residual

        Parameters
        ----------
        sig    : input signal of length N
        lam    : smoothness parameter. Larger lam = smoother baseline.
                 Typical range: 1e2 (rough) to 1e7 (very smooth).
                 Default 1e4 suits most biomedical signals.
        p      : asymmetry parameter in (0, 1). Smaller p = baseline
                 closer to signal floor. Default 0.01 (strongly asymmetric).
        n_iter : number of IRLS iterations. 10 is typically sufficient.

        Returns
        -------
        baseline : np.ndarray of shape (N,)
            The estimated baseline. To remove it: sig - baseline.

        Examples
        --------
        baseline = preprocessor.als_baseline(sig, lam=1e5, p=0.005)
        corrected = sig - baseline

        # Or via detrend:
        corrected = preprocessor.detrend(sig, method='als')
        """
        sig = validate_signal_1d(sig, name='sig', min_length=3)
        if lam <= 0:
            raise ValueError(f"lam must be > 0; got {lam}.")
        if not 0 < p < 1:
            raise ValueError(f"p must be in (0, 1); got {p}.")
        if n_iter < 1:
            raise ValueError(f"n_iter must be >= 1; got {n_iter}.")
        N   = len(sig)

        # Second-order difference matrix D of size (N-2, N)
        # Used to penalise curvature of the baseline
        D = _second_diff_matrix(N)
        DtD = (D.T).dot(D)           # (N, N) sparse matrix

        # Initialise weights uniformly
        w = np.ones(N)

        baseline = sig.copy()
        for _ in range(n_iter):
            W   = diags(w, 0, shape=(N, N), format='csr')
            Z   = W + lam * DtD
            baseline = spsolve(Z, w * sig)

            # Asymmetric reweighting:
            #   above baseline → weight p  (penalise lightly)
            #   below baseline → weight 1-p (penalise heavily)
            w = np.where(sig > baseline, p, 1.0 - p)

        return baseline

    # ------------------------------------------------------------------
    # Normalisation (original, unchanged)
    # ------------------------------------------------------------------

    def normalize(self, sig: np.ndarray, method: str = 'zscore') -> np.ndarray:
        """
        Normalise signal amplitude.

        Parameters
        ----------
        method : 'zscore' — zero mean, unit std
                 'minmax' — scale to [0, 1]
                 'robust' — subtract median, divide by IQR
        """
        sig = validate_signal_1d(sig, name='sig')
        if method == 'zscore':
            mu, sigma = np.mean(sig), np.std(sig)
            return (sig - mu) / (sigma + 1e-10)
        elif method == 'minmax':
            lo, hi = np.min(sig), np.max(sig)
            return (sig - lo) / (hi - lo + 1e-10)
        elif method == 'robust':
            med = np.median(sig)
            iqr = np.percentile(sig, 75) - np.percentile(sig, 25)
            return (sig - med) / (iqr + 1e-10)
        else:
            raise ValueError(
                f"Unknown normalize method '{method}'. "
                "Choose from: 'zscore', 'minmax', 'robust'."
            )

    # ------------------------------------------------------------------
    # Resampling (original, unchanged)
    # ------------------------------------------------------------------

    def resample(self, sig: np.ndarray, original_fs: float,
                 target_fs: float, method: str = 'fourier') -> np.ndarray:
        """
        Resample signal to a different sampling frequency.

        Parameters
        ----------
        method : 'fourier' — scipy.signal.resample (anti-aliased, recommended)
                 'linear'  — linear interpolation (no anti-aliasing; not
                             recommended for downsampling)
        """
        sig = validate_signal_1d(sig, name='sig')
        original_fs = validate_sampling_rate(original_fs, name='original_fs')
        target_fs = validate_sampling_rate(target_fs, name='target_fs')
        target_len = int(len(sig) * target_fs / original_fs)

        if method == 'fourier':
            return scipy_signal.resample(sig, target_len)
        elif method == 'linear':
            original_t = np.linspace(0, 1, len(sig))
            target_t   = np.linspace(0, 1, target_len)
            return np.interp(target_t, original_t, sig)
        else:
            raise ValueError(
                f"Unknown resample method '{method}'. Choose 'fourier' or 'linear'."
            )


# ------------------------------------------------------------------
# Module-level helper: sparse second-difference matrix
# ------------------------------------------------------------------

def _second_diff_matrix(n: int):
    """
    Sparse second-order difference matrix of shape (n-2, n).

    D[i] = [0,...,0, 1, -2, 1, 0,...,0] with the non-zeros at columns i, i+1, i+2.
    Used by ALS baseline to penalise curvature.
    """
    e    = np.ones(n)
    D    = diags([e[:-2], -2 * e[:-1], e], offsets=[0, 1, 2],
                 shape=(n - 2, n), format='csr')
    return D


def _validate_filtfilt_length(sig: np.ndarray, b: np.ndarray, a: np.ndarray):
    """Raise a clear error when a signal is too short for zero-phase filtering."""
    min_len = 3 * max(len(a), len(b))
    if len(sig) <= min_len:
        raise ValueError(
            f"sig must contain more than {min_len} samples for zero-phase filtering; "
            f"got {len(sig)}."
        )
