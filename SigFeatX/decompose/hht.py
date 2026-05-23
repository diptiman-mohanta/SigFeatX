"""
HHT — Hilbert-Huang Transform
==============================
Reference: Huang et al. (1998), "The empirical mode decomposition and the
           Hilbert spectrum for nonlinear and non-stationary time series
           analysis", Proc. Royal Society London A 454:903-995.

HHT = EMD (or CEEMDAN) decomposition followed by per-IMF Hilbert analysis,
giving instantaneous amplitude a_k(t) and instantaneous frequency f_k(t)
for each mode. The collection {a_k(t), f_k(t)} forms the Hilbert spectrum,
and integrating over time gives the marginal Hilbert spectrum h(f).

This module:
  - Computes a_k(t), phi_k(t), f_k(t) for each IMF using the analytic signal.
  - Builds the Hilbert spectrum on a user-supplied time-frequency grid.
  - Computes the marginal Hilbert spectrum (frequency-resolved energy).

For feature extraction, the most useful outputs are:
  - Mean instantaneous frequency per IMF
  - Bandwidth (std of instantaneous frequency) per IMF
  - Energy-weighted mean frequency per IMF
  - Marginal-spectrum peak and centroid
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import hilbert

from .._validation import validate_sampling_rate, validate_signal_1d
from .emd import EMD


class HHT:
    """
    Hilbert-Huang Transform.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    decomposer : object or None
        Object with ``.decompose(sig) -> List[np.ndarray]``. Defaults to EMD.
        Can be CEEMDAN or any compatible mode-extractor.
    """

    def __init__(self, fs: float = 1.0, decomposer=None):
        self.fs = validate_sampling_rate(fs)
        self.decomposer = decomposer if decomposer is not None else EMD()

    # ------------------------------------------------------------------
    # Per-mode Hilbert analysis
    # ------------------------------------------------------------------

    @staticmethod
    def instantaneous_attributes(
        imf: np.ndarray,
        fs: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute (instantaneous amplitude, instantaneous phase, instantaneous
        frequency) for a single mode.

        Phase is unwrapped. Frequency is the discrete derivative of phase
        divided by 2*pi, multiplied by fs. The frequency array has length
        ``len(imf) - 1`` — pad if you need same-length output.
        """
        imf = validate_signal_1d(imf, name='imf')
        fs = validate_sampling_rate(fs)

        analytic = hilbert(imf)
        amp = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))
        inst_freq = np.diff(phase) / (2.0 * np.pi) * fs
        return amp, phase, inst_freq

    # ------------------------------------------------------------------
    # Full Hilbert spectrum
    # ------------------------------------------------------------------

    def hilbert_spectrum(
        self,
        imfs: List[np.ndarray],
        n_freq_bins: int = 128,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the Hilbert spectrum H(t, f) by aggregating IMF energy onto
        a time-frequency grid.

        Parameters
        ----------
        imfs : list of 1D arrays of equal length N.
        n_freq_bins : int
            Number of frequency bins.
        f_min, f_max : float
            Frequency range. f_max defaults to Nyquist (fs / 2).

        Returns
        -------
        t : np.ndarray, length N
        f : np.ndarray, length n_freq_bins
        H : np.ndarray, shape (n_freq_bins, N)
        """
        if not imfs:
            raise ValueError("imfs must be a non-empty list.")
        N = len(imfs[0])
        if any(len(m) != N for m in imfs):
            raise ValueError("All IMFs must have the same length.")
        if f_max is None:
            f_max = self.fs / 2.0
        if f_max <= f_min:
            raise ValueError(f"f_max ({f_max}) must be > f_min ({f_min}).")

        t = np.arange(N) / self.fs
        f = np.linspace(f_min, f_max, n_freq_bins)
        H = np.zeros((n_freq_bins, N), dtype=float)

        df = f[1] - f[0]

        for imf in imfs:
            amp, _phase, inst_freq = self.instantaneous_attributes(imf, self.fs)
            # inst_freq has length N-1; map to time index 1..N-1 with the
            # corresponding amplitude. Drop the first sample for alignment.
            for i, freq_val in enumerate(inst_freq):
                if not np.isfinite(freq_val):
                    continue
                if freq_val < f_min or freq_val >= f_max:
                    continue
                bin_idx = int((freq_val - f_min) / df)
                if 0 <= bin_idx < n_freq_bins:
                    H[bin_idx, i + 1] += amp[i + 1] ** 2

        return t, f, H

    # ------------------------------------------------------------------
    # Marginal Hilbert spectrum
    # ------------------------------------------------------------------

    def marginal_spectrum(
        self,
        imfs: List[np.ndarray],
        n_freq_bins: int = 128,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the marginal Hilbert spectrum h(f) by integrating H(t, f)
        across time.

        Returns
        -------
        f : np.ndarray, length n_freq_bins
        h : np.ndarray, length n_freq_bins
        """
        _t, f, H = self.hilbert_spectrum(imfs, n_freq_bins, f_min, f_max)
        h = np.sum(H, axis=1)
        return f, h

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(self, sig: np.ndarray) -> Dict[str, float]:
        """
        End-to-end: decompose ``sig`` with the configured decomposer, run
        Hilbert analysis on each IMF, return a flat feature dict.

        Features per IMF k:
          imf_k_mean_inst_freq, imf_k_std_inst_freq
          imf_k_mean_inst_amp,  imf_k_std_inst_amp
          imf_k_weighted_mean_freq  (amp-weighted)

        Marginal-spectrum features:
          marginal_peak_freq, marginal_centroid, marginal_bandwidth,
          marginal_entropy
        """
        sig = validate_signal_1d(sig, name='sig')
        imfs = self.decomposer.decompose(sig)
        imfs = [np.asarray(m, dtype=float) for m in imfs if len(m) == len(sig)]
        if not imfs:
            return {}

        features: Dict[str, float] = {}

        for k, imf in enumerate(imfs):
            amp, _phase, ifreq = self.instantaneous_attributes(imf, self.fs)
            valid = np.isfinite(ifreq) & (ifreq >= 0) & (ifreq <= self.fs / 2.0)
            ifreq_v = ifreq[valid]
            amp_v = amp[1:][valid]               # align amp to ifreq length

            features[f'hht_imf_{k}_mean_inst_freq'] = (
                float(np.mean(ifreq_v)) if ifreq_v.size else 0.0
            )
            features[f'hht_imf_{k}_std_inst_freq'] = (
                float(np.std(ifreq_v)) if ifreq_v.size else 0.0
            )
            features[f'hht_imf_{k}_mean_inst_amp'] = float(np.mean(amp))
            features[f'hht_imf_{k}_std_inst_amp'] = float(np.std(amp))

            if ifreq_v.size and amp_v.size:
                w = amp_v ** 2
                wsum = float(np.sum(w))
                features[f'hht_imf_{k}_weighted_mean_freq'] = (
                    float(np.sum(w * ifreq_v) / (wsum + 1e-12))
                )
            else:
                features[f'hht_imf_{k}_weighted_mean_freq'] = 0.0

        # Marginal spectrum
        f, h = self.marginal_spectrum(imfs)
        if np.sum(h) > 1e-12:
            h_norm = h / np.sum(h)
            features['hht_marginal_peak_freq'] = float(f[int(np.argmax(h))])
            centroid = float(np.sum(f * h_norm))
            features['hht_marginal_centroid'] = centroid
            features['hht_marginal_bandwidth'] = float(
                np.sqrt(np.sum(((f - centroid) ** 2) * h_norm))
            )
            features['hht_marginal_entropy'] = float(
                -np.sum(h_norm * np.log2(h_norm + 1e-12))
            )
        else:
            features['hht_marginal_peak_freq'] = 0.0
            features['hht_marginal_centroid'] = 0.0
            features['hht_marginal_bandwidth'] = 0.0
            features['hht_marginal_entropy'] = 0.0

        return features
