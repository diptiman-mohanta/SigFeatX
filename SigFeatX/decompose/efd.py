"""
EFD — Empirical Fourier Decomposition
Reference: Xu et al. (2021), Mechanical Systems and Signal Processing 161:107952

Bugs fixed vs original SigFeatX version:

BUG 1 — Boundaries at spectrum PEAKS instead of MINIMA  [CRITICAL, already fixed]
  Paper: boundary between mode k and k+1 is placed at the local minimum between
  the k-th and (k+1)-th peak of the Fourier magnitude spectrum.
  Original: used peak positions directly as boundaries.
  Fix: find peaks first, then find the minimum between each consecutive pair.

BUG 2 — Non-symmetric (non-zero-phase) filter  [MODERATE, already fixed]
  Paper: EFD applies an ideal zero-phase brick-wall bandpass filter by masking
  BOTH positive AND negative frequency bands in the FFT, then IFFT.
  Original: only masked positive frequencies.
  Fix: apply mask symmetrically to ±freq bands.

BUG 3 — Float while-loop boundary drift  [MINOR, FIXED IN THIS VERSION]
  PREVIOUS:
      while len(boundaries) < self.n_modes + 1:
          last = boundaries[-1]
          top  = float(np.max(freqs))
          gap  = (top - last) / (self.n_modes + 1 - len(boundaries) + 1)
          boundaries.append(last + gap)

  Problem: repeated floating-point additions accumulate rounding error.
  Combined with the set() deduplication step, this can produce duplicate
  entries and leave the final boundary list shorter than n_modes+1, causing
  some modes to cover zero bandwidth and return all-zeros silently.

  FIX (this version):
      Replace the while-loop with np.linspace which divides the remaining
      range in one step with no accumulation error.

      if len(boundaries) < self.n_modes + 1:
          last  = boundaries[-1]
          top   = float(np.max(freqs))
          n_extra = self.n_modes + 1 - len(boundaries)
          extra = np.linspace(last, top, n_extra + 1)[1:]  # exclude 'last'
          boundaries.extend(extra.tolist())
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.fft import fft, ifft, fftfreq

from SigFeatX._validation import validate_signal_1d


class EFD:
    """
    Empirical Fourier Decomposition (Xu et al. 2021).

    Segments the Fourier magnitude spectrum by finding local minima between
    dominant peaks, then applies an ideal zero-phase bandpass filter per segment.

    Parameters
    ----------
    n_modes          : number of modes (frequency bands) to extract.
    peak_prominence  : minimum prominence for a peak to be considered.
                       Expressed as a fraction of the maximum spectral magnitude.
                       Default 0.1 (peaks at least 10% of max).
    """

    def __init__(self, n_modes: int = 5, peak_prominence: float = 0.1):
        if n_modes < 1:
            raise ValueError(f"n_modes must be >= 1; got {n_modes}.")
        if peak_prominence < 0:
            raise ValueError(
                f"peak_prominence must be >= 0; got {peak_prominence}."
            )
        self.n_modes         = n_modes
        self.peak_prominence = peak_prominence

    def decompose(self, sig: np.ndarray) -> np.ndarray:
        """
        Decompose signal using EFD.

        Returns
        -------
        np.ndarray of shape (n_modes, N)
        """
        sig = validate_signal_1d(sig, name='sig')
        N   = len(sig)

        fft_sig  = fft(sig)
        freqs    = fftfreq(N)

        # Work on the one-sided (positive frequency) magnitude spectrum
        pos_mask = freqs >= 0
        pos_freq = freqs[pos_mask]
        pos_mag  = np.abs(fft_sig[pos_mask])

        # ── Step 1: Find peaks in positive magnitude spectrum ──────────────
        min_prominence = np.max(pos_mag) * self.peak_prominence
        peaks, _       = scipy_signal.find_peaks(
            pos_mag, height=min_prominence, prominence=min_prominence
        )

        # ── Step 2: Find local minima BETWEEN consecutive peaks ────────────
        boundaries = self._find_boundaries(pos_mag, pos_freq, peaks)

        # ── Step 3: Apply zero-phase ideal bandpass per segment ────────────
        modes = np.zeros((self.n_modes, N))

        for k in range(self.n_modes):
            if k < len(boundaries) - 1:
                f_low  = boundaries[k]
                f_high = boundaries[k + 1]

                # Zero-phase ideal filter: mask BOTH +freq and -freq bands
                mask = np.zeros(N, dtype=complex)
                for i, f in enumerate(freqs):
                    if f_low <= abs(f) < f_high:
                        mask[i] = fft_sig[i]

                modes[k] = np.real(ifft(mask))

        return modes

    def _find_boundaries(self, magnitude: np.ndarray, freqs: np.ndarray,
                         peaks: np.ndarray) -> np.ndarray:
        """
        Determine n_modes+1 frequency boundaries.

        Boundaries are placed at:
          - 0 (start)
          - the local minimum between each consecutive pair of peaks
          - max(freqs) (end)

        If fewer peaks than needed, fills the remaining range with
        np.linspace (FIX: replaces the while-loop float accumulation).
        """
        boundaries = [0.0]

        # For each consecutive pair of peaks, find the minimum between them
        for i in range(min(len(peaks) - 1, self.n_modes - 1)):
            left  = peaks[i]
            right = peaks[i + 1]
            valley_section = magnitude[left:right + 1]
            valley_idx     = left + int(np.argmin(valley_section))
            boundaries.append(float(freqs[valley_idx]))

        # FIX: replace while-loop with linspace to avoid float accumulation
        # that caused duplicate boundaries and silent zero-bandwidth modes.
        if len(boundaries) < self.n_modes + 1:
            last    = boundaries[-1]
            top     = float(np.max(freqs))
            n_extra = self.n_modes + 1 - len(boundaries)
            extra   = np.linspace(last, top, n_extra + 1)[1:]   # exclude 'last'
            boundaries.extend(extra.tolist())

        boundaries.append(float(np.max(freqs)))

        # Deduplicate, sort, trim/pad to exactly n_modes+1 values
        boundaries = sorted(set(round(b, 8) for b in boundaries))
        if len(boundaries) > self.n_modes + 1:
            boundaries = boundaries[:self.n_modes + 1]

        # Safety pad (should never be needed after linspace fix)
        while len(boundaries) < self.n_modes + 1:
            boundaries.append(float(np.max(freqs)))

        return np.array(boundaries)

    def reconstruct(self, modes: np.ndarray) -> np.ndarray:
        return np.sum(modes, axis=0)