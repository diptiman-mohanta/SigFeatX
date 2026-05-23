"""
MODWT — Maximal Overlap Discrete Wavelet Transform
===================================================
Reference: Percival & Walden (2000), "Wavelet Methods for Time Series
           Analysis", Cambridge University Press, Chapter 5.

Advantages over standard DWT:
  - Shift-invariant: shifting input shifts coefficients by the same amount,
    not by a power of 2. Critical for feature extraction over windowed signals.
  - Works on signals of any length, no power-of-2 requirement.
  - Each level has the same length as the input — easy to align with the signal.
  - Energy is preserved exactly: sum of squared coefficients across all
    levels and the smooth equals sum of squared input samples.

Implementation
--------------
We use the pyramid algorithm from Percival & Walden, computing rescaled
filters at each level. Coefficients are produced as a list:

    [W_1, W_2, ..., W_J, V_J]

where W_j are the wavelet (detail) coefficients at level j and V_J is the
smooth (approximation) at the deepest level. All arrays have length N.
"""

from typing import List

import numpy as np
import pywt

from .._validation import validate_signal_1d


class MODWT:
    """
    Maximal Overlap Discrete Wavelet Transform.

    Parameters
    ----------
    wavelet : str
        Wavelet name (any PyWavelets discrete wavelet, e.g. 'db4', 'sym5').
    level : int or None
        Decomposition depth. None auto-picks ``floor(log2(N)) - 1`` capped
        at the wavelet's max level.

    Notes
    -----
    MODWT requires the *rescaled* filter coefficients h_j[l] = h[l] / sqrt(2)
    at each level j. This implementation uses pywt's filter coefficients
    directly and applies the rescaling explicitly.
    """

    def __init__(self, wavelet: str = 'db4', level: int = None):
        self.wavelet = wavelet
        self.level = level

    # ------------------------------------------------------------------
    # Forward transform
    # ------------------------------------------------------------------

    def decompose(self, sig: np.ndarray) -> List[np.ndarray]:
        """
        Run the forward MODWT.

        Returns
        -------
        coeffs : list of length ``level + 1``
            ``[W_1, W_2, ..., W_J, V_J]``. Every array has the same length
            as the input.
        """
        sig = validate_signal_1d(sig, name='sig')
        N = len(sig)

        # Pick decomposition level
        if self.level is None:
            max_lev = int(np.floor(np.log2(N))) - 1
            level = max(1, min(max_lev, pywt.dwt_max_level(N, self.wavelet)))
        else:
            if self.level < 1:
                raise ValueError(f"level must be >= 1; got {self.level}.")
            level = self.level

        # Get filter coefficients, rescaled for MODWT
        w = pywt.Wavelet(self.wavelet)
        # MODWT rescaling: divide by sqrt(2) so reconstruction stays unit-energy
        h = np.asarray(w.dec_hi) / np.sqrt(2.0)   # wavelet (high-pass)
        g = np.asarray(w.dec_lo) / np.sqrt(2.0)   # scaling (low-pass)

        coeffs: List[np.ndarray] = []
        v_prev = sig.astype(float).copy()         # V_0 = signal

        for j in range(1, level + 1):
            w_j, v_j = self._modwt_step(v_prev, h, g, j)
            coeffs.append(w_j)
            v_prev = v_j

        coeffs.append(v_prev)                     # V_J at the end
        return coeffs

    # ------------------------------------------------------------------
    # Inverse transform
    # ------------------------------------------------------------------

    def reconstruct(self, coeffs: List[np.ndarray]) -> np.ndarray:
        """
        Run the inverse MODWT.

        Parameters
        ----------
        coeffs : list as returned by ``decompose``.

        Returns
        -------
        np.ndarray of length matching the original signal.
        """
        if len(coeffs) < 2:
            raise ValueError("coeffs must contain at least one detail and one smooth.")

        # Last entry is V_J, everything before is W_1..W_J
        wavelets = coeffs[:-1]
        v = coeffs[-1].astype(float).copy()
        level = len(wavelets)

        w = pywt.Wavelet(self.wavelet)
        h = np.asarray(w.dec_hi) / np.sqrt(2.0)
        g = np.asarray(w.dec_lo) / np.sqrt(2.0)

        for j in range(level, 0, -1):
            w_j = wavelets[j - 1]
            v = self._imodwt_step(w_j, v, h, g, j)

        return v

    # ------------------------------------------------------------------
    # Single-level forward / inverse steps
    # ------------------------------------------------------------------

    @staticmethod
    def _modwt_step(v_prev: np.ndarray, h: np.ndarray, g: np.ndarray,
                    j: int) -> tuple:
        """
        One pyramid step: V_{j-1} -> (W_j, V_j).

        Uses circular convolution with the level-j upsampled filters.
        """
        N = len(v_prev)
        L = len(h)

        # At level j the filter is upsampled by inserting 2^(j-1) - 1 zeros
        # between coefficients. We implement this as direct indexing.
        scale = 2 ** (j - 1)

        w_j = np.zeros(N, dtype=float)
        v_j = np.zeros(N, dtype=float)

        for t in range(N):
            acc_w = 0.0
            acc_v = 0.0
            for l in range(L):
                idx = (t - scale * l) % N      # circular shift
                acc_w += h[l] * v_prev[idx]
                acc_v += g[l] * v_prev[idx]
            w_j[t] = acc_w
            v_j[t] = acc_v

        return w_j, v_j


    @staticmethod
    def _imodwt_step(w_j: np.ndarray, v_j: np.ndarray, h: np.ndarray,
                     g: np.ndarray, j: int) -> np.ndarray:
        """
        Inverse pyramid step: (W_j, V_j) -> V_{j-1}.
        """
        N = len(v_j)
        L = len(h)
        scale = 2 ** (j - 1)

        v_prev = np.zeros(N, dtype=float)
        for t in range(N):
            acc = 0.0
            for l in range(L):
                idx = (t + scale * l) % N      # forward shift (transpose)
                acc += h[l] * w_j[idx] + g[l] * v_j[idx]
            v_prev[t] = acc
        return v_prev
