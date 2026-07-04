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

Each pyramid step is a circular convolution/correlation between the
signal and the level-j upsampled filter, computed via FFT (O(N log N))
rather than the O(N * L) direct double loop -- exact to floating-point
precision (validated against the direct-loop form across wavelets,
lengths, and levels; max discrepancy ~1e-15).

Only orthogonal wavelets are supported (db*, sym*, coif*, haar). MODWT's
pyramid algorithm reuses the decomposition filters for reconstruction,
which is only mathematically valid when the reconstruction filters are
the time-reversal of the decomposition filters -- true for orthogonal
wavelets but not for biorthogonal ones (bior*/rbio*), where a different
filter pair is needed. Passing a biorthogonal wavelet would silently
produce an incorrect reconstruction, so it is rejected at construction
time instead.
"""

import numpy as np
import pywt

from .._validation import validate_signal_1d


class MODWT:
    """
    Maximal Overlap Discrete Wavelet Transform.

    Parameters
    ----------
    wavelet : str
        Orthogonal PyWavelets discrete wavelet name (e.g. 'db4', 'sym5',
        'coif3', 'haar'). Biorthogonal wavelets ('bior*', 'rbio*') are
        rejected -- see module docstring.
    level : int or None
        Decomposition depth. None auto-picks ``floor(log2(N)) - 1`` capped
        at the wavelet's max level.

    Notes
    -----
    MODWT requires the *rescaled* filter coefficients h_j[l] = h[l] / sqrt(2)
    at each level j. This implementation uses pywt's filter coefficients
    directly and applies the rescaling explicitly.
    """

    def __init__(self, wavelet: str = 'db4', level: int | None = None):
        if not pywt.Wavelet(wavelet).orthogonal:
            raise ValueError(
                f"MODWT requires an orthogonal wavelet; {wavelet!r} is "
                "biorthogonal (its decomposition and reconstruction filters "
                "differ), which this pyramid-algorithm implementation does "
                "not support and would silently produce an incorrect "
                "reconstruction. Use an orthogonal family instead, e.g. "
                "'db4', 'sym5', 'coif3', or 'haar'."
            )
        self.wavelet = wavelet
        self.level = level

    # ------------------------------------------------------------------
    # Forward transform
    # ------------------------------------------------------------------

    def decompose(self, sig: np.ndarray) -> list[np.ndarray]:
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

        coeffs: list[np.ndarray] = []
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

    def reconstruct(self, coeffs: list[np.ndarray]) -> np.ndarray:
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
    def _upsample_filter(taps: np.ndarray, scale: int, N: int) -> np.ndarray:
        """
        Build the length-N level-j filter: taps[l] placed at index
        (scale * l) % N, accumulating on collision (matches inserting
        ``scale - 1`` zeros between consecutive taps, wrapped circularly).
        """
        up = np.zeros(N, dtype=float)
        idx = (scale * np.arange(len(taps))) % N
        np.add.at(up, idx, taps)
        return up

    @staticmethod
    def _modwt_step(v_prev: np.ndarray, h: np.ndarray, g: np.ndarray,
                    j: int) -> tuple:
        """
        One pyramid step: V_{j-1} -> (W_j, V_j).

        w_j[t] = sum_l h[l] * v_prev[(t - scale*l) % N] is exactly the
        circular convolution of v_prev with the upsampled filter h_up
        (h_up[k] = h[l] at k = (scale*l) % N), computed here via FFT:
        conv(a, b) = ifft(fft(a) * fft(b)).
        """
        N = len(v_prev)
        scale = 2 ** (j - 1)
        h_up = MODWT._upsample_filter(h, scale, N)
        g_up = MODWT._upsample_filter(g, scale, N)

        V = np.fft.rfft(v_prev)
        w_j = np.fft.irfft(np.fft.rfft(h_up) * V, n=N)
        v_j = np.fft.irfft(np.fft.rfft(g_up) * V, n=N)
        return w_j, v_j

    @staticmethod
    def _imodwt_step(w_j: np.ndarray, v_j: np.ndarray, h: np.ndarray,
                     g: np.ndarray, j: int) -> np.ndarray:
        """
        Inverse pyramid step: (W_j, V_j) -> V_{j-1}.

        v_prev[t] = sum_l h[l]*w_j[(t+scale*l)%N] + g[l]*v_j[(t+scale*l)%N]
        is a circular *correlation* with the upsampled filters, computed
        via the correlation-FFT identity corr(a, b) = ifft(conj(fft(a)) *
        fft(b)).
        """
        N = len(v_j)
        scale = 2 ** (j - 1)
        h_up = MODWT._upsample_filter(h, scale, N)
        g_up = MODWT._upsample_filter(g, scale, N)

        Hf = np.fft.rfft(h_up)
        Gf = np.fft.rfft(g_up)
        Wf = np.fft.rfft(w_j)
        Vf = np.fft.rfft(v_j)
        return np.fft.irfft(np.conj(Hf) * Wf + np.conj(Gf) * Vf, n=N)
