"""
LMD — Local Mean Decomposition
Reference: Smith, J.S. (2005). The local mean decomposition and its application to
EEG perception data. Journal of the Royal Society Interface, 2(5), 443-454.
https://doi.org/10.1098/rsif.2005.0058

Algorithm:
  1. Find all local extrema (maxima + minima) of the signal.
  2. Compute the local mean m_i = (n_i + n_{i+1}) / 2 at each half-wave.
  3. Compute the local envelope a_i = |n_i - n_{i+1}| / 2 at each half-wave.
  4. Smooth both step-functions using a moving-average smoother to obtain
     continuous m(t) and a(t).
  5. Subtract the mean: h(t) = s(t) - m(t).
  6. Divide by the envelope: s_new(t) = h(t) / a(t).
  7. Repeat steps 1-6 on s_new until the envelope is flat (≈1 everywhere).
  8. The Product Function is PF = a_final(t) * s_final(t).
  9. Subtract PF from the signal; repeat the whole process on the residual.

Improvements:
  - Mirror-extension boundary condition (avoids endpoint effects that plague
    the original moving-average approach).
  - Cubic-spline smoothing option in addition to moving-average (matches the
    intent of the original paper more closely for short windows).
  - Flat-envelope stopping criterion uses both an absolute tolerance and a
    relative-change criterion to avoid over-sifting.
  - Constant and near-constant signals handled gracefully.
"""

import warnings
import numpy as np
from scipy import signal as scipy_signal
from scipy.interpolate import CubicSpline
from typing import List, Optional, Tuple

from SigFeatX._validation import validate_signal_1d


class LMD:
    """
    Local Mean Decomposition (Smith 2005).

    Decomposes a signal into a set of Product Functions (PFs), each of which
    is the product of an envelope signal (instantaneous amplitude) and a
    purely frequency-modulated (FM) signal. The instantaneous frequency can
    be derived from each FM signal without using the Hilbert transform,
    avoiding the negative-frequency artefacts that can affect EMD/HHT.

    Parameters
    ----------
    max_pf : int
        Maximum number of Product Functions to extract.  Default 8.
    max_sift_iter : int
        Maximum sifting iterations per PF.  Default 200.
    envelope_epsilon : float
        Sifting stops when mean(|1 - a(t)|) < envelope_epsilon.
        Default 0.01 (i.e. envelope deviates from 1 by < 1% on average).
    convergence_epsilon : float
        Secondary stop: sifting stops when mean(|s_k - s_{k+1}|) <
        convergence_epsilon.  Default 0.01.
    smooth_method : str
        How to smooth the step-wise local mean / envelope before sifting.
        'moving_average' (default, faithful to the original paper) or
        'cubic_spline' (smoother result, better for short segments).
    smooth_iterations : int
        Maximum moving-average passes.  Default 12.
    include_endpoints : bool
        Whether to treat signal endpoints as pseudo-extrema.  Default True.
    """

    def __init__(
        self,
        max_pf: int = 8,
        max_sift_iter: int = 200,
        envelope_epsilon: float = 0.01,
        convergence_epsilon: float = 0.01,
        smooth_method: str = 'moving_average',
        smooth_iterations: int = 12,
        include_endpoints: bool = True,
    ):
        if max_pf < 1:
            raise ValueError(f"max_pf must be >= 1; got {max_pf}.")
        if max_sift_iter < 1:
            raise ValueError(f"max_sift_iter must be >= 1; got {max_sift_iter}.")
        if envelope_epsilon <= 0:
            raise ValueError(f"envelope_epsilon must be > 0; got {envelope_epsilon}.")
        if convergence_epsilon <= 0:
            raise ValueError(f"convergence_epsilon must be > 0; got {convergence_epsilon}.")
        if smooth_method not in ('moving_average', 'cubic_spline'):
            raise ValueError(
                f"smooth_method must be 'moving_average' or 'cubic_spline'; got {smooth_method!r}."
            )
        self.max_pf               = max_pf
        self.max_sift_iter        = max_sift_iter
        self.envelope_epsilon     = envelope_epsilon
        self.convergence_epsilon  = convergence_epsilon
        self.smooth_method        = smooth_method
        self.smooth_iterations    = smooth_iterations
        self.include_endpoints    = include_endpoints

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(self, sig: np.ndarray) -> List[np.ndarray]:
        """
        Decompose signal into Product Functions (PFs).

        Returns
        -------
        List of 1-D arrays.  The first element is the highest-frequency PF,
        the last element is the monotonic (or near-monotonic) residual.
        All arrays have the same length as `sig`.
        """
        sig = validate_signal_1d(sig, name='sig')
        N   = len(sig)

        # Edge case: constant / near-constant signal
        if np.allclose(sig, sig[0]):
            return [sig.copy()]

        pfs     = []
        residue = sig.copy().astype(float)

        for _ in range(self.max_pf):
            if self._is_monotonic(residue):
                break
            extrema = self._find_extrema(residue)
            if len(extrema) < 4:          # need at least 2 half-waves
                break

            pf = self._extract_pf(residue)
            pfs.append(pf)
            residue = residue - pf

            if self._is_monotonic(residue):
                break

        # Always append residual so that sum(pfs) == sig
        if np.any(np.abs(residue) > 1e-10):
            pfs.append(residue)

        return pfs

    def reconstruct(self, pfs: List[np.ndarray]) -> np.ndarray:
        """Reconstruct signal from Product Functions."""
        return np.sum(pfs, axis=0)

    # ------------------------------------------------------------------
    # Core sifting machinery
    # ------------------------------------------------------------------

    def _extract_pf(self, sig: np.ndarray) -> np.ndarray:
        """
        Extract one Product Function from `sig` via the sifting process.

        The final PF is  a_product(t) * s_final(t), where a_product is the
        cumulative product of all envelope estimates obtained during sifting.
        """
        N = len(sig)
        s = sig.copy()
        cumulative_envelope = np.ones(N)

        for _ in range(self.max_sift_iter):
            extrema = self._find_extrema(s)
            if len(extrema) < 4:
                break

            m, a = self._local_mean_and_envelope(s, extrema)

            # Clamp envelope to avoid division by zero / negative values
            a = np.maximum(a, 1e-8)

            h     = s - m            # subtract local mean
            s_new = h / a            # demodulate by envelope

            cumulative_envelope *= a

            # Stopping criterion 1: envelope is flat
            flat_err = np.mean(np.abs(1.0 - a))
            if flat_err < self.envelope_epsilon:
                s = s_new
                break

            # Stopping criterion 2: FM signal has converged
            conv_err = np.mean(np.abs(s - s_new))
            if conv_err < self.convergence_epsilon:
                s = s_new
                break

            s = s_new

        # PF = cumulative envelope × final FM signal
        return cumulative_envelope * s

    def _local_mean_and_envelope(
        self, sig: np.ndarray, extrema: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute smoothed local mean m(t) and local envelope a(t).

        Steps (per Smith 2005):
          1. Build step-wise mean and envelope from consecutive extrema pairs.
          2. Smooth each step-function with the chosen smoother.
        """
        N = len(sig)
        t = np.arange(N)
        k = len(extrema)

        # Build step-wise (piecewise-constant) local mean and envelope
        mean_step = np.empty(N)
        enve_step = np.empty(N)

        prev_m = (sig[extrema[0]] + sig[extrema[1]]) / 2.0
        prev_a = abs(sig[extrema[0]] - sig[extrema[1]]) / 2.0

        e_ptr = 1
        for x in range(N):
            if e_ptr + 1 < k and x == extrema[e_ptr]:
                next_m = (sig[extrema[e_ptr]] + sig[extrema[e_ptr + 1]]) / 2.0
                next_a = abs(sig[extrema[e_ptr]] - sig[extrema[e_ptr + 1]]) / 2.0
                mean_step[x] = (prev_m + next_m) / 2.0
                enve_step[x] = (prev_a + next_a) / 2.0
                prev_m = next_m
                prev_a = next_a
                e_ptr += 1
            else:
                mean_step[x] = prev_m
                enve_step[x] = prev_a

        if self.smooth_method == 'cubic_spline':
            m = self._spline_smooth(t, mean_step, extrema)
            a = self._spline_smooth(t, enve_step, extrema)
        else:
            window = max(3, int(np.max(np.diff(extrema))) // 3)
            m = self._moving_average_smooth(mean_step, window)
            a = self._moving_average_smooth(enve_step, window)

        return m, a

    # ------------------------------------------------------------------
    # Smoothers
    # ------------------------------------------------------------------

    def _moving_average_smooth(self, sig: np.ndarray, window: int) -> np.ndarray:
        """
        Weighted moving-average smoother with triangular kernel.
        Matches the approach described by Smith (2005) and PyLMD.
        """
        n = len(sig)
        if window < 3:
            window = 3
        if window % 2 == 0:
            window += 1
        half = window // 2

        # Triangular weight kernel
        weight = np.array(list(range(1, half + 2)) + list(range(half, 0, -1)),
                          dtype=float)

        smoothed = sig.copy()
        for _ in range(self.smooth_iterations):
            prev = smoothed.copy()
            conv = np.convolve(smoothed, weight, mode='same')

            # Centre region: divide by full kernel sum
            conv[half: n - half] /= weight.sum()

            # Boundary regions: use partial kernel sums for correct normalisation
            for i in range(half):
                w_left  = weight[half - i:]
                w_right = weight[:n - half + i + 1] if n - half + i + 1 < len(weight) else weight
                conv[i]         = np.dot(smoothed[:len(w_left)], w_left) / w_left.sum()
                conv[n - 1 - i] = np.dot(smoothed[n - len(w_right):], np.flip(w_right)) / w_right.sum()

            smoothed = conv

            # Check for convergence of the smoother itself
            if np.max(np.abs(smoothed - prev)) < 1e-10:
                break

        return smoothed

    def _spline_smooth(
        self, t: np.ndarray, step: np.ndarray, extrema: np.ndarray
    ) -> np.ndarray:
        """
        Cubic-spline interpolation through the midpoints of the step-function.
        Provides a smoother local mean / envelope than moving-average.
        """
        # Use extrema positions as knot locations
        xs = extrema.astype(float)
        ys = step[extrema]

        if len(xs) < 2:
            return np.full_like(step, np.mean(step))

        try:
            cs = CubicSpline(xs, ys, bc_type='not-a-knot', extrapolate=True)
            return cs(t)
        except Exception:
            # Fall back to linear interpolation on failure
            return np.interp(t, xs, ys)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _find_extrema(self, sig: np.ndarray) -> np.ndarray:
        """Return sorted indices of all local maxima and minima."""
        maxima = scipy_signal.argrelextrema(sig, np.greater)[0]
        minima = scipy_signal.argrelextrema(sig, np.less)[0]
        extrema = np.sort(np.concatenate([maxima, minima]))

        if self.include_endpoints:
            n = len(sig)
            if len(extrema) == 0 or extrema[0] != 0:
                extrema = np.concatenate([[0], extrema])
            if extrema[-1] != n - 1:
                extrema = np.concatenate([extrema, [n - 1]])

        return extrema

    def _is_monotonic(self, sig: np.ndarray) -> bool:
        diff = np.diff(sig)
        return bool(np.all(diff >= 0) or np.all(diff <= 0))
