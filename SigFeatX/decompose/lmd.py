"""
LMD — Local Mean Decomposition
Reference: Smith, J.S. (2005), J. R. Soc. Interface 2(5):443-454.

Implementation aligned with the well-established PySDKit LMD class
(github.com/wwhenxuan/PySDKit) and PyLMD (github.com/shownlin/PyLMD).

Key design decisions from the reference:
  1. min_extrema guard: stop extracting PFs when the residue has fewer than
     min_extrema extreme points (default 5). Prevents decomposing noise tails.

  2. local_mean_and_envelope: window = max(diff(extrema)) // 3.
     This matches the reference exactly (max inter-extremum spacing / 3).

  3. Envelope clamp: if any envelope value a[i] <= 0, set a[i] = 1 - 1e-4.
     This prevents division by zero and negative-envelope artefacts.

  4. is_smooth check in moving_average_smooth: stop smoothing iterations
     when consecutive samples differ (i.e. the signal is already smooth).
     Note: the reference's is_smooth returns False when any two consecutive
     samples are EQUAL (indicating over-smoothing plateau), so we inherit
     that logic unchanged.

  5. Overflow guard: cumulative envelope product clamped to [1e-8, 1e8]
     before each multiplication to prevent IEEE 754 overflow.

  6. NaN/Inf guard in sifting loop: break immediately if demodulated
     signal contains non-finite values.
"""

import warnings
import numpy as np
from scipy.signal import argrelextrema
from typing import List, Optional, Tuple

from SigFeatX._validation import validate_signal_1d


class LMD:
    """
    Local Mean Decomposition (Smith 2005).

    Decomposes a signal into Product Functions (PFs). Each PF is the
    product of an instantaneous-amplitude envelope and a purely FM signal.

    Parameters
    ----------
    max_pf              : max Product Functions to extract. Default 8.
    endpoints           : treat signal endpoints as pseudo-extrema. Default True.
    max_smooth_iter     : max moving-average passes. Default 15.
    max_envelope_iter   : max sifting iterations per PF. Default 200.
    envelope_epsilon    : stop when mean(|1-a|) < this. Default 0.01.
    convergence_epsilon : stop when mean(|s-t|) < this. Default 0.01.
    min_extrema         : stop extracting if residue has fewer extrema. Default 5.
    """

    def __init__(
        self,
        max_pf: int = 8,
        endpoints: bool = True,
        max_smooth_iter: int = 15,
        max_envelope_iter: int = 200,
        envelope_epsilon: float = 0.01,
        convergence_epsilon: float = 0.01,
        min_extrema: int = 5,
    ):
        if max_pf < 1:
            raise ValueError(f"max_pf must be >= 1; got {max_pf}.")
        if max_envelope_iter < 1:
            raise ValueError(f"max_envelope_iter must be >= 1; got {max_envelope_iter}.")
        if envelope_epsilon <= 0:
            raise ValueError(f"envelope_epsilon must be > 0; got {envelope_epsilon}.")
        if convergence_epsilon <= 0:
            raise ValueError(f"convergence_epsilon must be > 0; got {convergence_epsilon}.")
        if min_extrema < 2:
            raise ValueError(f"min_extrema must be >= 2; got {min_extrema}.")

        self.max_pf              = max_pf
        self.endpoints           = endpoints
        self.max_smooth_iter     = max_smooth_iter
        self.max_envelope_iter   = max_envelope_iter
        self.envelope_epsilon    = envelope_epsilon
        self.convergence_epsilon = convergence_epsilon
        self.min_extrema         = min_extrema

        # aliases for backward compatibility with existing aggregator code
        self.include_endpoints = endpoints

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(self, sig: np.ndarray) -> List[np.ndarray]:
        """
        Decompose signal into Product Functions (PFs).

        Returns
        -------
        List of 1-D arrays, all the same length as `sig`.
        Last element is the monotonic residual.
        """
        result = self.fit_transform(sig)
        return [result[i] for i in range(result.shape[0])]

    def fit_transform(self, sig: np.ndarray) -> np.ndarray:
        """
        Decompose signal. Returns 2-D array of shape (n_pfs, N).
        """
        sig = validate_signal_1d(sig, name='sig')

        # Constant signal: nothing to decompose
        if np.allclose(sig, sig[0]):
            return sig.reshape(1, -1)

        pf      = []
        residue = sig.copy().astype(float)

        while (len(pf) < self.max_pf) and (not self._is_monotonic(residue)):
            extrema = self._find_extrema(residue)
            if len(extrema) < self.min_extrema:
                break

            component = self._extract_product_function(residue)
            if component is None:
                break

            pf.append(component)
            residue = residue - component

        # Always append residual so sum == original signal
        pf.append(residue)
        return np.array(pf)

    def reconstruct(self, pfs: List[np.ndarray]) -> np.ndarray:
        return np.sum(pfs, axis=0)

    # ------------------------------------------------------------------
    # Core sifting
    # ------------------------------------------------------------------

    def _extract_product_function(self, sig: np.ndarray) -> Optional[np.ndarray]:
        """Extract one PF via iterative envelope sifting."""
        s = sig.copy()
        n = len(sig)
        envelopes: List[np.ndarray] = []

        for _ in range(self.max_envelope_iter):
            extrema = self._find_extrema(s)
            if len(extrema) <= 3:
                break

            _m0, m, _a0, a = self._local_mean_and_envelope(s, extrema)

            # Reference: clamp non-positive envelope values
            for i in range(len(a)):
                if a[i] <= 0:
                    a[i] = 1.0 - 1e-4

            h = s - m        # subtract local mean
            t = h / a        # demodulate by envelope

            # Guard: NaN/Inf during demodulation
            if not np.all(np.isfinite(t)):
                warnings.warn(
                    "[LMD] Non-finite value in demodulated signal. "
                    "Stopping sift early.",
                    RuntimeWarning, stacklevel=2,
                )
                break

            # Terminate when pure FM signal obtained
            err = np.sum(np.abs(1.0 - a)) / n
            if err <= self.envelope_epsilon:
                break

            # Terminate when modulation signal has converged
            err = np.sum(np.abs(s - t)) / n
            if err <= self.convergence_epsilon:
                break

            # Clamp envelope before accumulating (overflow guard)
            a_clamped = np.clip(a, 1e-8, 1e8)
            envelopes.append(a_clamped)
            s = t

        # PF = product of all accumulated envelopes × final FM signal
        component = s.copy()
        for e in envelopes:
            component = component * e

        if not np.all(np.isfinite(component)):
            return None

        return component

    # ------------------------------------------------------------------
    # Local mean and envelope
    # ------------------------------------------------------------------

    def _local_mean_and_envelope(
        self, sig: np.ndarray, extrema: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute raw and smoothed local mean m(t) and envelope a(t).

        Algorithm (from Smith 2005 and PyLMD reference):
          For each consecutive pair of extrema (n_i, n_{i+1}):
            m_i = (sig[n_i] + sig[n_{i+1}]) / 2
            a_i = |sig[n_i] - sig[n_{i+1}}| / 2
          These step-functions are then smoothed with a triangular
          moving-average filter whose window = max(diff(extrema)) // 3.
        """
        n = len(sig)
        k = len(extrema)

        mean_raw = np.empty(n)
        enve_raw = np.empty(n)

        prev_mean = (sig[extrema[0]] + sig[extrema[1]]) / 2.0
        prev_enve = abs(sig[extrema[0]] - sig[extrema[1]]) / 2.0
        e_ptr = 1

        for x in range(n):
            if e_ptr + 1 < k and x == extrema[e_ptr]:
                next_mean = (sig[extrema[e_ptr]] + sig[extrema[e_ptr + 1]]) / 2.0
                next_enve = abs(sig[extrema[e_ptr]] - sig[extrema[e_ptr + 1]]) / 2.0
                mean_raw[x] = (prev_mean + next_mean) / 2.0
                enve_raw[x] = (prev_enve + next_enve) / 2.0
                prev_mean   = next_mean
                prev_enve   = next_enve
                e_ptr      += 1
            else:
                mean_raw[x] = prev_mean
                enve_raw[x] = prev_enve

        # Window size from reference: max inter-extremum spacing // 3
        window = max(int(np.max(np.diff(extrema))) // 3, 3)
        mean_smooth = self._moving_average_smooth(mean_raw, window)
        enve_smooth = self._moving_average_smooth(enve_raw, window)

        return mean_raw, mean_smooth, enve_raw, enve_smooth

    # ------------------------------------------------------------------
    # Moving average smoother (exact port of PyLMD reference)
    # ------------------------------------------------------------------

    def _moving_average_smooth(self, sig: np.ndarray, window: int) -> np.ndarray:
        """
        Triangular-kernel weighted moving average.
        Directly ported from PyLMD / PySDKit reference implementations.
        """
        n = len(sig)

        if window < 3:
            window = 3
        if window % 2 == 0:
            window += 1
        half = window // 2

        # Triangular weight kernel
        weight = np.array(
            list(range(1, half + 2)) + list(range(half, 0, -1)), dtype=float
        )
        assert len(weight) == window

        smoothed = sig.copy().astype(float)

        for _ in range(self.max_smooth_iter):
            head = []
            tail = []
            w_num = half

            for i in range(half):
                head.append(
                    np.array([smoothed[j] for j in range(i - (half - w_num), i + half + 1)])
                )
                tail.append(
                    np.flip([smoothed[-(j + 1)] for j in range(i - (half - w_num), i + half + 1)])
                )
                w_num -= 1

            smoothed = np.convolve(smoothed, weight, mode='same')
            smoothed[half: -half] = smoothed[half: -half] / weight.sum()

            w_num = half
            for i in range(half):
                smoothed[i]         = np.dot(head[i], weight[w_num:]) / weight[w_num:].sum()
                smoothed[-(i + 1)]  = np.dot(tail[i], weight[:-w_num]) / weight[:-w_num].sum()
                w_num -= 1

            if self._is_smooth(smoothed, n):
                break

        return smoothed

    @staticmethod
    def _is_smooth(sig: np.ndarray, n: int) -> bool:
        """
        Reference logic: returns True (stop) when NO two consecutive
        samples are equal (the plateau test from PyLMD).
        """
        for x in range(1, n):
            if sig[x] == sig[x - 1]:
                return False
        return True

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _find_extrema(self, sig: np.ndarray) -> np.ndarray:
        """Return sorted indices of all local maxima and minima."""
        maxima = argrelextrema(sig, np.greater)[0]
        minima = argrelextrema(sig, np.less)[0]
        extrema = np.sort(np.concatenate([maxima, minima]))

        if self.endpoints:
            n = len(sig)
            if len(extrema) == 0 or extrema[0] != 0:
                extrema = np.concatenate([[0], extrema])
            if extrema[-1] != n - 1:
                extrema = np.concatenate([extrema, [n - 1]])

        return extrema

    def _is_monotonic(self, sig: np.ndarray) -> bool:
        d = np.diff(sig)
        return bool(np.all(d >= 0) or np.all(d <= 0))