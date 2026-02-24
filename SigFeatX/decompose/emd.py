"""
EMD — Empirical Mode Decomposition
Reference: Huang et al. (1998), Proc. Royal Society London A, 454:903-995

Bugs fixed vs original:
  1. Linear interpolation → cubic spline (Huang 1998 explicitly requires cubic spline)
  2. Stopping threshold 1e-6 → 0.2  (Huang 1998 SD criterion, threshold 0.2-0.3)
  3. Boundary extrema now use nearest-extremum value instead of raw signal endpoint
     to avoid incorrect envelope pinning at signal edges
"""

import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline
from typing import List, Optional


class EMD:
    """Empirical Mode Decomposition (Huang et al. 1998)."""

    def __init__(self, max_imf: int = 10, max_iter: int = 100,
                 sd_threshold: float = 0.2):
        """
        Parameters
        ----------
        max_imf      : maximum number of IMFs to extract
        max_iter     : maximum sifting iterations per IMF
        sd_threshold : Huang (1998) SD stopping criterion threshold.
                       Recommended range 0.2–0.3. Original code used 1e-6
                       which caused severe over-sifting.
        """
        self.max_imf = max_imf
        self.max_iter = max_iter
        self.sd_threshold = sd_threshold

    def decompose(self, sig: np.ndarray) -> List[np.ndarray]:
        """
        Decompose signal into IMFs.

        Returns
        -------
        List of IMFs in order from highest to lowest frequency,
        followed by the monotonic residual.
        """
        imfs = []
        residual = sig.copy().astype(float)

        for _ in range(self.max_imf):
            imf = self._extract_imf(residual)
            if imf is None:
                break

            imfs.append(imf)
            residual = residual - imf

            if self._is_monotonic(residual) or np.sum(np.abs(residual)) < 1e-10:
                break

        if len(residual) > 0 and np.sum(np.abs(residual)) > 1e-10:
            imfs.append(residual)

        return imfs

    def _extract_imf(self, sig: np.ndarray) -> Optional[np.ndarray]:
        """Extract a single IMF using the sifting process."""
        h = sig.copy()
        n = len(h)
        t = np.arange(n)

        for _ in range(self.max_iter):
            # ── Find extrema ───────────────────────────────────────────────
            max_idx = signal.argrelextrema(h, np.greater)[0]
            min_idx = signal.argrelextrema(h, np.less)[0]

            if len(max_idx) < 2 or len(min_idx) < 2:
                return None

            # ── Boundary extension (Bug 3 fix) ─────────────────────────────
            # Use the value of the nearest interior extremum at the boundary
            # rather than the raw signal endpoint, which may not be an extremum.
            # This prevents incorrect envelope pinning at edges.
            left_max_val  = h[max_idx[0]]
            right_max_val = h[max_idx[-1]]
            left_min_val  = h[min_idx[0]]
            right_min_val = h[min_idx[-1]]

            max_t   = np.concatenate(([0],        max_idx, [n - 1]))
            max_v   = np.concatenate(([left_max_val],  h[max_idx], [right_max_val]))
            min_t   = np.concatenate(([0],        min_idx, [n - 1]))
            min_v   = np.concatenate(([left_min_val],  h[min_idx], [right_min_val]))

            # ── Cubic spline envelopes (Bug 1 fix) ─────────────────────────
            # Huang (1998) explicitly requires cubic spline interpolation.
            # Original code used np.interp() which is linear — wrong.
            upper_env = CubicSpline(max_t, max_v)(t)
            lower_env = CubicSpline(min_t, min_v)(t)

            # ── Mean envelope and proto-IMF ────────────────────────────────
            mean_env = (upper_env + lower_env) / 2.0
            h_new    = h - mean_env

            # ── Huang (1998) SD stopping criterion (Bug 2 fix) ─────────────
            # SD = sum[(h_{k-1} - h_k)^2 / h_{k-1}^2]
            # Stop when SD < sd_threshold (0.2–0.3 per Huang 1998).
            # Original threshold was 1e-6 — 5 orders of magnitude too tight,
            # causing over-sifting that destroys amplitude modulation.
            sd = np.sum((h - h_new) ** 2) / (np.sum(h ** 2) + 1e-10)
            h = h_new

            if sd < self.sd_threshold:
                return h

        return h

    def _is_monotonic(self, sig: np.ndarray) -> bool:
        diff = np.diff(sig)
        return np.all(diff >= 0) or np.all(diff <= 0)

    def reconstruct(self, imfs: List[np.ndarray]) -> np.ndarray:
        return np.sum(imfs, axis=0)