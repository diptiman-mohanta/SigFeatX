"""
EMD — Empirical Mode Decomposition
Reference: Huang et al. (1998), Proc. Royal Society London A, 454:903-995

Implementation aligned with the well-established PyEMD library
(github.com/laszukdawid/PyEMD) and the PySDKit EMD class.

Key differences from our previous naive version:
  1. Boundary extension via symmetric extrema mirroring (nbsym=2):
     Instead of creating synthetic boundary points at a single position,
     the reference mirrors the last nbsym interior extrema symmetrically
     about each endpoint. This produces a smooth, non-oscillating
     envelope at the boundaries.

  2. Multiple stopping criteria tested in order:
       a) scaled variance (svar_thr)
       b) standard deviation (std_thr)
       c) energy ratio (energy_ratio_thr)
     All consistent with Huang 1998 and PyEMD.

  3. Hard maximum iteration cap per IMF (MAX_ITERATION).

  4. End condition: stops when residue range < range_thr or
     residue power < total_power_thr.

  5. Cubic spline via scipy.interpolate.CubicSpline (not-a-knot,
     the standard choice in signal processing).
"""

import warnings
import numpy as np
from scipy import signal as scipy_signal
from scipy.interpolate import CubicSpline
from typing import List, Optional, Tuple

from SigFeatX._validation import validate_signal_1d


class EMD:
    """
    Empirical Mode Decomposition (Huang et al. 1998).

    Parameters
    ----------
    max_imf        : maximum IMFs to extract (-1 = unlimited). Default -1.
    max_iteration  : max sifting iterations per IMF. Default 1000.
    nbsym          : number of extrema to mirror at each boundary. Default 2.
    std_thr        : standard-deviation stopping threshold. Default 0.2.
    svar_thr       : scaled-variance stopping threshold. Default 0.001.
    energy_ratio_thr : energy-ratio stopping threshold. Default 0.2.
    total_power_thr  : end condition: residue power threshold. Default 0.005.
    range_thr        : end condition: residue range threshold. Default 0.001.
    """

    def __init__(
        self,
        max_imf: int = -1,
        max_iteration: int = 1000,
        nbsym: int = 2,
        std_thr: float = 0.2,
        svar_thr: float = 0.001,
        energy_ratio_thr: float = 0.2,
        total_power_thr: float = 0.005,
        range_thr: float = 0.001,
        sd_threshold: float = 0.2,   # kept for backward-compat alias
    ):
        if max_iteration < 1:
            raise ValueError(f"max_iteration must be >= 1; got {max_iteration}.")
        if nbsym < 1:
            raise ValueError(f"nbsym must be >= 1; got {nbsym}.")

        self.max_imf           = max_imf
        # backward-compat alias
        self.max_iter          = max_iteration
        self.MAX_ITERATION     = max_iteration
        self.nbsym             = nbsym
        self.std_thr           = std_thr
        self.svar_thr          = svar_thr
        self.energy_ratio_thr  = energy_ratio_thr
        self.total_power_thr   = total_power_thr
        self.range_thr         = range_thr

        # storage for last run
        self.imfs    = None
        self.residue = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(self, sig: np.ndarray) -> List[np.ndarray]:
        """
        Decompose signal into IMFs.

        Returns
        -------
        List of 1-D arrays: IMFs from highest to lowest frequency,
        followed by the residual if non-zero.
        """
        result = self.fit_transform(sig)
        return [result[i] for i in range(result.shape[0])]

    def fit_transform(self, sig: np.ndarray) -> np.ndarray:
        """
        Decompose signal. Returns 2-D array of shape (n_imfs, N).
        Also stores self.imfs and self.residue for later access.
        """
        sig = validate_signal_1d(sig, name='sig')
        N   = len(sig)
        t   = np.arange(N, dtype=float)

        residue = sig.copy().astype(float)
        imf_no  = 0
        ext_no  = -1
        IMF     = np.empty((0, N), dtype=float)
        finished = False

        while not finished:
            residue = sig - np.sum(IMF, axis=0) if imf_no > 0 else sig.copy()
            imf     = residue.copy()
            mean    = np.zeros(N)

            for n in range(1, self.MAX_ITERATION + 1):
                max_pos, max_val, min_pos, min_val, _ = self._find_extrema(t, imf)
                ext_no = len(max_pos) + len(min_pos)

                if ext_no > 2:
                    max_env, min_env = self._build_envelopes(
                        t, imf, max_pos, max_val, min_pos, min_val
                    )
                    if max_env is None:
                        finished = True
                        break

                    mean[:] = 0.5 * (max_env + min_env)
                    imf_old = imf.copy()
                    imf    -= mean

                    if self._check_imf(imf, imf_old, max_val, min_val):
                        break
                else:
                    finished = True
                    break

            IMF    = np.vstack([IMF, imf])
            imf_no += 1

            if self._end_condition(sig, IMF):
                finished = True

            if self.max_imf > 0 and imf_no >= self.max_imf - 1:
                finished = True

        # If last IMF has ≤ 2 extrema it is the trend/residue
        if ext_no <= 2 and IMF.shape[0] > 0:
            IMF = IMF[:-1]

        self.imfs    = IMF.copy()
        self.residue = sig - np.sum(self.imfs, axis=0)

        if not np.allclose(self.residue, 0):
            IMF = np.vstack([IMF, self.residue])

        return IMF

    def reconstruct(self, imfs: List[np.ndarray]) -> np.ndarray:
        return np.sum(imfs, axis=0)

    def get_imfs_and_residue(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.imfs is None:
            raise ValueError("Run fit_transform first.")
        return self.imfs, self.residue

    # ------------------------------------------------------------------
    # Stopping criteria
    # ------------------------------------------------------------------

    def _check_imf(
        self,
        imf_new: np.ndarray,
        imf_old: np.ndarray,
        max_val: np.ndarray,
        min_val: np.ndarray,
    ) -> bool:
        """
        True if the current proto-IMF satisfies any stopping criterion.

        Criteria (from Huang 1998 / PyEMD):
          1. All local maxima positive, all local minima negative.
          2. Scaled variance (svar_thr).
          3. Standard deviation (std_thr).
          4. Energy ratio (energy_ratio_thr).
        """
        if len(max_val) > 0 and np.any(max_val < 0):
            return False
        if len(min_val) > 0 and np.any(min_val > 0):
            return False
        if np.sum(imf_new ** 2) < 1e-10:
            return False

        diff              = imf_new - imf_old
        diff_sq_sum       = np.sum(diff * diff)
        imf_range         = max(imf_old) - min(imf_old)

        if imf_range > 1e-10:
            svar = diff_sq_sum / imf_range
            if svar < self.svar_thr:
                return True

        safe_imf = np.where(np.abs(imf_new) > 1e-10, imf_new, np.nan)
        std = float(np.nansum((diff / safe_imf) ** 2))
        if std < self.std_thr:
            return True

        denom = np.sum(imf_old ** 2)
        if denom > 1e-10:
            energy_ratio = diff_sq_sum / denom
            if energy_ratio < self.energy_ratio_thr:
                return True

        return False

    def _end_condition(self, sig: np.ndarray, IMF: np.ndarray) -> bool:
        """True if the residue is too small to decompose further."""
        tmp = sig - np.sum(IMF, axis=0)
        if np.max(tmp) - np.min(tmp) < self.range_thr:
            return True
        if np.sum(np.abs(tmp)) < self.total_power_thr:
            return True
        return False

    # ------------------------------------------------------------------
    # Extrema detection
    # ------------------------------------------------------------------

    def _find_extrema(
        self, t: np.ndarray, sig: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Find local maxima, minima, and zero-crossings.

        Returns
        -------
        max_pos, max_val, min_pos, min_val, ind_zer
        """
        max_idx = scipy_signal.argrelextrema(sig, np.greater)[0]
        min_idx = scipy_signal.argrelextrema(sig, np.less)[0]

        # Zero crossings (sign changes)
        zer_idx = np.where(np.diff(np.sign(sig)))[0]

        return (
            t[max_idx], sig[max_idx],
            t[min_idx], sig[min_idx],
            t[zer_idx],
        )

    # ------------------------------------------------------------------
    # Envelope construction
    # ------------------------------------------------------------------

    def _build_envelopes(
        self,
        t: np.ndarray,
        sig: np.ndarray,
        max_pos: np.ndarray,
        max_val: np.ndarray,
        min_pos: np.ndarray,
        min_val: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Build upper and lower envelopes using cubic spline with
        symmetric boundary mirroring (nbsym extrema reflected at each end).
        """
        try:
            max_t, max_v = self._mirror_extrema(t, max_pos, max_val)
            min_t, min_v = self._mirror_extrema(t, min_pos, min_val)
        except ValueError:
            return None, None

        if len(max_t) < 2 or len(min_t) < 2:
            return None, None

        try:
            upper = CubicSpline(max_t, max_v, bc_type='not-a-knot')(t)
            lower = CubicSpline(min_t, min_v, bc_type='not-a-knot')(t)
        except Exception:
            return None, None

        return upper, lower

    def _mirror_extrema(
        self,
        t: np.ndarray,
        pos: np.ndarray,
        val: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extend extrema arrays by reflecting the nbsym outermost extrema
        symmetrically about each endpoint of the signal.

        This is the standard boundary extension used by PyEMD and prevents
        the cubic spline from developing a large slope at the signal edges.
        """
        if len(pos) < 2:
            raise ValueError("Not enough extrema to mirror.")

        n = self.nbsym
        t_start, t_end = t[0], t[-1]

        # Reflect about left boundary
        left_t = 2 * t_start - pos[:n][::-1]
        left_v = val[:n][::-1]

        # Reflect about right boundary
        right_t = 2 * t_end - pos[-n:][::-1]
        right_v = val[-n:][::-1]

        ext_t = np.concatenate([left_t, pos, right_t])
        ext_v = np.concatenate([left_v, val, right_v])

        # Ensure strictly increasing positions (required by CubicSpline)
        _, unique_idx = np.unique(ext_t, return_index=True)
        return ext_t[unique_idx], ext_v[unique_idx]

    def _is_monotonic(self, sig: np.ndarray) -> bool:
        d = np.diff(sig)
        return bool(np.all(d >= 0) or np.all(d <= 0))