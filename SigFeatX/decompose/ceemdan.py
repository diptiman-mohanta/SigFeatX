"""
CEEMDAN — Complete Ensemble Empirical Mode Decomposition with Adaptive Noise
==============================================================================
Reference: Torres et al. (2011), "A complete ensemble empirical mode
           decomposition with adaptive noise", ICASSP 2011, pp. 4144-4147.

Fixes EMD's two well-known problems:
  - Mode mixing: a single IMF contains oscillations of widely different scales
  - Boundary artefacts amplified by sifting

Algorithm
---------
Let E_k(.) extract the k-th IMF via EMD. Generate I noise realisations
w^{(i)}(t) and define ensemble averages:

  IMF_1 = (1/I) sum_i E_1(x + eps_0 * w^{(i)})
  r_1   = x - IMF_1

For k >= 2:
  IMF_k = (1/I) sum_i E_1(r_{k-1} + eps_{k-1} * E_k(w^{(i)}))
  r_k   = r_{k-1} - IMF_k

Implementation
--------------
We use the existing EMD class for E_1 extraction. CEEMDAN is much heavier
than vanilla EMD (typical factor ~50 for I=50, max_imf=10), so the default
trial count is conservative. Bump it for production analyses.
"""

import numpy as np
from typing import List, Optional

from .._validation import validate_signal_1d
from .emd import EMD


class CEEMDAN:
    """
    Complete Ensemble EMD with Adaptive Noise.

    Parameters
    ----------
    trials : int
        Number of noise realisations (ensemble size). Higher = smoother
        results, slower. Typical: 50-100. Default 50.
    noise_amp : float
        Initial noise standard deviation as a fraction of the signal std.
        Typical: 0.005 - 0.05. Default 0.02.
    max_imf : int
        Maximum IMFs to extract. -1 means unlimited (limited by stopping
        criteria). Default 10.
    rng : np.random.Generator or int or None
        Random source for noise. Pass an int for reproducibility.

    Notes
    -----
    Each ensemble member runs a full EMD on a noise-perturbed residual.
    Decomposition cost scales as ``trials * max_imf * cost(EMD)``.
    """

    def __init__(
        self,
        trials: int = 50,
        noise_amp: float = 0.02,
        max_imf: int = 10,
        rng=None,
    ):
        if trials < 2:
            raise ValueError(f"trials must be >= 2; got {trials}.")
        if noise_amp <= 0:
            raise ValueError(f"noise_amp must be > 0; got {noise_amp}.")
        if max_imf != -1 and max_imf < 1:
            raise ValueError(f"max_imf must be -1 or >=1; got {max_imf}.")

        self.trials = trials
        self.noise_amp = noise_amp
        self.max_imf = max_imf
        self.rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
        self._emd = EMD(max_imf=-1)               # used for E_1 extraction

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(self, sig: np.ndarray) -> List[np.ndarray]:
        """
        Decompose ``sig`` into CEEMDAN IMFs plus a final residual.

        Returns
        -------
        list of 1D arrays, each the same length as ``sig``.
        """
        sig = validate_signal_1d(sig, name='sig')
        N = len(sig)
        sig_std = float(np.std(sig)) + 1e-12

        # Pre-generate I noise realisations, scaled by signal std
        white = self.rng.standard_normal((self.trials, N))
        white = white * (self.noise_amp * sig_std)

        # --- Stage 1: first IMF ------------------------------------------
        imf1_components = []
        for i in range(self.trials):
            imfs_i = self._emd_first_imf(sig + white[i])
            imf1_components.append(imfs_i)
        imf1 = np.mean(imf1_components, axis=0)

        imfs: List[np.ndarray] = [imf1]
        residue = sig - imf1

        # --- Stage 2..K: subsequent IMFs ---------------------------------
        k = 1
        while True:
            if self.max_imf != -1 and k >= self.max_imf:
                break
            if self._is_monotonic(residue):
                break
            if np.allclose(residue, 0.0, atol=1e-10):
                break

            # For stage k+1 we need E_k(w_i), the k-th IMF of each noise
            ek_noises = self._kth_imf_of_noises(white, k)
            if ek_noises is None:
                break

            # Compute new ensemble of first-IMFs on residue + scaled noise
            next_imf_components = []
            for i in range(self.trials):
                next_imf_components.append(
                    self._emd_first_imf(residue + ek_noises[i])
                )
            next_imf = np.mean(next_imf_components, axis=0)

            imfs.append(next_imf)
            residue = residue - next_imf
            k += 1

        # Append residual as the last component
        imfs.append(residue.copy())
        return imfs

    def reconstruct(self, imfs: List[np.ndarray]) -> np.ndarray:
        return np.sum(imfs, axis=0)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _emd_first_imf(self, signal: np.ndarray) -> np.ndarray:
        """Run EMD and return only the first IMF, zero-padded if needed."""
        try:
            imfs = self._emd.decompose(signal)
        except Exception:
            return np.zeros_like(signal)
        if len(imfs) == 0:
            return np.zeros_like(signal)
        first = np.asarray(imfs[0], dtype=float)
        if len(first) != len(signal):
            # Defensive: align lengths just in case
            out = np.zeros_like(signal)
            n = min(len(first), len(signal))
            out[:n] = first[:n]
            return out
        return first

    def _kth_imf_of_noises(self, noises: np.ndarray, k: int) -> Optional[np.ndarray]:
        """
        Return E_k(w_i) for every noise realisation.

        Returns None if more than half the noises fail to produce k IMFs.
        """
        out = []
        n_failed = 0
        for i in range(noises.shape[0]):
            try:
                imfs = self._emd.decompose(noises[i])
            except Exception:
                imfs = []
            if len(imfs) <= k:
                n_failed += 1
                out.append(np.zeros_like(noises[i]))
            else:
                out.append(np.asarray(imfs[k], dtype=float))

        if n_failed > noises.shape[0] // 2:
            return None
        return np.asarray(out)

    @staticmethod
    def _is_monotonic(sig: np.ndarray) -> bool:
        d = np.diff(sig)
        return bool(np.all(d >= 0) or np.all(d <= 0))
