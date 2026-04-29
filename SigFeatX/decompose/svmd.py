"""
SVMD — Successive Variational Mode Decomposition
Reference: Nazari & Sakhaei (2020), Signal Processing 174:107610.

Uses the corrected VMD (with mirror extension and fftshift reconstruction).
The successive strategy extracts K=1 mode at a time from the residual.

Changes from previous version:
  - tol default changed from 1e-7 to 0.01 (energy ratio criterion):
    stop when residual energy < tol × original energy.
  - Inherits VMD's mirror extension and correct amplitude reconstruction.
"""

import numpy as np
from .vmd import VMD
from SigFeatX._validation import validate_signal_1d


class SVMD:
    """
    Successive Variational Mode Decomposition.

    Parameters
    ----------
    alpha    : VMD bandwidth parameter. Default 2000.
    K_max    : maximum modes to extract. Default 10.
    tol      : stop when residual_energy / original_energy < tol. Default 0.01.
    max_iter : maximum ADMM iterations per VMD call. Default 500.
    """

    def __init__(
        self,
        alpha: float = 2000,
        K_max: int = 10,
        tol: float = 0.01,
        max_iter: int = 500,
    ):
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0; got {alpha}.")
        if K_max < 1:
            raise ValueError(f"K_max must be >= 1; got {K_max}.")
        if tol < 0:
            raise ValueError(f"tol must be >= 0; got {tol}.")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1; got {max_iter}.")

        self.alpha    = alpha
        self.K_max    = K_max
        self.tol      = tol
        self.max_iter = max_iter

    def decompose(self, sig: np.ndarray) -> np.ndarray:
        """
        Decompose signal using successive VMD (K=1 at a time).

        Returns
        -------
        np.ndarray of shape (n_modes, N)
        """
        sig = validate_signal_1d(sig, name='sig')
        modes        = []
        residual     = sig.copy().astype(float)
        orig_energy  = float(np.sum(sig ** 2))

        if orig_energy <= 1e-12:
            return residual.reshape(1, -1)

        for _ in range(self.K_max):
            vmd  = VMD(alpha=self.alpha, K=1, tol=1e-7, max_iter=self.max_iter)
            mode = vmd.decompose(residual)

            # mode has shape (1, N_out); N_out may differ from len(sig) by 1
            # due to even-length enforcement in VMD — align lengths
            mode_1d = mode[0]
            n_out   = mode_1d.shape[0]
            n_sig   = len(residual)

            if n_out != n_sig:
                # Trim or zero-pad to match residual length
                if n_out > n_sig:
                    mode_1d = mode_1d[:n_sig]
                else:
                    mode_1d = np.pad(mode_1d, (0, n_sig - n_out))

            modes.append(mode_1d)
            residual = residual - mode_1d

            # Early stopping: residual energy below threshold
            res_energy = float(np.sum(residual ** 2))
            if res_energy / orig_energy < self.tol:
                break

        if np.sum(np.abs(residual)) > 1e-10:
            modes.append(residual)

        return np.array(modes)

    def reconstruct(self, modes: np.ndarray) -> np.ndarray:
        return np.sum(modes, axis=0)