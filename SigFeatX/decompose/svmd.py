"""
SVMD — Successive Variational Mode Decomposition
Based on: Dragomiretskiy & Zosso (2014), applied successively with K=1.

Bugs fixed vs original:
  1. tol default was 1e-7 (way too tight — SVMD never stopped early).
     Fixed to 0.01 (stop when residual has < 1% energy of original).
  2. Inherits corrected VMD (true ADMM, not filterbank).
"""

import numpy as np
from .vmd import VMD


class SVMD:
    """
    Successive Variational Mode Decomposition.

    Extracts one VMD mode at a time from the residual, stopping when the
    residual energy drops below `tol` fraction of the original signal energy.

    Parameters
    ----------
    alpha    : VMD bandwidth parameter (passed to VMD). Default 2000.
    K_max    : maximum number of modes to extract. Default 10.
    tol      : stop when residual energy / original energy < tol. Default 0.01.
               Original code used 1e-7 which effectively disabled early stopping.
    max_iter : maximum ADMM iterations per VMD call. Default 500.
    """

    def __init__(self, alpha: float = 2000, K_max: int = 10,
                 tol: float = 0.01, max_iter: int = 500):
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
        modes    = []
        residual = sig.copy().astype(float)
        orig_energy = np.sum(sig ** 2)

        for _ in range(self.K_max):
            vmd  = VMD(alpha=self.alpha, K=1, tol=1e-7, max_iter=self.max_iter)
            mode = vmd.decompose(residual)[0]

            modes.append(mode)
            residual = residual - mode

            # Stop when residual energy is less than tol fraction of original
            if orig_energy > 0 and np.sum(residual ** 2) / orig_energy < self.tol:
                break

        if np.sum(np.abs(residual)) > 1e-10:
            modes.append(residual)

        return np.array(modes)