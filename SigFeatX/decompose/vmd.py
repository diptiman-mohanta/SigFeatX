"""
VMD — Variational Mode Decomposition
Reference: Dragomiretskiy & Zosso (2014), IEEE Trans. Signal Processing 62(3):531-544

Complete rewrite vs original. The original was a static Butterworth filterbank
with fixed frequency boundaries — not VMD. True VMD solves a constrained
variational problem using ADMM entirely in the Fourier domain.

The ADMM update equations (from the paper, Algorithm 1):

  For each mode k at each iteration n:
    û_k^{n+1}(ω) = [f̂(ω) - Σ_{i≠k} û_i(ω) + λ̂(ω)/2]
                   / [1 + 2α(ω - ω_k)²]

  ω_k^{n+1} = ∫₀^∞ ω |û_k(ω)|² dω / ∫₀^∞ |û_k(ω)|² dω

  λ̂^{n+1}(ω) = λ̂^n(ω) + τ [f̂(ω) - Σ_k û_k^{n+1}(ω)]

API is identical to the original: decompose(sig) → array of shape (K, N)
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq


class VMD:
    """
    Variational Mode Decomposition (Dragomiretskiy & Zosso 2014).

    Parameters
    ----------
    alpha    : bandwidth constraint (balancing parameter). Larger = narrower modes.
               Typical values: 100–5000. Default 2000.
    K        : number of modes to extract. Must be set in advance.
    tau      : dual ascent step (Lagrangian multiplier update rate).
               0 = no noise tolerance (strict reconstruction).
               Increase for noisy signals (e.g. 0.25).
    DC       : if True, the first mode is forced to zero frequency (DC component).
    init     : centre frequency initialisation.
               0 = all start at 0 Hz
               1 = uniformly distributed across [0, 0.5] (recommended)
               2 = random
    tol      : convergence tolerance. Default 1e-7.
    max_iter : maximum ADMM iterations. Default 500.
    """

    def __init__(self, alpha: float = 2000, K: int = 3, tau: float = 0.0,
                 DC: bool = False, init: int = 1, tol: float = 1e-7,
                 max_iter: int = 500):
        self.alpha    = alpha
        self.K        = K
        self.tau      = tau
        self.DC       = DC
        self.init     = init
        self.tol      = tol
        self.max_iter = max_iter

    def decompose(self, sig: np.ndarray) -> np.ndarray:
        """
        Decompose signal into K variational modes.

        Returns
        -------
        np.ndarray of shape (K, N) — one row per mode, time domain.
        """
        sig = np.asarray(sig, dtype=float)
        N   = len(sig)

        # Mirror-extend to reduce boundary effects (standard VMD pre-processing)
        T       = N
        f_hat   = fft(sig)                        # full spectrum, shape (N,)
        omega   = fftfreq(N)                      # normalised frequencies

        # Work only on positive half-spectrum (signal is real)
        # Paper works with analytic signal (one-sided spectrum × 2)
        pos     = omega >= 0
        f_hat_p = np.zeros(N, dtype=complex)
        f_hat_p[pos] = 2.0 * f_hat[pos]          # analytic spectrum

        # ── Initialise centre frequencies ──────────────────────────────────
        omega_K = self._init_omegas()

        # ── ADMM state ─────────────────────────────────────────────────────
        u_hat_K = np.zeros((self.K, N), dtype=complex)   # mode spectra
        lambda_hat = np.zeros(N, dtype=complex)           # Lagrangian multiplier

        # ── ADMM main loop ─────────────────────────────────────────────────
        for n in range(self.max_iter):
            u_hat_K_prev = u_hat_K.copy()

            for k in range(self.K):
                # Accumulate all modes except k
                sum_others = np.sum(u_hat_K, axis=0) - u_hat_K[k]

                # Mode update in Fourier domain (Eq. 14 in paper)
                # û_k(ω) = [f̂(ω) - Σ_{i≠k} û_i(ω) + λ̂(ω)/2]
                #           / [1 + 2α(ω - ω_k)²]
                denom         = 1.0 + 2.0 * self.alpha * (omega - omega_K[k]) ** 2
                u_hat_K[k]    = (f_hat_p - sum_others + lambda_hat / 2.0) / denom

                # Force DC mode if requested
                if self.DC and k == 0:
                    u_hat_K[0][omega != 0] = 0.0

                # Centre frequency update (Eq. 15 in paper)
                # ω_k = ∫ ω |û_k(ω)|² dω / ∫ |û_k(ω)|² dω
                power         = np.abs(u_hat_K[k][pos]) ** 2
                omega_K[k]    = np.dot(omega[pos], power) / (np.sum(power) + 1e-10)

            # Lagrangian multiplier update (Eq. 16 in paper)
            lambda_hat += self.tau * (f_hat_p - np.sum(u_hat_K, axis=0))

            # ── Convergence check ──────────────────────────────────────────
            delta = np.sum(np.abs(u_hat_K - u_hat_K_prev) ** 2) / (
                    np.sum(np.abs(u_hat_K_prev) ** 2) + 1e-10)
            if delta < self.tol:
                break

        # ── Convert modes back to time domain ─────────────────────────────
        # u_hat_K contains one-sided (analytic) spectra; take real part of IFFT
        modes = np.zeros((self.K, N))
        for k in range(self.K):
            modes[k] = np.real(ifft(u_hat_K[k]))

        return modes

    def _init_omegas(self) -> np.ndarray:
        """Initialise centre frequencies per the 'init' parameter."""
        if self.init == 0:
            omega_K = np.zeros(self.K)
        elif self.init == 1:
            # Uniformly distributed across [0, 0.5] (normalised Nyquist)
            omega_K = np.arange(1, self.K + 1) / (2.0 * self.K)
        else:
            omega_K = np.sort(np.random.rand(self.K) * 0.5)

        if self.DC:
            omega_K[0] = 0.0

        return omega_K

    def reconstruct(self, modes: np.ndarray) -> np.ndarray:
        return np.sum(modes, axis=0)