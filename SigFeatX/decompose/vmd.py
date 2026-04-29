"""
VMD — Variational Mode Decomposition
Reference: Dragomiretskiy & Zosso (2014), IEEE Trans. Signal Processing 62(3):531-544

Implementation ported from the well-established Python reference by
Vinícius Rezende Carvalho (vrcarva@gmail.com), itself based on Dominique
Zosso's original MATLAB code.

Key design decisions (all from the reference):
  1. Mirror extension: the signal is extended with flip(f[:N//2]) prepended
     and flip(f[-N//2:]) appended before taking the FFT. This prevents
     boundary Gibbs phenomena that plague the naive one-shot FFT approach.
  2. fftshift / ifftshift: the spectrum is shifted so DC is at the centre.
     Frequency axis is t - 0.5 - 1/T (NOT scipy fftfreq).
  3. One-sided positive spectrum: f_hat_plus[:T//2] = 0 zeroes the negative
     half after shifting, giving the analytic signal.
  4. Reconstruction: the final u_hat is built by conjugate-mirroring the
     positive half back onto the negative half, then ifft(ifftshift(...)).
     This is the standard way to recover a real signal from a one-sided
     analytic spectrum.
  5. De-mirroring: only the central quarter u[:, T//4 : 3*T//4] is kept.
"""

import numpy as np
from SigFeatX._validation import validate_signal_1d


class VMD:
    """
    Variational Mode Decomposition (Dragomiretskiy & Zosso 2014).

    Parameters
    ----------
    alpha    : bandwidth constraint (balancing parameter). Default 2000.
    K        : number of modes to extract.
    tau      : dual ascent step. 0 = noise-tolerant. Default 0.0.
    DC       : if True, force first mode to zero frequency (DC). Default False.
    init     : centre-frequency init: 0=zeros, 1=uniform, 2=random. Default 1.
    tol      : convergence tolerance. Default 1e-7.
    max_iter : maximum ADMM iterations. Default 500.
    """

    def __init__(
        self,
        alpha: float = 2000,
        K: int = 3,
        tau: float = 0.0,
        DC: bool = False,
        init: int = 1,
        tol: float = 1e-7,
        max_iter: int = 500,
    ):
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0; got {alpha}.")
        if K < 1:
            raise ValueError(f"K must be >= 1; got {K}.")
        if tau < 0:
            raise ValueError(f"tau must be >= 0; got {tau}.")
        if init not in (0, 1, 2):
            raise ValueError(f"init must be 0, 1, or 2; got {init}.")
        if tol <= 0:
            raise ValueError(f"tol must be > 0; got {tol}.")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1; got {max_iter}.")

        self.alpha    = alpha
        self.K        = K
        self.tau      = tau
        self.DC       = DC
        self.init     = init
        self.tol      = tol
        self.max_iter = max_iter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(self, sig: np.ndarray) -> np.ndarray:
        """
        Decompose signal into K variational modes.

        Parameters
        ----------
        sig : 1-D numpy array

        Returns
        -------
        modes : np.ndarray of shape (K, len(sig))
            One row per mode, time domain, same length as input.
        """
        sig = validate_signal_1d(sig, name='sig')

        # Ensure even length (reference implementation requirement)
        if len(sig) % 2:
            sig = sig[:-1]

        N  = len(sig)
        fs = 1.0 / N

        # ── Mirror extension ───────────────────────────────────────────
        # Prepend flip(sig[:N//2]) and append flip(sig[-N//2:])
        ltemp  = N // 2
        f_mirr = np.concatenate([
            np.flip(sig[:ltemp]),
            sig,
            np.flip(sig[-ltemp:]),
        ])

        T = len(f_mirr)                       # = 2*N
        t = np.arange(1, T + 1) / T
        freqs = t - 0.5 - (1.0 / T)          # centred frequency axis

        # ── Build analytic (positive-only) spectrum ────────────────────
        f_hat      = np.fft.fftshift(np.fft.fft(f_mirr))
        f_hat_plus = f_hat.copy()
        f_hat_plus[:T // 2] = 0               # zero negative frequencies

        # ── Initialise centre frequencies ──────────────────────────────
        omega_plus = np.zeros((self.max_iter, self.K))
        if self.init == 1:
            for i in range(self.K):
                omega_plus[0, i] = (0.5 / self.K) * i
        elif self.init == 2:
            omega_plus[0, :] = np.sort(
                np.exp(
                    np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand(self.K)
                )
            )
        # else init==0: omega stays at 0 (already initialised)

        if self.DC:
            omega_plus[0, 0] = 0.0

        # ── ADMM state ─────────────────────────────────────────────────
        lambda_hat = np.zeros((self.max_iter, T), dtype=complex)
        u_hat_plus = np.zeros((self.max_iter, T, self.K), dtype=complex)

        Alpha   = self.alpha * np.ones(self.K)
        u_diff  = self.tol + np.spacing(1)
        n       = 0
        sum_uk  = 0

        # ── Main ADMM loop ─────────────────────────────────────────────
        while u_diff > self.tol and n < self.max_iter - 1:

            # Update first mode
            k      = 0
            sum_uk = (u_hat_plus[n, :, self.K - 1]
                      + sum_uk
                      - u_hat_plus[n, :, 0])

            u_hat_plus[n + 1, :, k] = (
                (f_hat_plus - sum_uk - lambda_hat[n, :] / 2.0)
                / (1.0 + Alpha[k] * (freqs - omega_plus[n, k]) ** 2)
            )

            if not self.DC:
                pos_mask = freqs[T // 2 : T]
                pos_u    = np.abs(u_hat_plus[n + 1, T // 2 : T, k]) ** 2
                omega_plus[n + 1, k] = (
                    np.dot(pos_mask, pos_u) / (np.sum(pos_u) + 1e-10)
                )

            # Update remaining modes
            for k in range(1, self.K):
                sum_uk = (u_hat_plus[n + 1, :, k - 1]
                          + sum_uk
                          - u_hat_plus[n, :, k])

                u_hat_plus[n + 1, :, k] = (
                    (f_hat_plus - sum_uk - lambda_hat[n, :] / 2.0)
                    / (1.0 + Alpha[k] * (freqs - omega_plus[n, k]) ** 2)
                )

                pos_mask = freqs[T // 2 : T]
                pos_u    = np.abs(u_hat_plus[n + 1, T // 2 : T, k]) ** 2
                omega_plus[n + 1, k] = (
                    np.dot(pos_mask, pos_u) / (np.sum(pos_u) + 1e-10)
                )

            # Dual ascent
            lambda_hat[n + 1, :] = (
                lambda_hat[n, :]
                + self.tau * (
                    np.sum(u_hat_plus[n + 1, :, :], axis=1) - f_hat_plus
                )
            )

            n += 1

            # Convergence check
            u_diff = np.spacing(1)
            for i in range(self.K):
                diff   = u_hat_plus[n, :, i] - u_hat_plus[n - 1, :, i]
                u_diff += (1.0 / T) * np.real(np.dot(diff, np.conj(diff)))
            u_diff = np.abs(u_diff)

        # ── Reconstruction ─────────────────────────────────────────────
        # Discard empty rows if converged early
        Niter = min(self.max_iter, n)
        omega = omega_plus[:Niter, :]           # not used externally here

        # Build two-sided spectrum by conjugate mirroring
        idxs   = np.flip(np.arange(1, T // 2 + 1))
        u_hat  = np.zeros((T, self.K), dtype=complex)
        u_hat[T // 2 : T, :] = u_hat_plus[Niter - 1, T // 2 : T, :]
        u_hat[idxs, :]        = np.conj(u_hat_plus[Niter - 1, T // 2 : T, :])
        u_hat[0, :]           = np.conj(u_hat[-1, :])

        # Back to time domain via ifft(ifftshift)
        u_full = np.zeros((self.K, T))
        for k in range(self.K):
            u_full[k, :] = np.real(
                np.fft.ifft(np.fft.ifftshift(u_hat[:, k]))
            )

        # Remove mirror padding: keep central quarter
        modes = u_full[:, T // 4 : 3 * T // 4]

        # Ensure output length exactly matches original input length
        out_len = min(modes.shape[1], N)
        return modes[:, :out_len]

    def reconstruct(self, modes: np.ndarray) -> np.ndarray:
        """Sum all modes to reconstruct the signal."""
        return np.sum(modes, axis=0)