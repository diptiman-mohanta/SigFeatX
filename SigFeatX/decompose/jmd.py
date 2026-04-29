"""
JMD — Jump Plus AM-FM Mode Decomposition
Reference: Nazari, M., Korshøj, A. R., & Rehman, N. (2025).
    IEEE Transactions on Signal Processing, 73, 1081-1093.
    arXiv: 2407.07800

Implementation aligned with the PySDKit JMD class
(github.com/wwhenxuan/PySDKit) and the authors' MATLAB reference code.

Key design decisions from the reference:
  1. Mirror extension: same as VMD — prepend/append flipped halves.
  2. Increasing alpha schedule: alpha is NOT constant. The reference uses
     a quadratic ramp phi(t) = (-a2/2)t² + sqrt(2*a2)*t up to the
     inflection point, then 1 thereafter. alpha[n] = Alpha_param * phi[n].
     This makes the bandwidth constraint tighten gradually each iteration,
     improving convergence stability significantly.
  3. ifftshift before ifft for reconstruction (same as VMD reference).
  4. MC proximal operator matches the reference's _max/_min clipping:
       x = clip(clip(1/(1-mu*b) - mu*sqrt(2b)/((1-mu*b)*|h|), 0, None), None, 1) * h
  5. v update: linalg.solve (dense). D is a T×T forward-difference matrix
     with zero last row (NOT scipy sparse, matching the MATLAB reference).
  6. Lagrangian update for rho: rho = rho - gamma*(x - Dv)
     (subtraction, not addition — matches the MATLAB sign convention).
  7. v mean-correction after each update to prevent DC drift.
  8. De-mirroring: same as VMD — keep u[:, T//4 : 3*T//4].
"""

import numpy as np
from numpy import linalg
from typing import Tuple, Union, Optional

from SigFeatX._validation import validate_signal_1d


class JMD:
    """
    Jump Plus AM-FM Mode Decomposition (Nazari et al. 2025).

    Parameters
    ----------
    K        : number of AM-FM oscillatory modes.
    alpha    : bandwidth constraint (balancing parameter). Default 5000.
    init     : centre-frequency init: 'zero', 'uniform', 'random'. Default 'zero'.
    tol      : convergence tolerance. Default 1e-6.
    beta     : jump regularisation (≈ 1/expected_jumps). Default 0.03.
    b_bar    : MC-penalty shape parameter. Default 0.45.
    tau      : dual-ascent step for Lagrangian λ. Default 5.
    max_iter : maximum ADMM iterations. Default 2000.
    """

    def __init__(
        self,
        K: int,
        alpha: float = 5000.0,
        init: str = 'zero',
        tol: float = 1e-6,
        beta: float = 0.03,
        b_bar: float = 0.45,
        tau: float = 5.0,
        max_iter: int = 2000,
    ):
        if K < 1:
            raise ValueError(f"K must be >= 1; got {K}.")
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0; got {alpha}.")
        if beta < 0:
            raise ValueError(f"beta must be >= 0; got {beta}.")
        if b_bar <= 0:
            raise ValueError(f"b_bar must be > 0; got {b_bar}.")
        if init not in ('zero', 'uniform', 'random'):
            raise ValueError(f"init must be 'zero', 'uniform', or 'random'; got {init!r}.")
        if tol <= 0:
            raise ValueError(f"tol must be > 0; got {tol}.")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1; got {max_iter}.")

        self.K        = K
        self.alpha    = alpha
        self.init     = init
        self.tol      = tol
        self.beta     = beta
        self.b_bar    = b_bar
        self.tau      = tau
        self.max_iter = max_iter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(self, sig: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose signal into K AM-FM modes + jump component.

        Returns
        -------
        modes : np.ndarray of shape (K, N)
        jump  : np.ndarray of shape (N,)

        Reconstruction: modes.sum(axis=0) + jump ≈ sig
        """
        u, v, _ = self.fit_transform(sig, return_all=True)
        return u, v

    def fit_transform(
        self,
        sig: np.ndarray,
        return_all: bool = True,
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """
        Decompose signal.

        Parameters
        ----------
        sig        : 1-D input signal
        return_all : if True return (modes, jump, omega);
                     if False return modes only.
        """
        sig = validate_signal_1d(sig, name='sig')

        # DC shift (reference saves shift and adds it back to v)
        shift = float(np.mean(sig))

        # Period / sampling frequency
        save_T = len(sig)
        fs     = 1.0 / save_T

        # ── Mirror extension ───────────────────────────────────────────
        f = self._enc_fmirror(sig, save_T)
        T = len(f)                              # = 2 * save_T
        half_T = T // 2
        t = np.arange(1, T + 1) / T
        freqs = t - 0.5 - 1.0 / T             # centred frequency axis

        # ── Increasing alpha schedule ──────────────────────────────────
        # Reference: phi ramps from 0 up to 1, then stays at 1.
        # alpha[n] = Alpha_param * phi[n]
        a2   = 50
        t2   = np.arange(0.01, np.sqrt(2.0 / a2) + 0.001, 0.001)
        phi1 = (-a2 / 2.0) * t2 ** 2 + np.sqrt(2.0 * a2) * t2
        if self.max_iter > phi1.shape[0]:
            phi = np.concatenate([phi1, np.ones(self.max_iter - phi1.shape[0])])
        else:
            phi = phi1.copy()
        Alpha = self.alpha * phi               # shape (max_iter,)

        # ── Analytic (positive-only) spectrum ──────────────────────────
        f_hat      = np.fft.fftshift(np.fft.fft(f))
        f_hat_plus = f_hat.copy()
        f_hat_plus[:half_T] = 0

        # ── Mode spectrum storage ──────────────────────────────────────
        u_hat_plus = np.zeros((self.max_iter, T, self.K), dtype=complex)

        # ── Centre frequency initialisation ───────────────────────────
        omega_plus = self._init_omega(fs)

        # ── Jump component initialisation ──────────────────────────────
        b, v, x, D, DTD, SPDiag, j_hat_plus, rho, coef1, mu, gamma = \
            self._init_jump(freqs, T)

        # ── ADMM loop ──────────────────────────────────────────────────
        u_diff = self.tol + np.spacing(1)
        n      = 0
        sum_uk = 0

        while u_diff > self.tol and n < self.max_iter - 1:

            # Mode updates (same as VMD with scheduled Alpha)
            k      = 0
            sum_uk = (u_hat_plus[n, :, self.K - 1]
                      + sum_uk
                      - u_hat_plus[n, :, 0])

            for k in range(self.K):
                if k > 0:
                    sum_uk = (u_hat_plus[n + 1, :, k - 1]
                              + sum_uk
                              - u_hat_plus[n, :, k])

                u_hat_plus[n + 1, :, k] = (
                    (f_hat_plus - sum_uk - j_hat_plus[n, :])
                    / (1.0 + Alpha[n] * (freqs - omega_plus[n, k]) ** 2)
                )

                pos_u = np.abs(u_hat_plus[n + 1, half_T:T, k]) ** 2
                omega_plus[n + 1, k] = (
                    np.dot(freqs[half_T:T], pos_u)
                    / (np.sum(pos_u) + 1e-10)
                )

            # Back to time domain for v update
            u_hat_td = np.zeros((T, self.K), dtype=complex)
            for k in range(self.K):
                u_hat_td[half_T:T, k] = u_hat_plus[n + 1, half_T:T, k]
                conj_v = np.conj(u_hat_plus[n + 1, half_T:T, k])[::-1]
                u_hat_td[1: half_T + 1, k] = conj_v
                u_hat_td[0, k] = np.conj(u_hat_td[-1, k])

            u_td = np.zeros((self.K, T))
            for k in range(self.K):
                u_td[k, :] = np.real(
                    np.fft.ifft(np.fft.ifftshift(u_hat_td[:, k]))
                )

            # Jump component v update
            rhs = (
                gamma * D.T.dot(x)
                - D.T.dot(rho)
                + f
                - np.sum(u_td, axis=0)
            )
            lhs = SPDiag + gamma * DTD
            v   = linalg.solve(lhs, rhs)

            # Auxiliary variable x update (MC proximal — reference formula)
            Dv = D.dot(v)
            h  = Dv + coef1 * rho

            abs_h = np.abs(h)
            # Avoid division by zero in the proximal computation
            safe_abs_h = np.where(abs_h > 1e-10, abs_h, 1e-10)

            inner = (
                (1.0 / (1.0 - mu * b)) * np.ones_like(abs_h)
                - (mu * np.sqrt(2.0 * b) / (1.0 - mu * b)) / safe_abs_h
            )
            inner_clamped = np.clip(inner, 0.0, None)   # max(·, 0)
            x = np.clip(inner_clamped, None, 1.0) * h   # min(·, 1) × h

            # Dual update for rho (reference: subtract, not add)
            rho = rho - gamma * (x - Dv)

            # Mean-correction of v (prevents DC drift)
            v = v - (np.mean(v) - np.mean(f))

            # Jump to frequency domain
            j_hat_plus[n + 1, :] = np.fft.fftshift(np.fft.fft(v))
            j_hat_plus[n + 1, :half_T] = 0

            n += 1

            # Convergence check
            u_diff = np.spacing(1)
            for i in range(self.K):
                d      = u_hat_plus[n, :, i] - u_hat_plus[n - 1, :, i]
                u_diff += (1.0 / T) * np.real(np.dot(d, np.conj(d)))
            d_j    = j_hat_plus[n, :] - j_hat_plus[n - 1, :]
            u_diff += (1.0 / T) * np.real(np.dot(d_j, np.conj(d_j)))
            u_diff  = np.abs(u_diff)

        # ── Final reconstruction ───────────────────────────────────────
        N_iter  = min(n, self.max_iter)
        omega   = omega_plus[N_iter, :]

        u_hat_final = np.zeros((T, self.K), dtype=complex)
        for k in range(self.K):
            u_hat_final[half_T:T, k] = u_hat_plus[N_iter, half_T:T, k]
            conj_v = np.conj(u_hat_plus[N_iter, half_T:T, k])[::-1]
            u_hat_final[1: half_T + 1, k] = conj_v
            u_hat_final[0, k] = np.conj(u_hat_final[-1, k])

        u_full = np.zeros((self.K, T))
        for k in range(self.K):
            u_full[k, :] = np.real(
                np.fft.ifft(np.fft.ifftshift(u_hat_final[:, k]))
            )

        # De-mirror
        pos = T // 4
        u   = u_full[:, pos: 3 * pos]
        v   = v[pos: 3 * pos] + shift

        if return_all:
            return u, v, omega
        return u

    def reconstruct(self, modes: np.ndarray, jump: np.ndarray) -> np.ndarray:
        """Reconstruct signal from modes and jump component."""
        return np.sum(modes, axis=0) + jump

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _enc_fmirror(sig: np.ndarray, T: int) -> np.ndarray:
        """Mirror extension: flip(sig[:T//2]) + sig + flip(sig[T//2:])."""
        return np.concatenate([
            sig[:T // 2][::-1],
            sig,
            sig[T // 2:][::-1],
        ])

    def _init_omega(self, fs: float) -> np.ndarray:
        """Initialise centre-frequency matrix."""
        omega = np.zeros((self.max_iter, self.K))
        if self.init == 'zero':
            pass   # already zeros
        elif self.init == 'uniform':
            for i in range(1, self.K + 1):
                omega[0, i - 1] = (0.5 / self.K) * (i - 1)
        elif self.init == 'random':
            omega[0, :] = np.sort(
                np.exp(
                    np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand(self.K)
                )
            )
        return omega

    def _init_jump(self, freqs: np.ndarray, T: int):
        """
        Initialise all jump-component variables.

        Returns
        -------
        b, v, x, D, DTD, SPDiag, j_hat_plus, rho, coef1, mu, gamma
        """
        # b from b_bar
        b = 2.0 / (self.b_bar ** 2)

        # gamma from tau and b (reference formula)
        gamma = self.tau * (0.5 * b * self.beta)

        # Jump signal (time domain)
        v = np.zeros(T)

        # First-difference matrix D: T×T, last row = 0
        d = np.ones(T)
        D = np.diag(-d, 0) + np.diag(d[:-1], 1)
        D = D.astype(float)
        D[-1, :] = 0.0               # zero boundary condition

        DTD    = D.T.dot(D)
        SPDiag = np.eye(T, dtype=float)

        # Auxiliary variable and Lagrangian for jump
        x   = np.zeros(T)
        rho = np.zeros(T)

        # Pre-computed scalars
        coef1 = 1.0 / gamma
        mu    = 2.0 * self.beta / gamma

        # Jump spectrum storage
        j_hat_plus = np.zeros((self.max_iter, len(freqs)), dtype=complex)

        return b, v, x, D, DTD, SPDiag, j_hat_plus, rho, coef1, mu, gamma