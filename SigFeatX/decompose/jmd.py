"""
JMD — Jump Plus AM-FM Mode Decomposition
Reference: Nazari, M., Korshøj, A. R., & Rehman, N. (2025).
    Jump Plus AM-FM Mode Decomposition.
    IEEE Transactions on Signal Processing, 73, 1081-1093.
    https://doi.org/10.1109/TSP.2025.3535822
    arXiv: 2407.07800

Mathematical model:
    f(t) = Σ_k u_k(t) + v(t) + n(t)

    where:
      u_k(t) — AM-FM oscillatory modes (band-limited, like VMD)
      v(t)   — piecewise-constant jump component
      n(t)   — noise

The joint variational problem is solved via ADMM:

  Oscillatory modes (identical to VMD update):
      û_k^{n+1}(ω) = [f̂(ω) - Σ_{i≠k} û_i(ω) - v̂(ω) + λ̂(ω)/2]
                      / [1 + 2α(ω - ω_k)²]

  Centre-frequency update (identical to VMD):
      ω_k^{n+1} = ∫₀^∞ ω|û_k(ω)|² dω / ∫₀^∞ |û_k(ω)|² dω

  Jump component (minimax-concave penalty on first derivative):
      v^{n+1} = (γD^T D + 2I)^{-1} [D^T ρ + γD^T x + 2(f - Σ_k u_k) - λ]

  Auxiliary variable (element-wise proximal step):
      x_j^{n+1} = prox_{β/γ · ϕ}(Dv + ρ/γ)_j

  Lagrangian multiplier updates:
      λ^{n+1} = λ + τ₁ (f̂ - v̂ - Σ_k û_k)
      ρ^{n+1} = ρ  + τ₂ (Dv - x)

Key difference from VMD: the jump component v is explicitly extracted
alongside the oscillatory modes, preventing jumps from contaminating the
AM-FM modes with spurious high-frequency artefacts.

Note on the MC penalty proximal operator:
    The minimax-concave (MC) penalty's proximal step for a scalar z:
        prox(z) = sign(z) * max(|z| - 1/(γ b), 0)  if |z| < √(2/b)
                = z                                  otherwise
    For b→0 this reduces to the soft-threshold (L1/LASSO).
    For b→∞ it approaches hard thresholding (L0).
"""

import warnings
import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.sparse import diags as sp_diags
from scipy.sparse.linalg import spsolve
from typing import List, Optional, Tuple

from SigFeatX._validation import validate_signal_1d


class JMD:
    """
    Jump Plus AM-FM Mode Decomposition (Nazari et al. 2025).

    Jointly extracts K band-limited AM-FM oscillatory modes AND a
    piecewise-constant jump component from a nonstationary signal, using
    an ADMM-based variational optimization.

    This is the key advantage over VMD: when a signal contains sudden
    discontinuities (step changes, spikes, geomagnetic jerks, ECG artefacts),
    VMD smears the jump energy across all modes.  JMD explicitly separates
    the jump, yielding cleaner oscillatory modes.

    Parameters
    ----------
    K : int
        Number of AM-FM oscillatory modes.  Default 3.
    alpha : float
        Bandwidth constraint (same role as in VMD).  Larger = narrower modes.
        Typical range: 100–5000.  Default 2000.
    beta : float
        Weight of the jump regularisation term.  Larger = stronger jump
        suppression in the oscillatory modes.  Default 0.5.
    b : float
        Non-convexity parameter of the minimax-concave (MC) penalty.
        b→0 → L1 (soft threshold); b→∞ → L0 (hard threshold).
        Default 1.0 (moderately non-convex).
    tau1 : float
        Dual-ascent step for the oscillatory Lagrangian multiplier λ.
        0 = noise-tolerant mode.  Default 0.0.
    tau2 : float
        Dual-ascent step for the jump auxiliary Lagrangian multiplier ρ.
        Default 0.1.
    gamma : float
        ADMM penalty scalar for the jump sub-problem.  Default 1.0.
    init : int
        Centre-frequency initialisation: 0=zeros, 1=uniform, 2=random.
        Default 1.
    tol : float
        Convergence tolerance.  Default 1e-7.
    max_iter : int
        Maximum ADMM iterations.  Default 500.
    DC : bool
        If True, force the first mode to zero frequency.  Default False.
    """

    def __init__(
        self,
        K: int = 3,
        alpha: float = 2000.0,
        beta: float = 0.5,
        b: float = 1.0,
        tau1: float = 0.0,
        tau2: float = 0.1,
        gamma: float = 1.0,
        init: int = 1,
        tol: float = 1e-7,
        max_iter: int = 500,
        DC: bool = False,
    ):
        if K < 1:
            raise ValueError(f"K must be >= 1; got {K}.")
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0; got {alpha}.")
        if beta < 0:
            raise ValueError(f"beta must be >= 0; got {beta}.")
        if b <= 0:
            raise ValueError(f"b must be > 0; got {b}.")
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0; got {gamma}.")
        if tol <= 0:
            raise ValueError(f"tol must be > 0; got {tol}.")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1; got {max_iter}.")
        if init not in (0, 1, 2):
            raise ValueError(f"init must be 0, 1, or 2; got {init}.")

        self.K        = K
        self.alpha    = alpha
        self.beta     = beta
        self.b        = b
        self.tau1     = tau1
        self.tau2     = tau2
        self.gamma    = gamma
        self.init     = init
        self.tol      = tol
        self.max_iter = max_iter
        self.DC       = DC

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(self, sig: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose signal into K AM-FM modes + jump component.

        Returns
        -------
        modes : np.ndarray of shape (K, N)
            Oscillatory AM-FM modes (time domain), one row per mode.
        jump : np.ndarray of shape (N,)
            Piecewise-constant jump component.

        Notes
        -----
        To reconstruct the signal: modes.sum(axis=0) + jump ≈ sig
        (residual noise is not recovered by design).
        """
        sig = validate_signal_1d(sig, name='sig')
        N   = len(sig)

        # ── Fourier domain setup ──────────────────────────────────────────
        f_hat = fft(sig)
        omega = fftfreq(N)                     # normalised frequencies [-0.5, 0.5)

        pos   = omega >= 0
        # Analytic spectrum (one-sided × 2) — same as VMD
        f_hat_p              = np.zeros(N, dtype=complex)
        f_hat_p[pos]         = 2.0 * f_hat[pos]

        # ── Initialise centre frequencies ─────────────────────────────────
        omega_k = self._init_omegas()

        # ── ADMM state variables ──────────────────────────────────────────
        u_hat_k   = np.zeros((self.K, N), dtype=complex)  # mode spectra
        lambda_   = np.zeros(N, dtype=complex)             # Lagrangian for modes
        v         = np.zeros(N)                            # jump component (time)
        x         = np.zeros(N - 1)                        # auxiliary = D*v
        rho       = np.zeros(N - 1)                        # Lagrangian for jump

        # ── Sparse first-difference matrix D of shape (N-1, N) ───────────
        # D v = v[1:] - v[:-1]  (forward differences)
        D, DtD, DtD_mat = self._build_diff_matrices(N)

        # ── ADMM main loop ────────────────────────────────────────────────
        for iteration in range(self.max_iter):
            u_hat_prev = u_hat_k.copy()

            # -- Step 1: Update oscillatory modes and centre frequencies ---
            for k in range(self.K):
                sum_others = np.sum(u_hat_k, axis=0) - u_hat_k[k]
                v_hat      = fft(v)

                denom       = 1.0 + 2.0 * self.alpha * (omega - omega_k[k]) ** 2
                u_hat_k[k]  = (
                    f_hat_p - sum_others - v_hat + lambda_ / 2.0
                ) / denom

                if self.DC and k == 0:
                    u_hat_k[0][omega != 0] = 0.0

                power      = np.abs(u_hat_k[k][pos]) ** 2
                omega_k[k] = np.dot(omega[pos], power) / (np.sum(power) + 1e-10)

            # -- Step 2: Update jump component v ---------------------------
            # v = (γ D^T D + 2I)^{-1} [D^T ρ + γ D^T x + 2(f - Σu_k) - λ_time]
            u_sum_time    = np.sum(
                [np.real(ifft(u_hat_k[k])) for k in range(self.K)], axis=0
            )
            lambda_time   = np.real(ifft(lambda_))
            rhs = (
                D.T.dot(rho)
                + self.gamma * D.T.dot(x)
                + 2.0 * (sig - u_sum_time)
                - lambda_time
            )
            lhs = self.gamma * DtD_mat + 2.0 * sp_diags(
                np.ones(N), 0, shape=(N, N), format='csc')
            v = spsolve(lhs, rhs)

            # -- Step 3: Update auxiliary variable x (MC proximal step) ----
            Dv = D.dot(v)
            z  = Dv + rho / self.gamma
            x  = self._mc_proximal(z, threshold=self.beta / self.gamma, b=self.b)

            # -- Step 4: Update Lagrangian multipliers ----------------------
            v_hat     = fft(v)
            lambda_  += self.tau1 * (f_hat_p - v_hat - np.sum(u_hat_k, axis=0))
            rho      += self.tau2 * (Dv - x)

            # -- Convergence check ------------------------------------------
            delta = np.sum(np.abs(u_hat_k - u_hat_prev) ** 2) / (
                np.sum(np.abs(u_hat_prev) ** 2) + 1e-10
            )
            if delta < self.tol:
                break

        # ── Convert modes to time domain ──────────────────────────────────
        modes = np.zeros((self.K, N))
        for k in range(self.K):
            modes[k] = np.real(ifft(u_hat_k[k]))

        return modes, v

    def reconstruct(self, modes: np.ndarray, jump: np.ndarray) -> np.ndarray:
        """Reconstruct signal from modes and jump component."""
        return np.sum(modes, axis=0) + jump

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_omegas(self) -> np.ndarray:
        """Initialise centre frequencies (same as VMD)."""
        if self.init == 0:
            omega_k = np.zeros(self.K)
        elif self.init == 1:
            omega_k = np.arange(1, self.K + 1) / (2.0 * self.K)
        else:
            omega_k = np.sort(np.random.rand(self.K) * 0.5)
        if self.DC:
            omega_k[0] = 0.0
        return omega_k

    @staticmethod
    def _build_diff_matrices(N: int):
        """
        Build sparse first-difference matrix D ∈ R^{(N-1)×N} and D^T D.
        D v  = v[1:] - v[:-1]
        """
        ones = np.ones(N)
        D    = sp_diags([-ones[:-1], ones[:-1]], offsets=[0, 1],
                        shape=(N - 1, N), format='csr')
        DtD  = D.T.dot(D)
        return D, DtD, DtD.tocsc()

    @staticmethod
    def _mc_proximal(z: np.ndarray, threshold: float, b: float) -> np.ndarray:
        """
        Element-wise proximal operator for the minimax-concave (MC) penalty
        applied to the derivative of the jump component.

        For each z_j:
            if |z_j| < sqrt(2/b):
                prox = sign(z_j) * max(|z_j| - threshold, 0)   # soft-threshold
            else:
                prox = z_j                                        # pass through

        This follows from the subdifferential of ϕ(|·|; b) in Eq. (6) of the
        JMD paper.  As b → 0, reverts to pure L1 soft-thresholding.
        As b → ∞, the threshold region collapses → hard thresholding.
        """
        abs_z    = np.abs(z)
        boundary = np.sqrt(2.0 / (b + 1e-10))

        result   = np.where(
            abs_z < boundary,
            np.sign(z) * np.maximum(abs_z - threshold, 0.0),   # soft threshold
            z,                                                    # pass through
        )
        return result
