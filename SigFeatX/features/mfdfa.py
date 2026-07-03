"""
MFDFA — Multifractal Detrended Fluctuation Analysis
=====================================================
Reference: Kantelhardt et al. (2002), "Multifractal detrended fluctuation
           analysis of nonstationary time series", Physica A 316: 87-114.

Generalises DFA to a continuum of moment orders q. For each q the algorithm
fits a power law F_q(s) ~ s^h(q), giving the generalised Hurst exponent h(q).
A non-trivial h(q) (i.e. q-dependent) is the signature of a multifractal.

Features extracted:
  - h(q) for a user-selected set of moments
  - Width of the singularity spectrum  alpha_max - alpha_min
  - alpha_0 (position of the spectrum maximum)
  - Asymmetry of f(alpha)
"""

from collections.abc import Sequence

import numpy as np

from .._validation import validate_signal_1d


class MFDFAFeatures:
    """Multifractal DFA features."""

    @staticmethod
    def _profile(sig: np.ndarray) -> np.ndarray:
        """Cumulative mean-deviation profile Y(i)."""
        return np.cumsum(sig - np.mean(sig))

    @staticmethod
    def _scales(N: int, n_scales: int) -> np.ndarray:
        """Log-spaced scales between 8 and N // 4."""
        s_min = 8
        s_max = max(s_min + 1, N // 4)
        return np.unique(
            np.round(np.logspace(np.log10(s_min), np.log10(s_max), n_scales))
        ).astype(int)

    @staticmethod
    def _Fq(profile: np.ndarray, scale: int, q_values: np.ndarray) -> np.ndarray:  # noqa: N802
        """Compute F_q(s) for every q in q_values at scale s."""
        N = len(profile)
        n_boxes = N // scale
        if n_boxes < 4:
            return np.full(len(q_values), np.nan)

        boxes = profile[: n_boxes * scale].reshape(n_boxes, scale)
        x = np.arange(scale)
        F2 = np.zeros(n_boxes)
        for k in range(n_boxes):
            coeffs = np.polyfit(x, boxes[k], 1)
            fit = np.polyval(coeffs, x)
            F2[k] = np.mean((boxes[k] - fit) ** 2)

        # Reverse-pass: include boxes starting from the end
        F2_rev = np.zeros(n_boxes)
        boxes_rev = profile[-n_boxes * scale:].reshape(n_boxes, scale)
        for k in range(n_boxes):
            coeffs = np.polyfit(x, boxes_rev[k], 1)
            fit = np.polyval(coeffs, x)
            F2_rev[k] = np.mean((boxes_rev[k] - fit) ** 2)

        F2_all = np.concatenate([F2, F2_rev])
        F2_all = F2_all[F2_all > 0]
        if F2_all.size == 0:
            return np.full(len(q_values), np.nan)

        out = np.zeros(len(q_values))
        for i, q in enumerate(q_values):
            if q == 0:
                out[i] = np.exp(0.5 * np.mean(np.log(F2_all)))
            else:
                out[i] = (np.mean(F2_all ** (q / 2.0))) ** (1.0 / q)
        return out

    # ------------------------------------------------------------------
    # Public extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract(
        sig: np.ndarray,
        q_values: Sequence[float] | None = None,
        n_scales: int = 16,
    ) -> dict[str, float]:
        """
        Compute multifractal features.

        Parameters
        ----------
        sig : 1D array.
        q_values : moment orders. Default ``[-5, -3, -1, 0, 1, 3, 5]``.
        n_scales : number of log-spaced scales. Default 16.
        """
        sig = validate_signal_1d(sig, name='sig')
        if len(sig) < 64:
            # Too short for reliable MFDFA — return zeros without crashing.
            zeros: dict[str, float] = {
                'mfdfa_width': 0.0,
                'mfdfa_alpha0': 0.0,
                'mfdfa_asymmetry': 0.0,
            }
            for q in (q_values or [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0]):
                zeros[f'mfdfa_h_q{q:+.0f}'.replace('+', 'p').replace('-', 'n')] = 0.0
            return zeros

        q_arr = np.asarray(
            q_values if q_values is not None else [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0],
            dtype=float,
        )

        profile = MFDFAFeatures._profile(sig)
        scales = MFDFAFeatures._scales(len(sig), n_scales)
        if scales.size < 3:
            scales = np.array([8, 16, 32], dtype=int)

        # F_q(s) matrix
        Fq_matrix = np.zeros((len(scales), len(q_arr)))
        for si, s in enumerate(scales):
            Fq_matrix[si] = MFDFAFeatures._Fq(profile, int(s), q_arr)

        # h(q): slope of log(F_q) vs log(s) per q
        log_s = np.log(scales).astype(float)
        h = np.zeros(len(q_arr))
        for qi in range(len(q_arr)):
            yi = Fq_matrix[:, qi]
            mask = (yi > 0) & np.isfinite(yi)
            if mask.sum() < 2:
                h[qi] = np.nan
                continue
            log_F = np.log(yi[mask])
            slope, _intercept = np.polyfit(log_s[mask], log_F, 1)
            h[qi] = slope

        # Singularity-spectrum width via Legendre transform
        tau = q_arr * h - 1.0
        # Numerical derivative alpha = d tau / d q
        alpha = np.gradient(tau, q_arr)
        f_alpha = q_arr * alpha - tau

        valid_alpha = alpha[np.isfinite(alpha)]
        if valid_alpha.size == 0:
            width = 0.0
            alpha0 = 0.0
            asymmetry = 0.0
        else:
            width = float(np.max(valid_alpha) - np.min(valid_alpha))
            valid_f = f_alpha[np.isfinite(f_alpha)]
            if valid_f.size:
                alpha0 = float(alpha[np.argmax(f_alpha)])
                a_max = float(np.max(valid_alpha))
                a_min = float(np.min(valid_alpha))
                mid = 0.5 * (a_max + a_min)
                asymmetry = (alpha0 - mid) / (0.5 * width + 1e-12) if width > 0 else 0.0
            else:
                alpha0 = 0.0
                asymmetry = 0.0

        out: dict[str, float] = {
            'mfdfa_width': float(width),
            'mfdfa_alpha0': float(alpha0),
            'mfdfa_asymmetry': float(asymmetry),
        }
        for q_val, h_val in zip(q_arr, h, strict=True):
            # Compose stable key
            sign = 'p' if q_val >= 0 else 'n'
            out[f'mfdfa_h_q{sign}{abs(int(q_val))}'] = (
                float(h_val) if np.isfinite(h_val) else 0.0
            )
        return out
