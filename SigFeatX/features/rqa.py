"""
RQA — Recurrence Quantification Analysis
==========================================
References:
  - Eckmann, Kamphorst, Ruelle (1987), "Recurrence Plots of Dynamical
    Systems", Europhysics Letters 4(9): 973-977.
  - Marwan et al. (2007), "Recurrence plots for the analysis of complex
    systems", Physics Reports 438(5-6): 237-329.

Builds a recurrence matrix R(i, j) = Theta(eps - ||x_i - x_j||) from a
time-delay embedding of the signal, then extracts:

  RR    — Recurrence Rate (density of recurrence points)
  DET   — Determinism (fraction of recurrence points on diagonal lines)
  LAM   — Laminarity (fraction on vertical lines)
  L     — Average diagonal line length
  L_max — Longest diagonal line
  TT    — Trapping Time (average vertical line length)
  V_max — Longest vertical line
  ENTR  — Shannon entropy of diagonal-line length distribution
  DIV   — Divergence (1 / L_max)
"""

from typing import Dict

import numpy as np

from .._validation import validate_signal_1d


class RQAFeatures:
    """
    Recurrence Quantification Analysis features.

    Static methods so callers can do ``RQAFeatures.extract(sig)``.
    """

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    @staticmethod
    def _embed(sig: np.ndarray, m: int, tau: int) -> np.ndarray:
        """Build the time-delay embedding of dimension m, delay tau."""
        N = len(sig)
        n_vec = N - (m - 1) * tau
        if n_vec <= 0:
            raise ValueError(
                f"signal too short for m={m}, tau={tau}; need N > {(m-1)*tau}."
            )
        out = np.zeros((n_vec, m), dtype=float)
        for j in range(m):
            out[:, j] = sig[j * tau : j * tau + n_vec]
        return out

    # ------------------------------------------------------------------
    # Recurrence matrix
    # ------------------------------------------------------------------

    @staticmethod
    def _recurrence_matrix(embedded: np.ndarray, eps: float) -> np.ndarray:
        """Compute Theta(eps - distance(x_i, x_j)) for the embedded series."""
        diff = embedded[:, None, :] - embedded[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        return (dist <= eps).astype(np.int8)

    @staticmethod
    def _auto_eps(embedded: np.ndarray, target_rr: float = 0.1) -> float:
        """Pick eps so the resulting RR is roughly `target_rr`."""
        diff = embedded[:, None, :] - embedded[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        # Pick the target_rr-th percentile of the off-diagonal distances
        N = dist.shape[0]
        mask = ~np.eye(N, dtype=bool)
        return float(np.percentile(dist[mask], target_rr * 100))

    # ------------------------------------------------------------------
    # Line-distribution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _diagonal_line_lengths(R: np.ndarray, l_min: int = 2) -> np.ndarray:
        """All diagonal line lengths >= l_min (excluding the main diagonal)."""
        N = R.shape[0]
        lengths = []
        for offset in range(1, N):
            diag = np.diag(R, k=offset)
            lengths.extend(RQAFeatures._runs_of_ones(diag, l_min))
        return np.asarray(lengths, dtype=int)

    @staticmethod
    def _vertical_line_lengths(R: np.ndarray, v_min: int = 2) -> np.ndarray:
        """All vertical line lengths >= v_min."""
        N = R.shape[0]
        lengths = []
        for col in range(N):
            lengths.extend(RQAFeatures._runs_of_ones(R[:, col], v_min))
        return np.asarray(lengths, dtype=int)

    @staticmethod
    def _runs_of_ones(arr: np.ndarray, min_len: int) -> list:
        """Find runs of 1s in a 1D binary array; return lengths >= min_len."""
        if arr.size == 0:
            return []
        out = []
        run = 0
        for v in arr:
            if v == 1:
                run += 1
            else:
                if run >= min_len:
                    out.append(run)
                run = 0
        if run >= min_len:
            out.append(run)
        return out

    # ------------------------------------------------------------------
    # Public extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract(
        sig: np.ndarray,
        m: int = 3,
        tau: int = 1,
        eps: float = None,
        l_min: int = 2,
        v_min: int = 2,
    ) -> Dict[str, float]:
        """
        Compute RQA features.

        Parameters
        ----------
        sig : 1D array.
        m : embedding dimension. Default 3.
        tau : embedding delay. Default 1.
        eps : recurrence threshold. None picks an eps targeting RR ~10%.
        l_min : minimum diagonal line length. Default 2.
        v_min : minimum vertical line length. Default 2.
        """
        sig = validate_signal_1d(sig, name='sig')
        embedded = RQAFeatures._embed(sig, m, tau)
        if eps is None:
            eps = RQAFeatures._auto_eps(embedded, target_rr=0.1)
        R = RQAFeatures._recurrence_matrix(embedded, eps)
        N = R.shape[0]

        # Recurrence Rate
        n_recurrent = int(np.sum(R))
        rr = n_recurrent / float(N * N)

        # Diagonal lines
        diag_lengths = RQAFeatures._diagonal_line_lengths(R, l_min)
        n_on_diag = int(np.sum(diag_lengths))
        # Exclude main diagonal from total recurrences for DET denominator
        n_offdiag = int(np.sum(R) - N)
        det = n_on_diag / float(n_offdiag) if n_offdiag > 0 else 0.0

        # Vertical lines
        vert_lengths = RQAFeatures._vertical_line_lengths(R, v_min)
        n_on_vert = int(np.sum(vert_lengths))
        lam = n_on_vert / float(np.sum(R)) if np.sum(R) > 0 else 0.0

        # Average diagonal / vertical line length
        l_avg = float(np.mean(diag_lengths)) if diag_lengths.size else 0.0
        l_max = int(np.max(diag_lengths)) if diag_lengths.size else 0
        tt = float(np.mean(vert_lengths)) if vert_lengths.size else 0.0
        v_max = int(np.max(vert_lengths)) if vert_lengths.size else 0

        # Entropy of diagonal-line length distribution
        if diag_lengths.size:
            hist = np.bincount(diag_lengths)
            hist = hist[hist > 0]
            p = hist / float(np.sum(hist))
            entr = float(-np.sum(p * np.log2(p)))
        else:
            entr = 0.0

        div = 1.0 / l_max if l_max > 0 else 0.0

        return {
            'rqa_rr':    float(rr),
            'rqa_det':   float(det),
            'rqa_lam':   float(lam),
            'rqa_l_avg': l_avg,
            'rqa_l_max': float(l_max),
            'rqa_tt':    tt,
            'rqa_v_max': float(v_max),
            'rqa_entr':  entr,
            'rqa_div':   float(div),
            'rqa_eps':   float(eps),
        }
