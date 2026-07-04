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

Each trial's EMD call is independent of every other trial (both within a
stage and, since EMD is a pure function of its input signal, across the
whole ensemble), so the trial loop is embarrassingly parallel. Pass
``n_jobs`` to use multiple processes -- follows the same convention as
``FeatureAggregator.extract_batch`` (1 = sequential/default, -1 = all
cores, N = N cores), and is bit-for-bit identical to the sequential
result for the same ``rng`` seed, since ``Executor.map`` preserves input
order and EMD itself has no internal randomness.
"""

import os
import warnings
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

from .._validation import validate_n_jobs, validate_signal_1d
from .emd import EMD

# ---------------------------------------------------------------------------
# Module-level workers (must be top-level to be picklable by ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _worker_first_imf(signal: np.ndarray, emd: EMD) -> np.ndarray:
    """Run EMD and return only the first IMF, zero-padded/truncated if needed."""
    try:
        imfs = emd.decompose(signal)
    except Exception:
        return np.zeros_like(signal)
    if len(imfs) == 0:
        return np.zeros_like(signal)
    first = np.asarray(imfs[0], dtype=float)
    if len(first) != len(signal):
        out = np.zeros_like(signal)
        n = min(len(first), len(signal))
        out[:n] = first[:n]
        return out
    return first


def _worker_full_decompose(signal: np.ndarray, emd: EMD) -> list:
    """Run full EMD and return all IMFs (empty list on failure)."""
    try:
        return emd.decompose(signal)
    except Exception:
        return []


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
    n_jobs : int
        1 (default) runs trials sequentially. -1 uses all available CPU
        cores; N uses N worker processes. Falls back to threads if
        process-based parallelism is unavailable (e.g. sandboxed CI).
        Output is identical regardless of n_jobs for the same rng seed.

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
        n_jobs: int = 1,
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
        self.n_jobs = validate_n_jobs(n_jobs)
        self._emd = EMD(max_imf=-1)               # used for E_1 extraction

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(self, sig: np.ndarray) -> list[np.ndarray]:
        """
        Decompose ``sig`` into CEEMDAN IMFs plus a final residual.

        Returns
        -------
        list of 1D arrays, each the same length as ``sig``.
        """
        sig = validate_signal_1d(sig, name='sig')

        if self.n_jobs == 1:
            return self._decompose_with_executor(sig, None)

        max_workers = (os.cpu_count() or 1) if self.n_jobs == -1 else self.n_jobs
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                return self._decompose_with_executor(sig, executor)
        except (OSError, PermissionError):
            warnings.warn(
                "[SigFeatX] Process-based parallelism is unavailable in this "
                "environment; falling back to threads for CEEMDAN.",
                RuntimeWarning, stacklevel=2,
            )
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                return self._decompose_with_executor(sig, executor)

    def reconstruct(self, imfs: list[np.ndarray]) -> np.ndarray:
        return np.sum(imfs, axis=0)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _decompose_with_executor(
        self, sig: np.ndarray, executor: Executor | None
    ) -> list[np.ndarray]:
        N = len(sig)
        sig_std = float(np.std(sig)) + 1e-12

        # Pre-generate I *unit-amplitude* noise realisations. Kept unscaled
        # so E_k(white[i]) can be rescaled fresh at every stage relative to
        # the current residue's std -- this is the "Adaptive Noise" (AN) in
        # CEEMDAN: without it, the noise amplitude added at deep stages
        # stays pinned to the original signal's std instead of tracking the
        # (typically much smaller) residue, degrading high-order IMFs.
        white = self.rng.standard_normal((self.trials, N))

        # --- Stage 1: first IMF ------------------------------------------
        imf1_components = self._map_trials(
            executor, _worker_first_imf,
            [(sig + self.noise_amp * sig_std * white[i], self._emd)
             for i in range(self.trials)],
        )
        imf1 = np.mean(imf1_components, axis=0)

        imfs: list[np.ndarray] = [imf1]
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
            ek_noises = self._kth_imf_of_noises(white, k, executor)
            if ek_noises is None:
                break

            # Adaptive rescale: noise amplitude tracks the current residue's
            # std, not the original signal's.
            beta_k = self.noise_amp * (float(np.std(residue)) + 1e-12)

            # Compute new ensemble of first-IMFs on residue + scaled noise
            next_imf_components = self._map_trials(
                executor, _worker_first_imf,
                [(residue + beta_k * ek_noises[i], self._emd)
                 for i in range(self.trials)],
            )
            next_imf = np.mean(next_imf_components, axis=0)

            imfs.append(next_imf)
            residue = residue - next_imf
            k += 1

        # Append residual as the last component
        imfs.append(residue.copy())
        return imfs

    @staticmethod
    def _map_trials(executor: Executor | None, fn, args_list: list[tuple]) -> list:
        """Apply fn(*args) over args_list, sequentially or via executor.

        ``Executor.map`` preserves input order, so results (and therefore
        the ensemble average) are identical regardless of executor choice.
        """
        if not args_list:
            return []
        if executor is None:
            return [fn(*args) for args in args_list]
        return list(executor.map(fn, *zip(*args_list, strict=True)))

    def _kth_imf_of_noises(
        self, noises: np.ndarray, k: int, executor: Executor | None
    ) -> np.ndarray | None:
        """
        Return E_k(w_i) for every noise realisation.

        Returns None if more than half the noises fail to produce k IMFs.
        """
        full_decomps = self._map_trials(
            executor, _worker_full_decompose,
            [(noises[i], self._emd) for i in range(noises.shape[0])],
        )

        out = []
        n_failed = 0
        for imfs in full_decomps:
            if len(imfs) <= k:
                n_failed += 1
                out.append(np.zeros(noises.shape[1]))
            else:
                out.append(np.asarray(imfs[k], dtype=float))

        if n_failed > noises.shape[0] // 2:
            return None
        return np.asarray(out)

    @staticmethod
    def _is_monotonic(sig: np.ndarray) -> bool:
        d = np.diff(sig)
        return bool(np.all(d >= 0) or np.all(d <= 0))
