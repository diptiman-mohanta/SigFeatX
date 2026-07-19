"""
Advanced entropy features
==========================
Modern entropy measures complementing the Shannon/Sample/Permutation/
Approximate entropies already shipped in features.features.

Includes:
  - Dispersion Entropy   (Rostaghi & Azami 2016)
  - Fuzzy Entropy        (Chen et al. 2007)
  - Lempel-Ziv Complexity (Lempel & Ziv 1976)
  - Bubble Entropy       (Manis et al. 2017)
"""


import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import norm

from .._validation import validate_signal_1d


class AdvancedEntropyFeatures:
    """
    Collection of newer entropy and complexity measures.

    Use either the class-level ``extract(sig, ...)`` for the whole bundle,
    or call individual static methods.
    """

    # ==================================================================
    # Dispersion Entropy
    # ==================================================================

    @staticmethod
    def dispersion_entropy(
        sig: np.ndarray,
        m: int = 3,
        c: int = 6,
        tau: int = 1,
        normalize: bool = True,
    ) -> float:
        """
        Dispersion Entropy.

        Reference: Rostaghi & Azami (2016), "Dispersion Entropy: A Measure
        for Time-Series Analysis", IEEE Signal Processing Letters 23(5).

        Algorithm:
          1. Map signal to normal CDF, then to integer classes 1..c.
          2. Build embedded vectors of dimension m with delay tau.
          3. Each vector gets a dispersion pattern (integer in 1..c^m).
          4. Compute Shannon entropy over the pattern distribution.

        Parameters
        ----------
        m : embedding dimension. Default 3.
        c : number of dispersion classes. Default 6.
        tau : embedding delay. Default 1.
        normalize : divide by log(c^m) so result is in [0, 1]. Default True.
        """
        sig = validate_signal_1d(sig, name='sig')
        N = len(sig)
        if N < (m - 1) * tau + 1:
            return 0.0

        sigma = float(np.std(sig))
        if sigma < 1e-12:
            return 0.0
        mu = float(np.mean(sig))
        # Map to N(0,1) CDF, then to classes 1..c
        y = norm.cdf((sig - mu) / sigma)
        # Avoid edges by clipping
        z = np.clip(np.round(c * y + 0.5).astype(int), 1, c)

        # Build dispersion patterns
        n_vec = N - (m - 1) * tau
        patterns = np.zeros(n_vec, dtype=int)
        for j in range(m):
            patterns = patterns * c + (z[j * tau : j * tau + n_vec] - 1)

        # Probability of each pattern
        _, counts = np.unique(patterns, return_counts=True)
        p = counts / float(counts.sum())
        ent = float(-np.sum(p * np.log(p + 1e-12)))

        if normalize:
            ent /= float(np.log(c ** m))
        return ent

    # ==================================================================
    # Fuzzy Entropy
    # ==================================================================

    @staticmethod
    def fuzzy_entropy(
        sig: np.ndarray,
        m: int = 2,
        r: float | None = None,
        n: int = 2,
    ) -> float:
        """
        Fuzzy Entropy.

        Reference: Chen et al. (2007), "Characterization of surface EMG
        signal based on fuzzy entropy", IEEE Trans. Neural Syst. Rehabil.
        Eng. 15(2): 266-272.

        Replaces SampEn's Heaviside step with a smooth exponential
        similarity, giving more stable results on short noisy signals.

        Parameters
        ----------
        m : embedding dimension. Default 2.
        r : similarity radius. Default 0.2 * std(sig).
        n : steepness of the fuzzy membership function. Default 2.
        """
        sig = validate_signal_1d(sig, name='sig')
        N = len(sig)
        if N < m + 2:
            return 0.0

        if r is None:
            r = 0.2 * float(np.std(sig))
        if r <= 0:
            return 0.0

        def _phi(length: int) -> float:
            count = N - length + 1
            # Centre each pattern by its mean (Chen's definition)
            patterns = sliding_window_view(sig, length)
            patterns = patterns - patterns.mean(axis=1, keepdims=True)
            # Block-wise Chebyshev distances + fuzzy membership exp(-(d/r)^n).
            # (KD-trees don't apply here: the smooth membership needs the
            # actual distance values, not just a neighbour count.)
            mu = 0.0
            # small blocks keep the distance matrix in cache; in-place ops
            # avoid materialising full-size temporaries for each step
            block = max(1, min(32, count))
            for start in range(0, count, block):
                chunk = patterns[start : start + block]
                # running maximum over the (small) embedding dimension keeps
                # the working set 2-D instead of materialising a 3-D array
                d = np.abs(chunk[:, 0, None] - patterns[None, :, 0])
                for j in range(1, length):
                    np.maximum(
                        d,
                        np.abs(chunk[:, j, None] - patterns[None, :, j]),
                        out=d,
                    )
                d /= r
                d **= n
                np.negative(d, out=d)
                np.exp(d, out=d)
                mu += float(d.sum())
            # Subtract self-matches (count) and average
            return float((mu - count) / (count * (count - 1)))

        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)
        if phi_m <= 0 or phi_m1 <= 0:
            return 0.0
        return float(np.log(phi_m / phi_m1))

    # ==================================================================
    # Lempel-Ziv Complexity
    # ==================================================================

    @staticmethod
    def lempel_ziv_complexity(
        sig: np.ndarray,
        binarize: str = 'median',
        normalize: bool = True,
    ) -> float:
        """
        Lempel-Ziv Complexity.

        Reference: Lempel & Ziv (1976), "On the complexity of finite
        sequences", IEEE Trans. Information Theory 22(1): 75-81.

        Counts the number of distinct patterns encountered while parsing
        the binary version of the signal left-to-right.

        Parameters
        ----------
        binarize : 'median' (default) or 'mean' — threshold rule.
        normalize : divide by the theoretical maximum n / log2(n).
        """
        sig = validate_signal_1d(sig, name='sig')
        N = len(sig)
        if N < 2:
            return 0.0

        threshold = np.median(sig) if binarize == 'median' else np.mean(sig)
        s = ''.join('1' if v > threshold else '0' for v in sig)

        # Standard LZ76 (Kaspar-Schuster) parsing, using C-speed substring
        # search: the phrase s[i:i+k] stays reproducible while it occurs
        # starting at some position < i (the occurrence may overlap the
        # phrase itself, ending anywhere up to i+k-2). An unfinished final
        # phrase still counts, per the original algorithm.
        c = 0
        i = 0
        while i < N:
            k = 1
            while i + k <= N and s.find(s[i : i + k], 0, i + k - 1) != -1:
                k += 1
            c += 1
            i += k

        if normalize:
            b = N / np.log2(N + 1e-12)
            return float(c / b)
        return float(c)

    # ==================================================================
    # Bubble Entropy
    # ==================================================================

    @staticmethod
    def bubble_entropy(sig: np.ndarray, m: int = 10) -> float:
        """
        Bubble Entropy.

        Reference: Manis et al. (2017), "Bubble Entropy: An Entropy Almost
        Free of Parameters", IEEE Trans. Biomedical Engineering 64(11).

        Counts swaps a bubble sort would need on each embedded vector, then
        takes the *Renyi entropy of order 2* (not Shannon entropy) of that
        count distribution, and normalises by log((m+1)/(m-1)) per Eq. 8-9
        of the paper. Matches the reference EntropyHub/PyEntropy formula.

        Parameters
        ----------
        m : embedding dimension >= 2. Default 10 (works well for many signals).
        """
        sig = validate_signal_1d(sig, name='sig')
        if m < 2:
            raise ValueError(f"m must be >= 2; got {m}.")
        N = len(sig)
        if N < m + 1:
            return 0.0

        def _swap_dist(length: int) -> np.ndarray:
            # Bubble-sort swap count == number of inversions: pairs (j, k)
            # with j < k and v[j] > v[k]. Count them for all windows at once.
            windows = sliding_window_view(sig, length)
            ju, ku = np.triu_indices(length, k=1)
            return np.sum(windows[:, ju] > windows[:, ku], axis=1)

        s_m = _swap_dist(m)
        s_m1 = _swap_dist(m + 1)

        def _renyi2_entropy(arr: np.ndarray) -> float:
            _, counts = np.unique(arr, return_counts=True)
            p = counts / float(counts.sum())
            return float(-np.log(np.sum(p ** 2)))

        h_m = _renyi2_entropy(s_m)
        h_m1 = _renyi2_entropy(s_m1)
        return float((h_m1 - h_m) / np.log((m + 1) / (m - 1)))

    # ==================================================================
    # Bundle
    # ==================================================================

    @staticmethod
    def extract(sig: np.ndarray) -> dict[str, float]:
        """
        Compute all four advanced entropies with sensible defaults.
        """
        sig = validate_signal_1d(sig, name='sig')
        return {
            'dispersion_entropy': AdvancedEntropyFeatures.dispersion_entropy(sig),
            'fuzzy_entropy':      AdvancedEntropyFeatures.fuzzy_entropy(sig),
            'lz_complexity':      AdvancedEntropyFeatures.lempel_ziv_complexity(sig),
            'bubble_entropy':     AdvancedEntropyFeatures.bubble_entropy(sig),
        }
