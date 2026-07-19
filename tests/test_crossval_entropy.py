"""
Cross-validation of entropy and nonlinear features.

Two independent lines of evidence that the implementations are correct:

1. **Closed-form theory** (always runs, no extra dependencies):
   - the logistic map at r=4 has largest Lyapunov exponent exactly ln 2;
   - white noise has Hurst exponent and DFA scaling exponent 0.5
     (R/S Hurst is tolerated up toward 0.65 — the classic Anis–Lloyd
     small-sample bias of the rescaled-range estimator);
   - permutation entropy of i.i.d. noise approaches its maximum
     log2(order!);
   - normalised dispersion entropy of white noise approaches 1.

2. **Reference library agreement** (skipped unless ``antropy`` is
   installed, mirroring the PyEMD cross-validation setup): SampEn, ApEn,
   permutation entropy, and Lempel-Ziv complexity computed on identical
   fixed-seed signals must agree with antropy. Measured agreement:
   LZ76 bit-identical, ApEn/PermEn ~1e-7, SampEn within 3e-3 at N=1000
   (antropy uses N-m+1 templates vs. our N-m per Richman & Moorman);
   thresholds below leave margin over those measurements without being
   vacuous.
"""

import numpy as np
import pytest

from SigFeatX.features.advanced_entropy import AdvancedEntropyFeatures
from SigFeatX.features.features import EntropyFeatures, NonlinearFeatures

RNG = np.random.default_rng(2024)
WHITE = RNG.standard_normal(5000)


def _logistic_map(n: int, x0: float = 0.34567) -> np.ndarray:
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = 4.0 * x[i - 1] * (1.0 - x[i - 1])
    return x


# ======================================================================
# 1. Closed-form theory
# ======================================================================


class TestTheory:
    def test_lyapunov_logistic_map_is_ln2(self):
        # Largest Lyapunov exponent of x -> 4x(1-x) is exactly ln 2.
        lam = NonlinearFeatures._lyapunov_exponent(_logistic_map(5000))
        assert abs(lam - np.log(2.0)) < 0.05, f"lambda={lam}, expected ln2"

    def test_dfa_white_noise_is_half(self):
        alpha = NonlinearFeatures._dfa(WHITE)
        assert abs(alpha - 0.5) < 0.1, f"DFA alpha={alpha}, expected 0.5"

    def test_hurst_white_noise_near_half(self):
        # R/S estimator has a known upward small-sample bias (Anis-Lloyd),
        # so the band is asymmetric around the theoretical 0.5.
        h = NonlinearFeatures._hurst_exponent(WHITE)
        assert 0.4 < h < 0.7, f"Hurst={h}, expected near 0.5"

    def test_permutation_entropy_iid_is_maximal(self):
        pe = EntropyFeatures._permutation_entropy(WHITE, order=3, delay=1)
        max_pe = np.log2(6.0)  # log2(3!) — all ordinal patterns equally likely
        assert max_pe - pe < 0.01, f"PE={pe}, max={max_pe}"

    def test_dispersion_entropy_white_noise_is_maximal(self):
        de = AdvancedEntropyFeatures.dispersion_entropy(WHITE)
        assert de > 0.98, f"normalised DispEn={de}, expected ~1"

    def test_entropy_orders_periodic_below_noise(self):
        # Any sane entropy must rank a pure sine below white noise.
        t = np.linspace(0, 10 * np.pi, 2000)
        sine = np.sin(t)
        noise = RNG.standard_normal(2000)
        assert (EntropyFeatures._sample_entropy(sine)
                < EntropyFeatures._sample_entropy(noise))
        assert (EntropyFeatures._permutation_entropy(sine)
                < EntropyFeatures._permutation_entropy(noise))
        assert (AdvancedEntropyFeatures.fuzzy_entropy(sine)
                < AdvancedEntropyFeatures.fuzzy_entropy(noise))
        assert (AdvancedEntropyFeatures.lempel_ziv_complexity(sine)
                < AdvancedEntropyFeatures.lempel_ziv_complexity(noise))


# ======================================================================
# 2. Agreement with the antropy reference library
# ======================================================================

antropy = pytest.importorskip("antropy")

SIGNALS = [RNG.standard_normal(n) for n in (500, 1000, 3000)]


class TestAntropyAgreement:
    @pytest.mark.parametrize("idx", range(len(SIGNALS)))
    def test_sample_entropy(self, idx):
        x = SIGNALS[idx]
        ours = EntropyFeatures._sample_entropy(x, m=2)
        ref = antropy.sample_entropy(x, order=2)
        assert abs(ours - ref) < 0.02, f"ours={ours}, antropy={ref}"

    @pytest.mark.parametrize("idx", range(len(SIGNALS)))
    def test_approximate_entropy(self, idx):
        x = SIGNALS[idx]
        ours = EntropyFeatures._approximate_entropy(x, m=2)
        ref = antropy.app_entropy(x, order=2)
        assert abs(ours - ref) < 1e-5, f"ours={ours}, antropy={ref}"

    @pytest.mark.parametrize("idx", range(len(SIGNALS)))
    def test_permutation_entropy(self, idx):
        x = SIGNALS[idx]
        ours = EntropyFeatures._permutation_entropy(x, order=3, delay=1)
        ref = antropy.perm_entropy(x, order=3, delay=1, normalize=False)
        assert abs(ours - ref) < 1e-6, f"ours={ours}, antropy={ref}"

    @pytest.mark.parametrize("idx", range(len(SIGNALS)))
    def test_lempel_ziv_complexity(self, idx):
        x = SIGNALS[idx]
        ours = AdvancedEntropyFeatures.lempel_ziv_complexity(x)
        binary = "".join("1" if v > np.median(x) else "0" for v in x)
        ref = antropy.lziv_complexity(binary, normalize=True)
        assert abs(ours - ref) < 1e-12, f"ours={ours}, antropy={ref}"
