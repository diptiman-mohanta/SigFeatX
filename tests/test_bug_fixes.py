"""
tests/test_bug_fixes.py
=======================
Regression tests for all 8 bugs identified and fixed:

  Fix 1  vmd.py            — mirror extension + ifftshift reconstruction
  Fix 2  features.py       — spectral_flatness uses power, not magnitude
  Fix 3  features.py       — component entropy: probability mass not density
  Fix 4  emd.py            — boundary extension via nbsym mirror
  Fix 5  efd.py            — linspace boundary fill (no float drift)
  Fix 6  features.py       — shannon_entropy renormalised probability
  Fix 7  features.py       — KL divergence epsilon placement
  Fix 8  features.py       — spectral_kurtosis guard max(ss,1e-6)^4

Run:
    pytest tests/test_bug_fixes.py -v
or:
    python tests/test_bug_fixes.py
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from SigFeatX.decompose.vmd import VMD
from SigFeatX.decompose.emd import EMD
from SigFeatX.decompose.efd import EFD
from SigFeatX.features.features import (
    FrequencyDomainFeatures,
    EntropyFeatures,
    DecompositionFeatures,
)


# ── Signal factories ─────────────────────────────────────────────────────────

def _sine(amp=2.0, freq=10.0, fs=1000, dur=2.0):
    n = int(fs * dur)
    t = np.arange(n) / fs
    return amp * np.sin(2.0 * np.pi * freq * t)


def _multisine(fs=1000, dur=2.0):
    n = int(fs * dur)
    t = np.arange(n) / fs
    return (
        2.0 * np.sin(2.0 * np.pi * 10  * t)
        + 1.0 * np.sin(2.0 * np.pi * 50  * t)
        + 0.5 * np.sin(2.0 * np.pi * 120 * t)
    )


# ============================================================================
# Fix 1 — VMD mirror extension + correct amplitude
# ============================================================================

class TestVMDMirrorFix:
    """
    The reference VMD (Carvalho / Zosso MATLAB) uses mirror extension and
    ifftshift for reconstruction. Without the mirror, boundary Gibbs ringing
    corrupts the mode edges. The amplitude is recovered correctly by the
    conjugate-mirror reconstruction.
    """

    def test_modes_finite(self):
        modes = VMD(K=3, max_iter=200).decompose(_multisine())
        assert np.all(np.isfinite(modes))

    def test_reconstruction_energy_ratio(self):
        """sum(modes) should have comparable energy to input."""
        sig   = _multisine()
        modes = VMD(K=3, alpha=2000, max_iter=300).decompose(sig)
        recon = np.sum(modes, axis=0)
        n     = min(len(recon), len(sig))
        ratio = float(np.sum(recon[:n] ** 2)) / (float(np.sum(sig[:n] ** 2)) + 1e-10)
        assert ratio > 0.5, f"Energy ratio {ratio:.3f} < 0.5"

    def test_single_mode_amplitude_correct(self):
        """K=1 mode RMS should be close to A/sqrt(2) for a pure sine."""
        amp   = 3.0
        sig   = _sine(amp=amp, freq=20.0)
        modes = VMD(K=1, alpha=2000, max_iter=500).decompose(sig)
        n     = min(modes.shape[1], len(sig))
        actual_rms   = float(np.sqrt(np.mean(modes[0, :n] ** 2)))
        expected_rms = amp / np.sqrt(2.0)
        assert np.isclose(actual_rms, expected_rms, rtol=0.20), (
            f"VMD RMS {actual_rms:.4f} vs expected {expected_rms:.4f} (rtol=20%)"
        )

    def test_amplitude_scales_linearly(self):
        """Doubling input amplitude should approximately double output RMS."""
        m1 = VMD(K=1, alpha=2000, max_iter=300).decompose(_sine(amp=1.0, freq=15.0))
        m2 = VMD(K=1, alpha=2000, max_iter=300).decompose(_sine(amp=2.0, freq=15.0))
        r1 = float(np.sqrt(np.mean(m1[0] ** 2)))
        r2 = float(np.sqrt(np.mean(m2[0] ** 2)))
        ratio = r2 / (r1 + 1e-10)
        assert np.isclose(ratio, 2.0, rtol=0.15), f"Ratio {ratio:.3f} != 2.0"

    def test_output_length_matches_input(self):
        sig   = _multisine()
        modes = VMD(K=3, max_iter=100).decompose(sig)
        # Mirror extension ensures output length = input (even) length
        assert modes.shape[1] >= len(sig) - 2


# ============================================================================
# Fix 2 — spectral_flatness: power not magnitude
# ============================================================================

class TestSpectralFlatness:

    def test_always_in_0_1(self):
        rng = np.random.default_rng(99)
        for _ in range(20):
            sig = rng.standard_normal(1000)
            sf  = FrequencyDomainFeatures.extract(sig, fs=500)['spectral_flatness']
            assert 0.0 <= sf <= 1.0 + 1e-9, f"spectral_flatness {sf} outside [0,1]"

    def test_white_noise_high(self):
        rng = np.random.default_rng(42)
        sf  = FrequencyDomainFeatures.extract(
            rng.standard_normal(8000), fs=1000)['spectral_flatness']
        assert sf > 0.5, f"White noise flatness {sf:.4f} <= 0.5"

    def test_sine_low(self):
        sf = FrequencyDomainFeatures.extract(
            _sine(amp=1.0, freq=50.0), fs=1000)['spectral_flatness']
        assert sf < 0.3, f"Pure sine flatness {sf:.4f} >= 0.3"

    def test_noise_greater_than_sine(self):
        rng     = np.random.default_rng(7)
        sf_n    = FrequencyDomainFeatures.extract(
            rng.standard_normal(4000), fs=1000)['spectral_flatness']
        sf_s    = FrequencyDomainFeatures.extract(
            _sine(amp=1.0, freq=30.0, dur=4.0), fs=1000)['spectral_flatness']
        assert sf_n > sf_s, f"noise {sf_n:.4f} not > sine {sf_s:.4f}"


# ============================================================================
# Fix 3 — component entropy: probability mass
# ============================================================================

class TestComponentEntropy:

    def test_non_negative(self):
        components = [_sine(amp=1.0, freq=f) for f in [5.0, 20.0, 50.0]]
        feats = DecompositionFeatures.extract_from_components(components, 'c')
        for k, v in feats.items():
            if 'entropy' in k:
                assert v >= 0.0, f"{k} = {v:.6f} is negative"

    def test_amplitude_invariant(self):
        """Shannon entropy must not change when signal is scaled."""
        sig    = _sine(amp=1.0, freq=10.0, dur=2.0)
        e1     = DecompositionFeatures.extract_from_components([sig],        'c')['c_0_entropy']
        e10    = DecompositionFeatures.extract_from_components([sig * 10.0], 'c')['c_0_entropy']
        e100   = DecompositionFeatures.extract_from_components([sig * 100.0],'c')['c_0_entropy']
        assert np.isclose(e1, e10,  atol=0.05), f"e1={e1:.4f}, e10={e10:.4f}"
        assert np.isclose(e1, e100, atol=0.05), f"e1={e1:.4f}, e100={e100:.4f}"

    def test_bounded_above_log2_30(self):
        rng  = np.random.default_rng(0)
        sig  = rng.standard_normal(5000)
        ent  = DecompositionFeatures.extract_from_components([sig], 'c')['c_0_entropy']
        assert ent <= np.log2(30) + 1e-6, f"entropy {ent:.4f} > log2(30)"

    def test_uniform_higher_than_sine(self):
        """Uniform[-1,1] has higher entropy than a deterministic sine over same range."""
        rng     = np.random.default_rng(42)
        sine    = _sine(amp=1.0, freq=10.0, fs=1000, dur=2.0)
        uniform = rng.uniform(-1.0, 1.0, len(sine))
        e_s = DecompositionFeatures.extract_from_components([sine],    'c')['c_0_entropy']
        e_u = DecompositionFeatures.extract_from_components([uniform], 'c')['c_0_entropy']
        assert e_u > e_s, f"uniform {e_u:.4f} not > sine {e_s:.4f}"


# ============================================================================
# Fix 4 — EMD boundary: nbsym mirror extension
# ============================================================================

class TestEMDBoundary:

    def test_reconstruction_exact(self):
        sig  = _multisine()
        emd  = EMD(max_imf=10)
        imfs = emd.decompose(sig)
        recon = np.sum(imfs, axis=0)
        rmse  = float(np.sqrt(np.mean((sig - recon) ** 2)))
        assert rmse < 1e-5, f"EMD reconstruction RMSE {rmse:.2e}"

    def test_no_endpoint_spikes_on_significant_imfs(self):
        """
        Only check IMFs whose interior std > 1% of signal std.
        High-order near-zero IMFs are numerical residuals.
        """
        sig      = _multisine()
        sig_std  = float(np.std(sig))
        emd      = EMD(max_imf=8)
        imfs     = emd.decompose(sig)
        N        = len(sig)
        edge     = max(1, N // 20)

        for idx, imf in enumerate(imfs[:-1]):
            interior_std = float(np.std(imf[edge:-edge]))
            if interior_std < 0.01 * sig_std:
                continue
            left_max  = float(np.max(np.abs(imf[:edge])))
            right_max = float(np.max(np.abs(imf[-edge:])))
            ratio_l   = left_max  / (interior_std + 1e-10)
            ratio_r   = right_max / (interior_std + 1e-10)
            assert ratio_l < 8.0, \
                f"IMF {idx} left spike {ratio_l:.2f}x interior std"
            assert ratio_r < 8.0, \
                f"IMF {idx} right spike {ratio_r:.2f}x interior std"

    def test_imfs_all_finite(self):
        imfs = EMD(max_imf=6).decompose(_multisine())
        for i, imf in enumerate(imfs):
            assert np.all(np.isfinite(imf)), f"IMF {i} non-finite"


# ============================================================================
# Fix 5 — EFD boundary linspace
# ============================================================================

class TestEFDBoundary:

    def test_correct_mode_count(self):
        sig = _multisine()
        for n in [3, 5, 7]:
            assert EFD(n_modes=n).decompose(sig).shape[0] == n

    def test_all_modes_finite(self):
        assert np.all(np.isfinite(EFD(n_modes=5).decompose(_multisine())))

    def test_reconstruction_reasonable(self):
        sig   = _multisine()
        efd   = EFD(n_modes=5)
        rmse  = float(
            np.sqrt(np.mean((sig - efd.reconstruct(efd.decompose(sig))) ** 2))
            / (np.std(sig) + 1e-10)
        )
        assert rmse < 0.15, f"EFD RMSE {rmse:.4f}"

    def test_significant_modes_at_least_3_of_5(self):
        """
        Multisine has 3 frequency components; at least 3 of 5 EFD modes
        should carry meaningful energy (> 1% of max mode energy).
        """
        sig    = _multisine()
        modes  = EFD(n_modes=5).decompose(sig)
        rms    = [float(np.sqrt(np.mean(m ** 2))) for m in modes]
        max_r  = max(rms)
        n_sig  = sum(r > 0.01 * max_r for r in rms)
        assert n_sig >= 3, \
            f"Only {n_sig}/5 significant modes; rms={[f'{r:.4f}' for r in rms]}"


# ============================================================================
# Fix 6 — shannon_entropy renormalisation
# ============================================================================

class TestShannonEntropy:

    def test_non_negative(self):
        rng = np.random.default_rng(1)
        for _ in range(30):
            assert EntropyFeatures.extract(rng.standard_normal(500))['shannon_entropy'] >= 0.0

    def test_constant_zero(self):
        assert EntropyFeatures.extract(np.full(200, 5.0))['shannon_entropy'] == 0.0

    def test_bounded_above(self):
        rng = np.random.default_rng(42)
        ent = EntropyFeatures.extract(rng.standard_normal(5000))['shannon_entropy']
        assert ent <= np.log2(50) + 1e-6

    def test_uniform_higher_than_gaussian(self):
        rng  = np.random.default_rng(0)
        e_g  = EntropyFeatures.extract(rng.standard_normal(5000))['shannon_entropy']
        e_u  = EntropyFeatures.extract(rng.uniform(-3, 3, 5000))['shannon_entropy']
        assert e_u > e_g, f"uniform {e_u:.4f} not > gaussian {e_g:.4f}"


# ============================================================================
# Fix 7 — KL divergence epsilon
# ============================================================================

class TestKLDivergence:

    def test_non_negative(self):
        rng = np.random.default_rng(5)
        for _ in range(20):
            p  = rng.standard_normal(500)
            q  = rng.standard_normal(500) + 0.5
            assert DecompositionFeatures._kl_divergence(p, q) >= 0.0

    def test_identical_near_zero(self):
        rng = np.random.default_rng(10)
        sig = rng.standard_normal(1000)
        kl  = DecompositionFeatures._kl_divergence(sig, sig.copy())
        assert kl < 1e-6, f"KL(P||P) = {kl:.2e}, expected ≈ 0"

    def test_increases_with_divergence(self):
        rng     = np.random.default_rng(7)
        p       = rng.standard_normal(500)
        q_close = rng.standard_normal(500) + 0.1
        q_far   = rng.standard_normal(500) + 3.0
        assert (DecompositionFeatures._kl_divergence(p, q_far) >
                DecompositionFeatures._kl_divergence(p, q_close))


# ============================================================================
# Fix 8 — spectral_kurtosis guard
# ============================================================================

class TestSpectralKurtosisGuard:

    def test_finite_on_dc(self):
        rng = np.random.default_rng(0)
        sig = np.full(1000, 5.0) + 1e-4 * rng.standard_normal(1000)
        fd  = FrequencyDomainFeatures.extract(sig, fs=1000)
        assert np.isfinite(fd['spectral_kurtosis'])
        assert fd['spectral_kurtosis'] < 1e12

    def test_finite_on_all_types(self):
        rng = np.random.default_rng(42)
        sigs = {
            'noise'  : rng.standard_normal(2000),
            'sine'   : _sine(freq=50.0),
            'dc'     : np.ones(1000) + 1e-5 * rng.standard_normal(1000),
            'multi'  : _multisine(),
        }
        for name, sig in sigs.items():
            fd = FrequencyDomainFeatures.extract(sig, fs=1000)
            assert np.isfinite(fd['spectral_kurtosis']), \
                f"spectral_kurtosis not finite for {name}"


# ============================================================================
# Standalone runner
# ============================================================================

if __name__ == '__main__':
    import traceback

    classes = [
        TestVMDMirrorFix,
        TestSpectralFlatness,
        TestComponentEntropy,
        TestEMDBoundary,
        TestEFDBoundary,
        TestShannonEntropy,
        TestKLDivergence,
        TestSpectralKurtosisGuard,
    ]
    passed = failed = 0

    for cls in classes:
        inst = cls()
        print(f"\n{'─'*60}\n  {cls.__name__}\n{'─'*60}")
        for name in sorted(m for m in dir(inst) if m.startswith('test_')):
            try:
                getattr(inst, name)()
                print(f"  ✓  {name}")
                passed += 1
            except Exception as exc:
                print(f"  ✗  {name}: {exc}")
                traceback.print_exc()
                failed += 1

    print(f"\n{'='*60}\n  {passed} passed, {failed} failed\n{'='*60}")
    sys.exit(0 if failed == 0 else 1)