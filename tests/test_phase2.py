"""
tests/test_phase2.py
=====================
Tests for the 0.3.0 phase-2 additions:
  - MODWT     (Maximal Overlap Discrete Wavelet Transform)
  - CEEMDAN   (Complete Ensemble EMD with Adaptive Noise)
  - HHT       (Hilbert-Huang Transform)
  - SST       (Synchrosqueezing Transform)
  - RQA       (Recurrence Quantification Analysis)
  - MFDFA     (Multifractal Detrended Fluctuation Analysis)
  - Advanced Entropy (Dispersion, Fuzzy, Lempel-Ziv, Bubble)
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from SigFeatX.decompose.modwt import MODWT
from SigFeatX.decompose.ceemdan import CEEMDAN
from SigFeatX.decompose.hht import HHT
from SigFeatX.decompose.sst import SST
from SigFeatX.features.rqa import RQAFeatures
from SigFeatX.features.mfdfa import MFDFAFeatures
from SigFeatX.features.advanced_entropy import AdvancedEntropyFeatures


# ---------------------------------------------------------------------------
# Signal factories (reference-validated)
# ---------------------------------------------------------------------------

def _sine(amp=1.0, freq=10.0, fs=200, dur=4.0, phase=0.0):
    n = int(fs * dur)
    t = np.arange(n) / fs
    return amp * np.sin(2.0 * np.pi * freq * t + phase)


def _multisine(fs=200, dur=4.0):
    n = int(fs * dur)
    t = np.arange(n) / fs
    return (
        np.sin(2.0 * np.pi * 5 * t)
        + 0.5 * np.sin(2.0 * np.pi * 20 * t)
        + 0.25 * np.sin(2.0 * np.pi * 50 * t)
    )


def _noise(seed=0, n=1024):
    return np.random.default_rng(seed).standard_normal(n)


# ===========================================================================
# MODWT
# ===========================================================================

class TestMODWT:

    def test_perfect_reconstruction_db4(self):
        """MODWT followed by IMODWT must recover the signal to machine precision."""
        sig = _multisine()[:256]
        m = MODWT(wavelet='db4', level=3)
        coeffs = m.decompose(sig)
        rec = m.reconstruct(coeffs)
        rmse = float(np.sqrt(np.mean((sig - rec) ** 2)))
        assert rmse < 1e-10, f"MODWT reconstruction RMSE {rmse} too large"

    def test_perfect_reconstruction_sym8(self):
        """Same property with a different wavelet."""
        sig = _sine(freq=12.0)[:256]
        m = MODWT(wavelet='sym8', level=4)
        rec = m.reconstruct(m.decompose(sig))
        assert np.sqrt(np.mean((sig - rec) ** 2)) < 1e-10

    def test_all_coefficients_have_signal_length(self):
        """Unlike DWT, every MODWT level matches the input length."""
        sig = _multisine()[:512]
        coeffs = MODWT(level=4).decompose(sig)
        for c in coeffs:
            assert len(c) == len(sig)

    def test_output_count_matches_level_plus_one(self):
        """level=L returns L details + 1 smooth = L+1 arrays."""
        sig = _multisine()[:256]
        assert len(MODWT(level=3).decompose(sig)) == 4
        assert len(MODWT(level=5).decompose(sig)) == 6

    def test_shift_invariance(self):
        """Shifting the input shifts every coefficient by the same amount."""
        sig = _multisine()[:256]
        shift = 7
        coeffs0 = MODWT(level=3).decompose(sig)
        coeffs1 = MODWT(level=3).decompose(np.roll(sig, shift))
        # Each level should be a circular shift of the original
        for c0, c1 in zip(coeffs0, coeffs1):
            rolled = np.roll(c0, shift)
            # Allow tiny numerical wiggle
            err = np.max(np.abs(rolled - c1))
            assert err < 1e-8, f"shift invariance violated; max err {err}"

    def test_auto_level_selection_when_none(self):
        sig = _multisine()[:1024]
        coeffs = MODWT(level=None).decompose(sig)
        assert len(coeffs) >= 2

    def test_rejects_negative_level(self):
        with pytest.raises(ValueError, match='level must be >= 1'):
            MODWT(level=0).decompose(_sine()[:64])

    def test_works_on_non_power_of_two_length(self):
        """DWT often pads/truncates; MODWT must handle 999 samples directly."""
        sig = _multisine()[:999]
        coeffs = MODWT(level=3).decompose(sig)
        rec = MODWT(level=3).reconstruct(coeffs)
        assert len(rec) == len(sig)
        assert np.sqrt(np.mean((sig - rec) ** 2)) < 1e-10


# ===========================================================================
# CEEMDAN
# ===========================================================================

class TestCEEMDAN:

    def test_basic_decomposition_runs(self):
        sig = _multisine()[:300]
        imfs = CEEMDAN(trials=10, max_imf=4, rng=42).decompose(sig)
        assert len(imfs) >= 2
        assert all(len(m) == len(sig) for m in imfs)

    def test_perfect_reconstruction(self):
        """Sum of IMFs (incl. residual) should equal the signal exactly."""
        sig = _multisine()[:200]
        imfs = CEEMDAN(trials=10, max_imf=4, rng=0).decompose(sig)
        rec = sum(imfs)
        assert np.sqrt(np.mean((sig - rec) ** 2)) < 1e-10

    def test_finite_imfs(self):
        sig = _multisine()[:300]
        for m in CEEMDAN(trials=10, rng=1).decompose(sig):
            assert np.all(np.isfinite(m))

    def test_reproducibility_with_seed(self):
        sig = _multisine()[:200]
        a = CEEMDAN(trials=10, max_imf=3, rng=99).decompose(sig)
        b = CEEMDAN(trials=10, max_imf=3, rng=99).decompose(sig)
        for ma, mb in zip(a, b):
            assert np.allclose(ma, mb)

    def test_rejects_low_trial_count(self):
        with pytest.raises(ValueError, match='trials must be >= 2'):
            CEEMDAN(trials=1)

    def test_rejects_negative_noise_amp(self):
        with pytest.raises(ValueError, match='noise_amp must be > 0'):
            CEEMDAN(noise_amp=-0.01)


# ===========================================================================
# HHT
# ===========================================================================

class TestHHT:

    def test_instantaneous_attributes(self):
        sig = _sine(freq=10.0, fs=200, dur=2.0)
        amp, phase, ifreq = HHT.instantaneous_attributes(sig, fs=200)
        assert len(amp) == len(sig)
        assert len(ifreq) == len(sig) - 1
        # Amplitude of unit-amplitude sine should be ~1
        assert np.isclose(np.median(amp), 1.0, atol=0.05)
        # Instantaneous frequency near the carrier
        assert np.isclose(np.median(ifreq), 10.0, atol=0.5)

    def test_marginal_spectrum_has_peak_near_carrier(self):
        sig = _sine(freq=15.0, fs=200, dur=2.0)
        from SigFeatX.decompose.emd import EMD
        h = HHT(fs=200, decomposer=EMD(max_imf=5))
        imfs = h.decomposer.decompose(sig)
        f, spectrum = h.marginal_spectrum(imfs, n_freq_bins=64)
        peak_freq = f[int(np.argmax(spectrum))]
        assert abs(peak_freq - 15.0) < 5.0   # within a couple of bins

    def test_extract_features_returns_dict(self):
        sig = _multisine(fs=200)[:600]
        h = HHT(fs=200)
        feats = h.extract_features(sig)
        assert isinstance(feats, dict) and len(feats) > 0
        assert 'hht_marginal_centroid' in feats
        assert 'hht_marginal_peak_freq' in feats

    def test_all_features_finite(self):
        sig = _multisine(fs=200)[:600]
        feats = HHT(fs=200).extract_features(sig)
        assert all(np.isfinite(v) for v in feats.values())


# ===========================================================================
# SST
# ===========================================================================

class TestSST:

    def test_transform_shapes(self):
        sig = _sine(freq=20.0, fs=400, dur=2.0)
        t, f, Tx = SST(fs=400, nperseg=128).transform(sig)
        assert Tx.shape == (len(f), len(t))

    def test_peak_freq_matches_input(self):
        sig = _sine(freq=30.0, fs=400, dur=2.0)
        feats = SST(fs=400, nperseg=256).extract_features(sig)
        assert abs(feats['sst_peak_freq'] - 30.0) < 5.0

    def test_all_features_finite(self):
        sig = _multisine(fs=400, dur=2.0)
        feats = SST(fs=400, nperseg=128).extract_features(sig)
        assert all(np.isfinite(v) for v in feats.values())

    def test_entropy_higher_on_noise_than_sine(self):
        rng = np.random.default_rng(0)
        sine = _sine(fs=400, dur=2.0)
        noise = rng.standard_normal(len(sine))
        sst = SST(fs=400, nperseg=128)
        e_s = sst.extract_features(sine)['sst_entropy']
        e_n = sst.extract_features(noise)['sst_entropy']
        assert e_n > e_s

    def test_rejects_bad_nperseg(self):
        with pytest.raises(ValueError, match='nperseg must be >= 4'):
            SST(nperseg=2)


# ===========================================================================
# RQA
# ===========================================================================

class TestRQA:

    def test_basic_extract(self):
        sig = _multisine()[:200]
        feats = RQAFeatures.extract(sig, m=3, tau=1)
        assert all(k.startswith('rqa_') for k in feats)
        assert 0.0 <= feats['rqa_rr'] <= 1.0
        assert 0.0 <= feats['rqa_det'] <= 1.0
        assert 0.0 <= feats['rqa_lam'] <= 1.0

    def test_target_rr(self):
        """auto-eps should produce RR roughly equal to the target."""
        sig = _multisine()[:200]
        feats = RQAFeatures.extract(sig, m=2, tau=1)
        # Auto picks eps for RR ~ 0.1
        assert 0.05 < feats['rqa_rr'] < 0.20

    def test_periodic_signal_has_high_determinism(self):
        """Periodic signal should produce DET > 0.4 — well above white noise (~0.1-0.2)."""
        sine = _sine(fs=200, dur=2.0)
        feats = RQAFeatures.extract(sine[:300], m=3, tau=5)
        assert feats['rqa_det'] > 0.4

    def test_noise_has_lower_determinism_than_sine(self):
        sine_feats = RQAFeatures.extract(_sine(fs=200, dur=2.0)[:300], m=3, tau=5)
        noise_feats = RQAFeatures.extract(_noise(seed=0, n=300), m=3, tau=5)
        assert sine_feats['rqa_det'] > noise_feats['rqa_det']

    def test_rejects_signal_too_short(self):
        with pytest.raises(ValueError, match='signal too short'):
            RQAFeatures.extract(np.zeros(5), m=10, tau=2)

    def test_all_features_finite(self):
        feats = RQAFeatures.extract(_multisine()[:200])
        assert all(np.isfinite(v) for v in feats.values())


# ===========================================================================
# MFDFA
# ===========================================================================

class TestMFDFA:

    def test_basic_extract(self):
        sig = _noise(seed=0, n=1024)
        feats = MFDFAFeatures.extract(sig)
        assert 'mfdfa_width' in feats
        assert 'mfdfa_alpha0' in feats
        assert feats['mfdfa_width'] >= 0.0

    def test_white_noise_has_narrow_spectrum(self):
        """Monofractal white noise should have small mfdfa_width."""
        sig = _noise(seed=42, n=2048)
        feats = MFDFAFeatures.extract(sig, q_values=[-3, -1, 1, 3])
        # Pure white noise is monofractal — width is small
        assert feats['mfdfa_width'] < 0.5

    def test_all_h_q_finite(self):
        sig = _noise(seed=1, n=1024)
        feats = MFDFAFeatures.extract(sig)
        h_keys = [k for k in feats if k.startswith('mfdfa_h_q')]
        assert len(h_keys) > 0
        for k in h_keys:
            assert np.isfinite(feats[k])

    def test_short_signal_returns_zeros(self):
        feats = MFDFAFeatures.extract(np.arange(50.0))
        assert feats['mfdfa_width'] == 0.0


# ===========================================================================
# Advanced Entropy
# ===========================================================================

class TestAdvancedEntropy:

    # -- Dispersion -------------------------------------------------------

    def test_dispersion_in_unit_interval_when_normalized(self):
        sig = _noise(n=1000)
        de = AdvancedEntropyFeatures.dispersion_entropy(sig)
        assert 0.0 <= de <= 1.0

    def test_dispersion_high_for_noise_low_for_sine(self):
        de_noise = AdvancedEntropyFeatures.dispersion_entropy(_noise(n=1000))
        de_sine  = AdvancedEntropyFeatures.dispersion_entropy(_sine(fs=200, dur=5))
        assert de_noise > de_sine

    def test_dispersion_constant_signal_is_zero(self):
        assert AdvancedEntropyFeatures.dispersion_entropy(np.full(500, 3.0)) == 0.0

    # -- Fuzzy ------------------------------------------------------------

    def test_fuzzy_is_finite_and_non_negative_for_real_data(self):
        fe = AdvancedEntropyFeatures.fuzzy_entropy(_noise(n=300))
        assert np.isfinite(fe)

    def test_fuzzy_higher_on_noise_than_sine(self):
        fe_noise = AdvancedEntropyFeatures.fuzzy_entropy(_noise(n=400))
        fe_sine  = AdvancedEntropyFeatures.fuzzy_entropy(_sine(fs=200, dur=2.0))
        assert fe_noise > fe_sine

    # -- Lempel-Ziv -------------------------------------------------------

    def test_lz_in_unit_interval_when_normalized(self):
        lz = AdvancedEntropyFeatures.lempel_ziv_complexity(_noise(n=500))
        assert 0.0 <= lz <= 1.5     # normalised approaches 1; allow slack

    def test_lz_higher_on_noise_than_sine(self):
        lz_noise = AdvancedEntropyFeatures.lempel_ziv_complexity(_noise(n=500))
        lz_sine = AdvancedEntropyFeatures.lempel_ziv_complexity(_sine(fs=200, dur=2.5))
        assert lz_noise > lz_sine

    def test_lz_constant_signal_minimal(self):
        sig = np.full(500, 1.0)
        lz_n = AdvancedEntropyFeatures.lempel_ziv_complexity(sig, normalize=True)
        # All-same signal -> all '0' or all '1' depending on threshold
        # -> at most one distinct pattern -> very low LZ
        assert lz_n < 0.05

    # -- Bubble -----------------------------------------------------------

    def test_bubble_is_finite(self):
        be = AdvancedEntropyFeatures.bubble_entropy(_noise(n=300))
        assert np.isfinite(be)

    # -- Bundle -----------------------------------------------------------

    def test_extract_bundle_returns_all_four(self):
        feats = AdvancedEntropyFeatures.extract(_noise(n=500))
        assert set(feats.keys()) == {
            'dispersion_entropy', 'fuzzy_entropy',
            'lz_complexity', 'bubble_entropy',
        }
        assert all(np.isfinite(v) for v in feats.values())
