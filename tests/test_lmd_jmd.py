"""
tests/test_lmd_jmd.py
=====================
Tests for LMD and JMD decomposition methods.

Run:
    pytest tests/test_lmd_jmd.py -v
or:
    python tests/test_lmd_jmd.py
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from SigFeatX.decompose.lmd import LMD
from SigFeatX.decompose.jmd import JMD


# ── Signal factories ─────────────────────────────────────────────────────────

def _sine(amp=1.0, freq=10.0, fs=500, dur=2.0):
    n = int(fs * dur)
    t = np.arange(n) / fs
    return amp * np.sin(2.0 * np.pi * freq * t)


def _multisine(fs=500, dur=2.0):
    n = int(fs * dur)
    t = np.arange(n) / fs
    return (
        np.sin(2.0 * np.pi * 10 * t)
        + 0.6 * np.sin(2.0 * np.pi * 40 * t)
        + 0.3 * np.sin(2.0 * np.pi * 100 * t)
    )


def _jumpy(fs=500, dur=2.0):
    n    = int(fs * dur)
    t    = np.arange(n) / fs
    osc  = np.sin(2.0 * np.pi * 10 * t) + 0.5 * np.sin(2.0 * np.pi * 30 * t)
    jump = np.zeros(n)
    mid  = n // 2
    jump[mid: mid + n // 10] = 2.0
    return osc + jump, jump


# ============================================================================
# LMD
# ============================================================================

class TestLMD:

    def test_returns_array(self):
        result = LMD(max_pf=4).fit_transform(_multisine())
        assert isinstance(result, np.ndarray) and result.ndim == 2

    def test_decompose_returns_list_same_length(self):
        sig = _multisine()
        pfs = LMD(max_pf=6).decompose(sig)
        assert isinstance(pfs, list) and len(pfs) >= 1
        for i, pf in enumerate(pfs):
            assert len(pf) == len(sig), f"PF {i} length mismatch"

    def test_reconstruction(self):
        sig = _multisine()
        lmd = LMD(max_pf=8)
        pfs = lmd.decompose(sig)
        rec  = lmd.reconstruct(pfs)
        rmse = float(np.sqrt(np.mean((sig - rec) ** 2)) / (np.std(sig) + 1e-10))
        assert rmse < 0.02, f"RMSE {rmse:.4f} > 0.02"

    def test_all_pfs_finite(self):
        sig = _multisine()
        for i, pf in enumerate(LMD(max_pf=6).decompose(sig)):
            assert np.all(np.isfinite(pf)), f"PF {i} non-finite"

    def test_constant_signal(self):
        sig = np.full(256, 3.0)
        pfs = LMD(max_pf=5).decompose(sig)
        assert len(pfs) == 1 and np.allclose(pfs[0], sig)

    def test_short_signal(self):
        sig = np.sin(np.linspace(0, 4 * np.pi, 50))
        pfs = LMD(max_pf=3).decompose(sig)
        assert len(pfs) >= 1
        assert all(np.all(np.isfinite(p)) for p in pfs)

    def test_max_pf_respected(self):
        lmd = LMD(max_pf=3)
        assert len(lmd.decompose(_multisine())) <= lmd.max_pf + 1

    def test_multisine_multiple_components(self):
        assert len(LMD(max_pf=8).decompose(_multisine())) >= 2

    def test_noisy_more_pfs(self):
        rng   = np.random.default_rng(42)
        pure  = _sine(freq=10.0)
        noisy = pure + 0.5 * rng.standard_normal(len(pure))
        lmd   = LMD(max_pf=8)
        assert len(lmd.decompose(noisy)) >= len(lmd.decompose(pure))

    def test_rejects_max_pf_zero(self):
        with pytest.raises(ValueError, match='max_pf must be >= 1'):
            LMD(max_pf=0)

    def test_rejects_bad_envelope_epsilon(self):
        with pytest.raises(ValueError):
            LMD(envelope_epsilon=-0.1)

    def test_fit_transform_shape(self):
        sig    = _multisine()
        result = LMD(max_pf=4).fit_transform(sig)
        assert result.shape[1] == len(sig)

    def test_white_noise_finite(self):
        rng = np.random.default_rng(0)
        sig = rng.standard_normal(1000)
        for pf in LMD(max_pf=6).decompose(sig):
            assert np.all(np.isfinite(pf))

    def test_endpoints_false(self):
        sig = _multisine()
        pfs = LMD(max_pf=5, endpoints=False).decompose(sig)
        assert all(np.all(np.isfinite(p)) for p in pfs)


# ============================================================================
# JMD
# ============================================================================

class TestJMD:

    def test_returns_modes_and_jump(self):
        sig, _ = _jumpy()
        modes, jump = JMD(K=3).decompose(sig)
        assert isinstance(modes, np.ndarray) and isinstance(jump, np.ndarray)

    def test_modes_shape(self):
        sig, _ = _jumpy()
        K = 3
        modes, jump = JMD(K=K).decompose(sig)
        assert modes.shape[0] == K
        # Mirror extension may shorten output by up to 2 samples
        assert modes.shape[1] >= len(sig) - 2
        assert jump.shape[0] >= len(sig) - 2

    def test_all_finite(self):
        sig, _ = _jumpy()
        modes, jump = JMD(K=3, max_iter=500).decompose(sig)
        assert np.all(np.isfinite(modes)) and np.all(np.isfinite(jump))

    def test_reconstruction(self):
        sig, _ = _jumpy()
        jmd = JMD(K=3, max_iter=500)
        modes, jump = jmd.decompose(sig)
        n   = min(modes.shape[1], len(sig))
        rec = jmd.reconstruct(modes, jump)
        rmse = float(
            np.sqrt(np.mean((sig[:n] - rec[:n]) ** 2))
            / (np.std(sig[:n]) + 1e-10)
        )
        assert rmse < 0.25, f"RMSE {rmse:.4f} > 0.25"

    def test_jump_energy_positive(self):
        sig, _ = _jumpy()
        _, jump = JMD(K=2, max_iter=500).decompose(sig)
        assert float(np.sum(jump ** 2)) > 0.0

    def test_jump_variance_less_than_signal(self):
        sig = _multisine()
        modes, jump = JMD(K=3, beta=0.03, max_iter=500).decompose(sig)
        n = min(len(jump), len(sig))
        assert float(np.var(jump[:n])) < float(np.var(sig[:n])), (
            f"jump var {np.var(jump[:n]):.4f} >= signal var {np.var(sig[:n]):.4f}"
        )

    def test_rejects_K_zero(self):
        with pytest.raises(ValueError, match='K must be >= 1'):
            JMD(K=0)

    def test_rejects_negative_alpha(self):
        with pytest.raises(ValueError, match='alpha must be > 0'):
            JMD(K=2, alpha=-1.0)

    def test_rejects_negative_beta(self):
        with pytest.raises(ValueError, match='beta must be >= 0'):
            JMD(K=2, beta=-0.1)

    def test_rejects_zero_b_bar(self):
        with pytest.raises(ValueError, match='b_bar must be > 0'):
            JMD(K=2, b_bar=0.0)

    def test_rejects_bad_init(self):
        with pytest.raises(ValueError, match='init must be'):
            JMD(K=2, init='bad')

    def test_k1_shape(self):
        sig, _ = _jumpy()
        modes, jump = JMD(K=1, max_iter=200).decompose(sig)
        assert modes.shape[0] == 1 and jump.ndim == 1

    def test_random_init_finite(self):
        np.random.seed(99)
        sig, _ = _jumpy()
        modes, jump = JMD(K=3, init='random', max_iter=200).decompose(sig)
        assert np.all(np.isfinite(modes)) and np.all(np.isfinite(jump))

    def test_uniform_init_finite(self):
        sig, _ = _jumpy()
        modes, jump = JMD(K=3, init='uniform', max_iter=200).decompose(sig)
        assert np.all(np.isfinite(modes)) and np.all(np.isfinite(jump))

    def test_fit_transform_return_all_false(self):
        sig, _ = _jumpy()
        result = JMD(K=2, max_iter=100).fit_transform(sig, return_all=False)
        assert isinstance(result, np.ndarray) and result.ndim == 2
        assert result.shape[0] == 2

    def test_fit_transform_return_all_true(self):
        sig, _ = _jumpy()
        u, v, omega = JMD(K=2, max_iter=100).fit_transform(sig, return_all=True)
        assert u.shape[0] == 2 and omega.shape[0] == 2 and v.ndim == 1


# ============================================================================
# Boundary / numerical stability
# ============================================================================

class TestBoundaryHandling:

    def test_lmd_endpoints_finite(self):
        rng = np.random.default_rng(7)
        sig = rng.standard_normal(512)
        for pf in LMD(max_pf=5, endpoints=True).decompose(sig):
            assert np.isfinite(pf[0]) and np.isfinite(pf[-1])

    def test_jmd_boundary_finite(self):
        sig, _ = _jumpy()
        modes, jump = JMD(K=3, max_iter=200).decompose(sig)
        assert np.isfinite(modes[:, 0]).all() and np.isfinite(modes[:, -1]).all()
        assert np.isfinite(jump[0]) and np.isfinite(jump[-1])


# ============================================================================
# Aggregator integration
# ============================================================================

class TestAggregatorIntegration:

    def test_aggregator_lmd(self):
        from SigFeatX import FeatureAggregator
        agg  = FeatureAggregator(fs=500)
        feat = agg.extract_all_features(
            _multisine(), decomposition_methods=['lmd'],
            preprocess_signal=False, validate=False, check_consistency=False)
        assert any(k.startswith('lmd_') for k in feat)

    def test_aggregator_jmd(self):
        from SigFeatX import FeatureAggregator
        sig, _ = _jumpy()
        agg    = FeatureAggregator(fs=500)
        feat   = agg.extract_all_features(
            sig, decomposition_methods=['jmd'],
            preprocess_signal=False, validate=False, check_consistency=False)
        assert any(k.startswith('jmd_') for k in feat)

    def test_aggregator_lmd_batch(self):
        from SigFeatX import FeatureAggregator
        signals = [_sine(freq=f) for f in [5.0, 10.0, 20.0]]
        agg     = FeatureAggregator(fs=500)
        result  = agg.extract_batch(
            signals, decomposition_methods=['lmd'],
            preprocess_signal=False, validate=False, n_jobs=1)
        assert result.n_success == 3 and result.n_failed == 0


# ============================================================================
# Standalone runner
# ============================================================================

if __name__ == '__main__':
    import traceback

    test_classes = [TestLMD, TestJMD, TestBoundaryHandling]
    passed = failed = 0

    for cls in test_classes:
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