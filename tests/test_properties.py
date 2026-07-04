"""
Property-based tests (Hypothesis).

These complement the fixed-example unit tests elsewhere: instead of
checking a handful of hand-picked signals, they generate many random
signals and check invariants that must hold for *any* valid input --
perfect-reconstruction identities and value bounds. Fixed toy-signal
tests can pass even when a normalisation constant is wrong (as happened
with RQA's DET, which was silently deflated 2x); property tests widen
the input space these mistakes have to survive.
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from SigFeatX.decompose.emd import EMD
from SigFeatX.decompose.modwt import MODWT
from SigFeatX.features.advanced_entropy import AdvancedEntropyFeatures
from SigFeatX.features.mfdfa import MFDFAFeatures
from SigFeatX.features.rqa import RQAFeatures

_finite_signal = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=64, max_value=256),
    elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False, width=64),
)


@given(
    sig=_finite_signal,
    wavelet=st.sampled_from(['db2', 'db4', 'sym5', 'coif3']),
    level=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=30, deadline=None)
def test_modwt_perfect_reconstruction(sig, wavelet, level):
    m = MODWT(wavelet=wavelet, level=level)
    coeffs = m.decompose(sig)
    rec = m.reconstruct(coeffs)
    assert rec.shape == sig.shape
    assert np.max(np.abs(rec - sig)) < 1e-6


@given(sig=_finite_signal)
@settings(max_examples=25, deadline=None)
def test_emd_perfect_reconstruction_and_finite(sig):
    emd = EMD(max_imf=6)
    imfs = emd.decompose(sig)
    assert all(len(m) == len(sig) for m in imfs)
    assert all(np.all(np.isfinite(m)) for m in imfs)
    rec = emd.reconstruct(imfs)
    assert np.max(np.abs(rec - sig)) < 1e-6


@given(
    sig=_finite_signal,
    m=st.integers(min_value=2, max_value=4),
    tau=st.integers(min_value=1, max_value=2),
)
@settings(max_examples=30, deadline=None)
def test_rqa_bounds(sig, m, tau):
    if len(sig) <= (m - 1) * tau:
        return
    feats = RQAFeatures.extract(sig, m=m, tau=tau)
    assert all(np.isfinite(v) for v in feats.values())
    assert 0.0 <= feats['rqa_rr'] <= 1.0
    assert 0.0 <= feats['rqa_det'] <= 1.0 + 1e-9
    assert 0.0 <= feats['rqa_lam'] <= 1.0 + 1e-9
    assert feats['rqa_eps'] >= 0.0
    assert feats['rqa_l_max'] >= 0.0
    assert feats['rqa_v_max'] >= 0.0


@given(sig=_finite_signal)
@settings(max_examples=20, deadline=None)
def test_mfdfa_finite_and_nonneg_width(sig):
    feats = MFDFAFeatures.extract(sig)
    assert all(np.isfinite(v) for v in feats.values())
    assert feats['mfdfa_width'] >= 0.0


@given(sig=_finite_signal)
@settings(max_examples=20, deadline=None)
def test_advanced_entropy_finite_and_bounded(sig):
    feats = AdvancedEntropyFeatures.extract(sig)
    assert all(np.isfinite(v) for v in feats.values())
    assert 0.0 <= feats['dispersion_entropy'] <= 1.0 + 1e-9
