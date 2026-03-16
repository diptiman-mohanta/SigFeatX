import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from SigFeatX.decompose import EFD, EMD, SVMD, VMD, WaveletDecomposer


def test_emd_constant_signal_returns_single_residual():
    """Constant signals should fall back to a single residual component."""
    sig = np.full(256, 3.5)
    emd = EMD(max_imf=5)

    imfs = emd.decompose(sig)

    assert len(imfs) == 1
    assert np.allclose(imfs[0], sig)


def test_svmd_zero_signal_returns_single_zero_mode():
    """Zero-energy input should not expand into many redundant zero modes."""
    sig = np.zeros(256)
    svmd = SVMD(K_max=5)

    modes = svmd.decompose(sig)

    assert modes.shape == (1, len(sig))
    assert np.allclose(modes[0], 0.0)


def test_wavelet_cwt_short_signal_uses_default_scale_safely():
    """Very short signals should still produce a valid default CWT output."""
    sig = np.array([1.0])
    wavelet = WaveletDecomposer(wavelet='db4')

    with pytest.warns(RuntimeWarning, match='falling back to'):
        coeffs = wavelet.cwt(sig)

    assert coeffs.shape[1] == len(sig)
    assert coeffs.shape[0] >= 1


def test_vmd_rejects_invalid_mode_count():
    with pytest.raises(ValueError, match='K must be >= 1'):
        VMD(K=0)


def test_efd_rejects_invalid_mode_count():
    with pytest.raises(ValueError, match='n_modes must be >= 1'):
        EFD(n_modes=0)


def test_wavelet_rejects_invalid_level():
    sig = np.sin(np.linspace(0.0, 2.0 * np.pi, 32))
    wavelet = WaveletDecomposer()

    with pytest.raises(ValueError, match='level must be >= 1'):
        wavelet.dwt(sig, level=0)
