"""
Cross-validation against the reference PyEMD implementation.

Validates SigFeatX's EMD against the widely used ``EMD-signal`` package
(github.com/laszukdawid/PyEMD) — the de-facto reference implementation —
rather than only against our own expectations.

EMD implementations legitimately differ in boundary handling and stopping
criteria, so IMFs are not expected to match sample-for-sample. What *must*
agree, and what these tests check on a clean well-separated two-tone
signal:

  1. both split the signal into the same modes (IMF-to-IMF correlation),
  2. both recover the known ground-truth tones (we constructed the
     signal, so the true components are available exactly),
  3. both reconstruct the input to floating-point precision.

Thresholds are set with margin below measured values (e.g. IMF1-vs-IMF1
correlation measured at 1.0000, asserted > 0.95) so the test is robust to
reference-library version drift without being vacuous.

Requires ``EMD-signal`` (skipped if not installed — note it cannot
coexist with the unrelated ``pyemd`` package on case-insensitive
filesystems, since ``PyEMD/`` and ``pyemd/`` collide there).
"""

import numpy as np
import pytest

PyEMD = pytest.importorskip("PyEMD")

from SigFeatX.decompose.emd import EMD as SigFeatXEMD  # noqa: E402, N811

FS = 200.0
T = np.arange(0, 4, 1 / FS)
HI_TRUE = 0.5 * np.sin(2 * np.pi * 25.0 * T)
LO_TRUE = 1.0 * np.sin(2 * np.pi * 3.0 * T)
SIGNAL = HI_TRUE + LO_TRUE


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    return abs(float(np.corrcoef(a[:n], b[:n])[0, 1]))


@pytest.fixture(scope="module")
def decompositions():
    ours = SigFeatXEMD(max_imf=-1).decompose(SIGNAL)
    ref = PyEMD.EMD()(SIGNAL)
    return ours, ref


def test_comparable_imf_count(decompositions):
    ours, ref = decompositions
    assert abs(len(ours) - len(ref)) <= 2, (
        f"IMF counts diverge: ours={len(ours)}, reference={len(ref)}"
    )


def test_first_imf_agrees_with_reference(decompositions):
    ours, ref = decompositions
    c = _corr(ours[0], ref[0])
    assert c > 0.95, f"IMF1 correlation with reference PyEMD only {c:.4f}"


def test_both_recover_ground_truth_tones(decompositions):
    ours, ref = decompositions
    # IMF1 should be the fast (25 Hz) tone, IMF2 the slow (3 Hz) tone
    assert _corr(ours[0], HI_TRUE) > 0.99
    assert _corr(ref[0], HI_TRUE) > 0.99
    assert _corr(ours[1], LO_TRUE) > 0.95
    assert _corr(ref[1], LO_TRUE) > 0.95


def test_both_reconstruct_to_machine_precision(decompositions):
    ours, ref = decompositions
    assert np.max(np.abs(np.sum(ours, axis=0) - SIGNAL)) < 1e-8
    assert np.max(np.abs(np.sum(ref, axis=0) - SIGNAL)) < 1e-8
