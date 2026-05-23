# SigFeatX 0.3.0 — Phase 2 Upgrade

Adds **four research-grade decompositions** and **three feature classes**
(eleven new features total). All wire into `FeatureAggregator` and
`Pipeline` via simple method-name strings.

## Install

Same as phase 1 — no new mandatory dependencies. Optional extras unchanged.

```bash
pip install -e .
```

## New decompositions

### MODWT — Maximal Overlap Discrete Wavelet Transform

Shift-invariant version of DWT. Critical when extracting features over
sliding windows: a small input shift produces the same shift in every
coefficient instead of a power-of-2 jump.

```python
from SigFeatX.decompose import MODWT

m = MODWT(wavelet='db4', level=4)
coeffs = m.decompose(signal)           # list of length 5 (4 details + 1 smooth)
reconstructed = m.reconstruct(coeffs)  # perfect to machine precision
```

Properties:
- Works on signals of any length (no power-of-2 requirement)
- All coefficient arrays have the same length as the input
- Energy is preserved exactly
- Reconstruction error ~1e-16

### CEEMDAN — Complete Ensemble EMD with Adaptive Noise

Fixes EMD's mode-mixing problem by averaging EMD outputs across noise
realisations.

```python
from SigFeatX.decompose import CEEMDAN

c = CEEMDAN(trials=50, noise_amp=0.02, max_imf=10, rng=42)
imfs = c.decompose(signal)
```

Heavier than vanilla EMD (~50× for `trials=50`). Use a small `trials` for
exploration; bump to 100+ for production analysis.

### HHT — Hilbert-Huang Transform

Pairs EMD or CEEMDAN with Hilbert analysis to give instantaneous amplitude
and frequency per IMF, plus the marginal Hilbert spectrum.

```python
from SigFeatX.decompose import HHT, CEEMDAN

h = HHT(fs=1000, decomposer=CEEMDAN(trials=30))
feats = h.extract_features(signal)
# returns: mean/std/weighted inst-freq per IMF + marginal spectrum stats
```

Features include `hht_marginal_peak_freq`, `hht_marginal_centroid`,
`hht_marginal_bandwidth`, `hht_marginal_entropy`, plus per-IMF stats.

### SST — Synchrosqueezing Transform

Sharpens an STFT-style time-frequency representation by reassigning energy
to the instantaneous frequency estimated from the Auger-Flandrin operator
(no naive phase differencing — handles tonal signals correctly).

```python
from SigFeatX.decompose import SST

s = SST(fs=1000, nperseg=256)
t, f, Tx = s.transform(signal)
feats = s.extract_features(signal)
```

## New feature classes

### RQAFeatures — Recurrence Quantification Analysis

```python
from SigFeatX.features import RQAFeatures

feats = RQAFeatures.extract(signal, m=3, tau=1)
# 10 features: rr, det, lam, l_avg, l_max, tt, v_max, entr, div, eps
```

`eps` is the recurrence threshold; auto-picked to target RR ≈ 10% when not
specified.

### MFDFAFeatures — Multifractal DFA

```python
from SigFeatX.features import MFDFAFeatures

feats = MFDFAFeatures.extract(signal, q_values=[-5, -3, -1, 1, 3, 5])
# returns h(q) for each q + singularity-spectrum width, alpha0, asymmetry
```

### AdvancedEntropyFeatures — modern complexity measures

```python
from SigFeatX.features import AdvancedEntropyFeatures

feats = AdvancedEntropyFeatures.extract(signal)
# returns: dispersion_entropy, fuzzy_entropy, lz_complexity, bubble_entropy
```

Or call individual methods:

```python
AdvancedEntropyFeatures.dispersion_entropy(sig, m=3, c=6, normalize=True)
AdvancedEntropyFeatures.fuzzy_entropy(sig, m=2, r=None, n=2)
AdvancedEntropyFeatures.lempel_ziv_complexity(sig, binarize='median')
AdvancedEntropyFeatures.bubble_entropy(sig, m=10)
```

## Aggregator integration

After applying `PATCH_aggregator_phase2.md`, the new decompositions become
first-class methods:

```python
from SigFeatX import FeatureAggregator

agg = FeatureAggregator(fs=1000)
feats = agg.extract_all_features(
    signal,
    decomposition_methods=['fourier', 'modwt', 'ceemdan', 'hht', 'sst'],
)
```

The raw-feature dict automatically includes RQA, MFDFA, and the four
advanced entropies — no opt-in flag needed.

In Pipelines:

```python
from SigFeatX import Pipeline

df = (
    Pipeline(fs=1000)
    .denoise(method='bandpass', low_hz=1, high_hz=40)
    .decompose(['modwt', 'ceemdan', 'sst'])
    .with_parallel(n_jobs=-1)
    .extract_batch(signals)
).dataframe
```

## Test results

```
43 passed in 2.98s
```

Coverage:
- 8 MODWT tests (perfect reconstruction, shift invariance, multiple wavelets)
- 6 CEEMDAN tests (reconstruction, reproducibility, validation)
- 4 HHT tests (carrier-frequency tracking, marginal spectrum)
- 5 SST tests (peak detection, entropy ordering)
- 6 RQA tests (RR targeting, periodicity discrimination)
- 4 MFDFA tests (white-noise monofractal check, finiteness)
- 10 advanced-entropy tests (noise vs sine discrimination, bounds)

## Files added / changed

```
SigFeatX/decompose/
├── modwt.py        [NEW]   Shift-invariant DWT
├── ceemdan.py      [NEW]   Robust ensemble EMD
├── hht.py          [NEW]   Hilbert-Huang Transform
├── sst.py          [NEW]   Synchrosqueezing
└── __init__.py     [UPDATE] Export the 4 new classes

SigFeatX/features/
├── rqa.py              [NEW]   Recurrence Quantification
├── mfdfa.py            [NEW]   Multifractal DFA
├── advanced_entropy.py [NEW]   Dispersion / Fuzzy / LZ / Bubble
└── __init__.py         [UPDATE] Export the 3 new classes

SigFeatX/aggregator.py  [PATCH]  See PATCH_aggregator_phase2.md
tests/test_phase2.py    [NEW]    43 tests
examples/phase2_usage.py [NEW]   End-to-end demo
PATCH_aggregator_phase2.md [NEW] Aggregator wiring instructions
PHASE2_README.md        [NEW]    This file
```

## Backward compatibility

All existing APIs unchanged. Adding the new decompositions and features
to `decomposition_methods` is opt-in. Adding RQA/MFDFA/AdvancedEntropy to
raw features (after the patch) does extend the default feature set —
expect ~20 extra columns in `extract_all_features` output. If that breaks
downstream code, you can pass `extract_raw=False` or call the individual
extractors directly to control which features land in your batch.
