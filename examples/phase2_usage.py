"""
SigFeatX — Phase 2 Upgrade Examples
=====================================
Demonstrates the seven new capabilities shipped in 0.3.0:
  1. MODWT     — shift-invariant wavelet decomposition
  2. CEEMDAN   — robust EMD variant
  3. HHT       — Hilbert-Huang Transform
  4. SST       — Synchrosqueezing Transform
  5. RQA       — Recurrence Quantification Analysis
  6. MFDFA     — Multifractal DFA
  7. Advanced Entropy — Dispersion, Fuzzy, LZ, Bubble

Run:
    python examples/phase2_usage.py
"""

import numpy as np

from SigFeatX.decompose import CEEMDAN, HHT, MODWT, SST
from SigFeatX.features import AdvancedEntropyFeatures, MFDFAFeatures, RQAFeatures

# ------------------------------------------------------------------------
# Synthetic test signal: multi-component non-stationary
# ------------------------------------------------------------------------
np.random.seed(42)
fs = 500
t = np.arange(fs * 4) / fs
sig = (
    np.sin(2 * np.pi * 5 * t)                              # slow component
    + 0.7 * np.sin(2 * np.pi * (20 + 10 * t) * t)          # chirp
    + 0.4 * np.sin(2 * np.pi * 80 * t) * (t > 2)           # gated tone
    + 0.05 * np.random.randn(len(t))
)


print("=" * 64)
print("  1. MODWT  (shift-invariant wavelet)")
print("=" * 64)
modwt = MODWT(wavelet='db4', level=4)
coeffs = modwt.decompose(sig)
print(f"  Levels: {len(coeffs)}, each length = {len(coeffs[0])}")
rec = modwt.reconstruct(coeffs)
print(f"  Reconstruction RMSE: {np.sqrt(np.mean((sig - rec)**2)):.2e}")
print("  Detail-band energies:")
for i, c in enumerate(coeffs[:-1]):
    print(f"    Level {i+1}: {np.sum(c**2):.2f}")
print(f"  Smooth band energy: {np.sum(coeffs[-1]**2):.2f}")


print("\n" + "=" * 64)
print("  2. CEEMDAN  (robust EMD)")
print("=" * 64)
ceemdan = CEEMDAN(trials=20, max_imf=5, rng=0)
imfs = ceemdan.decompose(sig[:1000])
print(f"  Number of IMFs: {len(imfs)}")
print(f"  Reconstruction RMSE: {np.sqrt(np.mean((sig[:1000] - sum(imfs))**2)):.2e}")


print("\n" + "=" * 64)
print("  3. HHT  (Hilbert-Huang)")
print("=" * 64)
hht = HHT(fs=fs)
hht_feats = hht.extract_features(sig)
print(f"  Total features: {len(hht_feats)}")
print(f"  Marginal peak frequency: {hht_feats['hht_marginal_peak_freq']:.2f} Hz")
print(f"  Marginal centroid:       {hht_feats['hht_marginal_centroid']:.2f} Hz")
print(f"  Marginal bandwidth:      {hht_feats['hht_marginal_bandwidth']:.2f} Hz")


print("\n" + "=" * 64)
print("  4. SST  (Synchrosqueezing)")
print("=" * 64)
sst = SST(fs=fs, nperseg=256)
sst_feats = sst.extract_features(sig)
print(f"  Peak frequency:  {sst_feats['sst_peak_freq']:.2f} Hz")
print(f"  Centroid:        {sst_feats['sst_centroid']:.2f} Hz")
print(f"  Bandwidth:       {sst_feats['sst_bandwidth']:.2f} Hz")
print(f"  Entropy:         {sst_feats['sst_entropy']:.3f}")
print(f"  Concentration:   {sst_feats['sst_concentration']:.1f}")


print("\n" + "=" * 64)
print("  5. RQA  (Recurrence Quantification)")
print("=" * 64)
rqa_feats = RQAFeatures.extract(sig[:400], m=3, tau=2)
for k, v in rqa_feats.items():
    print(f"  {k}: {v:.4f}")


print("\n" + "=" * 64)
print("  6. MFDFA  (Multifractal DFA)")
print("=" * 64)
mfdfa_feats = MFDFAFeatures.extract(sig)
print(f"  Singularity-spectrum width:   {mfdfa_feats['mfdfa_width']:.4f}")
print(f"  Spectrum peak position alpha0: {mfdfa_feats['mfdfa_alpha0']:.4f}")
print(f"  Asymmetry:                     {mfdfa_feats['mfdfa_asymmetry']:.4f}")
print("  h(q) values:")
for k, v in mfdfa_feats.items():
    if k.startswith('mfdfa_h_q'):
        print(f"    {k}: {v:.4f}")


print("\n" + "=" * 64)
print("  7. Advanced Entropy")
print("=" * 64)
ent_feats = AdvancedEntropyFeatures.extract(sig[:1000])
for k, v in ent_feats.items():
    print(f"  {k}: {v:.4f}")


print("\n" + "=" * 64)
print("  8. INTEGRATED  (via FeatureAggregator)")
print("=" * 64)
# Once the aggregator integration is applied, you can do:
# agg = FeatureAggregator(fs=fs)
# feats = agg.extract_all_features(
#     sig,
#     decomposition_methods=['fourier', 'dwt', 'modwt', 'ceemdan', 'hht', 'sst'],
# )
print("  After applying PATCH_aggregator_phase2.md:")
print("  FeatureAggregator + Pipeline support all 4 new decomposers as method")
print("  names: 'modwt', 'ceemdan', 'hht', 'sst'.")
print()
print("  Raw features automatically include RQA, MFDFA, and the 4 advanced")
print("  entropies — no opt-in flag needed.")

print("\n" + "=" * 64)
print("  All phase-2 modules working.")
print("=" * 64)
