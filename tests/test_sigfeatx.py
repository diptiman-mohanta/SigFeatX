"""
SigFeatX – Test Suite
======================
Run from the repo root:
    python tests/test_sigfeatx.py

Three test suites, one per signal type:
  Suite A – Multi-sine  (10 Hz + 50 Hz + 100 Hz)
  Suite B – Chirp       (frequency sweep 5→200 Hz)
  Suite C – ECG-like    (synthetic QRS + P + T waves)

Each suite runs:
  1. Preprocessing test          (detrend / denoise / normalise)
  2. Decomposition tests         (EMD, VMD, SVMD, EFD, DWT, STFT)
  3. Feature extraction tests    (time, frequency, entropy, nonlinear)
  4. Decomposition quality tests (SNR, RMSE, energy preservation)
  5. Cross-method consistency    (EMD vs VMD features)
  6. Pipeline metadata test      (run_pipeline audit trail)

Figures saved to  tests/figures/:
  fig1_signals.png          – raw + preprocessed for all three signal types
  fig2_imfs_multisine.png   – EMD IMFs on multi-sine
  fig3_imfs_chirp.png       – EMD IMFs on chirp
  fig4_imfs_ecg.png         – EMD IMFs on ECG
  fig5_modes_vmd.png        – VMD modes for all three signals
  fig6_spectrogram.png      – STFT spectrograms for all three signals
  fig7_quality_metrics.png  – SNR / RMSE / energy bar charts per method & signal
  fig8_feature_comparison.png – EMD vs VMD feature bar chart
"""

import os
import sys
import warnings
import traceback
import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as scipy_signal

# ── Make sure SigFeatX is importable when run from repo root ────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from SigFeatX.preprocess import SignalPreprocessor
from SigFeatX.decompose import (
    EMD, VMD, SVMD, EFD,
    WaveletDecomposer, ShortTimeFourierTransform, FourierTransform,
)
from SigFeatX.features.features import (
    TimeDomainFeatures, FrequencyDomainFeatures,
    EntropyFeatures, NonlinearFeatures, DecompositionFeatures,
)
from SigFeatX.decomposition_validator import DecompositionValidator
from SigFeatX.feature_consistency import validate_feature_dict, CrossMethodChecker
from SigFeatX.aggregator import FeatureAggregator

# ── Output directory ────────────────────────────────────────────────────────
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

FS   = 1000          # Hz
DUR  = 2.0           # seconds
N    = int(FS * DUR)
t    = np.linspace(0, DUR, N, endpoint=False)

PASS = "  ✓ PASS"
FAIL = "  ✗ FAIL"

# ============================================================================
# 0.  SIGNAL GENERATORS
# ============================================================================

def make_multisine() -> np.ndarray:
    """10 Hz + 50 Hz + 100 Hz + light noise."""
    sig  = (1.0  * np.sin(2 * np.pi * 10  * t)
          + 0.5  * np.sin(2 * np.pi * 50  * t)
          + 0.25 * np.sin(2 * np.pi * 100 * t)
          + 0.05 * np.random.default_rng(0).standard_normal(N))
    # Add a mild linear trend so detrend has something to do
    sig += 0.3 * t
    return sig


def make_chirp() -> np.ndarray:
    """Linear frequency sweep 5 → 200 Hz."""
    sig = scipy_signal.chirp(t, f0=5, f1=200, t1=DUR, method="linear")
    sig += 0.05 * np.random.default_rng(1).standard_normal(N)
    sig += 0.2 * t
    return sig


def make_ecg() -> np.ndarray:
    """
    Synthetic ECG: superposition of a narrow QRS complex, a broad T wave,
    and a small P wave, repeated at 60 bpm.
    """
    rng  = np.random.default_rng(2)
    hr   = 60          # beats per minute → 1 beat/second
    rr   = FS // hr    # samples per beat

    beat = np.zeros(rr)
    tb   = np.linspace(-0.5, 0.5, rr)

    # P wave
    beat += 0.15 * np.exp(-((tb + 0.25) ** 2) / (2 * 0.015 ** 2))
    # QRS
    beat += ( 1.0  * np.exp(-((tb)        ** 2) / (2 * 0.006 ** 2))
            - 0.3  * np.exp(-((tb + 0.02) ** 2) / (2 * 0.005 ** 2))
            - 0.25 * np.exp(-((tb - 0.02) ** 2) / (2 * 0.005 ** 2)))
    # T wave
    beat += 0.35 * np.exp(-((tb - 0.2) ** 2) / (2 * 0.025 ** 2))

    n_beats = N // rr + 1
    sig     = np.tile(beat, n_beats)[:N]
    sig    += 0.03 * rng.standard_normal(N)
    sig    += 0.1 * t
    return sig


SIGNALS = {
    "MultiSine" : make_multisine(),
    "Chirp"     : make_chirp(),
    "ECG"       : make_ecg(),
}

# ============================================================================
# 1.  TEST HELPERS
# ============================================================================

_results = []   # list of (suite, test_name, passed, message)


def check(suite: str, name: str, condition: bool, msg: str = ""):
    status = PASS if condition else FAIL
    label  = f"[{suite}] {name}"
    print(f"{status}  {label}" + (f" — {msg}" if msg else ""))
    _results.append((suite, name, condition, msg))
    return condition


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ============================================================================
# 2.  INDIVIDUAL TESTS
# ============================================================================

def test_preprocessing(suite: str, sig: np.ndarray):
    section(f"[{suite}] Preprocessing")
    pp = SignalPreprocessor()

    detrended  = pp.detrend(sig.copy(), method="linear")
    denoised   = pp.denoise(sig.copy(), method="wavelet")
    normalised = pp.normalize(sig.copy(), method="zscore")

    check(suite, "detrend  – same length",
          len(detrended) == len(sig))
    check(suite, "detrend  – trend reduced",
          np.abs(np.polyfit(t, detrended, 1)[0]) < np.abs(np.polyfit(t, sig, 1)[0]),
          f"slope before={np.polyfit(t,sig,1)[0]:.4f}, after={np.polyfit(t,detrended,1)[0]:.4f}")
    check(suite, "denoise  – same length",
          len(denoised) == len(sig))
    check(suite, "denoise  – output finite",
          np.all(np.isfinite(denoised)))
    check(suite, "normalise – zero mean (zscore)",
          abs(np.mean(normalised)) < 0.01,
          f"mean={np.mean(normalised):.4f}")
    check(suite, "normalise – unit std  (zscore)",
          abs(np.std(normalised) - 1.0) < 0.01,
          f"std={np.std(normalised):.4f}")

    return detrended, denoised, normalised


def test_decomposition(suite: str, sig: np.ndarray):
    section(f"[{suite}] Decomposition")
    results = {}

    # ── EMD ─────────────────────────────────────────────────────────────────
    emd  = EMD(max_imf=8)
    imfs = emd.decompose(sig)
    check(suite, "EMD – returns list",        isinstance(imfs, list))
    check(suite, "EMD – at least 2 IMFs",     len(imfs) >= 2,       f"got {len(imfs)}")
    check(suite, "EMD – all IMFs same length",
          all(len(c) == N for c in imfs))
    check(suite, "EMD – all finite",
          all(np.all(np.isfinite(c)) for c in imfs))
    results["EMD"] = imfs

    # ── VMD ─────────────────────────────────────────────────────────────────
    vmd   = VMD(K=4)
    modes = vmd.decompose(sig)
    check(suite, "VMD – 2D array",            modes.ndim == 2)
    check(suite, "VMD – correct n_modes",     modes.shape[0] == 4, f"shape={modes.shape}")
    check(suite, "VMD – correct length",      modes.shape[1] == N)
    check(suite, "VMD – all finite",          np.all(np.isfinite(modes)))
    results["VMD"] = modes

    # ── SVMD ────────────────────────────────────────────────────────────────
    svmd  = SVMD(K_max=5)
    smodes= svmd.decompose(sig)
    check(suite, "SVMD – 2D array",           smodes.ndim == 2)
    check(suite, "SVMD – at least 1 mode",    smodes.shape[0] >= 1)
    check(suite, "SVMD – all finite",         np.all(np.isfinite(smodes)))
    results["SVMD"] = smodes

    # ── EFD ─────────────────────────────────────────────────────────────────
    efd   = EFD(n_modes=4)
    emodes= efd.decompose(sig)
    check(suite, "EFD – 2D array",            emodes.ndim == 2)
    check(suite, "EFD – correct n_modes",     emodes.shape[0] == 4)
    check(suite, "EFD – correct length",      emodes.shape[1] == N)
    check(suite, "EFD – all finite",          np.all(np.isfinite(emodes)))
    results["EFD"] = emodes

    # ── DWT ─────────────────────────────────────────────────────────────────
    wav    = WaveletDecomposer(wavelet="db4")
    coeffs = wav.dwt(sig, level=4)
    check(suite, "DWT – returns list",        isinstance(coeffs, list))
    check(suite, "DWT – correct n_levels",    len(coeffs) == 5,     f"got {len(coeffs)}")
    check(suite, "DWT – all finite",
          all(np.all(np.isfinite(c)) for c in coeffs))

    # DWT reconstruction via waverec (the correct approach).
    # Raw coefficients are sub-band arrays at different scales — summing them
    # directly is meaningless. The inverse transform must be used.
    try:
        rec = wav.idwt(coeffs)[:N]
        dwt_rmse = float(np.sqrt(np.mean((sig - rec) ** 2))
                         / (np.sqrt(np.mean(sig ** 2)) + 1e-30))
        check(suite, "DWT – waverec RMSE < 0.001",
              dwt_rmse < 0.001, f"RMSE={dwt_rmse:.8f}")
    except Exception as e:
        check(suite, "DWT – waverec RMSE < 0.001", False, str(e))

    results["DWT"] = coeffs

    # ── STFT ────────────────────────────────────────────────────────────────
    stft_obj = ShortTimeFourierTransform(fs=FS, nperseg=256)
    f, t_stft, Zxx = stft_obj.transform(sig)
    check(suite, "STFT – freqs non-negative", np.all(f >= 0))
    check(suite, "STFT – magnitudes finite",  np.all(np.isfinite(Zxx)))
    check(suite, "STFT – correct freq bins",
          len(f) == 256 // 2 + 1, f"got {len(f)}")
    results["STFT"] = (f, t_stft, Zxx)

    return results


def test_features(suite: str, sig: np.ndarray):
    section(f"[{suite}] Feature Extraction")

    # Detrend before frequency-domain tests so a linear trend doesn't push
    # all power into the DC bin and make dominant_frequency report 0 Hz.
    pp  = SignalPreprocessor()
    sig_dt = pp.detrend(sig.copy(), method="linear")

    td = TimeDomainFeatures.extract(sig)          # time features on raw signal
    fd = FrequencyDomainFeatures.extract(sig_dt, fs=FS)   # freq features on detrended
    en = EntropyFeatures.extract(sig)
    nl = NonlinearFeatures.extract(sig)

    # Basic sanity
    check(suite, "TimeDomain   – rms ≥ 0",
          td["rms"] >= 0,              f"rms={td['rms']:.4f}")
    check(suite, "TimeDomain   – variance ≥ 0",
          td["variance"] >= 0,         f"var={td['variance']:.4f}")
    check(suite, "TimeDomain   – zcr in [0,1]",
          0 <= td["zero_crossing_rate"] <= 1,
          f"zcr={td['zero_crossing_rate']:.4f}")
    check(suite, "FreqDomain   – dom_freq > 0",
          fd["dominant_frequency"] > 0,
          f"dom={fd['dominant_frequency']:.2f} Hz")
    check(suite, "FreqDomain   – spectral_centroid > 0",
          fd["spectral_centroid"] > 0,
          f"sc={fd['spectral_centroid']:.2f} Hz")
    check(suite, "FreqDomain   – spectral_flatness in [0,1]",
          0 <= fd["spectral_flatness"] <= 1,
          f"sf={fd['spectral_flatness']:.4f}")
    check(suite, "Entropy      – sample_entropy ≥ 0",
          en["sample_entropy"] >= 0,
          f"se={en['sample_entropy']:.4f}")
    check(suite, "Entropy      – permutation_entropy ≥ 0",
          en["permutation_entropy"] >= 0)
    check(suite, "Nonlinear    – hurst in [0,2]",
          0 <= nl["hurst_exponent"] <= 2,
          f"H={nl['hurst_exponent']:.4f}")
    check(suite, "Nonlinear    – dfa_alpha finite",
          np.isfinite(nl["dfa_alpha"]),
          f"dfa={nl['dfa_alpha']:.4f}")

    # Contract validation (should return no hard violations on clean features)
    all_feats = {**td, **fd, **en, **nl}
    violations = validate_feature_dict(all_feats, method=suite)
    hard_violations = [v for v in violations
                       if "≥" in v.reason and float(v.value) < 0]
    check(suite, "Contract     – no hard violations",
          len(hard_violations) == 0,
          f"{len(hard_violations)} hard violation(s)" if hard_violations else "all clear")

    return td, fd, en, nl


def test_quality(suite: str, sig: np.ndarray, decomp_results: dict):
    section(f"[{suite}] Decomposition Quality")
    quality = {}

    method_map = {
        "EMD" : ("EMD",  decomp_results["EMD"]),
        "VMD" : ("VMD",  decomp_results["VMD"]),
        "SVMD": ("SVMD", decomp_results["SVMD"]),
        "EFD" : ("EFD",  decomp_results["EFD"]),
        "DWT" : ("DWT",  decomp_results["DWT"]),
    }

    for label, (method, components) in method_map.items():
        report = DecompositionValidator.evaluate(sig, components, method=method)
        quality[label] = report

        check(suite, f"{label:5s} – report has snr_db",
              np.isfinite(report.snr_db) or report.snr_db == np.inf,
              f"SNR={'inf (perfect)' if report.snr_db == np.inf else f'{report.snr_db:.1f} dB'}")
        check(suite, f"{label:5s} – reconstruction_error finite",
              np.isfinite(report.reconstruction_error),
              f"RMSE={report.reconstruction_error:.6f}")
        check(suite, f"{label:5s} – energy_preservation finite",
              np.isfinite(report.energy_preservation),
              f"EP={report.energy_preservation:.4f}")
        # EMD is strictly additive in time domain so sum(IMFs) == original exactly.
        # DWT coefficients are sub-band arrays at different scales/lengths —
        # summing them raw is not a valid reconstruction. The validator pads them
        # to equal length which introduces error by design. Only assert RMSE
        # for EMD here; DWT reconstruction is tested via pywt.waverec separately.
        if label == "EMD":
            check(suite, f"{label:5s} – reconstruction_error < 0.01",
                  report.reconstruction_error < 0.01,
                  f"RMSE={report.reconstruction_error:.6f}")

        print(report.summary())

    return quality


def test_consistency(suite: str, decomp_results: dict):
    section(f"[{suite}] Cross-Method Consistency (EMD vs VMD)")

    emd_feats = DecompositionFeatures.extract_from_components(
        decomp_results["EMD"], prefix="comp"
    )
    vmd_feats = DecompositionFeatures.extract_from_components(
        [decomp_results["VMD"][i] for i in range(decomp_results["VMD"].shape[0])],
        prefix="comp"
    )

    checker = CrossMethodChecker(tolerance=0.5)   # 50% — modes differ in count
    checker.add_raw("EMD", emd_feats)
    checker.add_raw("VMD", vmd_feats)

    comparisons = checker.compare(features=["comp_0_rms", "comp_0_energy", "comp_0_std"])
    check(suite, "CrossMethod  – comparison ran",
          len(comparisons) > 0, f"{len(comparisons)} feature(s) compared")
    for c in comparisons:
        check(suite, f"CrossMethod  – {c.feature} spread finite",
              np.isfinite(c.relative_spread), f"spread={c.relative_spread:.3f}")

    return comparisons


def test_pipeline(suite: str, sig: np.ndarray):
    section(f"[{suite}] Pipeline Metadata")

    agg = FeatureAggregator(fs=FS)
    features, meta = agg.run_pipeline(
        sig,
        preprocess_params={"denoise_method": "wavelet", "normalize_method": "zscore"},
        decomposition_methods=["emd", "vmd", "dwt"],
        validate=True,
    )

    check(suite, "Pipeline – features dict non-empty",
          len(features) > 0,          f"{len(features)} features")
    check(suite, "Pipeline – metadata has stages",
          len(meta.stages) >= 4,      f"{len(meta.stages)} stages")
    check(suite, "Pipeline – metadata fs correct",
          meta.fs == FS,              f"fs={meta.fs}")
    check(suite, "Pipeline – all feature values finite",
          all(
              np.isfinite(v)
              for k, v in features.items()
              if isinstance(v, float) and "snr_db" not in k   # SNR of 999.0 is clipped in to_dict
          ),
          "all finite (snr_db keys use 999.0 sentinel for perfect reconstruction)")

    # Quality keys injected into feature dict
    has_snr = any("snr_db" in k for k in features)
    check(suite, "Pipeline – SNR quality keys present", has_snr)

    print(meta)
    return features, meta


# ============================================================================
# 3.  FIGURE GENERATORS
# ============================================================================

COLORS = {
    "MultiSine": "#2196F3",
    "Chirp"    : "#FF5722",
    "ECG"      : "#4CAF50",
}
METHOD_COLORS = ["#3F51B5","#E91E63","#009688","#FF9800","#9C27B0"]


def fig1_signals(preprocessed: dict):
    """Raw + preprocessed for each signal type."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 9))
    fig.suptitle("Fig 1 — Raw, Detrended & Normalised Signals", fontsize=14, fontweight="bold")

    titles = ["Raw", "Detrended", "Normalised (z-score)"]
    for row, (name, sig) in enumerate(SIGNALS.items()):
        data = [sig,
                preprocessed[name]["detrended"],
                preprocessed[name]["normalised"]]
        for col, (d, title) in enumerate(zip(data, titles)):
            ax = axes[row][col]
            ax.plot(t[:1000], d[:1000], color=COLORS[name], lw=0.8)
            ax.set_title(f"{name} — {title}", fontsize=9)
            ax.set_xlabel("Time (s)", fontsize=7)
            ax.set_ylabel("Amplitude", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig1_signals.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  → Saved {path}")


def fig2_imfs(all_decomp: dict, signal_name: str, fignum: int):
    """EMD IMFs for one signal."""
    imfs = all_decomp[signal_name]["EMD"]
    n    = min(len(imfs), 8)
    fig, axes = plt.subplots(n + 1, 1, figsize=(14, 2.2 * (n + 1)))
    fig.suptitle(f"Fig {fignum} — EMD IMFs  [{signal_name}]",
                 fontsize=13, fontweight="bold")

    axes[0].plot(t[:2000], SIGNALS[signal_name][:2000],
                 color=COLORS[signal_name], lw=0.9)
    axes[0].set_title("Original Signal", fontsize=9)
    axes[0].set_ylabel("Amp", fontsize=7)
    axes[0].grid(True, alpha=0.3)

    for i in range(n):
        axes[i + 1].plot(t[:2000], imfs[i][:2000],
                         color=METHOD_COLORS[i % len(METHOD_COLORS)], lw=0.8)
        axes[i + 1].set_title(f"IMF {i+1}", fontsize=9)
        axes[i + 1].set_ylabel("Amp", fontsize=7)
        axes[i + 1].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)", fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"fig{fignum}_imfs_{signal_name.lower()}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → Saved {path}")


def fig5_vmd_modes(all_decomp: dict):
    """VMD modes for all three signals side by side."""
    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    fig.suptitle("Fig 5 — VMD Modes (K=4)", fontsize=13, fontweight="bold")

    for row, name in enumerate(SIGNALS):
        modes = all_decomp[name]["VMD"]
        for col in range(4):
            ax = axes[row][col]
            ax.plot(t[:2000], modes[col][:2000],
                    color=METHOD_COLORS[col], lw=0.8)
            ax.set_title(f"{name}  Mode {col+1}", fontsize=8)
            ax.set_xlabel("Time (s)", fontsize=7)
            ax.set_ylabel("Amp", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig5_modes_vmd.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → Saved {path}")


def fig6_spectrograms(all_decomp: dict):
    """STFT spectrograms."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Fig 6 — STFT Spectrograms", fontsize=13, fontweight="bold")

    for ax, name in zip(axes, SIGNALS):
        f_arr, t_arr, Zxx = all_decomp[name]["STFT"]
        img = ax.pcolormesh(t_arr, f_arr, 20 * np.log10(Zxx + 1e-10),
                            shading="gouraud", cmap="inferno")
        ax.set_title(f"{name}", fontsize=10)
        ax.set_ylabel("Frequency (Hz)", fontsize=8)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylim(0, FS / 2)
        ax.tick_params(labelsize=7)
        cb = plt.colorbar(img, ax=ax)
        cb.set_label("dB", fontsize=7)
        cb.ax.tick_params(labelsize=7)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig6_spectrogram.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → Saved {path}")


def fig7_quality_metrics(all_quality: dict):
    """
    SNR / RMSE / energy preservation grouped bar charts.
    One group per (signal_type × method).
    """
    methods      = ["EMD", "VMD", "SVMD", "EFD", "DWT"]
    signal_names = list(SIGNALS.keys())
    n_methods    = len(methods)
    x            = np.arange(n_methods)
    width        = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Fig 7 — Decomposition Quality Metrics", fontsize=13, fontweight="bold")

    metrics = ["snr_db", "reconstruction_error", "energy_preservation"]
    ylabels = ["SNR (dB)", "Normalised RMSE", "Energy Preservation Ratio"]
    ideal   = [None, 0.0, 1.0]

    for col, (metric, ylabel, id_val) in enumerate(zip(metrics, ylabels, ideal)):
        ax = axes[col]
        for si, sname in enumerate(signal_names):
            vals = []
            for m in methods:
                report = all_quality[sname][m]
                vals.append(getattr(report, metric))
            offset = (si - 1) * width
            bars = ax.bar(x + offset, vals, width,
                          label=sname, color=list(COLORS.values())[si], alpha=0.8)
            # Value labels
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.01,
                        f"{v:.2g}", ha="center", va="bottom", fontsize=6)

        if id_val is not None:
            ax.axhline(id_val, color="black", lw=1, ls="--",
                       label=f"Ideal = {id_val}")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig7_quality_metrics.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → Saved {path}")


def fig8_feature_comparison(all_decomp: dict):
    """
    EMD vs VMD per-component features (rms, energy, std) for all three signals.
    """
    features_to_plot = ["rms", "energy", "std"]
    signal_names     = list(SIGNALS.keys())

    fig, axes = plt.subplots(3, 3, figsize=(16, 10))
    fig.suptitle("Fig 8 — EMD vs VMD Per-Component Feature Comparison",
                 fontsize=13, fontweight="bold")

    for row, sname in enumerate(signal_names):
        imfs  = all_decomp[sname]["EMD"]
        modes = all_decomp[sname]["VMD"]

        emd_feats = DecompositionFeatures.extract_from_components(imfs,  prefix="comp")
        vmd_feats = DecompositionFeatures.extract_from_components(
            [modes[i] for i in range(modes.shape[0])], prefix="comp"
        )

        for col, feat in enumerate(features_to_plot):
            ax = axes[row][col]

            # Collect per-component values
            emd_vals = [emd_feats[k] for k in sorted(emd_feats)
                        if k.endswith(f"_{feat}") and "ratio" not in k
                        and "corr" not in k and "kl" not in k]
            vmd_vals = [vmd_feats[k] for k in sorted(vmd_feats)
                        if k.endswith(f"_{feat}") and "ratio" not in k
                        and "corr" not in k and "kl" not in k]

            x_emd = np.arange(len(emd_vals))
            x_vmd = np.arange(len(vmd_vals))

            ax.bar(x_emd - 0.2, emd_vals, 0.4, label="EMD",
                   color="#3F51B5", alpha=0.8)
            ax.bar(x_vmd + 0.2, vmd_vals, 0.4, label="VMD",
                   color="#E91E63", alpha=0.8)

            ax.set_title(f"{sname} — {feat}", fontsize=9)
            ax.set_xlabel("Component index", fontsize=7)
            ax.set_ylabel(feat, fontsize=7)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig8_feature_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → Saved {path}")


# ============================================================================
# 4.  MAIN RUNNER
# ============================================================================

def run_all():
    print("=" * 60)
    print("  SigFeatX — Full Test Suite")
    print(f"  Signal: {DUR}s @ {FS} Hz  ({N} samples)")
    print("=" * 60)

    np.random.seed(42)

    preprocessed = {}
    all_decomp   = {}
    all_quality  = {}

    # ── Run all suites ───────────────────────────────────────────────────────
    for name, sig in SIGNALS.items():
        print(f"\n{'█'*60}")
        print(f"  SUITE: {name}")
        print(f"{'█'*60}")

        try:
            detrended, denoised, normalised = test_preprocessing(name, sig)
            preprocessed[name] = {
                "detrended" : detrended,
                "denoised"  : denoised,
                "normalised": normalised,
            }
        except Exception as e:
            print(f"{FAIL}  [{name}] preprocessing crashed: {e}")
            traceback.print_exc()
            preprocessed[name] = {
                "detrended" : sig, "denoised": sig, "normalised": sig
            }

        try:
            decomp = test_decomposition(name, sig)
            all_decomp[name] = decomp
        except Exception as e:
            print(f"{FAIL}  [{name}] decomposition crashed: {e}")
            traceback.print_exc()
            all_decomp[name] = {}
            continue

        try:
            test_features(name, sig)
        except Exception as e:
            print(f"{FAIL}  [{name}] features crashed: {e}")
            traceback.print_exc()

        try:
            quality = test_quality(name, sig, decomp)
            all_quality[name] = quality
        except Exception as e:
            print(f"{FAIL}  [{name}] quality check crashed: {e}")
            traceback.print_exc()
            all_quality[name] = {}

        try:
            test_consistency(name, decomp)
        except Exception as e:
            print(f"{FAIL}  [{name}] consistency check crashed: {e}")
            traceback.print_exc()

        try:
            test_pipeline(name, sig)
        except Exception as e:
            print(f"{FAIL}  [{name}] pipeline crashed: {e}")
            traceback.print_exc()

    # ── Generate figures ─────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Generating figures …")
    print(f"{'─'*60}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fig1_signals(preprocessed)
        except Exception as e:
            print(f"  fig1 failed: {e}")

        for i, name in enumerate(SIGNALS, start=2):
            try:
                fig2_imfs(all_decomp, name, fignum=i)
            except Exception as e:
                print(f"  fig{i} ({name} IMFs) failed: {e}")

        try:
            fig5_vmd_modes(all_decomp)
        except Exception as e:
            print(f"  fig5 failed: {e}")

        try:
            fig6_spectrograms(all_decomp)
        except Exception as e:
            print(f"  fig6 failed: {e}")

        try:
            if all_quality:
                fig7_quality_metrics(all_quality)
        except Exception as e:
            print(f"  fig7 failed: {e}")

        try:
            fig8_feature_comparison(all_decomp)
        except Exception as e:
            print(f"  fig8 failed: {e}")

    # ── Summary ──────────────────────────────────────────────────────────────
    total  = len(_results)
    passed = sum(1 for r in _results if r[2])
    failed = total - passed

    print(f"\n{'='*60}")
    print(f"  RESULTS:  {passed}/{total} passed  |  {failed} failed")

    if failed:
        print("\n  Failed tests:")
        for suite, name, ok, msg in _results:
            if not ok:
                print(f"    ✗  [{suite}] {name}" + (f" — {msg}" if msg else ""))

    print(f"\n  Figures saved to:  {os.path.abspath(FIG_DIR)}/")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)