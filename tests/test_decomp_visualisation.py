"""
SigFeatX — Decomposition Visualisation
========================================
Shows for each method (EMD, VMD, SVMD, EFD, DWT):
  - Row 0  : original signal
  - Rows 1…K: individual decomposed components (IMFs / modes / sub-bands)
  - Last row: reconstructed signal (sum of components) vs original
  - Title bar: SNR and RMSE of reconstruction

Run from the repo root:
    python tests/test_decomp_visualisation.py

Output: tests/figures/decomp_vis_<method>.png  (one file per method)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# ── Path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from SigFeatX.decompose.emd     import EMD
from SigFeatX.decompose.vmd     import VMD
from SigFeatX.decompose.svmd    import SVMD
from SigFeatX.decompose.efd     import EFD
from SigFeatX.decompose.wavelet import WaveletDecomposer


# ============================================================================
# Figures directory
# ============================================================================

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================================
# Signal definitions
# ============================================================================

FS      = 1000          # Hz
DURATION = 2.0          # seconds
t = np.linspace(0, DURATION, int(FS * DURATION), endpoint=False)

def _make_ecg(t):
    """Synthetic single-lead ECG: sum of Gaussian peaks at ~1 Hz."""
    sig = np.zeros_like(t)
    rng = np.random.default_rng(42)
    bpm = 60
    period = FS * 60 / bpm
    for beat_start in np.arange(0, len(t), period):
        for amp, offset, width in [
            (1.0,  0,   5),
            (-0.2, 10,  3),
            (2.5,  22,  2),
            (-0.3, 35,  3),
            (0.4,  45,  8),
        ]:
            idx = int(beat_start + offset)
            if 0 <= idx < len(t):
                x = np.arange(len(t)) - idx
                sig += amp * np.exp(-x**2 / (2 * width**2))
    sig += 0.05 * rng.standard_normal(len(t))
    return sig


SIGNALS = {
    'MultiSine': (
        np.sin(2 * np.pi * 10  * t)
        + 0.7 * np.sin(2 * np.pi * 50  * t)
        + 0.4 * np.sin(2 * np.pi * 120 * t)
        + 0.15 * np.random.default_rng(0).standard_normal(len(t))
    ),
    'Chirp': (
        np.sin(2 * np.pi * (5 + 90 * t / DURATION) * t)
        + 0.1 * np.random.default_rng(1).standard_normal(len(t))
    ),
    'ECG': _make_ecg(t),
}


# ============================================================================
# Reconstruction metrics
# ============================================================================

def snr_db(original, reconstructed):
    """SNR in dB. Returns np.inf for perfect reconstruction."""
    noise_power = np.sum((original - reconstructed) ** 2)
    if noise_power < 1e-20:
        return np.inf
    signal_power = np.sum(original ** 2)
    return 10 * np.log10(signal_power / (noise_power + 1e-30))


def rmse_normalised(original, reconstructed):
    """RMSE normalised by signal std."""
    return np.sqrt(np.mean((original - reconstructed) ** 2)) / (np.std(original) + 1e-10)


# ============================================================================
# Decomposers: unified interface returning list of 1D arrays
# ============================================================================

def decompose_emd(sig):
    imfs = EMD(max_imf=8).decompose(sig)
    return imfs, np.sum(imfs, axis=0), 'EMD', 'IMF'

def decompose_vmd(sig):
    modes = VMD(K=5, alpha=2000).decompose(sig)
    return list(modes), np.sum(modes, axis=0), 'VMD', 'Mode'

def decompose_svmd(sig):
    modes = SVMD(K_max=6).decompose(sig)
    return list(modes), np.sum(modes, axis=0), 'SVMD', 'Mode'

def decompose_efd(sig):
    modes = EFD(n_modes=5).decompose(sig)
    return list(modes), np.sum(modes, axis=0), 'EFD', 'Band'

def decompose_dwt(sig):
    """DWT: returns sub-band coefficients zero-padded to signal length for display,
    plus exact reconstruction via pywt.waverec."""
    import pywt
    wav    = WaveletDecomposer(wavelet='db4')
    coeffs = wav.dwt(sig, level=4)
    recon  = pywt.waverec(coeffs, 'db4')[:len(sig)]

    # Pad each coefficient array to len(sig) for a fair time-axis display
    padded = []
    for c in coeffs:
        p = np.zeros(len(sig))
        p[:min(len(c), len(sig))] = c[:min(len(c), len(sig))]
        padded.append(p)

    labels = (
        [f'cA{len(coeffs)-1}']
        + [f'cD{len(coeffs)-1-i}' for i in range(len(coeffs) - 1)]
    )
    return padded, recon, 'DWT (db4)', 'Level', labels

def decompose_dwt_labels(sig):
    import pywt
    wav    = WaveletDecomposer(wavelet='db4')
    coeffs = wav.dwt(sig, level=4)
    labels = (
        [f'cA{len(coeffs)-1}']
        + [f'cD{len(coeffs)-1-i}' for i in range(len(coeffs) - 1)]
    )
    return labels


# ============================================================================
# Colour palette (one per component row, cycles if needed)
# ============================================================================

COMPONENT_COLOURS = [
    '#E63946',   # red
    '#2A9D8F',   # teal
    '#E9C46A',   # amber
    '#457B9D',   # steel blue
    '#F4A261',   # orange
    '#6A4C93',   # purple
    '#06D6A0',   # green
    '#118AB2',   # blue
    '#A8DADC',   # light teal
    '#FFB703',   # gold
]

ORIGINAL_COLOUR      = '#1D3557'   # dark navy
RECONSTRUCTED_COLOUR = '#E63946'   # red


# ============================================================================
# Core plotting function
# ============================================================================

def plot_decomposition(sig, t, components, reconstructed,
                       method_name, component_label,
                       component_sublabels=None,
                       signal_name='Signal',
                       save_path=None):
    """
    Plot original signal, each decomposed component, and the reconstructed
    signal in a vertical stack.

    Parameters
    ----------
    sig                : original 1D signal
    t                  : time axis
    components         : list of 1D arrays, one per decomposed component
    reconstructed      : 1D array, sum of components
    method_name        : string shown in figure title, e.g. 'EMD'
    component_label    : row label prefix, e.g. 'IMF', 'Mode', 'Band'
    component_sublabels: optional list of per-component label overrides
    signal_name        : name of the test signal for the figure title
    save_path          : if given, save figure to this path
    """
    n_comp = len(components)

    # Total rows: original + n_comp components + reconstruction
    n_rows = 1 + n_comp + 1

    fig_height = max(8, 1.6 * n_rows)
    fig = plt.figure(figsize=(16, fig_height), facecolor='#FAFAFA')

    gs = gridspec.GridSpec(
        n_rows, 1,
        figure=fig,
        hspace=0.08,
        left=0.08, right=0.97,
        top=0.93,  bottom=0.06,
    )

    axes = [fig.add_subplot(gs[i]) for i in range(n_rows)]

    snr     = snr_db(sig, reconstructed)
    rmse    = rmse_normalised(sig, reconstructed)
    snr_str = 'inf (perfect)' if np.isinf(snr) else f'{snr:.2f} dB'

    # ── Figure title ──────────────────────────────────────────────────────
    fig.suptitle(
        f'{method_name}  —  {signal_name}\n'
        f'Reconstruction  SNR = {snr_str}     Normalised RMSE = {rmse:.6f}',
        fontsize=13, fontweight='bold', color='#1D3557', y=0.98,
    )

    # ── Row 0: Original signal ─────────────────────────────────────────────
    ax = axes[0]
    ax.plot(t, sig, color=ORIGINAL_COLOUR, linewidth=0.9, alpha=0.92)
    ax.set_ylabel('Original', fontsize=9, color=ORIGINAL_COLOUR,
                  fontweight='bold', rotation=0, labelpad=55, va='center')
    _style_ax(ax, t, last=(n_rows == 1))

    # ── Rows 1…n_comp: components ──────────────────────────────────────────
    for k, comp in enumerate(components):
        ax    = axes[k + 1]
        colour = COMPONENT_COLOURS[k % len(COMPONENT_COLOURS)]

        if component_sublabels is not None and k < len(component_sublabels):
            label = component_sublabels[k]
        else:
            label = f'{component_label} {k+1}'

        ax.plot(t, comp, color=colour, linewidth=0.8, alpha=0.88)
        ax.axhline(0, color='#CCCCCC', linewidth=0.5, linestyle='--', zorder=0)
        ax.set_ylabel(label, fontsize=8.5, color=colour,
                      fontweight='bold', rotation=0,
                      labelpad=55, va='center')
        _style_ax(ax, t, last=False)

    # ── Last row: reconstruction vs original ──────────────────────────────
    ax = axes[-1]
    ax.plot(t, sig,           color=ORIGINAL_COLOUR,      linewidth=1.1,
            alpha=0.5,  label='Original',        zorder=2)
    ax.plot(t, reconstructed, color=RECONSTRUCTED_COLOUR, linewidth=1.0,
            alpha=0.85, label='Reconstructed',   zorder=3, linestyle='--')
    ax.set_ylabel('Recon.', fontsize=9, color=RECONSTRUCTED_COLOUR,
                  fontweight='bold', rotation=0, labelpad=55, va='center')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.7, edgecolor='#DDDDDD')
    _style_ax(ax, t, last=True)

    # ── Shared x-label on bottom axis only ────────────────────────────────
    axes[-1].set_xlabel('Time (s)', fontsize=9, color='#444444')

    # ── Vertical divider lines between panels for readability ─────────────
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f'  -> Saved {save_path}')

    plt.close(fig)


def _style_ax(ax, t, last=False):
    """Shared axis style: clean, minimal tick marks."""
    ax.set_xlim(t[0], t[-1])
    ax.tick_params(axis='both', labelsize=7.5, length=3, color='#AAAAAA')
    ax.spines['left'].set_color('#DDDDDD')
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
    ax.set_facecolor('#FAFAFA')
    if not last:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)
    ax.grid(axis='y', color='#EEEEEE', linewidth=0.5, zorder=0)


# ============================================================================
# Run all decomposers × all signals
# ============================================================================

DECOMPOSERS = [
    ('emd',  decompose_emd),
    ('vmd',  decompose_vmd),
    ('svmd', decompose_svmd),
    ('efd',  decompose_efd),
    ('dwt',  decompose_dwt),
]

def run_all():
    sep = '-' * 62
    print()
    print('=' * 62)
    print('  SigFeatX — Decomposition Visualisation')
    print('=' * 62)

    for sig_name, sig in SIGNALS.items():
        print(f'\n{sep}')
        print(f'  Signal: {sig_name}')
        print(sep)

        for method_key, decompose_fn in DECOMPOSERS:
            print(f'  [{method_key.upper():5s}] decomposing ... ', end='', flush=True)

            try:
                result = decompose_fn(sig)
            except Exception as exc:
                print(f'FAILED: {exc}')
                continue

            if len(result) == 5:
                components, recon, method_name, comp_label, sublabels = result
            else:
                components, recon, method_name, comp_label = result
                sublabels = None

            # Trim to signal length (DWT waverec may add 1 sample)
            recon = recon[:len(sig)]

            snr  = snr_db(sig, recon)
            rmse = rmse_normalised(sig, recon)
            snr_str = 'inf' if np.isinf(snr) else f'{snr:.1f} dB'
            print(f'{len(components)} components   '
                  f'SNR={snr_str}   RMSE={rmse:.6f}')

            fname = f'decomp_vis_{sig_name.lower()}_{method_key}.png'
            save_path = os.path.join(FIGURES_DIR, fname)

            plot_decomposition(
                sig            = sig,
                t              = t,
                components     = components,
                reconstructed  = recon,
                method_name    = method_name,
                component_label= comp_label,
                component_sublabels = sublabels,
                signal_name    = sig_name,
                save_path      = save_path,
            )

    print()
    print('=' * 62)
    print(f'  Figures saved to: {FIGURES_DIR}/')
    print(f'  Total figures:    {len(SIGNALS) * len(DECOMPOSERS)}')
    print('=' * 62)
    print()


if __name__ == '__main__':
    run_all()