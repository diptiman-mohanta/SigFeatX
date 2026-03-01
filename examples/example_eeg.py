"""
EEG Signal Analysis Script
Load, clean, decompose using VMD (K=7), and extract features from EEG data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import seaborn as sns

# Add SigFeatX to path
sys.path.insert(0, r'D:\SigFeatX')

from SigFeatX.aggregator import FeatureAggregator, SignalPreprocessor
from SigFeatX.decompose import VMD
from SigFeatX.features.features import TimeDomainFeatures

# File path
filepath = r"D:\LieWaves dataset for lie detection based on EEG signals and wavelets\LieWaves dataset for lie detection based on EEG signals and wavelets\modified\S1S1_truth.csv"

print("=" * 80)
print("EEG Signal Analysis with VMD (K=7)")
print("=" * 80)

# Step 1: Load the data
print("\n1. Loading data...")
try:
    df = pd.read_csv(filepath)
    print(f"   Loaded CSV with shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Get the first column as signal
    signal = df.iloc[:, 0].values
    print(f"   Signal length: {len(signal)}")
    print(f"   Signal range: [{np.min(signal):.4f}, {np.max(signal):.4f}]")
    
except Exception as e:
    print(f"   Error loading file: {e}")
    sys.exit(1)

# Step 2: Clean the signal
print("\n2. Cleaning signal...")
preprocessor = SignalPreprocessor()

signal_detrended = preprocessor.detrend(signal, method='linear')
print(f"   ✓ Detrended signal")

signal_clean = preprocessor.denoise(signal_detrended, method='wavelet', 
                                   wavelet='db4', level=3, mode='soft')
print(f"   ✓ Denoised signal (wavelet db4, level 3)")

signal_clean = preprocessor.normalize(signal_clean, method='zscore')
print(f"   ✓ Normalized signal (z-score)")

# Step 3: Decompose using VMD with K=7
print("\n3. Decomposing signal using VMD (K=7)...")
vmd = VMD(K=7, alpha=2000, tau=0.0, DC=False, init=1, tol=1e-7, max_iter=500)
modes = vmd.decompose(signal_clean)
print(f"   ✓ Extracted {len(modes)} modes")

# Step 4: Extract features
print("\n4. Extracting statistical features...")

print("\n   Raw Signal Features:")
raw_features = TimeDomainFeatures.extract(signal)
feature_names = ['mean', 'std', 'rms', 'skewness', 'kurtosis']
for feat_name in feature_names:
    if feat_name in raw_features:
        print(f"      {feat_name:15s}: {raw_features[feat_name]:12.6f}")

print("\n   Clean Signal Features:")
clean_features = TimeDomainFeatures.extract(signal_clean)
for feat_name in feature_names:
    if feat_name in clean_features:
        print(f"      {feat_name:15s}: {clean_features[feat_name]:12.6f}")

print("\n   VMD Mode Features:")
mode_features = []
for i, mode in enumerate(modes):
    print(f"\n   Mode {i+1}:")
    mode_feat = TimeDomainFeatures.extract(mode)
    mode_features.append(mode_feat)
    for feat_name in feature_names:
        if feat_name in mode_feat:
            print(f"      {feat_name:15s}: {mode_feat[feat_name]:12.6f}")

# Step 5: Create visualizations
print("\n5. Creating visualizations...")

time = np.arange(len(signal)) / 128
time_clean = np.arange(len(signal_clean)) / 128
time_modes = np.arange(len(modes[0])) / 128

# Figure 1: Raw and Clean Signals
fig1 = plt.figure(figsize=(16, 8))

ax1 = plt.subplot(2, 2, 1)
ax1.plot(time, signal, 'b-', linewidth=0.5, alpha=0.7)
ax1.set_title('Raw EEG Signal (EEG.AF3)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Time (s)', fontsize=11)
ax1.set_ylabel('Amplitude (μV)', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, time[-1]])

ax2 = plt.subplot(2, 2, 2)
zoom_samples = min(2560, len(signal))
ax2.plot(time[:zoom_samples], signal[:zoom_samples], 'b-', linewidth=0.8)
ax2.set_title('Raw Signal (Zoomed - First 20s)', fontsize=13, fontweight='bold')
ax2.set_xlabel('Time (s)', fontsize=11)
ax2.set_ylabel('Amplitude (μV)', fontsize=11)
ax2.grid(True, alpha=0.3)

ax3 = plt.subplot(2, 2, 3)
ax3.plot(time_clean, signal_clean, 'g-', linewidth=0.5, alpha=0.7)
ax3.set_title('Preprocessed Signal (Detrended → Denoised → Normalized)', 
              fontsize=13, fontweight='bold')
ax3.set_xlabel('Time (s)', fontsize=11)
ax3.set_ylabel('Normalized Amplitude', fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_xlim([0, time_clean[-1]])

ax4 = plt.subplot(2, 2, 4)
zoom_samples = min(2560, len(signal_clean))
ax4.plot(time_clean[:zoom_samples], signal_clean[:zoom_samples], 'g-', linewidth=0.8)
ax4.set_title('Preprocessed Signal (Zoomed - First 20s)', fontsize=13, fontweight='bold')
ax4.set_xlabel('Time (s)', fontsize=11)
ax4.set_ylabel('Normalized Amplitude', fontsize=11)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('EEG_Raw_Clean_Signals.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved raw/clean signals to: EEG_Raw_Clean_Signals.png")

# Figure 2: VMD Decomposed Modes (K=7)
fig2 = plt.figure(figsize=(16, 14))

colors = ['#E74C3C', '#F39C12', '#9B59B6', '#8E44AD', '#E91E63', '#3498DB', '#16A085']
mode_names = ['Mode 1 (Highest Freq)', 'Mode 2', 'Mode 3', 'Mode 4', 'Mode 5', 'Mode 6', 'Mode 7 (Lowest Freq)']

for i in range(7):
    ax = plt.subplot(7, 1, i + 1)
    ax.plot(time_modes, modes[i], color=colors[i], linewidth=0.6, alpha=0.9)
    ax.set_title(f'VMD {mode_names[i]}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Amplitude', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, time_modes[-1]])
    
    energy = np.sum(modes[i]**2)
    rms = np.sqrt(np.mean(modes[i]**2))
    mean_val = np.mean(modes[i])
    std_val = np.std(modes[i])
    
    info_text = f'Energy: {energy:.2f}\nRMS: {rms:.4f}\nMean: {mean_val:.4f}\nStd: {std_val:.4f}'
    ax.text(0.99, 0.97, info_text,
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    if i == 6:
        ax.set_xlabel('Time (s)', fontsize=11)

plt.tight_layout()
plt.savefig('EEG_VMD_Decomposed_Modes.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved VMD modes to: EEG_VMD_Decomposed_Modes.png")

# Figure 3: Frequency Spectra - Raw and Clean
fig3 = plt.figure(figsize=(14, 5))

def plot_spectrum(ax, signal_data, title, color='blue', fs=128):
    from scipy.fft import fft, fftfreq
    n = len(signal_data)
    fft_vals = fft(signal_data)
    freqs = fftfreq(n, 1/fs)
    
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    magnitude = np.abs(fft_vals[pos_mask])
    
    freq_limit = 60
    freq_mask = freqs_pos <= freq_limit
    
    ax.plot(freqs_pos[freq_mask], magnitude[freq_mask], color=color, linewidth=1.2)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel('Magnitude', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, freq_limit])
    
    bands = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 
             'Beta': (13, 30), 'Gamma': (30, 60)}
    y_max = ax.get_ylim()[1]
    for band_name, (low, high) in bands.items():
        ax.axvspan(low, high, alpha=0.1, color='gray')
        ax.text((low + high) / 2, y_max * 0.95, band_name, 
                fontsize=8, ha='center', va='top')

ax1 = plt.subplot(1, 2, 1)
plot_spectrum(ax1, signal, 'Raw Signal - Frequency Spectrum', '#2E86DE')
ax1.set_xlabel('Frequency (Hz)', fontsize=10)

ax2 = plt.subplot(1, 2, 2)
plot_spectrum(ax2, signal_clean, 'Preprocessed Signal - Frequency Spectrum', '#10AC84')
ax2.set_xlabel('Frequency (Hz)', fontsize=10)

plt.tight_layout()
plt.savefig('EEG_Signal_Spectra.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved signal spectra to: EEG_Signal_Spectra.png")

# Figure 4: VMD Modes Frequency Spectra
fig4 = plt.figure(figsize=(16, 12))

for i in range(7):
    ax = plt.subplot(4, 2, i + 1)
    plot_spectrum(ax, modes[i], f'VMD Mode {i+1} - Frequency Spectrum', colors[i])
    if i >= 5:  # Bottom row
        ax.set_xlabel('Frequency (Hz)', fontsize=10)

# Remove the 8th empty subplot
if len(fig4.axes) > 7:
    fig4.delaxes(fig4.axes[7])

plt.tight_layout()
plt.savefig('EEG_VMD_Mode_Spectra.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved VMD mode spectra to: EEG_VMD_Mode_Spectra.png")

# Step 6: Save features
print("\n6. Saving features to CSV...")
features_df = pd.DataFrame({
    'Feature': feature_names,
    'Raw_Signal': [raw_features[f] for f in feature_names],
    'Clean_Signal': [clean_features[f] for f in feature_names],
    'Mode_1': [mode_features[0][f] for f in feature_names],
    'Mode_2': [mode_features[1][f] for f in feature_names],
    'Mode_3': [mode_features[2][f] for f in feature_names],
    'Mode_4': [mode_features[3][f] for f in feature_names],
    'Mode_5': [mode_features[4][f] for f in feature_names],
    'Mode_6': [mode_features[5][f] for f in feature_names],
    'Mode_7': [mode_features[6][f] for f in feature_names],
})

features_df.to_csv('EEG_VMD_Features.csv', index=False)
print(f"   ✓ Saved features to: EEG_VMD_Features.csv")

# Step 7: Create Feature Matrix Visualization
print("\n7. Creating feature matrix visualization...")

# Prepare data for heatmap
feature_matrix = np.zeros((len(feature_names), 9))  # 9 columns: Raw, Clean, 7 Modes
column_labels = ['Raw', 'Clean', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7']

for i, feat in enumerate(feature_names):
    feature_matrix[i, 0] = raw_features[feat]
    feature_matrix[i, 1] = clean_features[feat]
    for j in range(7):
        feature_matrix[i, j+2] = mode_features[j][feat]

# Create figure for feature matrix
fig5 = plt.figure(figsize=(14, 8))

# Normalize each feature row for better visualization
feature_matrix_norm = np.zeros_like(feature_matrix)
for i in range(len(feature_names)):
    row_min = feature_matrix[i, :].min()
    row_max = feature_matrix[i, :].max()
    if row_max - row_min > 0:
        feature_matrix_norm[i, :] = (feature_matrix[i, :] - row_min) / (row_max - row_min)
    else:
        feature_matrix_norm[i, :] = 0

# Create heatmap
ax = plt.subplot(1, 1, 1)
im = ax.imshow(feature_matrix_norm, cmap='YlOrRd', aspect='auto', interpolation='nearest')

# Set ticks and labels
ax.set_xticks(np.arange(len(column_labels)))
ax.set_yticks(np.arange(len(feature_names)))
ax.set_xticklabels(column_labels, fontsize=11)
ax.set_yticklabels([f.capitalize() for f in feature_names], fontsize=11)

# Rotate the tick labels for better readability
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Normalized Feature Value', rotation=270, labelpad=20, fontsize=11)

# Add text annotations with actual values
for i in range(len(feature_names)):
    for j in range(len(column_labels)):
        text = ax.text(j, i, f'{feature_matrix[i, j]:.3f}',
                      ha="center", va="center", color="black", fontsize=9, 
                      fontweight='bold')

ax.set_title('EEG Feature Matrix Heatmap (VMD K=7)\nNormalized by Feature Row', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Signal Components', fontsize=12, fontweight='bold')
ax.set_ylabel('Statistical Features', fontsize=12, fontweight='bold')

# Add grid
ax.set_xticks(np.arange(len(column_labels))-.5, minor=True)
ax.set_yticks(np.arange(len(feature_names))-.5, minor=True)
ax.grid(which="minor", color="white", linestyle='-', linewidth=2)

plt.tight_layout()
plt.savefig('EEG_Feature_Matrix.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved feature matrix to: EEG_Feature_Matrix.png")

# Create second feature matrix with raw values
fig6 = plt.figure(figsize=(14, 8))
ax2 = plt.subplot(1, 1, 1)

# Create bar chart comparison
x = np.arange(len(column_labels))
width = 0.15
multiplier = 0

colors_bar = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

for i, feature in enumerate(feature_names):
    offset = width * multiplier
    values = feature_matrix[i, :]
    ax2.bar(x + offset, values, width, label=feature.capitalize(), color=colors_bar[i], alpha=0.8)
    multiplier += 1

ax2.set_xlabel('Signal Components', fontsize=12, fontweight='bold')
ax2.set_ylabel('Feature Value', fontsize=12, fontweight='bold')
ax2.set_title('EEG Feature Matrix - Bar Chart Comparison (VMD K=7)', 
              fontsize=14, fontweight='bold')
ax2.set_xticks(x + width * 2)
ax2.set_xticklabels(column_labels, fontsize=11)
ax2.legend(loc='upper right', fontsize=10, ncol=5)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('EEG_Feature_Matrix_BarChart.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved feature matrix bar chart to: EEG_Feature_Matrix_BarChart.png")

# Summary
print("\n" + "=" * 80)
print("Analysis Complete!")
print("=" * 80)
print("\nGenerated files:")
print("  1. EEG_Raw_Clean_Signals.png          - Raw and preprocessed signals")
print("  2. EEG_VMD_Decomposed_Modes.png       - All 7 VMD decomposition modes")
print("  3. EEG_Signal_Spectra.png             - Frequency domain analysis")
print("  4. EEG_VMD_Mode_Spectra.png           - VMD mode frequency spectra")
print("  5. EEG_Feature_Matrix.png             - Feature matrix heatmap")
print("  6. EEG_Feature_Matrix_BarChart.png    - Feature matrix bar chart")
print("  7. EEG_VMD_Features.csv               - Extracted statistical features")
print("\n" + "=" * 80)
print("\nEEG Channel Information:")
print("  Analyzing: EEG.AF3 (Left frontal channel)")
print("  Sample rate: 128 Hz")
print(f"  Duration: {len(signal)/128:.2f} seconds")
print(f"  Total samples: {len(signal)}")
print(f"  VMD Modes: 7")
print("\n" + "=" * 80)
print("\nFeature Summary Table:")
print(features_df.to_string(index=False))
print("\n" + "=" * 80)
print("\nEEG Frequency Bands:")
print("  Delta (0.5-4 Hz):   Deep sleep, unconscious")
print("  Theta (4-8 Hz):     Drowsiness, meditation, creativity")
print("  Alpha (8-13 Hz):    Relaxed, calm, awake")
print("  Beta (13-30 Hz):    Active thinking, focus, alertness")
print("  Gamma (30-60 Hz):   High-level information processing")
print("\n" + "=" * 80)

print("\nDisplaying plots...")
plt.show()