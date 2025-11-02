# SigFeatX
A comprehensive Python library for extracting statistical features from 1D signals using advanced decomposition techniques and signal processing metrics.
## Features

### Signal Preprocessing
- **Denoising**: Wavelet, median filter, lowpass filter
- **Normalization**: Z-score, min-max, robust (MAD-based)
- **Detrending**: Linear and constant detrending
- **Resampling**: Linear and Fourier-based resampling

### Decomposition Methods
- **Fourier Transform (FT)**: Classical frequency domain decomposition
- **Short-Time Fourier Transform (STFT)**: Time-frequency analysis
- **Discrete Wavelet Transform (DWT)**: Multi-resolution analysis
- **Wavelet Packet Decomposition (WPD)**: Full wavelet tree decomposition
- **Empirical Mode Decomposition (EMD)**: Data-driven decomposition
- **Variational Mode Decomposition (VMD)**: Optimization-based decomposition
- **Successive VMD (SVMD)**: Sequential mode extraction
- **Empirical Fourier Decomposition (EFD)**: Adaptive frequency bands

### Statistical Features (100+)

#### Time Domain
- Basic: Mean, Std, Variance, Median, Mode, Min, Max, Range
- Energy: RMS, Energy, Power, Mean Absolute Value
- Shape: Crest Factor, Shape Factor, Impulse Factor, Clearance Factor
- Distribution: Skewness, Kurtosis, Percentiles, IQR
- Signal Characteristics: Zero Crossings, Peak-to-Peak

#### Frequency Domain
- Spectral: Centroid, Bandwidth, Rolloff, Flatness, Flux
- Energy Distribution: Energy in frequency bands (very low, low, medium, high, very high)
- Statistical: Spectral Entropy, Skewness, Kurtosis
- Dominant Frequency and Maximum Magnitude

#### Entropy Measures
- Shannon Entropy
- Sample Entropy
- Permutation Entropy
- Approximate Entropy
- Spectral Entropy

#### Nonlinear Dynamics
- Hjorth Parameters: Activity, Mobility, Complexity
- Fractal Dimensions: Higuchi, Petrosian
- Hurst Exponent (R/S analysis)
- Lyapunov Exponent
- Detrended Fluctuation Analysis (DFA)

#### Decomposition-Based Features
- Energy per level/component
- RMS per level
- Entropy per level
- Correlations between components
- Energy ratios between components
- Relative entropy (KL divergence) between components

## Installation

```bash
# Clone the repository
git clone https://github.com/diptiman-mohanta/SigFeatX.git
cd SigFeatX

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Requirements
```
numpy>=1.20.0
scipy>=1.7.0
PyWavelets>=1.1.1
pandas>=1.3.0
```

## Quick Start

```python
import numpy as np
from SigFeatX import FeatureAggregator

# Generate sample signal
t = np.linspace(0, 1, 1000)
signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)

# Initialize feature extractor
extractor = FeatureAggregator(fs=1000)

# Extract features
features = extractor.extract_all_features(
    signal,
    decomposition_methods=['fourier', 'dwt', 'emd'],
    preprocess_signal=True,
    denoise=True,
    normalize=True,
    detrend=True
)

print(f"Extracted {len(features)} features")
print(f"Mean: {features['raw_mean']:.4f}")
print(f"RMS: {features['raw_rms']:.4f}")
print(f"Dominant Frequency: {features['raw_dominant_frequency']:.2f} Hz")
```

## Usage Examples

### Basic Feature Extraction

```python
from SigFeatX import FeatureAggregator, SignalIO

# Load signal
io = SignalIO()
signal = io.load_signal('signal.npy')

# Extract features
extractor = FeatureAggregator(fs=1000)
features = extractor.extract_all_features(signal)

# Save features
io.save_features(features, 'features.json')
```

### Custom Preprocessing

```python
from SigFeatX.preprocess import SignalPreprocessor

preprocessor = SignalPreprocessor()

# Denoise with wavelet
clean_signal = preprocessor.denoise(signal, method='wavelet', 
                                   wavelet='db6', level=3)

# Normalize
normalized = preprocessor.normalize(clean_signal, method='robust')

# Detrend
detrended = preprocessor.detrend(normalized, method='linear')
```

### Individual Decomposition Methods

```python
from SigFeatX.decompose import WaveletDecomposer, EMD, VMD

# Wavelet decomposition
wavelet = WaveletDecomposer(wavelet='db4')
coeffs = wavelet.dwt(signal, level=4)

# EMD
emd = EMD(max_imf=10)
imfs = emd.decompose(signal)

# VMD
vmd = VMD(K=5, alpha=2000)
modes = vmd.decompose(signal)
```

### Specific Feature Categories

```python
from SigFeatX.features.features import (
    TimeDomainFeatures,
    FrequencyDomainFeatures,
    EntropyFeatures,
    NonlinearFeatures
)

# Extract specific feature categories
time_features = TimeDomainFeatures.extract(signal)
freq_features = FrequencyDomainFeatures.extract(signal, fs=1000)
entropy_features = EntropyFeatures.extract(signal)
nonlinear_features = NonlinearFeatures.extract(signal)
```

### Batch Processing

```python
import pandas as pd

signals = [signal1, signal2, signal3, ...]
all_features = []

for sig in signals:
    features = extractor.extract_all_features(sig)
    all_features.append(features)

# Convert to DataFrame
df = pd.DataFrame(all_features)
df.to_csv('features.csv', index=False)
```

### Using Utility Functions

```python
from SigFeatX.utils import SignalUtils

# Create sliding windows
windows = SignalUtils.sliding_window(signal, window_size=256, step_size=128)

# Segment signal
segments = SignalUtils.segment_signal(signal, n_segments=5)

# Detect peaks
peaks, properties = SignalUtils.detect_peaks(signal, height=0.5)

# Compute envelope
envelope = SignalUtils.compute_envelope(signal)

# Add noise
noisy_signal = SignalUtils.add_noise(signal, snr_db=10)
```

## API Reference

### FeatureAggregator

Main class for feature extraction pipeline.

```python
FeatureAggregator(fs=1.0)
```

**Parameters:**
- `fs`: Sampling frequency (default: 1.0)

**Methods:**

#### `extract_all_features(signal, decomposition_methods=None, preprocess_signal=True, extract_raw=True, **preprocess_kwargs)`

Extract all features from signal.

**Parameters:**
- `signal`: Input 1D numpy array
- `decomposition_methods`: List of decomposition methods to apply
  - Options: `['fourier', 'stft', 'dwt', 'wpd', 'emd', 'vmd', 'svmd', 'efd']`
- `preprocess_signal`: Whether to preprocess signal (default: True)
- `extract_raw`: Extract features from raw signal (default: True)
- `**preprocess_kwargs`: Preprocessing parameters
  - `denoise`: Apply denoising (default: True)
  - `normalize`: Apply normalization (default: True)
  - `detrend`: Apply detrending (default: True)
  - `denoise_method`: Denoising method ('wavelet', 'median', 'lowpass')
  - `normalize_method`: Normalization method ('zscore', 'minmax', 'robust')
  - `detrend_method`: Detrending method ('linear', 'constant')

**Returns:**
- Dictionary of feature names and values

### SignalPreprocessor

Preprocessing operations for signals.

```python
SignalPreprocessor()
```

**Methods:**

#### `denoise(sig, method='wavelet', **kwargs)`

Denoise signal using various methods.

#### `normalize(sig, method='zscore')`

Normalize signal.

#### `detrend(sig, method='linear')`

Remove trend from signal.

#### `resample(sig, target_length, method='linear')`

Resample signal to target length.

### Decomposition Classes

#### FourierTransform
```python
FourierTransform(fs=1.0)
```
- `transform(signal)`: Compute FFT
- `get_power_spectrum(signal)`: Get power spectrum
- `get_phase_spectrum(signal)`: Get phase spectrum

#### WaveletDecomposer
```python
WaveletDecomposer(wavelet='db4')
```
- `dwt(signal, level=None)`: Discrete Wavelet Transform
- `wpd(signal, level=3)`: Wavelet Packet Decomposition
- `cwt(signal, scales=None)`: Continuous Wavelet Transform
- `swt(signal, level=1)`: Stationary Wavelet Transform

#### EMD
```python
EMD(max_imf=10, max_iter=100)
```
- `decompose(signal)`: Decompose into IMFs
- `reconstruct(imfs)`: Reconstruct signal from IMFs

#### VMD
```python
VMD(alpha=2000, K=3, tau=0.0, DC=False, init=1, tol=1e-7, max_iter=500)
```
- `decompose(signal)`: Decompose into modes
- `reconstruct(modes)`: Reconstruct signal from modes

#### SVMD
```python
SVMD(alpha=2000, K_max=10, tol=1e-7, max_iter=500)
```
- `decompose(signal)`: Successive decomposition

#### EFD
```python
EFD(n_modes=5)
```
- `decompose(signal)`: Empirical Fourier decomposition

### Feature Extraction Classes

#### TimeDomainFeatures
```python
TimeDomainFeatures.extract(signal)
```
Returns dictionary with 25+ time domain features.

#### FrequencyDomainFeatures
```python
FrequencyDomainFeatures.extract(signal, fs=1.0)
```
Returns dictionary with 20+ frequency domain features.

#### EntropyFeatures
```python
EntropyFeatures.extract(signal)
```
Returns dictionary with 4 entropy measures.

#### NonlinearFeatures
```python
NonlinearFeatures.extract(signal)
```
Returns dictionary with 9+ nonlinear features.

#### DecompositionFeatures
```python
DecompositionFeatures.extract_from_components(components, prefix='comp')
```
Extract features from decomposition components.

### SignalIO

Handle signal data I/O operations.

```python
SignalIO()
```

**Methods:**
- `load_signal(filepath, file_format='npy')`: Load signal from file
- `save_signal(signal, filepath, file_format='npy')`: Save signal to file
- `save_features(features, filepath, file_format='json')`: Save features
- `load_features(filepath, file_format='json')`: Load features

### SignalUtils

Utility functions for signal processing.

```python
SignalUtils()
```

**Static Methods:**
- `sliding_window(sig, window_size, step_size)`: Create sliding windows
- `pad_signal(sig, target_length, mode='constant')`: Pad signal
- `segment_signal(sig, n_segments)`: Segment signal
- `compute_snr(sig, noise)`: Compute SNR
- `detect_peaks(sig, height=None, distance=None)`: Detect peaks
- `compute_envelope(sig)`: Compute signal envelope
- `compute_instantaneous_frequency(sig, fs=1.0)`: Compute inst. frequency
- `remove_outliers(sig, n_std=3.0)`: Remove outliers
- `add_noise(sig, snr_db, noise_type='gaussian')`: Add noise

## Feature List

### Time Domain (25 features)
- mean, std, variance, median, mode
- max, min, range, peak_to_peak
- skewness, kurtosis
- rms, energy, power
- mean_absolute, sum_absolute
- zero_crossings, zero_crossing_rate
- crest_factor, shape_factor, impulse_factor, clearance_factor
- q25, q75, iqr, coeff_variation

### Frequency Domain (20+ features)
- spectral_centroid, spectral_spread, spectral_bandwidth
- spectral_rolloff, dominant_frequency, max_magnitude
- spectral_flatness, spectral_entropy, spectral_flux
- spectral_kurtosis, spectral_skewness
- energy_very_low, energy_low, energy_medium, energy_high, energy_very_high
- energy_ratio_very_low, energy_ratio_low, energy_ratio_medium, energy_ratio_high, energy_ratio_very_high

### Entropy Features (4 features)
- shannon_entropy
- sample_entropy
- permutation_entropy
- approximate_entropy

### Nonlinear Features (9 features)
- hjorth_activity, hjorth_mobility, hjorth_complexity
- higuchi_fractal_dimension, petrosian_fractal_dimension
- hurst_exponent
- lyapunov_exponent
- dfa_alpha

### Decomposition Features (per component)
- energy, rms, mean, std, max, entropy
- energy_ratio
- correlations between components
- energy ratios between components
- kl_divergence between components

## Advanced Topics

### Custom Wavelet Selection

```python
# Available wavelets: db1-db20, sym2-sym20, coif1-coif17, bior, rbio, dmey
wavelet = WaveletDecomposer(wavelet='sym8')
coeffs = wavelet.dwt(signal, level=5)
```

### Optimizing VMD Parameters

```python
# For signals with known number of modes
vmd = VMD(K=3, alpha=2000)  # K = number of modes

# For noisy signals, increase alpha
vmd = VMD(K=3, alpha=5000)

# For signals with closely spaced frequencies
vmd = VMD(K=3, alpha=1000, init=2)  # Random initialization
```

### Memory-Efficient Batch Processing

```python
def extract_features_generator(signals, extractor):
    for sig in signals:
        yield extractor.extract_all_features(sig)

# Process large datasets
features_gen = extract_features_generator(large_signal_list, extractor)
for i, features in enumerate(features_gen):
    # Process or save features incrementally
    io.save_features(features, f'features_{i}.json')
```

### Parallel Processing

```python
from multiprocessing import Pool

def extract_wrapper(sig):
    extractor = FeatureAggregator(fs=1000)
    return extractor.extract_all_features(sig)

# Process signals in parallel
with Pool(processes=4) as pool:
    all_features = pool.map(extract_wrapper, signals)
```

## Performance Considerations

- **Signal Length**: Most methods work efficiently for signals up to 10,000 samples
- **Decomposition Methods**: EMD and VMD are computationally expensive; use DWT for faster processing
- **Feature Count**: Extracting all features with all decomposition methods can yield 200+ features
- **Memory**: VMD and SVMD require more memory for long signals

## Troubleshooting

### Common Issues

**Issue**: Features contain NaN or Inf values
```python
# Solution: Check signal quality and preprocessing
features = {k: v for k, v in features.items() if not (np.isnan(v) or np.isinf(v))}
```

**Issue**: EMD fails to converge
```python
# Solution: Reduce max_imf or smooth signal first
emd = EMD(max_imf=5, max_iter=50)
```

**Issue**: Memory error with large signals
```python
# Solution: Use sliding windows
windows = SignalUtils.sliding_window(signal, window_size=1000, step_size=500)
for window in windows:
    features = extractor.extract_all_features(window)
```

## Citation

If you use SigFeatX in your research, please cite:

```bibtex
@software{sigfeatx2024,
  title={SigFeatX: Signal Feature Extraction Library},
  author={Diptiman Mohanta},
  year={2024},
  url={https://github.com/diptiman-mohanta/SigFeatX}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyWavelets for wavelet transforms
- SciPy for signal processing functions
- NumPy for numerical computations

## Contact

For questions and support, please open an issue on GitHub.