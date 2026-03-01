import numpy as np
import sys
sys.path.insert(0, '..')

from SigFeatX.aggregator import FeatureAggregator, SignalPreprocessor
from SigFeatX.decompose import WaveletDecomposer, EMD, VMD
from SigFeatX.features.features import DecompositionFeatures

# Generate complex signal
t = np.linspace(0, 2, 2000)
signal = (np.sin(2 * np.pi * 5 * t) + 
          0.5 * np.sin(2 * np.pi * 15 * t) * np.exp(-t) +
          0.3 * np.sin(2 * np.pi * 25 * t) +
          0.2 * np.random.randn(2000))

print("Advanced Feature Extraction Example")
print("=" * 60)

# Preprocessing
preprocessor = SignalPreprocessor()
signal_clean = preprocessor.detrend(signal)
signal_clean = preprocessor.denoise(signal_clean, method='wavelet', level=3)
signal_clean = preprocessor.normalize(signal_clean, method='zscore')

print(f"Signal length: {len(signal)}")
print(f"Preprocessing steps: {preprocessor.get_history()}")

# Extract features using all decomposition methods
extractor = FeatureAggregator(fs=1000)

all_methods = ['fourier', 'stft', 'dwt', 'wpd', 'emd', 'vmd', 'svmd', 'efd']

print(f"\nExtracting features using methods: {all_methods}")

features = extractor.extract_all_features(
    signal_clean,
    decomposition_methods=all_methods,
    preprocess_signal=False,  # Already preprocessed
    extract_raw=True
)

print(f"\nTotal features extracted: {len(features)}")

# Analyze feature categories
categories = {}
for key in features.keys():
    category = key.split('_')[0]
    categories[category] = categories.get(category, 0) + 1

print("\nFeatures by category:")
for category, count in sorted(categories.items()):
    print(f"  {category}: {count} features")

# Display some key features
print("\nKey Features:")
print("-" * 60)
key_features = [
    'raw_mean', 'raw_std', 'raw_rms', 'raw_energy',
    'raw_shannon_entropy', 'raw_hurst_exponent',
    'raw_hjorth_mobility', 'raw_spectral_centroid',
    'dwt_0_energy', 'emd_0_energy', 'vmd_0_energy'
]

for feat in key_features:
    if feat in features:
        print(f"{feat:30s}: {features[feat]:12.4f}")