import numpy as np
import sys
sys.path.insert(0, '..')

from SigFeatX import FeatureAggregator, SignalIO

# Generate sample signal
t = np.linspace(0, 1, 1000)
signal = (np.sin(2 * np.pi * 5 * t) + 
          0.5 * np.sin(2 * np.pi * 10 * t) + 
          0.1 * np.random.randn(1000))

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

# Display results
print("Extracted Features:")
print("=" * 50)
for i, (key, value) in enumerate(features.items()):
    print(f"{key}: {value:.4f}")
    if i >= 19:  # Show first 20 features
        print(f"... and {len(features) - 20} more features")
        break

print(f"\nTotal features extracted: {len(features)}")

# Save features
io = SignalIO()
io.save_features(features, 'features.json', file_format='json')
print("\nFeatures saved to 'features.json'")