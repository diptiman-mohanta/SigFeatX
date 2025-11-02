import numpy as np
import sys
sys.path.insert(0, '..')

from SigFeatX import FeatureAggregator
from SigFeatX.utils import SignalUtils
import pandas as pd

# Generate multiple signals
n_signals = 10
signals = []

print("Batch Processing Example")
print("=" * 60)

for i in range(n_signals):
    t = np.linspace(0, 1, 1000)
    freq1 = 5 + i
    freq2 = 10 + i * 2
    sig = (np.sin(2 * np.pi * freq1 * t) + 
           0.5 * np.sin(2 * np.pi * freq2 * t) + 
           0.1 * np.random.randn(1000))
    signals.append(sig)

print(f"Generated {n_signals} signals")

# Initialize extractor
extractor = FeatureAggregator(fs=1000)

# Extract features from all signals
all_features = []
decomp_methods = ['fourier', 'dwt']

print(f"Extracting features using methods: {decomp_methods}")

for i, sig in enumerate(signals):
    features = extractor.extract_all_features(
        sig,
        decomposition_methods=decomp_methods,
        preprocess_signal=True
    )
    features['signal_id'] = i
    all_features.append(features)
    
    if (i + 1) % 5 == 0:
        print(f"Processed {i + 1}/{n_signals} signals...")

# Convert to DataFrame
df = pd.DataFrame(all_features)
print(f"\nFeature matrix shape: {df.shape}")
print(f"Columns: {df.shape[1]}")
print(f"Rows: {df.shape[0]}")

# Display summary statistics
print("\nSummary Statistics for Key Features:")
print("-" * 60)
key_cols = ['raw_mean', 'raw_std', 'raw_energy', 'raw_dominant_frequency']
print(df[key_cols].describe())

# Save to CSV
df.to_csv('batch_features.csv', index=False)
print("\nFeatures saved to 'batch_features.csv'")