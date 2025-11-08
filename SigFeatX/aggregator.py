import numpy as np
from typing import Dict, List, Optional
from .preprocess import SignalPreprocessor
from .decompose import (
    FourierTransform, ShortTimeFourierTransform, WaveletDecomposer,
    EMD, VMD, SVMD, EFD
)
from .features.features import (
    TimeDomainFeatures, FrequencyDomainFeatures,
    EntropyFeatures, NonlinearFeatures, DecompositionFeatures
)


class FeatureAggregator:
    """Main class for feature extraction pipeline."""
    
    def __init__(self, fs: float = 1.0):
        """
        Initialize feature aggregator.
        
        Args:
            fs: Sampling frequency
        """
        self.fs = fs
        self.preprocessor = SignalPreprocessor()
        
        # Initialize decomposers
        self.ft = FourierTransform(fs=fs)
        self.stft = ShortTimeFourierTransform(fs=fs)
        self.wavelet = WaveletDecomposer()
        self.emd = EMD()
        self.vmd = VMD()
        self.svmd = SVMD()
        self.efd = EFD()
        
        # Initialize feature extractors
        self.time_features = TimeDomainFeatures()
        self.freq_features = FrequencyDomainFeatures()
        self.entropy_features = EntropyFeatures()
        self.nonlinear_features = NonlinearFeatures()
        self.decomp_features = DecompositionFeatures()
    
    def preprocess(self, signal: np.ndarray, 
                   denoise: bool = True,
                   normalize: bool = True,
                   detrend: bool = True,
                   **kwargs) -> np.ndarray:
        """
        Preprocess signal.
        
        Args:
            signal: Input signal
            denoise: Apply denoising
            normalize: Apply normalization
            detrend: Apply detrending
            **kwargs: Additional preprocessing parameters
        
        Returns:
            Preprocessed signal
        """
        processed = signal.copy()
        
        if detrend:
            processed = self.preprocessor.detrend(
                processed, 
                method=kwargs.get('detrend_method', 'linear')
            )
        
        if denoise:
            processed = self.preprocessor.denoise(
                processed,
                method=kwargs.get('denoise_method', 'wavelet'),
                **kwargs.get('denoise_params', {})
            )
        
        if normalize:
            processed = self.preprocessor.normalize(
                processed,
                method=kwargs.get('normalize_method', 'zscore')
            )
        
        return processed
    
    def extract_all_features(self, signal: np.ndarray,
                            decomposition_methods: Optional[List[str]] = None,
                            preprocess_signal: bool = True,
                            extract_raw: bool = True,
                            **preprocess_kwargs) -> Dict[str, float]:
        """
        Extract all features from signal.
        
        Args:
            signal: Input signal
            decomposition_methods: List of decomposition methods
                Options: 'fourier', 'stft', 'dwt', 'wpd', 'emd', 'vmd', 'svmd', 'efd'
            preprocess_signal: Whether to preprocess
            extract_raw: Extract features from raw signal
            **preprocess_kwargs: Preprocessing parameters
        
        Returns:
            Dictionary of all features
        """
        if decomposition_methods is None:
            decomposition_methods = ['fourier', 'dwt']
        
        all_features = {}
        
        # Preprocess
        if preprocess_signal:
            sig = self.preprocess(signal, **preprocess_kwargs)
        else:
            sig = signal.copy()
        
        # Extract raw signal features
        if extract_raw:
            raw_features = self._extract_raw_features(sig)
            all_features.update(self._add_prefix(raw_features, 'raw'))
        
        # Extract decomposition-based features
        for method in decomposition_methods:
            decomp_features = self._extract_decomposition_features(sig, method)
            all_features.update(self._add_prefix(decomp_features, method))
        
        return all_features
    
    def _extract_raw_features(self, sig: np.ndarray) -> Dict[str, float]:
        """Extract features from raw signal."""
        features = {}
        
        # Time domain
        features.update(self.time_features.extract(sig))
        
        # Frequency domain
        features.update(self.freq_features.extract(sig, self.fs))
        
        # Entropy
        features.update(self.entropy_features.extract(sig))
        
        # Nonlinear
        features.update(self.nonlinear_features.extract(sig))
        
        return features
    
    def _extract_decomposition_features(self, sig: np.ndarray, 
                                       method: str) -> Dict[str, float]:
        """Extract features from decomposition."""
        features = {}
        
        if method == 'fourier':
            freqs, magnitude = self.ft.transform(sig)
            features['dominant_freq'] = freqs[np.argmax(magnitude)]
            features['mean_magnitude'] = np.mean(magnitude)
            features['std_magnitude'] = np.std(magnitude)
        
        elif method == 'stft':
            f, t, Zxx = self.stft.transform(sig)
            features['mean_power'] = np.mean(Zxx**2)
            features['std_power'] = np.std(Zxx**2)
            features['max_power'] = np.max(Zxx**2)
        
        elif method == 'dwt':
            coeffs = self.wavelet.dwt(sig)
            features.update(self.decomp_features.extract_from_components(coeffs, 'dwt'))
        
        elif method == 'wpd':
            wpd_dict = self.wavelet.wpd(sig)
            wpd_coeffs = list(wpd_dict.values())
            features.update(self.decomp_features.extract_from_components(wpd_coeffs, 'wpd'))
        
        elif method == 'emd':
            imfs = self.emd.decompose(sig)
            features.update(self.decomp_features.extract_from_components(imfs, 'emd'))
        
        elif method == 'vmd':
            modes = self.vmd.decompose(sig)
            modes_list = [modes[i] for i in range(len(modes))]
            features.update(self.decomp_features.extract_from_components(modes_list, 'vmd'))
        
        elif method == 'svmd':
            modes = self.svmd.decompose(sig)
            modes_list = [modes[i] for i in range(len(modes))]
            features.update(self.decomp_features.extract_from_components(modes_list, 'svmd'))
        
        elif method == 'efd':
            modes = self.efd.decompose(sig)
            modes_list = [modes[i] for i in range(len(modes))]
            features.update(self.decomp_features.extract_from_components(modes_list, 'efd'))
        
        return features
    
    @staticmethod
    def _add_prefix(features: Dict[str, float], prefix: str) -> Dict[str, float]:
        """Add prefix to feature names."""
        return {f'{prefix}_{k}': v for k, v in features.items()}
    
    def get_feature_names(self, decomposition_methods: Optional[List[str]] = None) -> List[str]:
        """
        Get list of all feature names that would be extracted.
        
        Args:
            decomposition_methods: List of decomposition methods
        
        Returns:
            List of feature names
        """
        if decomposition_methods is None:
            decomposition_methods = ['fourier', 'dwt']
        
        # Create dummy signal
        dummy_signal = np.random.randn(1000)
        
        # Extract features
        features = self.extract_all_features(
            dummy_signal,
            decomposition_methods=decomposition_methods,
            preprocess_signal=False
        )
        
        return list(features.keys())
