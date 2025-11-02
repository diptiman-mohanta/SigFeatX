import numpy as np
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from typing import Dict, List, Tuple


class TimeDomainFeatures:
    """Extract time domain statistical features."""
    
    @staticmethod
    def extract(sig: np.ndarray) -> Dict[str, float]:
        """
        Extract all time domain features.
        
        Args:
            sig: Input signal
        
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(sig)
        features['std'] = np.std(sig)
        features['variance'] = np.var(sig)
        features['median'] = np.median(sig)
        features['mode'] = float(stats.mode(sig, keepdims=True)[0][0])
        features['max'] = np.max(sig)
        features['min'] = np.min(sig)
        features['range'] = np.ptp(sig)
        features['peak_to_peak'] = features['range']
        
        # Higher order statistics
        features['skewness'] = stats.skew(sig)
        features['kurtosis'] = stats.kurtosis(sig)
        
        # RMS and energy
        features['rms'] = np.sqrt(np.mean(sig**2))
        features['energy'] = np.sum(sig**2)
        features['power'] = features['energy'] / len(sig)
        
        # Absolute values
        features['mean_absolute'] = np.mean(np.abs(sig))
        features['sum_absolute'] = np.sum(np.abs(sig))
        
        # Zero crossings
        features['zero_crossings'] = np.sum(np.diff(np.sign(sig)) != 0)
        features['zero_crossing_rate'] = features['zero_crossings'] / len(sig)
        
        # Shape factors
        rms_val = features['rms']
        mean_abs = features['mean_absolute']
        peak = np.max(np.abs(sig))
        
        features['crest_factor'] = peak / (rms_val + 1e-10)
        features['shape_factor'] = rms_val / (mean_abs + 1e-10)
        features['impulse_factor'] = peak / (mean_abs + 1e-10)
        features['clearance_factor'] = peak / ((np.mean(np.sqrt(np.abs(sig))))**2 + 1e-10)
        
        # Percentiles
        features['q25'] = np.percentile(sig, 25)
        features['q75'] = np.percentile(sig, 75)
        features['iqr'] = features['q75'] - features['q25']
        
        # Coefficient of variation
        features['coeff_variation'] = features['std'] / (np.abs(features['mean']) + 1e-10)
        
        return features


class FrequencyDomainFeatures:
    """Extract frequency domain features."""
    
    @staticmethod
    def extract(sig: np.ndarray, fs: float = 1.0) -> Dict[str, float]:
        """
        Extract frequency domain features.
        
        Args:
            sig: Input signal
            fs: Sampling frequency
        
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Compute FFT
        n = len(sig)
        fft_vals = fft(sig)
        freqs = fftfreq(n, 1/fs)
        
        # Positive frequencies only
        pos_mask = freqs >= 0
        freqs = freqs[pos_mask]
        fft_vals = fft_vals[pos_mask]
        
        # Magnitude and power
        magnitude = np.abs(fft_vals)
        power = magnitude ** 2
        power_norm = power / (np.sum(power) + 1e-10)
        
        # Spectral centroid
        features['spectral_centroid'] = np.sum(freqs * power_norm)
        
        # Spectral spread (bandwidth)
        features['spectral_spread'] = np.sqrt(
            np.sum(((freqs - features['spectral_centroid'])**2) * power_norm)
        )
        features['spectral_bandwidth'] = features['spectral_spread']
        
        # Spectral rolloff (95% of energy)
        cumsum_power = np.cumsum(power_norm)
        rolloff_idx = np.where(cumsum_power >= 0.95)[0]
        features['spectral_rolloff'] = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
        
        # Dominant frequency
        features['dominant_frequency'] = freqs[np.argmax(power)]
        features['max_magnitude'] = np.max(magnitude)
        
        # Spectral flatness
        geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
        arithmetic_mean = np.mean(magnitude)
        features['spectral_flatness'] = geometric_mean / (arithmetic_mean + 1e-10)
        
        # Spectral entropy
        features['spectral_entropy'] = -np.sum(power_norm * np.log2(power_norm + 1e-10))
        
        # Spectral kurtosis and skewness
        features['spectral_kurtosis'] = np.sum(
            ((freqs - features['spectral_centroid'])**4) * power_norm
        ) / (features['spectral_spread']**4 + 1e-10)
        
        features['spectral_skewness'] = np.sum(
            ((freqs - features['spectral_centroid'])**3) * power_norm
        ) / (features['spectral_spread']**3 + 1e-10)
        
        # Energy in frequency bands
        nyquist = fs / 2
        bands = {
            'very_low': (0, 0.05 * nyquist),
            'low': (0.05 * nyquist, 0.15 * nyquist),
            'medium': (0.15 * nyquist, 0.4 * nyquist),
            'high': (0.4 * nyquist, 0.7 * nyquist),
            'very_high': (0.7 * nyquist, nyquist)
        }
        
        total_energy = np.sum(power)
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs < high)
            band_energy = np.sum(power[mask])
            features[f'energy_{band_name}'] = band_energy
            features[f'energy_ratio_{band_name}'] = band_energy / (total_energy + 1e-10)
        
        # Spectral flux (rate of change)
        if len(magnitude) > 1:
            features['spectral_flux'] = np.sum(np.diff(magnitude)**2)
        else:
            features['spectral_flux'] = 0.0
        
        return features


class EntropyFeatures:
    """Extract entropy-based features."""
    
    @staticmethod
    def extract(sig: np.ndarray) -> Dict[str, float]:
        """
        Extract entropy features.
        
        Args:
            sig: Input signal
        
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Shannon entropy
        features['shannon_entropy'] = EntropyFeatures._shannon_entropy(sig)
        
        # Sample entropy
        features['sample_entropy'] = EntropyFeatures._sample_entropy(sig)
        
        # Permutation entropy
        features['permutation_entropy'] = EntropyFeatures._permutation_entropy(sig)
        
        # Approximate entropy
        features['approximate_entropy'] = EntropyFeatures._approximate_entropy(sig)
        
        return features
    
    @staticmethod
    def _shannon_entropy(sig: np.ndarray, n_bins: int = 50) -> float:
        """Compute Shannon entropy."""
        hist, _ = np.histogram(sig, bins=n_bins, density=True)
        hist = hist[hist > 0]
        bin_width = (np.max(sig) - np.min(sig)) / n_bins
        entropy = -np.sum(hist * bin_width * np.log2(hist * bin_width + 1e-10))
        return entropy
    
    @staticmethod
    def _sample_entropy(sig: np.ndarray, m: int = 2, r: float = None) -> float:
        """Compute Sample Entropy."""
        if r is None:
            r = 0.2 * np.std(sig)
        
        N = len(sig)
        
        def _maxdist(x_i, x_j):
            return np.max(np.abs(x_i - x_j))
        
        def _phi(m):
            patterns = np.array([sig[i:i+m] for i in range(N-m)])
            count = 0
            for i in range(len(patterns)):
                for j in range(len(patterns)):
                    if i != j and _maxdist(patterns[i], patterns[j]) <= r:
                        count += 1
            return count / ((N - m) * (N - m - 1) + 1e-10)
        
        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)
        
        return -np.log(phi_m1 / (phi_m + 1e-10) + 1e-10)
    
    @staticmethod
    def _permutation_entropy(sig: np.ndarray, order: int = 3, delay: int = 1) -> float:
        """Compute Permutation Entropy."""
        n = len(sig)
        permutations = {}
        
        for i in range(n - delay * (order - 1)):
            pattern = sig[i:i + delay * order:delay]
            sorted_indices = tuple(np.argsort(pattern))
            permutations[sorted_indices] = permutations.get(sorted_indices, 0) + 1
        
        total = sum(permutations.values())
        if total == 0:
            return 0.0
        
        probs = np.array([count / total for count in permutations.values()])
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    @staticmethod
    def _approximate_entropy(sig: np.ndarray, m: int = 2, r: float = None) -> float:
        """Compute Approximate Entropy."""
        if r is None:
            r = 0.2 * np.std(sig)
        
        N = len(sig)
        
        def _maxdist(x_i, x_j):
            return np.max(np.abs(x_i - x_j))
        
        def _phi(m):
            patterns = np.array([sig[i:i+m] for i in range(N-m+1)])
            C = np.zeros(len(patterns))
            for i in range(len(patterns)):
                count = 0
                for j in range(len(patterns)):
                    if _maxdist(patterns[i], patterns[j]) <= r:
                        count += 1
                C[i] = count / len(patterns)
            return np.sum(np.log(C + 1e-10)) / len(patterns)
        
        return _phi(m) - _phi(m + 1)


class NonlinearFeatures:
    """Extract nonlinear dynamics features."""
    
    @staticmethod
    def extract(sig: np.ndarray) -> Dict[str, float]:
        """
        Extract nonlinear features.
        
        Args:
            sig: Input signal
        
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Hjorth parameters
        hjorth = NonlinearFeatures._hjorth_parameters(sig)
        features.update(hjorth)
        
        # Fractal dimension
        features['higuchi_fractal_dimension'] = NonlinearFeatures._higuchi_fractal_dimension(sig)
        features['petrosian_fractal_dimension'] = NonlinearFeatures._petrosian_fractal_dimension(sig)
        
        # Hurst exponent
        features['hurst_exponent'] = NonlinearFeatures._hurst_exponent(sig)
        
        # Lyapunov exponent (simplified)
        features['lyapunov_exponent'] = NonlinearFeatures._lyapunov_exponent(sig)
        
        # Detrended Fluctuation Analysis
        features['dfa_alpha'] = NonlinearFeatures._dfa(sig)
        
        return features
    
    @staticmethod
    def _hjorth_parameters(sig: np.ndarray) -> Dict[str, float]:
        """Compute Hjorth parameters."""
        # Activity
        activity = np.var(sig)
        
        # Mobility
        diff1 = np.diff(sig)
        mobility = np.sqrt(np.var(diff1) / (activity + 1e-10))
        
        # Complexity
        diff2 = np.diff(diff1)
        complexity = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10)) / (mobility + 1e-10)
        
        return {
            'hjorth_activity': activity,
            'hjorth_mobility': mobility,
            'hjorth_complexity': complexity
        }
    
    @staticmethod
    def _higuchi_fractal_dimension(sig: np.ndarray, kmax: int = 10) -> float:
        """Compute Higuchi Fractal Dimension."""
        n = len(sig)
        lk = np.zeros(kmax)
        
        for k in range(1, kmax + 1):
            lm = np.zeros(k)
            for m in range(k):
                n_max = int(np.floor((n - m - 1) / k))
                if n_max > 0:
                    length = np.sum(np.abs(np.diff(sig[m::k])))
                    lm[m] = length * (n - 1) / (n_max * k * k)
            lk[k-1] = np.mean(lm)
        
        lk = lk[lk > 0]
        if len(lk) < 2:
            return 1.0
        
        x = np.log(1 / np.arange(1, len(lk) + 1))
        y = np.log(lk)
        
        return np.polyfit(x, y, 1)[0]
    
    @staticmethod
    def _petrosian_fractal_dimension(sig: np.ndarray) -> float:
        """Compute Petrosian Fractal Dimension."""
        n = len(sig)
        diff = np.diff(sig)
        n_delta = np.sum(diff[:-1] * diff[1:] < 0)
        
        return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * n_delta)))
    
    @staticmethod
    def _hurst_exponent(sig: np.ndarray) -> float:
        """Compute Hurst Exponent using R/S analysis."""
        n = len(sig)
        if n < 20:
            return 0.5
        
        lags = np.arange(2, min(n // 2, 100))
        rs = np.zeros(len(lags))
        
        for i, lag in enumerate(lags):
            splits = n // lag
            if splits < 2:
                continue
            
            rs_temp = []
            for j in range(splits):
                subset = sig[j * lag:(j + 1) * lag]
                mean = np.mean(subset)
                cumdev = np.cumsum(subset - mean)
                r = np.max(cumdev) - np.min(cumdev)
                s = np.std(subset)
                if s > 0:
                    rs_temp.append(r / s)
            
            if rs_temp:
                rs[i] = np.mean(rs_temp)
        
        rs = rs[rs > 0]
        lags = lags[:len(rs)]
        
        if len(lags) < 2:
            return 0.5
        
        return np.polyfit(np.log(lags), np.log(rs), 1)[0]
    
    @staticmethod
    def _lyapunov_exponent(sig: np.ndarray, emb_dim: int = 3, lag: int = 1) -> float:
        """Compute largest Lyapunov exponent (simplified)."""
        n = len(sig)
        if n < emb_dim * lag + 1:
            return 0.0
        
        # Create embedded vectors
        embedded = []
        for i in range(n - emb_dim * lag):
            embedded.append(sig[i:i + emb_dim * lag:lag])
        embedded = np.array(embedded)
        
        if len(embedded) < 2:
            return 0.0
        
        # Find nearest neighbors and track divergence
        divergences = []
        for i in range(len(embedded) - 1):
            distances = np.linalg.norm(embedded - embedded[i], axis=1)
            distances[i] = np.inf
            nearest_idx = np.argmin(distances)
            
            if nearest_idx < len(embedded) - 1:
                dist_0 = distances[nearest_idx]
                dist_1 = np.linalg.norm(embedded[i+1] - embedded[nearest_idx+1])
                if dist_0 > 0 and dist_1 > 0:
                    divergences.append(np.log(dist_1 / dist_0))
        
        return np.mean(divergences) if divergences else 0.0
    
    @staticmethod
    def _dfa(sig: np.ndarray) -> float:
        """Detrended Fluctuation Analysis."""
        n = len(sig)
        if n < 16:
            return 1.0
        
        # Integrate the signal
        y = np.cumsum(sig - np.mean(sig))
        
        # Define box sizes
        scales = np.unique(np.logspace(0.5, np.log10(n // 4), 20).astype(int))
        fluctuations = np.zeros(len(scales))
        
        for i, scale in enumerate(scales):
            # Divide into boxes
            n_boxes = n // scale
            boxes = []
            
            for j in range(n_boxes):
                box = y[j * scale:(j + 1) * scale]
                # Fit polynomial and calculate fluctuation
                x = np.arange(len(box))
                coeffs = np.polyfit(x, box, 1)
                fit = np.polyval(coeffs, x)
                boxes.append(np.sqrt(np.mean((box - fit)**2)))
            
            fluctuations[i] = np.mean(boxes)
        
        # Calculate scaling exponent
        valid = fluctuations > 0
        if np.sum(valid) < 2:
            return 1.0
        
        coeffs = np.polyfit(np.log(scales[valid]), np.log(fluctuations[valid]), 1)
        return coeffs[0]


class DecompositionFeatures:
    """Extract features from decomposed signals."""
    
    @staticmethod
    def extract_from_components(components: List[np.ndarray], 
                                prefix: str = 'comp') -> Dict[str, float]:
        """
        Extract features from decomposition components.
        
        Args:
            components: List of signal components
            prefix: Prefix for feature names
        
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Per-component features
        for i, comp in enumerate(components):
            comp_prefix = f'{prefix}_{i}'
            
            # Energy and power
            features[f'{comp_prefix}_energy'] = np.sum(comp**2)
            features[f'{comp_prefix}_rms'] = np.sqrt(np.mean(comp**2))
            features[f'{comp_prefix}_mean'] = np.mean(comp)
            features[f'{comp_prefix}_std'] = np.std(comp)
            features[f'{comp_prefix}_max'] = np.max(np.abs(comp))
            
            # Entropy
            hist, _ = np.histogram(comp, bins=30, density=True)
            hist = hist[hist > 0]
            features[f'{comp_prefix}_entropy'] = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Energy ratios
        total_energy = sum(features[f'{prefix}_{i}_energy'] for i in range(len(components)))
        for i in range(len(components)):
            features[f'{prefix}_{i}_energy_ratio'] = (
                features[f'{prefix}_{i}_energy'] / (total_energy + 1e-10)
            )
        
        # Cross-component features
        if len(components) > 1:
            cross_features = DecompositionFeatures._cross_component_features(
                components, prefix
            )
            features.update(cross_features)
        
        return features
    
    @staticmethod
    def _cross_component_features(components: List[np.ndarray], 
                                  prefix: str) -> Dict[str, float]:
        """Extract cross-component features."""
        features = {}
        n_comp = len(components)
        
        # Correlations
        for i in range(n_comp):
            for j in range(i + 1, n_comp):
                min_len = min(len(components[i]), len(components[j]))
                comp_i = components[i][:min_len]
                comp_j = components[j][:min_len]
                
                # Correlation
                corr = np.corrcoef(comp_i, comp_j)[0, 1]
                features[f'{prefix}_corr_{i}_{j}'] = corr if not np.isnan(corr) else 0.0
                
                # Energy ratio
                energy_i = np.sum(comp_i**2)
                energy_j = np.sum(comp_j**2)
                features[f'{prefix}_energy_ratio_{i}_{j}'] = (
                    energy_i / (energy_j + 1e-10)
                )
        
        # Relative entropy (KL divergence)
        for i in range(min(n_comp, 5)):  # Limit to first 5 components
            for j in range(i + 1, min(n_comp, 5)):
                min_len = min(len(components[i]), len(components[j]))
                comp_i = components[i][:min_len]
                comp_j = components[j][:min_len]
                
                kl = DecompositionFeatures._kl_divergence(comp_i, comp_j)
                features[f'{prefix}_kl_div_{i}_{j}'] = kl
        
        return features
    
    @staticmethod
    def _kl_divergence(p_data: np.ndarray, q_data: np.ndarray, n_bins: int = 30) -> float:
        """Compute KL divergence between two signals."""
        # Create histograms
        range_min = min(np.min(p_data), np.min(q_data))
        range_max = max(np.max(p_data), np.max(q_data))
        
        p_hist, _ = np.histogram(p_data, bins=n_bins, range=(range_min, range_max), density=True)
        q_hist, _ = np.histogram(q_data, bins=n_bins, range=(range_min, range_max), density=True)
        
        # Add small constant to avoid log(0)
        p_hist = p_hist + 1e-10
        q_hist = q_hist + 1e-10
        
        # Normalize
        p_hist = p_hist / np.sum(p_hist)
        q_hist = q_hist / np.sum(q_hist)
        
        return np.sum(p_hist * np.log(p_hist / q_hist))
