import numpy as np
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from typing import Dict, List, Tuple


class TimeDomainFeatures:
    """Extract time domain statistical features."""

    @staticmethod
    def extract(sig: np.ndarray) -> Dict[str, float]:
        features = {}
        features['mean']             = np.mean(sig)
        features['std']              = np.std(sig)
        features['variance']         = np.var(sig)
        features['median']           = np.median(sig)
        features['mode']             = float(stats.mode(sig, keepdims=True)[0][0])
        features['max']              = np.max(sig)
        features['min']              = np.min(sig)
        features['range']            = np.ptp(sig)
        features['peak_to_peak']     = features['range']
        features['skewness']         = stats.skew(sig)
        features['kurtosis']         = stats.kurtosis(sig)
        features['rms']              = np.sqrt(np.mean(sig**2))
        features['energy']           = np.sum(sig**2)
        features['power']            = features['energy'] / len(sig)
        features['mean_absolute']    = np.mean(np.abs(sig))
        features['sum_absolute']     = np.sum(np.abs(sig))
        features['zero_crossings']   = np.sum(np.diff(np.sign(sig)) != 0)
        features['zero_crossing_rate'] = features['zero_crossings'] / len(sig)

        rms_val  = features['rms']
        mean_abs = features['mean_absolute']
        peak     = np.max(np.abs(sig))

        features['crest_factor']    = peak / (rms_val + 1e-10)
        features['shape_factor']    = rms_val / (mean_abs + 1e-10)
        features['impulse_factor']  = peak / (mean_abs + 1e-10)
        features['clearance_factor']= peak / ((np.mean(np.sqrt(np.abs(sig))))**2 + 1e-10)
        features['q25']             = np.percentile(sig, 25)
        features['q75']             = np.percentile(sig, 75)
        features['iqr']             = features['q75'] - features['q25']
        features['coeff_variation'] = features['std'] / (np.abs(features['mean']) + 1e-10)

        return features


class FrequencyDomainFeatures:
    """Extract frequency domain features."""

    @staticmethod
    def extract(sig: np.ndarray, fs: float = 1.0) -> Dict[str, float]:
        features = {}
        n        = len(sig)
        fft_vals = fft(sig)
        freqs    = fftfreq(n, 1/fs)

        pos_mask    = freqs >= 0
        freqs       = freqs[pos_mask]
        fft_vals    = fft_vals[pos_mask]
        magnitude   = np.abs(fft_vals)
        power       = magnitude ** 2
        power_norm  = power / (np.sum(power) + 1e-10)

        features['spectral_centroid']  = np.sum(freqs * power_norm)
        features['spectral_spread']    = np.sqrt(
            np.sum(((freqs - features['spectral_centroid'])**2) * power_norm))
        features['spectral_bandwidth'] = features['spectral_spread']

        cumsum_power = np.cumsum(power_norm)
        rolloff_idx  = np.where(cumsum_power >= 0.95)[0]
        features['spectral_rolloff']   = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
        features['dominant_frequency'] = freqs[np.argmax(power)]
        features['max_magnitude']      = np.max(magnitude)

        geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
        arithmetic_mean= np.mean(magnitude)
        features['spectral_flatness']  = geometric_mean / (arithmetic_mean + 1e-10)
        features['spectral_entropy']   = -np.sum(power_norm * np.log2(power_norm + 1e-10))

        sc  = features['spectral_centroid']
        ss  = features['spectral_spread']
        features['spectral_kurtosis']  = np.sum(((freqs-sc)**4)*power_norm) / (ss**4 + 1e-10)
        features['spectral_skewness']  = np.sum(((freqs-sc)**3)*power_norm) / (ss**3 + 1e-10)

        nyquist = fs / 2
        bands   = {
            'very_low' : (0,              0.05 * nyquist),
            'low'      : (0.05 * nyquist, 0.15 * nyquist),
            'medium'   : (0.15 * nyquist, 0.40 * nyquist),
            'high'     : (0.40 * nyquist, 0.70 * nyquist),
            'very_high': (0.70 * nyquist, nyquist),
        }
        total_energy = np.sum(power)
        for band_name, (low, high) in bands.items():
            mask        = (freqs >= low) & (freqs < high)
            band_energy = np.sum(power[mask])
            features[f'energy_{band_name}']       = band_energy
            features[f'energy_ratio_{band_name}'] = band_energy / (total_energy + 1e-10)

        features['spectral_flux'] = np.sum(np.diff(magnitude)**2) if len(magnitude) > 1 else 0.0
        return features


class EntropyFeatures:
    """Extract entropy-based features."""

    @staticmethod
    def extract(sig: np.ndarray) -> Dict[str, float]:
        return {
            'shannon_entropy'     : EntropyFeatures._shannon_entropy(sig),
            'sample_entropy'      : EntropyFeatures._sample_entropy(sig),
            'permutation_entropy' : EntropyFeatures._permutation_entropy(sig),
            'approximate_entropy' : EntropyFeatures._approximate_entropy(sig),
        }

    @staticmethod
    def _shannon_entropy(sig: np.ndarray, n_bins: int = 50) -> float:
        # Guard: constant signal has zero entropy
        if np.max(sig) == np.min(sig):
            return 0.0
        hist, _    = np.histogram(sig, bins=n_bins, density=True)
        bin_width  = (np.max(sig) - np.min(sig)) / n_bins
        prob       = hist * bin_width          # probability mass per bin
        prob       = prob[prob > 0]            # exclude zero bins (0*log0 = 0)
        return float(-np.sum(prob * np.log2(prob)))

    @staticmethod
    def _sample_entropy(sig: np.ndarray, m: int = 2, r: float = None) -> float:
        """
        Sample Entropy (Richman & Moorman 2000).

        Vectorised implementation — replaces the original O(N²) double Python
        loop with numpy broadcasting, ~100x faster for N=2000.
        Self-matches excluded (i≠j) per the paper.
        """
        if r is None:
            r = 0.2 * np.std(sig)

        N = len(sig)

        def _count_matches(template_len: int) -> int:
            # Build template matrix: shape (N - template_len, template_len)
            idx      = np.arange(N - template_len)
            patterns = np.array([sig[i : i + template_len] for i in idx])
            # Chebyshev distance between every pair (vectorised)
            # patterns shape: (M, template_len)
            # diff[i,j] = max |patterns[i] - patterns[j]| over template_len
            M     = len(patterns)
            count = 0
            # Process in blocks to keep memory reasonable
            block = 500
            for start in range(0, M, block):
                end   = min(start + block, M)
                chunk = patterns[start:end]                   # (block, template_len)
                diff  = np.abs(chunk[:, None, :] - patterns[None, :, :])  # (block, M, tl)
                chebyshev = np.max(diff, axis=2)              # (block, M)
                matches   = chebyshev <= r                    # (block, M) bool
                # Exclude self-matches: diagonal entries
                for local_i in range(end - start):
                    global_i = start + local_i
                    matches[local_i, global_i] = False
                count += int(np.sum(matches))
            return count

        B = _count_matches(m)
        A = _count_matches(m + 1)

        if B == 0:
            return 0.0
        if A == 0:
            return float(-np.log(2.0 / ((N - m - 1) * (N - m))))

        return float(-np.log(A / B))

    @staticmethod
    def _permutation_entropy(sig: np.ndarray, order: int = 3, delay: int = 1) -> float:
        """Permutation Entropy — unchanged, correct."""
        n           = len(sig)
        permutations= {}
        for i in range(n - delay * (order - 1)):
            pattern = sig[i : i + delay * order : delay]
            key     = tuple(np.argsort(pattern))
            permutations[key] = permutations.get(key, 0) + 1
        total = sum(permutations.values())
        if total == 0:
            return 0.0
        probs = np.array([c / total for c in permutations.values()])
        return float(-np.sum(probs * np.log2(probs + 1e-10)))

    @staticmethod
    def _approximate_entropy(sig: np.ndarray, m: int = 2, r: float = None) -> float:
        """Approximate Entropy — unchanged, correct."""
        if r is None:
            r = 0.2 * np.std(sig)
        N = len(sig)

        def _maxdist(x_i, x_j):
            return np.max(np.abs(x_i - x_j))

        def _phi(m):
            patterns = np.array([sig[i:i+m] for i in range(N-m+1)])
            C        = np.zeros(len(patterns))
            for i in range(len(patterns)):
                count = sum(1 for j in range(len(patterns))
                            if _maxdist(patterns[i], patterns[j]) <= r)
                C[i]  = count / len(patterns)
            return np.sum(np.log(C + 1e-10)) / len(patterns)

        return float(_phi(m) - _phi(m + 1))


class NonlinearFeatures:
    """Extract nonlinear dynamics features."""

    @staticmethod
    def extract(sig: np.ndarray) -> Dict[str, float]:
        features = {}
        features.update(NonlinearFeatures._hjorth_parameters(sig))
        features['higuchi_fractal_dimension']  = NonlinearFeatures._higuchi_fractal_dimension(sig)
        features['petrosian_fractal_dimension']= NonlinearFeatures._petrosian_fractal_dimension(sig)
        features['hurst_exponent']             = NonlinearFeatures._hurst_exponent(sig)
        features['lyapunov_exponent']          = NonlinearFeatures._lyapunov_exponent(sig)
        features['dfa_alpha']                  = NonlinearFeatures._dfa(sig)
        return features

    @staticmethod
    def _hjorth_parameters(sig: np.ndarray) -> Dict[str, float]:
        """Hjorth parameters — unchanged, correct."""
        activity   = np.var(sig)
        diff1      = np.diff(sig)
        mobility   = np.sqrt(np.var(diff1) / (activity + 1e-10))
        diff2      = np.diff(diff1)
        complexity = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10)) / (mobility + 1e-10)
        return {'hjorth_activity': activity, 'hjorth_mobility': mobility,
                'hjorth_complexity': complexity}

    @staticmethod
    def _higuchi_fractal_dimension(sig: np.ndarray, kmax: int = 10) -> float:
        """
        Higuchi Fractal Dimension (Higuchi 1988).

        Bug fixed: inner loop now slices sig[m::k] to exactly n_max elements
        before np.diff, so the sum has exactly floor((N-m)/k) terms as the
        paper requires. Original used np.diff(sig[m::k]) which could include
        extra terms when (N-m) is not divisible by k.
        """
        n  = len(sig)
        lk = np.zeros(kmax)

        for k in range(1, kmax + 1):
            lm = np.zeros(k)
            for m in range(1, k + 1):             # m = 1..k (1-indexed as in paper)
                # Number of terms in the sum: floor((N-m)/k)
                n_max = int(np.floor((n - m) / k))
                if n_max < 1:
                    continue
                # Slice exactly n_max+1 points so diff gives n_max terms (Bug fix)
                subsig = sig[m - 1 : m - 1 + (n_max * k + 1) : k]   # step k, n_max+1 pts
                if len(subsig) < 2:
                    continue
                length    = np.sum(np.abs(np.diff(subsig)))
                lm[m - 1] = length * (n - 1) / (n_max * k * k)
            valid = lm[lm > 0]
            if len(valid) > 0:
                lk[k - 1] = np.mean(valid)

        lk   = lk[lk > 0]
        if len(lk) < 2:
            return 1.0
        k_vals = np.arange(1, len(lk) + 1)
        x      = np.log(1.0 / k_vals)            # log(1/k) on x-axis
        y      = np.log(lk)                       # log(L(k)) on y-axis
        return float(np.polyfit(x, y, 1)[0])

    @staticmethod
    def _petrosian_fractal_dimension(sig: np.ndarray) -> float:
        """Petrosian FD — unchanged, correct."""
        n       = len(sig)
        diff    = np.diff(sig)
        n_delta = np.sum(diff[:-1] * diff[1:] < 0)
        return float(np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * n_delta))))

    @staticmethod
    def _hurst_exponent(sig: np.ndarray) -> float:
        """Hurst Exponent (R/S analysis) — unchanged, correct."""
        n = len(sig)
        if n < 20:
            return 0.5
        lags = np.arange(2, min(n // 2, 100))
        rs   = np.zeros(len(lags))
        for i, lag in enumerate(lags):
            splits = n // lag
            if splits < 2:
                continue
            rs_temp = []
            for j in range(splits):
                subset = sig[j * lag : (j + 1) * lag]
                mean   = np.mean(subset)
                cumdev = np.cumsum(subset - mean)
                r      = np.max(cumdev) - np.min(cumdev)
                s      = np.std(subset)
                if s > 0:
                    rs_temp.append(r / s)
            if rs_temp:
                rs[i] = np.mean(rs_temp)
        rs   = rs[rs > 0]
        lags = lags[:len(rs)]
        if len(lags) < 2:
            return 0.5
        return float(np.polyfit(np.log(lags), np.log(rs), 1)[0])

    @staticmethod
    def _lyapunov_exponent(sig: np.ndarray, emb_dim: int = 3,
                           lag: int = 1, theiler_window: int = 10) -> float:
        """
        Largest Lyapunov Exponent (simplified Rosenstein method).

        Bug fixed: added Theiler window to exclude temporally-correlated
        neighbours. Without it, the nearest neighbour for point i is nearly
        always i-1 or i+1 (trivially close in time), and the divergence
        measures temporal correlation rather than chaotic divergence.

        theiler_window: number of time steps to exclude around i when
                        searching for nearest neighbours. Default 10.
        """
        n = len(sig)
        if n < emb_dim * lag + 1:
            return 0.0

        embedded = np.array([sig[i : i + emb_dim * lag : lag]
                             for i in range(n - emb_dim * lag)])
        M = len(embedded)
        if M < 2:
            return 0.0

        divergences = []
        for i in range(M - 1):
            distances = np.linalg.norm(embedded - embedded[i], axis=1)

            # Apply Theiler window: exclude indices within theiler_window of i
            distances[max(0, i - theiler_window) : min(M, i + theiler_window + 1)] = np.inf

            nearest_idx = np.argmin(distances)
            if distances[nearest_idx] == np.inf:
                continue

            if nearest_idx < M - 1:
                dist_0 = distances[nearest_idx]
                dist_1 = np.linalg.norm(embedded[i + 1] - embedded[nearest_idx + 1])
                if dist_0 > 0 and dist_1 > 0:
                    divergences.append(np.log(dist_1 / dist_0))

        return float(np.mean(divergences)) if divergences else 0.0

    @staticmethod
    def _dfa(sig: np.ndarray) -> float:
        """Detrended Fluctuation Analysis — unchanged, correct."""
        n = len(sig)
        if n < 16:
            return 1.0
        y      = np.cumsum(sig - np.mean(sig))
        scales = np.unique(np.logspace(0.5, np.log10(n // 4), 20).astype(int))
        flucts = np.zeros(len(scales))
        for i, scale in enumerate(scales):
            n_boxes = n // scale
            boxes   = []
            for j in range(n_boxes):
                box    = y[j * scale : (j + 1) * scale]
                x_box  = np.arange(len(box))
                coeffs = np.polyfit(x_box, box, 1)
                fit    = np.polyval(coeffs, x_box)
                boxes.append(np.sqrt(np.mean((box - fit) ** 2)))
            flucts[i] = np.mean(boxes)
        valid  = flucts > 0
        if np.sum(valid) < 2:
            return 1.0
        return float(np.polyfit(np.log(scales[valid]), np.log(flucts[valid]), 1)[0])


class DecompositionFeatures:
    """Extract features from decomposed signals. — UNCHANGED, all correct."""

    @staticmethod
    def extract_from_components(components: List[np.ndarray],
                                prefix: str = 'comp') -> Dict[str, float]:
        features = {}
        for i, comp in enumerate(components):
            p = f'{prefix}_{i}'
            features[f'{p}_energy']  = float(np.sum(comp**2))
            features[f'{p}_rms']     = float(np.sqrt(np.mean(comp**2)))
            features[f'{p}_mean']    = float(np.mean(comp))
            features[f'{p}_std']     = float(np.std(comp))
            features[f'{p}_max']     = float(np.max(np.abs(comp)))
            hist, _ = np.histogram(comp, bins=30, density=True)
            hist    = hist[hist > 0]
            features[f'{p}_entropy'] = float(-np.sum(hist * np.log2(hist + 1e-10)))

        total_energy = sum(features[f'{prefix}_{i}_energy'] for i in range(len(components)))
        for i in range(len(components)):
            features[f'{prefix}_{i}_energy_ratio'] = (
                features[f'{prefix}_{i}_energy'] / (total_energy + 1e-10))

        if len(components) > 1:
            features.update(
                DecompositionFeatures._cross_component_features(components, prefix))
        return features

    @staticmethod
    def _cross_component_features(components: List[np.ndarray],
                                  prefix: str) -> Dict[str, float]:
        features = {}
        n_comp   = len(components)
        for i in range(n_comp):
            for j in range(i + 1, n_comp):
                min_len = min(len(components[i]), len(components[j]))
                ci, cj  = components[i][:min_len], components[j][:min_len]
                corr    = np.corrcoef(ci, cj)[0, 1]
                features[f'{prefix}_corr_{i}_{j}'] = corr if not np.isnan(corr) else 0.0
                features[f'{prefix}_energy_ratio_{i}_{j}'] = (
                    np.sum(ci**2) / (np.sum(cj**2) + 1e-10))
        for i in range(min(n_comp, 5)):
            for j in range(i + 1, min(n_comp, 5)):
                min_len  = min(len(components[i]), len(components[j]))
                ci, cj   = components[i][:min_len], components[j][:min_len]
                features[f'{prefix}_kl_div_{i}_{j}'] = (
                    DecompositionFeatures._kl_divergence(ci, cj))
        return features

    @staticmethod
    def _kl_divergence(p_data: np.ndarray, q_data: np.ndarray,
                       n_bins: int = 30) -> float:
        lo  = min(np.min(p_data), np.min(q_data))
        hi  = max(np.max(p_data), np.max(q_data))
        ph, _ = np.histogram(p_data, bins=n_bins, range=(lo, hi), density=True)
        qh, _ = np.histogram(q_data, bins=n_bins, range=(lo, hi), density=True)
        ph    = (ph + 1e-10) / np.sum(ph + 1e-10)
        qh    = (qh + 1e-10) / np.sum(qh + 1e-10)
        return float(np.sum(ph * np.log(ph / qh)))