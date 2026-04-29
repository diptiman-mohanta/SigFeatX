import numpy as np
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from typing import Dict, List, Optional, Tuple

from SigFeatX._validation import validate_sampling_rate, validate_signal_1d
from SigFeatX.utils import SignalUtils


class TimeDomainFeatures:
    """Extract time domain statistical features."""

    @staticmethod
    def extract(sig: np.ndarray) -> Dict[str, float]:
        sig = validate_signal_1d(sig, name='sig')
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
        if len(sig) < 3 or np.allclose(sig, sig[0]):
            features['skewness'] = 0.0
            features['kurtosis'] = 0.0
        else:
            features['skewness'] = float(stats.skew(sig))
            features['kurtosis'] = float(stats.kurtosis(sig))
        features['rms']              = np.sqrt(np.mean(sig**2))
        features['energy']           = np.sum(sig**2)
        features['power']            = features['energy'] / len(sig)
        features['mean_absolute']    = np.mean(np.abs(sig))
        features['sum_absolute']     = np.sum(np.abs(sig))
        features['zero_crossings']   = np.sum(np.diff(np.sign(sig)) != 0)
        features['zero_crossing_rate'] = features['zero_crossings'] / len(sig)
        features['line_length']      = np.sum(np.abs(np.diff(sig)))

        acf_peak_lag, acf_peak_value = TimeDomainFeatures._autocorrelation_peak(sig)
        features['autocorrelation_peak_lag']   = acf_peak_lag
        features['autocorrelation_peak_value'] = acf_peak_value

        tkeo = TimeDomainFeatures._tkeo(sig)
        features['tkeo_mean'] = np.mean(tkeo)
        features['tkeo_std']  = np.std(tkeo)
        features['tkeo_max']  = np.max(tkeo)

        rms_val  = features['rms']
        mean_abs = features['mean_absolute']
        peak     = np.max(np.abs(sig))

        features['crest_factor']     = peak / (rms_val + 1e-10)
        features['shape_factor']     = rms_val / (mean_abs + 1e-10)
        features['impulse_factor']   = peak / (mean_abs + 1e-10)
        features['clearance_factor'] = peak / ((np.mean(np.sqrt(np.abs(sig))))**2 + 1e-10)
        features['q25']              = np.percentile(sig, 25)
        features['q75']              = np.percentile(sig, 75)
        features['iqr']              = features['q75'] - features['q25']
        features['coeff_variation']  = features['std'] / (np.abs(features['mean']) + 1e-10)

        return features

    @staticmethod
    def _tkeo(sig: np.ndarray) -> np.ndarray:
        if len(sig) < 3:
            return np.zeros(1, dtype=float)
        return sig[1:-1] ** 2 - sig[:-2] * sig[2:]

    @staticmethod
    def _autocorrelation_peak(sig: np.ndarray) -> Tuple[float, float]:
        if len(sig) < 3:
            return 0.0, 0.0
        centered = sig - np.mean(sig)
        denom = np.sum(centered ** 2)
        if denom <= 1e-12:
            return 0.0, 0.0
        acf_full  = np.correlate(centered, centered, mode='full')
        acf       = acf_full[len(sig) - 1:] / (denom + 1e-10)
        if len(acf) <= 1:
            return 0.0, 0.0
        peak_lag   = int(np.argmax(acf[1:]) + 1)
        peak_value = float(acf[peak_lag])
        return float(peak_lag), peak_value


class FrequencyDomainFeatures:
    """Extract frequency domain features."""

    @staticmethod
    def extract(sig: np.ndarray, fs: float = 1.0) -> Dict[str, float]:
        sig = validate_signal_1d(sig, name='sig')
        fs  = validate_sampling_rate(fs)
        features = {}
        n        = len(sig)
        fft_vals = np.asarray(fft(sig), dtype=np.complex128)
        freqs    = np.asarray(fftfreq(n, 1/fs), dtype=np.float64)

        pos_idx    = np.where(freqs >= 0)[0]
        freqs      = freqs[pos_idx]
        fft_vals   = fft_vals[pos_idx]
        magnitude  = np.abs(fft_vals)
        power      = magnitude ** 2
        power_norm = power / (np.sum(power) + 1e-10)

        features['spectral_centroid']  = np.sum(freqs * power_norm)
        features['spectral_spread']    = np.sqrt(
            np.sum(((freqs - features['spectral_centroid'])**2) * power_norm))
        features['spectral_bandwidth'] = features['spectral_spread']
        features['spectral_bandwidth_90'] = FrequencyDomainFeatures._fractional_power_bandwidth(
            freqs, power_norm, lower_fraction=0.05, upper_fraction=0.95
        )

        cumsum_power = np.cumsum(power_norm)
        rolloff_idx  = np.where(cumsum_power >= 0.95)[0]
        features['spectral_rolloff']   = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
        features['dominant_frequency'] = freqs[np.argmax(power)]
        features['max_magnitude']      = np.max(magnitude)

        # ── FIX: spectral_flatness must use POWER spectrum, not magnitude ──
        # Wiener's Spectral Flatness Measure (SFM) is defined as:
        #   SFM = geometric_mean(P(f)) / arithmetic_mean(P(f))
        # where P(f) = |X(f)|^2 is the power spectrum.
        # Using |X(f)| (magnitude) gives sqrt(SFM), which is wrong.
        # The ratio lies in [0, 1] by the AM-GM inequality applied to power.
        geometric_mean_power  = np.exp(np.mean(np.log(power + 1e-10)))
        arithmetic_mean_power = np.mean(power)
        features['spectral_flatness']  = float(
            geometric_mean_power / (arithmetic_mean_power + 1e-10)
        )
        # Clamp to [0, 1] — floating-point can push slightly above 1 for
        # near-flat spectra due to the log/exp round-trip.
        features['spectral_flatness']  = float(
            np.clip(features['spectral_flatness'], 0.0, 1.0)
        )

        features['spectral_entropy']   = max(
            0.0, float(-np.sum(power_norm * np.log2(power_norm + 1e-10))))
        features['spectral_slope']     = FrequencyDomainFeatures._spectral_slope(freqs, power)

        inst_freq_mean, inst_freq_std = FrequencyDomainFeatures._instantaneous_frequency_stats(sig, fs)
        features['instantaneous_freq_mean'] = inst_freq_mean
        features['instantaneous_freq_std']  = inst_freq_std

        sc = features['spectral_centroid']
        ss = features['spectral_spread']

        # ── FIX: spectral_kurtosis guard ───────────────────────────────────
        # Using ss**4 + 1e-10 gives enormous values when ss ≈ 0 (DC signal).
        # Use max(ss, 1e-6)**4 instead so the guard is meaningful.
        features['spectral_kurtosis']  = float(
            np.sum(((freqs - sc)**4) * power_norm) / (max(ss, 1e-6)**4)
        )
        features['spectral_skewness']  = float(
            np.sum(((freqs - sc)**3) * power_norm) / (max(ss, 1e-6)**3)
        )

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

        eeg_bandpowers = FrequencyDomainFeatures._eeg_bandpowers(freqs, power, fs)
        features.update(eeg_bandpowers)

        features['spectral_flux'] = np.sum(np.diff(magnitude)**2) if len(magnitude) > 1 else 0.0
        return features

    @staticmethod
    def _instantaneous_frequency_stats(
        sig: np.ndarray,
        fs: float,
        edge_fraction: float = 0.02,
        amplitude_threshold_ratio: float = 0.05,
    ) -> Tuple[float, float]:
        if len(sig) < 4 or not np.all(np.isfinite(sig)) or np.allclose(sig, sig[0]):
            return 0.0, 0.0

        inst_freq = SignalUtils.compute_instantaneous_frequency(sig, fs=fs)
        envelope  = SignalUtils.compute_envelope(sig)
        if len(inst_freq) == 0 or len(envelope) < 2:
            return 0.0, 0.0

        midpoint_envelope = 0.5 * (envelope[:-1] + envelope[1:])
        valid = np.isfinite(inst_freq) & np.isfinite(midpoint_envelope)
        if not np.any(valid):
            return 0.0, 0.0

        max_envelope = np.max(midpoint_envelope[valid])
        if max_envelope <= 1e-12:
            return 0.0, 0.0

        valid &= midpoint_envelope >= amplitude_threshold_ratio * max_envelope

        trim = int(np.floor(edge_fraction * len(inst_freq)))
        if trim > 0:
            valid[:trim]  = False
            valid[-trim:] = False

        nyquist = fs / 2.0
        valid  &= (inst_freq >= 0.0) & (inst_freq <= nyquist + 1e-10)

        inst_freq = inst_freq[valid]
        if len(inst_freq) == 0:
            return 0.0, 0.0

        return float(np.mean(inst_freq)), float(np.std(inst_freq))

    @staticmethod
    def _spectral_slope(freqs: np.ndarray, power: np.ndarray) -> float:
        mask = (freqs > 0.0) & np.isfinite(freqs) & np.isfinite(power) & (power > 1e-20)
        if np.sum(mask) < 2:
            return 0.0
        x = np.log10(freqs[mask])
        y = np.log10(power[mask])
        slope, _ = np.polyfit(x, y, deg=1)
        return float(slope)

    @staticmethod
    def _eeg_bandpowers(freqs: np.ndarray, power: np.ndarray, fs: float) -> Dict[str, float]:
        nyquist = fs / 2.0
        bands   = {
            'delta': (0.5,  4.0),
            'theta': (4.0,  8.0),
            'alpha': (8.0,  13.0),
            'beta' : (13.0, 30.0),
            'gamma': (30.0, 100.0),
        }
        total_pos_power = np.sum(power)
        out: Dict[str, float] = {}
        for name, (low, high) in bands.items():
            lo = max(0.0, low)
            hi = min(nyquist, high)
            if hi <= lo:
                band_power = 0.0
            else:
                mask       = (freqs >= lo) & (freqs < hi)
                band_power = float(np.sum(power[mask]))
            out[f'bandpower_{name}']     = band_power
            out[f'bandpower_{name}_rel'] = float(band_power / (total_pos_power + 1e-10))
        return out

    @staticmethod
    def _fractional_power_bandwidth(
        freqs: np.ndarray,
        power_norm: np.ndarray,
        lower_fraction: float,
        upper_fraction: float,
    ) -> float:
        if len(freqs) == 0 or len(power_norm) == 0:
            return 0.0
        cumsum_power = np.cumsum(power_norm)
        if cumsum_power[-1] <= 1e-12:
            return 0.0
        low_idx  = int(np.searchsorted(cumsum_power, lower_fraction, side='left'))
        high_idx = int(np.searchsorted(cumsum_power, upper_fraction, side='left'))
        low_idx  = min(max(low_idx,  0), len(freqs) - 1)
        high_idx = min(max(high_idx, 0), len(freqs) - 1)
        return float(max(0.0, freqs[high_idx] - freqs[low_idx]))


class EntropyFeatures:
    """Extract entropy-based features."""

    @staticmethod
    def extract(sig: np.ndarray) -> Dict[str, float]:
        sig = validate_signal_1d(sig, name='sig')
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

        hist, _ = np.histogram(sig, bins=n_bins, density=False)

        # ── FIX: renormalise to proper probability mass ────────────────────
        # density=False gives raw counts.  Divide by total to get probability
        # mass per bin.  Explicitly renormalise so np.sum(prob) == 1.0 exactly,
        # removing floating-point drift that previously produced slightly wrong
        # entropy values.  This also prevents log2 of values slightly above 1
        # that were possible with the old bin_width correction.
        total = hist.sum()
        if total == 0:
            return 0.0
        prob = hist.astype(float) / total     # probability mass, sums to 1.0
        prob = prob[prob > 0]                 # exclude zero bins (0·log0 = 0)
        return float(-np.sum(prob * np.log2(prob)))

    @staticmethod
    def _sample_entropy(sig: np.ndarray, m: int = 2, r: Optional[float] = None) -> float:
        """
        Sample Entropy (Richman & Moorman 2000).
        Vectorised implementation — self-matches excluded (i≠j) per the paper.
        """
        if r is None:
            r = float(0.2 * np.std(sig))
        N = len(sig)
        if N <= m + 1:
            return 0.0

        def _count_matches(template_len: int) -> int:
            idx      = np.arange(N - template_len)
            patterns = np.array([sig[i : i + template_len] for i in idx])
            M     = len(patterns)
            count = 0
            block = 500
            for start in range(0, M, block):
                end   = min(start + block, M)
                chunk = patterns[start:end]
                diff  = np.abs(chunk[:, None, :] - patterns[None, :, :])
                chebyshev = np.max(diff, axis=2)
                matches   = chebyshev <= r
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
        """Permutation Entropy (Bandt & Pompe 2002) — correct."""
        n = len(sig)
        if n < delay * (order - 1) + 1:
            return 0.0
        permutations = {}
        for i in range(n - delay * (order - 1)):
            pattern = sig[i : i + delay * order : delay]
            key     = tuple(np.argsort(pattern))
            permutations[key] = permutations.get(key, 0) + 1
        total = sum(permutations.values())
        if total == 0:
            return 0.0
        probs = np.array([c / total for c in permutations.values()])
        return float(max(0.0, -np.sum(probs * np.log2(probs + 1e-10))))

    @staticmethod
    def _approximate_entropy(sig: np.ndarray, m: int = 2, r: Optional[float] = None) -> float:
        """
        Approximate Entropy (Pincus 1991) — vectorised block-wise.
        Self-matches (i==j) included per the original ApEn definition.
        """
        if r is None:
            r = float(0.2 * np.std(sig))
        N = len(sig)
        if N <= m + 1:
            return 0.0

        def _phi(template_len: int) -> float:
            M        = N - template_len + 1
            patterns = np.array([sig[i : i + template_len] for i in range(M)])
            log_C    = np.empty(M)
            block    = 500
            for start in range(0, M, block):
                end       = min(start + block, M)
                chunk     = patterns[start:end]
                diff      = np.abs(chunk[:, None, :] - patterns[None, :, :])
                chebyshev = np.max(diff, axis=2)
                counts    = np.sum(chebyshev <= r, axis=1)
                log_C[start:end] = np.log(counts / M + 1e-10)
            return float(np.sum(log_C) / M)

        return _phi(m) - _phi(m + 1)


class NonlinearFeatures:
    """Extract nonlinear dynamics features."""

    @staticmethod
    def extract(sig: np.ndarray) -> Dict[str, float]:
        sig = validate_signal_1d(sig, name='sig')
        features = {}
        features.update(NonlinearFeatures._hjorth_parameters(sig))
        features['higuchi_fractal_dimension']   = NonlinearFeatures._higuchi_fractal_dimension(sig)
        features['petrosian_fractal_dimension'] = NonlinearFeatures._petrosian_fractal_dimension(sig)
        features['hurst_exponent']              = NonlinearFeatures._hurst_exponent(sig)
        features['lyapunov_exponent']           = NonlinearFeatures._lyapunov_exponent(sig)
        features['dfa_alpha']                   = NonlinearFeatures._dfa(sig)
        return features

    @staticmethod
    def _hjorth_parameters(sig: np.ndarray) -> Dict[str, float]:
        activity   = np.var(sig)
        diff1      = np.diff(sig)
        mobility   = np.sqrt(np.var(diff1) / (activity + 1e-10))
        diff2      = np.diff(diff1)
        complexity = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10)) / (mobility + 1e-10)
        return {
            'hjorth_activity'  : float(activity),
            'hjorth_mobility'  : float(mobility),
            'hjorth_complexity': float(complexity),
        }

    @staticmethod
    def _higuchi_fractal_dimension(sig: np.ndarray, kmax: int = 10) -> float:
        """
        Higuchi Fractal Dimension (Higuchi 1988).
        Verified correct: slice gives exactly n_max+1 points → n_max diffs.
        Normalisation (n-1)/(n_max·k²) matches Higuchi 1988 Eq. 2.
        """
        n  = len(sig)
        lk = np.zeros(kmax)

        for k in range(1, kmax + 1):
            lm = np.zeros(k)
            for m in range(1, k + 1):
                n_max = int(np.floor((n - m) / k))
                if n_max < 1:
                    continue
                subsig = sig[m - 1 : m - 1 + (n_max * k + 1) : k]
                if len(subsig) < 2:
                    continue
                length    = np.sum(np.abs(np.diff(subsig)))
                lm[m - 1] = length * (n - 1) / (n_max * k * k)
            valid = lm[lm > 0]
            if len(valid) > 0:
                lk[k - 1] = np.mean(valid)

        lk = lk[lk > 0]
        if len(lk) < 2:
            return 1.0
        k_vals = np.arange(1, len(lk) + 1)
        x      = np.log(1.0 / k_vals)
        y      = np.log(lk)
        return float(np.polyfit(x, y, 1)[0])

    @staticmethod
    def _petrosian_fractal_dimension(sig: np.ndarray) -> float:
        n = len(sig)
        if n < 2:
            return 1.0
        diff    = np.diff(sig)
        n_delta = np.sum(diff[:-1] * diff[1:] < 0)
        return float(np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * n_delta))))

    @staticmethod
    def _hurst_exponent(sig: np.ndarray) -> float:
        n = len(sig)
        if n < 20:
            return 0.5
        lags = np.arange(2, min(n // 2, 100))
        rs   = np.zeros(len(lags))
        for i, lag in enumerate(lags):
            splits  = n // lag
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
        """Largest Lyapunov Exponent (Rosenstein method) with Theiler window."""
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
        """Detrended Fluctuation Analysis — correct."""
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
        valid = flucts > 0
        if np.sum(valid) < 2:
            return 1.0
        return float(np.polyfit(np.log(scales[valid]), np.log(flucts[valid]), 1)[0])


class DecompositionFeatures:
    """Extract features from decomposed signals."""

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

            # ── FIX: component entropy ────────────────────────────────────
            # PREVIOUS (wrong):
            #   hist, _ = np.histogram(comp, bins=30, density=True)
            #   hist = hist[hist > 0]
            #   features[f'{p}_entropy'] = float(-np.sum(hist * np.log2(hist + 1e-10)))
            #
            # density=True gives probability DENSITY (units: 1/amplitude).
            # Computing -Σ density·log2(density) has wrong units and is not
            # Shannon entropy.  Shannon entropy requires probability MASS
            # (dimensionless values that sum to 1).
            #
            # FIX: use density=False and normalise to probability mass explicitly.
            hist, _ = np.histogram(comp, bins=30, density=False)
            total   = hist.sum()
            if total > 0:
                prob = hist.astype(float) / total   # probability mass, sums to 1
                prob = prob[prob > 0]               # exclude zero bins
                features[f'{p}_entropy'] = float(-np.sum(prob * np.log2(prob)))
            else:
                features[f'{p}_entropy'] = 0.0

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
                with np.errstate(invalid='ignore'):
                    corr = np.corrcoef(ci, cj)[0, 1]
                features[f'{prefix}_corr_{i}_{j}'] = corr if not np.isnan(corr) else 0.0
                features[f'{prefix}_energy_ratio_{i}_{j}'] = (
                    np.sum(ci**2) / (np.sum(cj**2) + 1e-10))

        for i in range(min(n_comp, 5)):
            for j in range(i + 1, min(n_comp, 5)):
                min_len = min(len(components[i]), len(components[j]))
                ci, cj  = components[i][:min_len], components[j][:min_len]
                features[f'{prefix}_kl_div_{i}_{j}'] = (
                    DecompositionFeatures._kl_divergence(ci, cj))
        return features

    @staticmethod
    def _kl_divergence(p_data: np.ndarray, q_data: np.ndarray,
                       n_bins: int = 30) -> float:
        lo  = min(np.min(p_data), np.min(q_data))
        hi  = max(np.max(p_data), np.max(q_data))
        ph, _ = np.histogram(p_data, bins=n_bins, range=(lo, hi), density=False)
        qh, _ = np.histogram(q_data, bins=n_bins, range=(lo, hi), density=False)

        # ── FIX: KL divergence normalisation ─────────────────────────────
        # PREVIOUS (wrong):
        #   ph = (ph + 1e-10) / np.sum(ph + 1e-10)
        #   qh = (qh + 1e-10) / np.sum(qh + 1e-10)
        #   return float(np.sum(ph * np.log(ph / qh)))
        #
        # Adding 1e-10 to every bin BEFORE normalising shifts all probabilities
        # (including non-zero bins), making the resulting distributions not sum
        # to 1.0 and introducing a systematic bias in KL(p||q).
        #
        # FIX: normalise raw counts first, then apply epsilon only inside
        # the log to prevent log(0).  This keeps np.sum(ph) == 1 exactly.
        ph = ph.astype(float) / (ph.sum() + 1e-30)
        qh = qh.astype(float) / (qh.sum() + 1e-30)

        # KL divergence: only sum where ph > 0 (0·log(0/q) = 0 by convention)
        mask = ph > 0
        return float(np.sum(ph[mask] * np.log((ph[mask] + 1e-10) / (qh[mask] + 1e-10))))