import numpy as np
from typing import Tuple, List, Optional
from scipy import signal


class SignalUtils:
    """Utility functions for signal processing."""
    
    @staticmethod
    def sliding_window(sig: np.ndarray, window_size: int, 
                      step_size: int) -> List[np.ndarray]:
        """
        Create sliding windows from signal.
        
        Args:
            sig: Input signal
            window_size: Size of each window
            step_size: Step size between windows
        
        Returns:
            List of signal windows
        """
        windows = []
        for i in range(0, len(sig) - window_size + 1, step_size):
            windows.append(sig[i:i + window_size])
        return windows
    
    @staticmethod
    def pad_signal(sig: np.ndarray, target_length: int, 
                  mode: str = 'constant') -> np.ndarray:
        """
        Pad signal to target length.
        
        Args:
            sig: Input signal
            target_length: Desired length
            mode: Padding mode ('constant', 'edge', 'reflect', 'symmetric')
        
        Returns:
            Padded signal
        """
        if len(sig) >= target_length:
            return sig[:target_length]
        
        pad_width = target_length - len(sig)
        return np.pad(sig, (0, pad_width), mode=mode)
    
    @staticmethod
    def segment_signal(sig: np.ndarray, n_segments: int) -> List[np.ndarray]:
        """
        Segment signal into equal parts.
        
        Args:
            sig: Input signal
            n_segments: Number of segments
        
        Returns:
            List of signal segments
        """
        segment_length = len(sig) // n_segments
        segments = []
        
        for i in range(n_segments):
            start = i * segment_length
            end = start + segment_length if i < n_segments - 1 else len(sig)
            segments.append(sig[start:end])
        
        return segments
    
    @staticmethod
    def compute_snr(sig: np.ndarray, noise: np.ndarray) -> float:
        """
        Compute Signal-to-Noise Ratio.
        
        Args:
            sig: Clean signal
            noise: Noise signal
        
        Returns:
            SNR in dB
        """
        signal_power = np.mean(sig ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power == 0:
            return np.inf
        
        return 10 * np.log10(signal_power / noise_power)
    
    @staticmethod
    def detect_peaks(sig: np.ndarray, height: Optional[float] = None,
                    distance: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """
        Detect peaks in signal.
        
        Args:
            sig: Input signal
            height: Minimum peak height
            distance: Minimum distance between peaks
        
        Returns:
            Tuple of (peak indices, peak properties)
        """
        peaks, properties = signal.find_peaks(sig, height=height, distance=distance)
        return peaks, properties
    
    @staticmethod
    def compute_envelope(sig: np.ndarray) -> np.ndarray:
        """
        Compute signal envelope using Hilbert transform.
        
        Args:
            sig: Input signal
        
        Returns:
            Signal envelope
        """
        analytic_signal = signal.hilbert(sig)
        return np.abs(analytic_signal)
    
    @staticmethod
    def compute_instantaneous_frequency(sig: np.ndarray, fs: float = 1.0) -> np.ndarray:
        """
        Compute instantaneous frequency.
        
        Args:
            sig: Input signal
            fs: Sampling frequency
        
        Returns:
            Instantaneous frequency
        """
        analytic_signal = signal.hilbert(sig)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * fs
        
        return instantaneous_frequency
    
    @staticmethod
    def remove_outliers(sig: np.ndarray, n_std: float = 3.0) -> np.ndarray:
        """
        Remove outliers from signal.
        
        Args:
            sig: Input signal
            n_std: Number of standard deviations for threshold
        
        Returns:
            Signal with outliers removed (replaced with median)
        """
        median = np.median(sig)
        std = np.std(sig)
        threshold = n_std * std
        
        outlier_mask = np.abs(sig - median) > threshold
        sig_clean = sig.copy()
        sig_clean[outlier_mask] = median
        
        return sig_clean
    
    @staticmethod
    def add_noise(sig: np.ndarray, snr_db: float, noise_type: str = 'gaussian') -> np.ndarray:
        """
        Add noise to signal.
        
        Args:
            sig: Input signal
            snr_db: Desired SNR in dB
            noise_type: 'gaussian' or 'uniform'
        
        Returns:
            Noisy signal
        """
        signal_power = np.mean(sig ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        if noise_type == 'gaussian':
            noise = np.random.normal(0, np.sqrt(noise_power), len(sig))
        elif noise_type == 'uniform':
            noise = np.random.uniform(-np.sqrt(3 * noise_power), 
                                     np.sqrt(3 * noise_power), len(sig))
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        return sig + noise
