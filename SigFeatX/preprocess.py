"""Signal preprocessing operations."""

import numpy as np
from scipy import signal
from typing import Optional
import pywt


class SignalPreprocessor:
    """Preprocessing operations for signals."""
    
    def __init__(self):
        self.history = []
    
    def denoise(self, sig: np.ndarray, method: str = 'wavelet', **kwargs) -> np.ndarray:
        """
        Denoise signal.
        
        Args:
            sig: Input signal
            method: 'wavelet', 'median', or 'lowpass'
            **kwargs: Method-specific parameters
        
        Returns:
            Denoised signal
        """
        if method == 'wavelet':
            wavelet = kwargs.get('wavelet', 'db4')
            level = kwargs.get('level', 1)
            mode = kwargs.get('mode', 'soft')
            
            coeffs = pywt.wavedec(sig, wavelet, level=level)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(sig)))
            
            coeffs[1:] = [pywt.threshold(c, threshold, mode=mode) for c in coeffs[1:]]
            denoised = pywt.waverec(coeffs, wavelet)
            
            # Handle length mismatch
            if len(denoised) > len(sig):
                denoised = denoised[:len(sig)]
            elif len(denoised) < len(sig):
                denoised = np.pad(denoised, (0, len(sig) - len(denoised)), mode='edge')
            
            self.history.append(('denoise', method, kwargs))
            return denoised
        
        elif method == 'median':
            kernel_size = kwargs.get('kernel_size', 5)
            if kernel_size % 2 == 0:
                kernel_size += 1
            denoised = signal.medfilt(sig, kernel_size=kernel_size)
            self.history.append(('denoise', method, kwargs))
            return denoised
        
        elif method == 'lowpass':
            cutoff = kwargs.get('cutoff', 0.3)
            order = kwargs.get('order', 5)
            b, a = signal.butter(order, cutoff, btype='low')
            denoised = signal.filtfilt(b, a, sig)
            self.history.append(('denoise', method, kwargs))
            return denoised
        
        else:
            raise ValueError(f"Unknown denoising method: {method}")
    
    def normalize(self, sig: np.ndarray, method: str = 'zscore') -> np.ndarray:
        """
        Normalize signal.
        
        Args:
            sig: Input signal
            method: 'zscore', 'minmax', or 'robust'
        
        Returns:
            Normalized signal
        """
        if method == 'zscore':
            mean = np.mean(sig)
            std = np.std(sig)
            normalized = (sig - mean) / (std + 1e-10)
        
        elif method == 'minmax':
            min_val = np.min(sig)
            max_val = np.max(sig)
            normalized = (sig - min_val) / (max_val - min_val + 1e-10)
        
        elif method == 'robust':
            median = np.median(sig)
            mad = np.median(np.abs(sig - median))
            normalized = (sig - median) / (mad + 1e-10)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        self.history.append(('normalize', method, {}))
        return normalized
    
    def detrend(self, sig: np.ndarray, method: str = 'linear') -> np.ndarray:
        """
        Remove trend from signal.
        
        Args:
            sig: Input signal
            method: 'linear' or 'constant'
        
        Returns:
            Detrended signal
        """
        detrended = signal.detrend(sig, type=method)
        self.history.append(('detrend', method, {}))
        return detrended
    
    def resample(self, sig: np.ndarray, target_length: int, method: str = 'linear') -> np.ndarray:
        """
        Resample signal to target length.
        
        Args:
            sig: Input signal
            target_length: Desired length
            method: 'linear' or 'fourier'
        
        Returns:
            Resampled signal
        """
        if method == 'linear':
            x_old = np.linspace(0, 1, len(sig))
            x_new = np.linspace(0, 1, target_length)
            resampled = np.interp(x_new, x_old, sig)
        elif method == 'fourier':
            resampled = signal.resample(sig, target_length)
        else:
            raise ValueError(f"Unknown resampling method: {method}")
        
        self.history.append(('resample', method, {'target_length': target_length}))
        return resampled
    
    def get_history(self) -> list:
        """Return preprocessing history."""
        return self.history
    
    def clear_history(self):
        """Clear preprocessing history."""
        self.history = []
