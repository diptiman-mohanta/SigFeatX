"""Fourier Transform decomposition."""

import numpy as np
from scipy.fft import fft, fftfreq, ifft
from typing import Tuple


class FourierTransform:
    """Fourier Transform decomposition."""
    
    def __init__(self, fs: float = 1.0):
        """
        Initialize Fourier Transform.
        
        Args:
            fs: Sampling frequency
        """
        self.fs = fs
        self.freqs = None
        self.fft_vals = None
    
    def transform(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Fourier Transform.
        
        Args:
            signal: Input signal
        
        Returns:
            Tuple of (frequencies, magnitude spectrum)
        """
        n = len(signal)
        self.fft_vals = fft(signal)
        self.freqs = fftfreq(n, 1/self.fs)
        
        # Return positive frequencies only
        pos_mask = self.freqs >= 0
        return self.freqs[pos_mask], np.abs(self.fft_vals[pos_mask])
    
    def inverse_transform(self) -> np.ndarray:
        """Compute inverse Fourier Transform."""
        if self.fft_vals is None:
            raise ValueError("Must call transform() first")
        return np.real(ifft(self.fft_vals))
    
    def get_power_spectrum(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectrum.
        
        Returns:
            Tuple of (frequencies, power spectrum)
        """
        freqs, magnitude = self.transform(signal)
        power = magnitude ** 2
        return freqs, power
    
    def get_phase_spectrum(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute phase spectrum.
        
        Returns:
            Tuple of (frequencies, phase spectrum)
        """
        self.transform(signal)
        pos_mask = self.freqs >= 0
        phase = np.angle(self.fft_vals[pos_mask])
        return self.freqs[pos_mask], phase