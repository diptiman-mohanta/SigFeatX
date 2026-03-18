"""Short-Time Fourier Transform."""

import numpy as np
from scipy import signal
from typing import Tuple, Optional

from SigFeatX._validation import validate_sampling_rate, validate_signal_1d


class ShortTimeFourierTransform:
    """Short-Time Fourier Transform (STFT)."""
    
    def __init__(self, fs: float = 1.0, nperseg: int = 256, noverlap: Optional[int] = None,
                 window: str = 'hann'):
        """
        Initialize STFT.
        
        Args:
            fs: Sampling frequency
            nperseg: Length of each segment
            noverlap: Number of points to overlap between segments
            window: Window function to use
        """
        self.fs = validate_sampling_rate(fs)
        if nperseg < 1:
            raise ValueError(f"nperseg must be >= 1; got {nperseg}.")
        self.nperseg = nperseg
        self.noverlap = noverlap if noverlap is not None else nperseg // 2
        if self.noverlap < 0 or self.noverlap >= self.nperseg:
            raise ValueError(
                f"noverlap must be in [0, nperseg); got noverlap={self.noverlap}, "
                f"nperseg={self.nperseg}."
            )
        self.window = window
    
    def transform(self, sig: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute STFT.
        
        Args:
            sig: Input signal
        
        Returns:
            Tuple of (frequencies, times, STFT magnitude)
        """
        sig = validate_signal_1d(sig, name='sig')
        nperseg = min(self.nperseg, len(sig))
        noverlap = min(self.noverlap, nperseg - 1)
        
        f, t, Zxx = signal.stft(sig, fs=self.fs, nperseg=nperseg, 
                                noverlap=noverlap, window=self.window)
        return f, t, np.abs(Zxx)
    
    def inverse_transform(self, Zxx: np.ndarray) -> np.ndarray:
        """
        Compute inverse STFT.
        
        Args:
            Zxx: STFT coefficients
        
        Returns:
            Reconstructed signal
        """
        _, reconstructed = signal.istft(Zxx, fs=self.fs, nperseg=self.nperseg,
                                        noverlap=self.noverlap, window=self.window)
        return reconstructed
    
    def get_spectrogram(self, sig: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spectrogram (power spectrum over time).
        
        Returns:
            Tuple of (frequencies, times, spectrogram)
        """
        f, t, Zxx = self.transform(sig)
        spectrogram = np.abs(Zxx) ** 2
        return f, t, spectrogram
