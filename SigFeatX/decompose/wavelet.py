"""Wavelet decomposition methods."""

import numpy as np
import pywt
import warnings
from typing import List, Dict, Optional

from SigFeatX._validation import validate_signal_1d


class WaveletDecomposer:
    """Wavelet-based decomposition methods."""
    
    def __init__(self, wavelet: str = 'db4'):
        """
        Initialize wavelet decomposer.
        
        Args:
            wavelet: Wavelet type (e.g., 'db4', 'sym5', 'coif3')
        """
        self.wavelet = wavelet
    
    def dwt(self, signal: np.ndarray, level: Optional[int] = None) -> List[np.ndarray]:
        """
        Discrete Wavelet Transform.
        
        Args:
            signal: Input signal
            level: Decomposition level (None for maximum)
        
        Returns:
            List of wavelet coefficients [cA_n, cD_n, cD_n-1, ..., cD_1]
        """
        signal = validate_signal_1d(signal, name='signal')
        if level is not None and level < 1:
            raise ValueError(f"level must be >= 1; got {level}.")
        if level is None:
            level = pywt.dwt_max_level(len(signal), self.wavelet)
        level = max(1, int(level))

        coeffs = pywt.wavedec(signal, self.wavelet, level=level)
        return coeffs
    
    def idwt(self, coeffs: List[np.ndarray]) -> np.ndarray:
        """
        Inverse Discrete Wavelet Transform.
        
        Args:
            coeffs: Wavelet coefficients
        
        Returns:
            Reconstructed signal
        """
        return pywt.waverec(coeffs, self.wavelet)
    
    def wpd(self, signal: np.ndarray, level: int = 3, order: str = 'freq') -> Dict[str, np.ndarray]:
        """
        Wavelet Packet Decomposition.
        
        Args:
            signal: Input signal
            level: Decomposition level
            order: Node ordering ('natural' or 'freq')
        
        Returns:
            Dictionary of node paths to coefficients
        """
        signal = validate_signal_1d(signal, name='signal')
        if level < 1:
            raise ValueError(f"level must be >= 1; got {level}.")
        wp = pywt.WaveletPacket(data=signal, wavelet=self.wavelet, maxlevel=level)
        nodes = [node.path for node in wp.get_level(level, order=order)]
        return {node: wp[node].data for node in nodes}
    
    def cwt(self, signal: np.ndarray, scales: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Continuous Wavelet Transform.
        
        Args:
            signal: Input signal
            scales: Scales for CWT (None for default)
        
        Returns:
            CWT coefficients
        """
        signal = validate_signal_1d(signal, name='signal')
        if scales is None:
            upper = max(2, min(128, len(signal) // 2))
            scales = np.arange(1, upper)
        scales = np.asarray(scales)
        if scales.ndim != 1 or len(scales) == 0:
            raise ValueError("scales must be a non-empty 1D array.")
        if np.any(scales <= 0):
            raise ValueError("scales must contain only positive values.")

        try:
            coeffs, _ = pywt.cwt(signal, scales, self.wavelet)
        except (AttributeError, ValueError):
            warnings.warn(
                f"Wavelet '{self.wavelet}' is not suitable for CWT; falling back to 'morl'.",
                RuntimeWarning,
                stacklevel=2,
            )
            coeffs, _ = pywt.cwt(signal, scales, 'morl')
        return coeffs
    
    def swt(self, signal: np.ndarray, level: int = 1) -> List[tuple[np.ndarray, np.ndarray]]:
        """
        Stationary Wavelet Transform.
        
        Args:
            signal: Input signal
            level: Decomposition level
        
        Returns:
            List of (approximation, detail) coefficient pairs
        """
        signal = validate_signal_1d(signal, name='signal')
        if level < 1:
            raise ValueError(f"level must be >= 1; got {level}.")
        return pywt.swt(signal, self.wavelet, level=level)
