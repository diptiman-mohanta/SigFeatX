import numpy as np
from scipy import signal
from typing import List, Optional


class EMD:
    """Empirical Mode Decomposition."""
    
    def __init__(self, max_imf: int = 10, max_iter: int = 100):
        """
        Initialize EMD.
        
        Args:
            max_imf: Maximum number of IMFs to extract
            max_iter: Maximum iterations for sifting
        """
        self.max_imf = max_imf
        self.max_iter = max_iter
    
    def decompose(self, sig: np.ndarray) -> List[np.ndarray]:
        """
        Decompose signal into IMFs.
        
        Args:
            sig: Input signal
        
        Returns:
            List of Intrinsic Mode Functions (IMFs)
        """
        imfs = []
        residual = sig.copy()
        
        for _ in range(self.max_imf):
            imf = self._extract_imf(residual)
            
            if imf is None:
                break
            
            imfs.append(imf)
            residual = residual - imf
            
            # Stop if residual is monotonic or too small
            if self._is_monotonic(residual) or np.sum(np.abs(residual)) < 1e-10:
                break
        
        # Add residual as last component
        if len(residual) > 0 and np.sum(np.abs(residual)) > 1e-10:
            imfs.append(residual)
        
        return imfs
    
    def _extract_imf(self, sig: np.ndarray) -> Optional[np.ndarray]:
        """Extract a single IMF using sifting process."""
        h = sig.copy()
        
        for _ in range(self.max_iter):
            # Find extrema
            max_peaks = signal.argrelextrema(h, np.greater)[0]
            min_peaks = signal.argrelextrema(h, np.less)[0]
            
            # Need at least 2 extrema of each type
            if len(max_peaks) < 2 or len(min_peaks) < 2:
                return None
            
            # Add boundary points for interpolation
            t = np.arange(len(h))
            max_peaks = np.concatenate(([0], max_peaks, [len(h)-1]))
            min_peaks = np.concatenate(([0], min_peaks, [len(h)-1]))
            
            # Interpolate upper and lower envelopes
            upper_env = np.interp(t, max_peaks, h[max_peaks])
            lower_env = np.interp(t, min_peaks, h[min_peaks])
            
            # Calculate mean envelope
            mean_env = (upper_env + lower_env) / 2
            
            # Subtract mean from signal
            h_new = h - mean_env
            
            # Check stopping criterion
            if np.sum((h - h_new) ** 2) / (np.sum(h ** 2) + 1e-10) < 1e-6:
                return h_new
            
            h = h_new
        
        return h
    
    def _is_monotonic(self, sig: np.ndarray) -> bool:
        """Check if signal is monotonic."""
        diff = np.diff(sig)
        return np.all(diff >= 0) or np.all(diff <= 0)
    
    def reconstruct(self, imfs: List[np.ndarray]) -> np.ndarray:
        """
        Reconstruct signal from IMFs.
        
        Args:
            imfs: List of IMFs
        
        Returns:
            Reconstructed signal
        """
        return np.sum(imfs, axis=0)