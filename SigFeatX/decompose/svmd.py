import numpy as np
from .vmd import VMD


class SVMD:
    """Successive Variational Mode Decomposition."""
    
    def __init__(self, alpha: float = 2000, K_max: int = 10, 
                 tol: float = 1e-7, max_iter: int = 500):
        """
        Initialize SVMD.
        
        Args:
            alpha: Balancing parameter
            K_max: Maximum number of modes
            tol: Tolerance
            max_iter: Maximum iterations
        """
        self.alpha = alpha
        self.K_max = K_max
        self.tol = tol
        self.max_iter = max_iter
    
    def decompose(self, sig: np.ndarray) -> np.ndarray:
        """
        Decompose signal using successive VMD.
        
        Args:
            sig: Input signal
        
        Returns:
            Array of modes
        """
        modes = []
        residual = sig.copy()
        
        for k in range(self.K_max):
            # Extract one mode at a time
            vmd = VMD(alpha=self.alpha, K=1, tol=self.tol, max_iter=self.max_iter)
            mode = vmd.decompose(residual)[0]
            
            modes.append(mode)
            residual = residual - mode
            
            # Stop if residual energy is small
            if np.sum(residual**2) / np.sum(sig**2) < self.tol:
                break
        
        # Add remaining residual
        if np.sum(np.abs(residual)) > 1e-10:
            modes.append(residual)
        
        return np.array(modes)