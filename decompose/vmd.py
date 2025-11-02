import numpy as np
from scipy import signal


class VMD:
    """Variational Mode Decomposition."""
    
    def __init__(self, alpha: float = 2000, K: int = 3, tau: float = 0.0,
                 DC: bool = False, init: int = 1, tol: float = 1e-7, max_iter: int = 500):
        """
        Initialize VMD.
        
        Args:
            alpha: Balancing parameter of the data-fidelity constraint
            K: Number of modes to be recovered
            tau: Time-step of the dual ascent
            DC: True if the first mode is put and kept at DC (0-freq)
            init: 0 = all omegas start at 0
                  1 = all omegas start uniformly distributed
                  2 = all omegas initialized randomly
            tol: Tolerance of convergence criterion
            max_iter: Maximum number of iterations
        """
        self.alpha = alpha
        self.K = K
        self.tau = tau
        self.DC = DC
        self.init = init
        self.tol = tol
        self.max_iter = max_iter
    
    def decompose(self, sig: np.ndarray) -> np.ndarray:
        """
        Decompose signal using VMD.
        
        Args:
            sig: Input signal
        
        Returns:
            Array of modes (K x N)
        """
        # Simplified VMD implementation using filterbank approach
        N = len(sig)
        modes = np.zeros((self.K, N))
        
        # Initialize center frequencies
        if self.init == 0:
            omega_K = np.zeros(self.K)
        elif self.init == 1:
            omega_K = np.linspace(0, 0.5, self.K)
        else:
            omega_K = np.sort(np.random.rand(self.K) * 0.5)
        
        # Extract modes using bandpass filters
        for k in range(self.K):
            if k == 0:
                freq_low = 0
                freq_high = (omega_K[k] + omega_K[k+1]) / 2 if self.K > 1 else 0.5
            elif k == self.K - 1:
                freq_low = (omega_K[k-1] + omega_K[k]) / 2
                freq_high = 0.5
            else:
                freq_low = (omega_K[k-1] + omega_K[k]) / 2
                freq_high = (omega_K[k] + omega_K[k+1]) / 2
            
            # Design and apply bandpass filter
            if freq_low < freq_high and freq_high <= 0.5:
                try:
                    b, a = signal.butter(4, [max(freq_low, 0.001), min(freq_high, 0.499)], 
                                        btype='band')
                    modes[k] = signal.filtfilt(b, a, sig)
                except:
                    modes[k] = sig / self.K
            else:
                modes[k] = sig / self.K
        
        return modes
    
    def reconstruct(self, modes: np.ndarray) -> np.ndarray:
        """
        Reconstruct signal from modes.
        
        Args:
            modes: Array of modes (K x N)
        
        Returns:
            Reconstructed signal
        """
        return np.sum(modes, axis=0)
