import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftfreq


class EFD:
    """Empirical Fourier Decomposition."""
    
    def __init__(self, n_modes: int = 5):
        """
        Initialize EFD.
        
        Args:
            n_modes: Number of modes to extract
        """
        self.n_modes = n_modes
    
    def decompose(self, sig: np.ndarray) -> np.ndarray:
        """
        Decompose signal using EFD.
        
        Args:
            sig: Input signal
        
        Returns:
            Array of modes (n_modes x N)
        """
        N = len(sig)
        
        # Compute FFT
        fft_sig = fft(sig)
        freqs = fftfreq(N)
        power = np.abs(fft_sig) ** 2
        
        # Find boundaries by detecting local maxima in spectrum
        pos_freqs = freqs[freqs >= 0]
        pos_power = power[freqs >= 0]
        
        # Find peaks in power spectrum
        peaks, _ = signal.find_peaks(pos_power, height=np.max(pos_power) * 0.1)
        
        # Sort peaks by power
        if len(peaks) > 0:
            peak_powers = pos_power[peaks]
            sorted_idx = np.argsort(peak_powers)[::-1]
            peaks = peaks[sorted_idx]
        
        # Determine frequency boundaries
        boundaries = self._determine_boundaries(pos_freqs, peaks, self.n_modes)
        
        # Extract modes using bandpass filters
        modes = np.zeros((self.n_modes, N))
        
        for k in range(self.n_modes):
            if k < len(boundaries) - 1:
                freq_low = boundaries[k]
                freq_high = boundaries[k + 1]
                
                # Create filter in frequency domain
                filter_fft = np.zeros_like(fft_sig, dtype=complex)
                mask = (np.abs(freqs) >= freq_low) & (np.abs(freqs) < freq_high)
                filter_fft[mask] = fft_sig[mask]
                
                # Inverse FFT to get mode
                modes[k] = np.real(ifft(filter_fft))
        
        return modes
    
    def _determine_boundaries(self, freqs: np.ndarray, peaks: np.ndarray, 
                             n_modes: int) -> np.ndarray:
        """Determine frequency boundaries for modes."""
        if len(peaks) == 0:
            # Uniform distribution if no peaks found
            return np.linspace(0, np.max(freqs), n_modes + 1)
        
        # Use peaks as boundary guides
        boundaries = [0]
        
        for i in range(min(n_modes - 1, len(peaks))):
            if i < len(peaks):
                boundaries.append(freqs[peaks[i]])
        
        boundaries.append(np.max(freqs))
        
        # Ensure we have n_modes+1 boundaries
        while len(boundaries) < n_modes + 1:
            # Add midpoints
            new_boundaries = []
            for i in range(len(boundaries) - 1):
                new_boundaries.append(boundaries[i])
                new_boundaries.append((boundaries[i] + boundaries[i+1]) / 2)
            new_boundaries.append(boundaries[-1])
            boundaries = new_boundaries[:n_modes + 1]
        
        return np.array(sorted(boundaries[:n_modes + 1]))
    
    def reconstruct(self, modes: np.ndarray) -> np.ndarray:
        """
        Reconstruct signal from modes.
        
        Args:
            modes: Array of modes
        
        Returns:
            Reconstructed signal
        """
        return np.sum(modes, axis=0)