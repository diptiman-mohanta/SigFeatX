"""Input/Output utilities for signal data."""

import numpy as np
from typing import Union, Dict, List
import json
import pickle


class SignalIO:
    """Handle signal data I/O operations."""
    
    @staticmethod
    def load_signal(filepath: str, file_format: str = 'npy') -> np.ndarray:
        """
        Load signal from file.
        
        Args:
            filepath: Path to signal file
            file_format: Format ('npy', 'txt', 'csv', 'pkl')
        
        Returns:
            Signal as numpy array
        """
        if file_format == 'npy':
            return np.load(filepath)
        elif file_format == 'txt' or file_format == 'csv':
            return np.loadtxt(filepath)
        elif file_format == 'pkl':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
    
    @staticmethod
    def save_signal(signal: np.ndarray, filepath: str, file_format: str = 'npy'):
        """Save signal to file."""
        if file_format == 'npy':
            np.save(filepath, signal)
        elif file_format == 'txt' or file_format == 'csv':
            np.savetxt(filepath, signal)
        elif file_format == 'pkl':
            with open(filepath, 'wb') as f:
                pickle.dump(signal, f)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
    
    @staticmethod
    def save_features(features: Dict[str, float], filepath: str, file_format: str = 'json'):
        """Save extracted features."""
        if file_format == 'json':
            with open(filepath, 'w') as f:
                json.dump(features, f, indent=2)
        elif file_format == 'pkl':
            with open(filepath, 'wb') as f:
                pickle.dump(features, f)
        elif file_format == 'csv':
            import csv
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Feature', 'Value'])
                for key, value in features.items():
                    writer.writerow([key, value])
    
    @staticmethod
    def load_features(filepath: str, file_format: str = 'json') -> Dict[str, float]:
        """Load features from file."""
        if file_format == 'json':
            with open(filepath, 'r') as f:
                return json.load(f)
        elif file_format == 'pkl':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {file_format}")