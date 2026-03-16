"""Input/Output utilities for signal data."""

import numpy as np
from pathlib import Path
from typing import Dict
import json
import pickle
import csv


class SignalIO:
    """Handle signal data I/O operations."""

    @staticmethod
    def _resolve_format(filepath: str, file_format: str) -> str:
        if file_format != 'auto':
            return file_format

        suffix = Path(filepath).suffix.lower()
        mapping = {
            '.npy': 'npy',
            '.txt': 'txt',
            '.csv': 'csv',
            '.pkl': 'pkl',
            '.pickle': 'pkl',
            '.json': 'json',
        }
        if suffix in mapping:
            return mapping[suffix]

        raise ValueError(
            f"Could not infer file format from '{filepath}'. "
            "Pass file_format explicitly."
        )

    @staticmethod
    def _coerce_scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        return value
    
    @staticmethod
    def load_signal(filepath: str, file_format: str = 'auto') -> np.ndarray:
        """
        Load signal from file.
        
        Args:
            filepath: Path to signal file
            file_format: Format ('npy', 'txt', 'csv', 'pkl')
        
        Returns:
            Signal as numpy array
        """
        file_format = SignalIO._resolve_format(filepath, file_format)

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
    def save_signal(signal: np.ndarray, filepath: str, file_format: str = 'auto'):
        """Save signal to file."""
        file_format = SignalIO._resolve_format(filepath, file_format)

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
    def save_features(features: Dict[str, float], filepath: str, file_format: str = 'auto'):
        """Save extracted features."""
        file_format = SignalIO._resolve_format(filepath, file_format)

        if file_format == 'json':
            serializable = {
                key: SignalIO._coerce_scalar(value)
                for key, value in features.items()
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable, f, indent=2)
        elif file_format == 'pkl':
            with open(filepath, 'wb') as f:
                pickle.dump(features, f)
        elif file_format == 'csv':
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Feature', 'Value'])
                for key, value in features.items():
                    writer.writerow([key, SignalIO._coerce_scalar(value)])
        else:
            raise ValueError(f"Unsupported format: {file_format}")
    
    @staticmethod
    def load_features(filepath: str, file_format: str = 'auto') -> Dict[str, float]:
        """Load features from file."""
        file_format = SignalIO._resolve_format(filepath, file_format)

        if file_format == 'json':
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif file_format == 'pkl':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif file_format == 'csv':
            with open(filepath, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                features = {}
                for row in reader:
                    value = row['Value']
                    try:
                        features[row['Feature']] = float(value)
                    except (TypeError, ValueError):
                        features[row['Feature']] = value
                return features
        else:
            raise ValueError(f"Unsupported format: {file_format}")
