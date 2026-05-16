"""
SigFeatX - Signal Feature Extraction Library
A comprehensive library for extracting statistical features from 1D signals.
"""

__version__ = "0.2.0"
__author__ = "Diptiman Mohanta"

from .preprocess import SignalPreprocessor
from .aggregator import FeatureAggregator, BatchResult
from .io import SignalIO
from .io_extensions import BatchIO
from .pipeline import Pipeline

# sklearn wrapper is optional; only export if sklearn is installed
try:
    from .sklearn_wrapper import SigFeatXTransformer
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

__all__ = [
    '__version__',
    'SignalPreprocessor',
    'FeatureAggregator',
    'BatchResult',
    'SignalIO',
    'BatchIO',
    'Pipeline',
]

if _SKLEARN_OK:
    __all__.append('SigFeatXTransformer')