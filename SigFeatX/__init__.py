"""
SigFeatX - Signal Feature Extraction Library
A comprehensive library for extracting statistical features from 1D signals.
"""

__version__ = "0.3.0"
__author__ = "Diptiman Mohanta"

from .aggregator import BatchResult, FeatureAggregator
from .io import SignalIO
from .io_extensions import BatchIO
from .pipeline import Pipeline
from .preprocess import SignalPreprocessor

# sklearn wrapper is optional; only export if sklearn is installed
try:
    from .sklearn_wrapper import SigFeatXTransformer  # noqa: F401
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

__all__ = [
    'BatchIO',
    'BatchResult',
    'FeatureAggregator',
    'Pipeline',
    'SignalIO',
    'SignalPreprocessor',
    '__version__',
]

if _SKLEARN_OK:
    __all__.append('SigFeatXTransformer')
