"""
SigFeatX - Signal Feature Extraction Library
A comprehensive library for extracting statistical features from 1D signals.
"""

__version__ = "0.4.0"
__author__ = "Diptiman Mohanta"

from .aggregator import BatchResult, FeatureAggregator
from .io import SignalIO
from .io_extensions import BatchIO
from .pipeline import Pipeline
from .preprocess import SignalPreprocessor

# sklearn wrapper is optional; only export if sklearn is installed. The
# module always imports cleanly (it defines placeholder base classes when
# sklearn is missing so SigFeatXTransformer itself stays definable), so
# check its own SKLEARN_AVAILABLE flag rather than catching ImportError
# here -- an import that no longer fails would otherwise always advertise
# the transformer as available even without scikit-learn installed.
from .sklearn_wrapper import SKLEARN_AVAILABLE as _SKLEARN_OK
from .sklearn_wrapper import SigFeatXTransformer  # noqa: F401

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
