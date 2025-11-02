"""
SigFeatX - Signal Feature Extraction Library
A comprehensive library for extracting statistical features from 1D signals.
"""

__version__ = "0.1.0"
__author__ = "Diptiman Mohanta"

from .preprocess import SignalPreprocessor
from .aggregator import FeatureAggregator
from .io import SignalIO

__all__ = [
    'SignalPreprocessor',
    'FeatureAggregator',
    'SignalIO',
]