"""
SigFeatX - pipeline.py
=======================
Fluent builder for chained preprocessing + decomposition + feature extraction.

Usage
-----
    from SigFeatX.pipeline import Pipeline

    # Single signal
    features = (
        Pipeline(fs=1000)
        .detrend(method='linear')
        .denoise(method='wavelet')
        .normalize(method='zscore')
        .decompose(['fourier', 'dwt'])
        .extract(signal)
    )

    # Batch
    df = (
        Pipeline(fs=1000)
        .detrend(method='als', lam=1e5)
        .denoise(method='bandpass', low_hz=1, high_hz=40)
        .normalize(method='robust')
        .decompose(['emd', 'vmd'])
        .with_parallel(n_jobs=-1)
        .with_progress()
        .extract_batch(signals)
    )

    # Windowed
    windowed = (
        Pipeline(fs=250)
        .decompose(['fourier'])
        .extract_windowed(signal, window_size=512, step_size=256)
    )
"""

from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from .aggregator import BatchResult, FeatureAggregator


class Pipeline:
    """
    Fluent pipeline for signal feature extraction.

    Every configuration method returns ``self`` so calls can be chained.
    The pipeline is lazy: nothing runs until ``.extract*`` is called.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    """

    def __init__(self, fs: float = 1.0):
        self.fs = fs
        self._detrend_cfg: Optional[Dict[str, Any]] = None
        self._denoise_cfg: Optional[Dict[str, Any]] = None
        self._normalize_cfg: Optional[Dict[str, Any]] = None
        self._decomposition_methods: List[str] = ['fourier', 'dwt']
        self._extract_raw: bool = True
        self._validate: bool = False
        self._check_consistency: bool = False
        self._n_jobs: int = 1
        self._on_error: str = 'warn'
        self._show_progress: bool = False
        self._preprocess_enabled: bool = False

    # ------------------------------------------------------------------
    # Preprocessing chain (each method enables preprocessing)
    # ------------------------------------------------------------------

    def detrend(self, method: str = 'linear', **kwargs) -> 'Pipeline':
        """
        Add a detrend step. Valid methods: 'linear', 'constant', 'als'.
        ALS params: ``lam``, ``p``, ``n_iter``.
        """
        self._detrend_cfg = {'method': method, **kwargs}
        self._preprocess_enabled = True
        return self

    def denoise(self, method: str = 'wavelet', **kwargs) -> 'Pipeline':
        """
        Add a denoise step. Valid methods: 'wavelet', 'median', 'lowpass',
        'bandpass', 'notch'.
        """
        self._denoise_cfg = {'method': method, **kwargs}
        self._preprocess_enabled = True
        return self

    def normalize(self, method: str = 'zscore') -> 'Pipeline':
        """Add a normalize step. Valid methods: 'zscore', 'minmax', 'robust'."""
        self._normalize_cfg = {'method': method}
        self._preprocess_enabled = True
        return self

    def no_preprocess(self) -> 'Pipeline':
        """Disable all preprocessing (explicit raw extraction)."""
        self._detrend_cfg = None
        self._denoise_cfg = None
        self._normalize_cfg = None
        self._preprocess_enabled = False
        return self

    # ------------------------------------------------------------------
    # Decomposition / feature config
    # ------------------------------------------------------------------

    def decompose(self, methods: Union[str, List[str]]) -> 'Pipeline':
        """
        Set decomposition methods. Pass a single name or list.
        Valid: 'fourier', 'stft', 'dwt', 'wpd', 'emd', 'vmd', 'svmd',
        'efd', 'lmd', 'jmd'.
        """
        if isinstance(methods, str):
            methods = [methods]
        self._decomposition_methods = list(methods)
        return self

    def include_raw(self, enabled: bool = True) -> 'Pipeline':
        """Toggle raw-signal feature extraction (default on)."""
        self._extract_raw = enabled
        return self

    def with_validation(self, enabled: bool = True) -> 'Pipeline':
        """Toggle decomposition quality validation (off by default)."""
        self._validate = enabled
        return self

    def with_consistency_check(self, enabled: bool = True) -> 'Pipeline':
        """Toggle cross-method consistency checking."""
        self._check_consistency = enabled
        return self

    # ------------------------------------------------------------------
    # Execution config
    # ------------------------------------------------------------------

    def with_parallel(self, n_jobs: int = -1) -> 'Pipeline':
        """Enable parallel batch/multichannel execution."""
        self._n_jobs = n_jobs
        return self

    def with_progress(self, enabled: bool = True) -> 'Pipeline':
        """Enable tqdm-style progress reporting for batch operations."""
        self._show_progress = enabled
        return self

    def on_error(self, mode: str = 'warn') -> 'Pipeline':
        """Set error policy. 'warn' fills failed rows with NaN; 'raise' stops."""
        if mode not in ('warn', 'raise'):
            raise ValueError(f"mode must be 'warn' or 'raise'; got {mode!r}.")
        self._on_error = mode
        return self

    # ------------------------------------------------------------------
    # Terminal operations
    # ------------------------------------------------------------------

    def extract(self, signal: np.ndarray) -> Dict[str, float]:
        """Run the configured pipeline on a single 1D signal."""
        agg = FeatureAggregator(fs=self.fs)
        return agg.extract_all_features(
            signal,
            decomposition_methods=self._decomposition_methods,
            preprocess_signal=self._preprocess_enabled,
            extract_raw=self._extract_raw,
            validate=self._validate,
            check_consistency=self._check_consistency,
            **self._preprocess_kwargs(),
        )

    def extract_batch(self, signals) -> BatchResult:
        """Run on a batch of signals (list of 1D arrays or 2D array)."""
        agg = FeatureAggregator(fs=self.fs)
        return agg.extract_batch(
            signals,
            decomposition_methods=self._decomposition_methods,
            preprocess_signal=self._preprocess_enabled,
            validate=self._validate,
            check_consistency=self._check_consistency,
            n_jobs=self._n_jobs,
            on_error=self._on_error,
            show_progress=self._show_progress,
            **self._preprocess_kwargs(),
        )

    def extract_windowed(
        self,
        signal: np.ndarray,
        window_size: int,
        step_size: int,
    ) -> BatchResult:
        """Run on overlapping windows of a single 1D signal."""
        agg = FeatureAggregator(fs=self.fs)
        return agg.extract_windowed(
            signal,
            window_size=window_size,
            step_size=step_size,
            decomposition_methods=self._decomposition_methods,
            preprocess_signal=self._preprocess_enabled,
            validate=self._validate,
            check_consistency=self._check_consistency,
            n_jobs=self._n_jobs,
            on_error=self._on_error,
            **self._preprocess_kwargs(),
        )

    def extract_multichannel(
        self,
        signals_2d: np.ndarray,
        channel_names: Optional[List[str]] = None,
        include_cross: bool = True,
    ) -> Dict[str, float]:
        """Run on a (n_channels, n_samples) array."""
        agg = FeatureAggregator(fs=self.fs)
        return agg.extract_multichannel(
            signals_2d,
            channel_names=channel_names,
            decomposition_methods=self._decomposition_methods,
            preprocess_signal=self._preprocess_enabled,
            validate=self._validate,
            include_cross=include_cross,
            n_jobs=self._n_jobs,
            **self._preprocess_kwargs(),
        )

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def clone(self) -> 'Pipeline':
        """Return a deep copy. Useful for grid search over configurations."""
        return deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the configuration."""
        return {
            'fs': self.fs,
            'detrend': self._detrend_cfg,
            'denoise': self._denoise_cfg,
            'normalize': self._normalize_cfg,
            'decomposition_methods': list(self._decomposition_methods),
            'extract_raw': self._extract_raw,
            'validate': self._validate,
            'check_consistency': self._check_consistency,
            'n_jobs': self._n_jobs,
            'on_error': self._on_error,
            'show_progress': self._show_progress,
            'preprocess_enabled': self._preprocess_enabled,
        }

    def __repr__(self) -> str:
        steps = []
        if self._detrend_cfg:
            steps.append(f"detrend({self._detrend_cfg['method']})")
        if self._denoise_cfg:
            steps.append(f"denoise({self._denoise_cfg['method']})")
        if self._normalize_cfg:
            steps.append(f"normalize({self._normalize_cfg['method']})")
        steps.append(f"decompose({self._decomposition_methods})")
        return f"Pipeline(fs={self.fs}, steps={' -> '.join(steps)})"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _preprocess_kwargs(self) -> Dict[str, Any]:
        """Translate fluent config into aggregator preprocess_kwargs."""
        if not self._preprocess_enabled:
            return {}

        out: Dict[str, Any] = {
            'detrend': self._detrend_cfg is not None,
            'denoise': self._denoise_cfg is not None,
            'normalize': self._normalize_cfg is not None,
        }

        if self._detrend_cfg is not None:
            cfg = dict(self._detrend_cfg)
            out['detrend_method'] = cfg.pop('method')
            if cfg:
                out['detrend_params'] = cfg

        if self._denoise_cfg is not None:
            cfg = dict(self._denoise_cfg)
            method = cfg.pop('method')
            out['denoise_method'] = method
            # bandpass / notch / lowpass all need the sampling rate; the
            # underlying preprocessor defaults fs=1.0 which makes any real
            # frequency cutoff fail the Nyquist check.
            if method in ('bandpass', 'notch', 'lowpass') and 'fs' not in cfg:
                cfg['fs'] = self.fs
            if cfg:
                out['denoise_params'] = cfg

        if self._normalize_cfg is not None:
            out['normalize_method'] = self._normalize_cfg['method']

        return out