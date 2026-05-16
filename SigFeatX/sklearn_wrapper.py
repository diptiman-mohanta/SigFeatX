"""
SigFeatX - sklearn_wrapper.py
==============================
Provides a sklearn-compatible TransformerMixin so SigFeatX plugs into
pipelines, grid search, and cross-validation seamlessly.

Usage
-----
    from SigFeatX.sklearn_wrapper import SigFeatXTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier

    pipe = Pipeline([
        ('features', SigFeatXTransformer(
            fs=1000,
            decomposition_methods=['fourier', 'dwt'],
            preprocess_signal=True,
            n_jobs=-1,
        )),
        ('clf', RandomForestClassifier()),
    ])
    pipe.fit(X_train, y_train)       # X_train: (n_samples, n_timesteps) or list
    pipe.predict(X_test)

Input shape
-----------
- 2D ndarray of shape (n_signals, signal_length)  -- standard sklearn convention
- list of 1D arrays (variable lengths allowed)
- 3D ndarray of shape (n_signals, n_channels, signal_length) -- multichannel
"""

import warnings
from typing import List, Optional, Union

import numpy as np

try:
    from sklearn.base import BaseEstimator, TransformerMixin
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    BaseEstimator = object
    TransformerMixin = object

from .aggregator import FeatureAggregator


class SigFeatXTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for SigFeatX feature extraction.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    decomposition_methods : list of str, optional
        Decomposition methods to apply. Default ['fourier', 'dwt'].
    preprocess_signal : bool, default True
        Whether to apply preprocessing pipeline.
    extract_raw : bool, default True
        Extract features from the raw (or preprocessed) signal.
    validate : bool, default False
        Run decomposition quality validation. Off by default for speed.
    check_consistency : bool, default False
        Run cross-method consistency checks.
    n_jobs : int, default 1
        Number of parallel workers. -1 = all cores.
    on_error : {'warn', 'raise'}, default 'warn'
        How to handle per-signal failures during transform.
    channel_names : list of str, optional
        Required when input X is 3D (multichannel).
    include_cross_channel : bool, default True
        Compute pairwise cross-channel features when multichannel.
    preprocess_kwargs : dict, optional
        Extra preprocessing arguments forwarded to FeatureAggregator.

    Attributes
    ----------
    feature_names_ : list of str
        Feature column names after fit. Populated on first transform.
    n_features_out_ : int
        Number of features produced. Populated on first transform.
    """

    def __init__(
        self,
        fs: float = 1.0,
        decomposition_methods: Optional[List[str]] = None,
        preprocess_signal: bool = True,
        extract_raw: bool = True,
        validate: bool = False,
        check_consistency: bool = False,
        n_jobs: int = 1,
        on_error: str = 'warn',
        channel_names: Optional[List[str]] = None,
        include_cross_channel: bool = True,
        preprocess_kwargs: Optional[dict] = None,
    ):
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for SigFeatXTransformer. "
                "Install with: pip install scikit-learn"
            )

        self.fs = fs
        self.decomposition_methods = decomposition_methods
        self.preprocess_signal = preprocess_signal
        self.extract_raw = extract_raw
        self.validate = validate
        self.check_consistency = check_consistency
        self.n_jobs = n_jobs
        self.on_error = on_error
        self.channel_names = channel_names
        self.include_cross_channel = include_cross_channel
        self.preprocess_kwargs = preprocess_kwargs

    # ------------------------------------------------------------------
    # sklearn interface
    # ------------------------------------------------------------------

    def fit(self, X, y=None):
        """Fit is a no-op; features are stateless. Returns self."""
        # Validate input shape early so users get fast feedback
        X_arr = self._validate_input(X)
        self._input_is_3d_ = X_arr.ndim == 3 if isinstance(X_arr, np.ndarray) else False
        return self

    def transform(self, X) -> np.ndarray:
        """
        Extract features from a batch of signals.

        Parameters
        ----------
        X : array-like of shape (n_signals, signal_length) or
            (n_signals, n_channels, signal_length) or list of 1D arrays

        Returns
        -------
        features : ndarray of shape (n_signals, n_features)
        """
        X_arr = self._validate_input(X)
        preprocess_kwargs = self.preprocess_kwargs or {}
        agg = FeatureAggregator(fs=self.fs)

        decomp = self.decomposition_methods
        if decomp is None:
            decomp = ['fourier', 'dwt']

        # Multichannel branch: 3D input
        if isinstance(X_arr, np.ndarray) and X_arr.ndim == 3:
            return self._transform_multichannel(X_arr, agg, decomp, preprocess_kwargs)

        # Standard 2D / list branch
        result = agg.extract_batch(
            X_arr,
            decomposition_methods=decomp,
            preprocess_signal=self.preprocess_signal,
            validate=self.validate,
            check_consistency=self.check_consistency,
            n_jobs=self.n_jobs,
            on_error=self.on_error,
            **preprocess_kwargs,
        )

        self.feature_names_ = result.feature_names
        self.n_features_out_ = len(result.feature_names)

        if result.n_failed > 0:
            warnings.warn(
                f"[SigFeatXTransformer] {result.n_failed}/{result.n_failed + result.n_success} "
                f"signals failed extraction; rows filled with NaN.",
                RuntimeWarning,
                stacklevel=2,
            )

        return result.dataframe.to_numpy(dtype=float)

    def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
        """Standard sklearn fit_transform; equivalent to fit().transform()."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """sklearn 1.0+ feature-name introspection."""
        if not hasattr(self, 'feature_names_'):
            raise RuntimeError(
                "feature_names_ not set. Call transform() at least once first."
            )
        return np.asarray(self.feature_names_, dtype=object)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _validate_input(self, X):
        """Accept 2D ndarray, 3D ndarray, or list-of-1D."""
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                return X.reshape(1, -1)
            if X.ndim in (2, 3):
                return X
            raise ValueError(
                f"X ndarray must be 1D, 2D, or 3D; got shape {X.shape}."
            )
        # list or similar iterable
        try:
            return list(X)
        except TypeError as exc:
            raise ValueError(
                "X must be a numpy array or an iterable of 1D signals."
            ) from exc

    def _transform_multichannel(
        self,
        X: np.ndarray,
        agg: FeatureAggregator,
        decomp: List[str],
        preprocess_kwargs: dict,
    ) -> np.ndarray:
        """Handle (n_signals, n_channels, signal_length) input."""
        n_signals, n_channels, _ = X.shape

        if self.channel_names is None:
            channel_names = [f'ch{i}' for i in range(n_channels)]
        else:
            channel_names = list(self.channel_names)
            if len(channel_names) != n_channels:
                raise ValueError(
                    f"channel_names length {len(channel_names)} != "
                    f"n_channels {n_channels}."
                )

        rows = []
        feature_names: Optional[List[str]] = None

        for i in range(n_signals):
            try:
                feats = agg.extract_multichannel(
                    X[i],
                    channel_names=channel_names,
                    decomposition_methods=decomp,
                    preprocess_signal=self.preprocess_signal,
                    validate=self.validate,
                    include_cross=self.include_cross_channel,
                    n_jobs=self.n_jobs,
                    **preprocess_kwargs,
                )
                if feature_names is None:
                    feature_names = list(feats.keys())
                rows.append([feats[k] for k in feature_names])
            except Exception as exc:
                if self.on_error == 'raise':
                    raise
                warnings.warn(
                    f"[SigFeatXTransformer] Multichannel signal {i} failed: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                rows.append(None)

        if feature_names is None:
            raise RuntimeError("All multichannel signals failed; no features produced.")

        # Fill failed rows with NaN
        out = np.full((n_signals, len(feature_names)), np.nan, dtype=float)
        for i, row in enumerate(rows):
            if row is not None:
                out[i] = row

        self.feature_names_ = feature_names
        self.n_features_out_ = len(feature_names)
        return out

    def _more_tags(self):
        """Tell sklearn we accept variable-length input."""
        return {
            'allow_nan': True,
            'requires_y': False,
            'X_types': ['2darray', '3darray'],
        }