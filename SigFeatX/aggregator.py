"""
SigFeatX - aggregator.py  (upgraded)
======================================
New additions in this version:

  Batch processing:
    extract_batch(signals, n_jobs=-1, on_error='warn')
      Processes a list or 2D array of signals and returns a pandas DataFrame.
      on_error='warn'  fills failed rows with NaN and continues (default).
      on_error='raise' propagates the first exception.

  Parallel execution:
    All methods accept n_jobs parameter.
      n_jobs=1  : sequential (default, safe for debugging)
      n_jobs=-1 : use all CPU cores
      n_jobs=N  : use N cores

  Multi-channel support:
    extract_multichannel(signals_2d, channel_names=None, include_cross=True)
      Accepts a 2D array of shape (n_channels, N).
      Extracts per-channel features with channel-prefixed keys.
      When include_cross=True, adds pairwise cross-channel features:
        - coherence (magnitude-squared, averaged across frequency)
        - cross-correlation (peak value and lag)
        - phase-locking value (PLV)

  Sliding-window analysis:
    extract_windowed(signal, window_size, step_size)
      Splits a 1D signal into overlapping windows and runs the standard
      feature pipeline on each window. Returns a BatchResult with window
      metadata columns for downstream modelling or visualisation.

All original APIs (extract_all_features, run_pipeline) are UNCHANGED.
"""

import warnings
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union

from ._validation import (
    validate_n_jobs,
    validate_sampling_rate,
    validate_signal_1d,
    validate_signal_batch,
    validate_signal_matrix,
    validate_unique_names,
)
from .preprocess import SignalPreprocessor
from .decompose import (
    FourierTransform, ShortTimeFourierTransform, WaveletDecomposer,
    EMD, VMD, SVMD, EFD, LMD, JMD,
)
from .features.features import (
    TimeDomainFeatures, FrequencyDomainFeatures,
    EntropyFeatures, NonlinearFeatures, DecompositionFeatures,
)
from .decomposition_validator import DecompositionValidator, DecompositionReport
from .feature_consistency import validate_feature_dict, CrossMethodChecker


# ---------------------------------------------------------------------------
# Pipeline metadata containers (unchanged)
# ---------------------------------------------------------------------------

@dataclass
class StageRecord:
    name: str
    params: Dict[str, Any]
    input_shape: tuple
    output_shape: tuple

    def __str__(self):
        return (
            f"  [{self.name}]  {self.input_shape} -> {self.output_shape}  "
            f"params={self.params}"
        )


@dataclass
class PipelineMetadata:
    fs: float
    window: Optional[str] = None
    nperseg: Optional[int] = None
    noverlap: Optional[int] = None
    stages: List[StageRecord] = field(default_factory=list)

    def __str__(self):
        lines = [
            "PipelineMetadata",
            f"  fs      = {self.fs} Hz",
            f"  window  = {self.window}",
            f"  nperseg = {self.nperseg}",
            f"  noverlap= {self.noverlap}",
            "  Stages:",
        ] + [str(s) for s in self.stages]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Batch result container
# ---------------------------------------------------------------------------

@dataclass
class BatchResult:
    """
    Result of extract_batch().

    Attributes
    ----------
    dataframe    : pandas DataFrame of shape (n_signals, n_features).
    errors       : dict mapping signal index to the exception that occurred.
    n_success    : number of signals processed without error.
    n_failed     : number of signals that produced errors.
    feature_names: list of feature column names.
    """
    dataframe: Any
    errors: Dict[int, Exception]
    n_success: int
    n_failed: int
    feature_names: List[str]

    def __repr__(self):
        return (
            f"BatchResult(n_signals={self.n_success + self.n_failed}, "
            f"n_success={self.n_success}, n_failed={self.n_failed}, "
            f"n_features={len(self.feature_names)})"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FeatureAggregator:
    """Main class for the SigFeatX feature extraction pipeline."""

    def __init__(self, fs: float = 1.0):
        self.fs = validate_sampling_rate(fs)
        self.preprocessor = SignalPreprocessor()

        self.ft      = FourierTransform(fs=self.fs)
        self.stft    = ShortTimeFourierTransform(fs=self.fs)
        self.wavelet = WaveletDecomposer()
        self.emd     = EMD()
        self.vmd     = VMD()
        self.svmd    = SVMD()
        self.efd     = EFD()
        self.lmd     = LMD()
        self.jmd     = JMD()

        self.time_features      = TimeDomainFeatures()
        self.freq_features      = FrequencyDomainFeatures()
        self.entropy_features   = EntropyFeatures()
        self.nonlinear_features = NonlinearFeatures()
        self.decomp_features    = DecompositionFeatures()

        self.last_quality_reports: Dict[str, DecompositionReport] = {}
        self.last_consistency_report: Optional[str] = None

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self, signal: np.ndarray,
                   denoise: bool = True,
                   normalize: bool = True,
                   detrend: bool = True,
                   **kwargs) -> np.ndarray:
        signal = validate_signal_1d(signal, name='signal')
        processed = signal.copy()
        if detrend:
            processed = self.preprocessor.detrend(
                processed,
                method=kwargs.get('detrend_method', 'linear'),
                **kwargs.get('detrend_params', {}))
        if denoise:
            processed = self.preprocessor.denoise(
                processed,
                method=kwargs.get('denoise_method', 'wavelet'),
                **kwargs.get('denoise_params', {}))
        if normalize:
            processed = self.preprocessor.normalize(
                processed, method=kwargs.get('normalize_method', 'zscore'))
        return processed

    # ------------------------------------------------------------------
    # Single-signal extraction (unchanged API)
    # ------------------------------------------------------------------

    def extract_all_features(
        self,
        signal: np.ndarray,
        decomposition_methods: Optional[List[str]] = None,
        preprocess_signal: bool = True,
        extract_raw: bool = True,
        validate: bool = True,
        check_consistency: bool = True,
        **preprocess_kwargs,
    ) -> Dict[str, float]:
        if decomposition_methods is None:
            decomposition_methods = ['fourier', 'dwt']
        signal = validate_signal_1d(signal, name='signal')

        all_features: Dict[str, float] = {}
        self.last_quality_reports = {}

        sig = self.preprocess(signal, **preprocess_kwargs) if preprocess_signal else signal.copy()

        if extract_raw:
            raw_features = self._extract_raw_features(sig)
            all_features.update(self._add_prefix(raw_features, 'raw'))

        for method in decomposition_methods:
            decomp_features, quality_report = self._extract_decomposition_features(
                sig, method, validate=validate)
            all_features.update(self._add_prefix(decomp_features, method))

            if quality_report is not None:
                self.last_quality_reports[method] = quality_report
                all_features.update(self._add_prefix(quality_report.to_dict(), method))
                if not quality_report.passed:
                    warnings.warn(
                        f"[SigFeatX] Decomposition quality FAILED for '{method}':\n"
                        + quality_report.summary(),
                        RuntimeWarning, stacklevel=2,
                    )

        component_methods = [
            m for m in decomposition_methods
            if m in ('emd', 'vmd', 'svmd', 'efd', 'dwt', 'wpd')
        ]
        if check_consistency and len(component_methods) >= 2:
            checker = CrossMethodChecker(tolerance=0.3)
            for m in component_methods:
                checker.add_from_aggregator_output(m, all_features, prefix=m)
            comparisons = checker.compare(
                features=["rms", "energy", "std", "kurtosis", "entropy"])
            inconsistent = [c for c in comparisons if not c.consistent]
            if inconsistent:
                feat_names = ", ".join(c.feature for c in inconsistent)
                warnings.warn(
                    f"[SigFeatX] Cross-method consistency warning: {feat_names}.",
                    UserWarning, stacklevel=2,
                )
            self.last_consistency_report = checker.report(
                features=["rms", "energy", "std", "kurtosis", "entropy"])

        return all_features

    # ------------------------------------------------------------------
    # NEW: Batch processing
    # ------------------------------------------------------------------

    def extract_batch(
        self,
        signals: Union[np.ndarray, List[np.ndarray]],
        decomposition_methods: Optional[List[str]] = None,
        preprocess_signal: bool = True,
        validate: bool = False,
        check_consistency: bool = False,
        n_jobs: int = 1,
        on_error: str = 'warn',
        show_progress: bool = False,
        **preprocess_kwargs,
    ) -> "BatchResult":
        """
        Extract features from a batch of signals.

        Parameters
        ----------
        signals      : list of 1D arrays, or 2D array of shape (n_signals, N).
        n_jobs       : 1 = sequential, -1 = all available CPU cores, N = N cores.
        on_error     : 'warn' fills failed rows with NaN; 'raise' stops on first error.
        show_progress: print progress counter to stdout.

        Returns
        -------
        BatchResult with .dataframe (pandas DataFrame, shape n_signals x n_features),
        .errors, .n_success, .n_failed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for extract_batch(). pip install pandas")

        if on_error not in {'warn', 'raise'}:
            raise ValueError(f"on_error must be 'warn' or 'raise'; got {on_error!r}.")

        signals = validate_signal_batch(signals, name='signals', require_finite=False)
        n_signals = len(signals)

        if n_jobs == -1:
            import os
            n_jobs = os.cpu_count() or 1
        else:
            n_jobs = validate_n_jobs(n_jobs)

        extract_kwargs = dict(
            decomposition_methods=decomposition_methods,
            preprocess_signal=preprocess_signal,
            validate=validate,
            check_consistency=check_consistency,
            **preprocess_kwargs,
        )

        results: Dict[int, Optional[Dict[str, float]]] = {}
        errors:  Dict[int, Exception] = {}

        if n_signals == 0:
            df = pd.DataFrame()
            df.index.name = 'signal_idx'
            return BatchResult(
                dataframe=df,
                errors=errors,
                n_success=0,
                n_failed=0,
                feature_names=[],
            )

        if n_jobs == 1:
            for i, sig in enumerate(signals):
                if show_progress:
                    print(f"\r  Batch: {i+1}/{n_signals}", end="", flush=True)
                try:
                    agg = FeatureAggregator(fs=self.fs)
                    results[i] = agg.extract_all_features(sig, **extract_kwargs)
                except Exception as exc:
                    if on_error == 'raise':
                        raise
                    warnings.warn(
                        f"[SigFeatX] extract_batch: signal {i} failed: {exc}",
                        RuntimeWarning, stacklevel=2,
                    )
                    results[i] = None
                    errors[i]  = exc
        else:
            task_results = _run_parallel_extract(
                {i: sig for i, sig in enumerate(signals)},
                fs=self.fs,
                extract_kwargs=extract_kwargs,
                max_workers=n_jobs,
                show_progress=show_progress,
                total=n_signals,
            )
            for i, outcome in task_results.items():
                if isinstance(outcome, Exception):
                    if on_error == 'raise':
                        raise outcome
                    warnings.warn(
                        f"[SigFeatX] extract_batch: signal {i} failed: {outcome}",
                        RuntimeWarning, stacklevel=2,
                    )
                    results[i] = None
                    errors[i] = outcome
                else:
                    results[i] = outcome

        if show_progress:
            print()

        feature_names: List[str] = []
        for i in range(n_signals):
            if results.get(i) is not None:
                feature_names = list(results[i].keys())
                break

        rows = []
        for i in range(n_signals):
            if results[i] is not None:
                rows.append(results[i])
            else:
                rows.append({k: float('nan') for k in feature_names})

        df = pd.DataFrame(rows, columns=feature_names if feature_names else None)
        df.index.name = 'signal_idx'

        n_failed  = len(errors)
        n_success = n_signals - n_failed

        return BatchResult(
            dataframe=df,
            errors=errors,
            n_success=n_success,
            n_failed=n_failed,
            feature_names=feature_names,
        )

    def extract_windowed(
        self,
        signal: np.ndarray,
        window_size: int,
        step_size: int,
        decomposition_methods: Optional[List[str]] = None,
        preprocess_signal: bool = True,
        validate: bool = False,
        check_consistency: bool = False,
        n_jobs: int = 1,
        on_error: str = 'warn',
        include_window_metadata: bool = True,
        **preprocess_kwargs,
    ) -> "BatchResult":
        """
        Extract features from overlapping sliding windows of a 1D signal.

        Parameters
        ----------
        signal                 : input 1D signal.
        window_size            : number of samples per window.
        step_size              : hop size between successive windows.
        include_window_metadata: prepend window/sample/time columns to the
                                 returned dataframe. `end_sample` is exclusive.

        Returns
        -------
        BatchResult where each row corresponds to one window.
        """
        signal = np.asarray(signal, dtype=float)
        signal = validate_signal_1d(signal, name='signal')
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0; got {window_size}.")
        if step_size <= 0:
            raise ValueError(f"step_size must be > 0; got {step_size}.")
        if len(signal) < window_size:
            raise ValueError(
                f"window_size ({window_size}) cannot exceed signal length ({len(signal)})."
            )

        starts = list(range(0, len(signal) - window_size + 1, step_size))
        windows = [signal[start : start + window_size] for start in starts]

        result = self.extract_batch(
            windows,
            decomposition_methods=decomposition_methods,
            preprocess_signal=preprocess_signal,
            validate=validate,
            check_consistency=check_consistency,
            n_jobs=n_jobs,
            on_error=on_error,
            **preprocess_kwargs,
        )

        if include_window_metadata:
            df = result.dataframe.copy()
            window_idx = np.arange(len(df), dtype=int)
            if len(df) > 0:
                try:
                    index_idx = df.index.to_numpy(dtype=int)
                    if np.all((index_idx >= 0) & (index_idx < len(starts))):
                        window_idx = index_idx
                except (TypeError, ValueError):
                    pass

            starts_arr = np.asarray(starts, dtype=int)
            end_arr = starts_arr + window_size
            df.insert(0, 'window_idx', window_idx)
            df.insert(1, 'start_sample', starts_arr[window_idx])
            df.insert(2, 'end_sample', end_arr[window_idx])
            df.insert(3, 'start_time_s', starts_arr[window_idx].astype(float) / self.fs)
            df.insert(4, 'end_time_s', end_arr[window_idx].astype(float) / self.fs)
            result.dataframe = df

        return result

    # ------------------------------------------------------------------
    # NEW: Multi-channel extraction
    # ------------------------------------------------------------------

    def extract_multichannel(
        self,
        signals_2d: np.ndarray,
        channel_names: Optional[List[str]] = None,
        decomposition_methods: Optional[List[str]] = None,
        preprocess_signal: bool = True,
        validate: bool = False,
        include_cross: bool = True,
        n_jobs: int = 1,
        **preprocess_kwargs,
    ) -> Dict[str, float]:
        """
        Extract features from a multi-channel signal.

        Parameters
        ----------
        signals_2d    : 2D array of shape (n_channels, N).
        channel_names : list of channel labels; defaults to ['ch0', 'ch1', ...].
        include_cross : compute pairwise cross-channel features (coherence,
                        cross-correlation, phase-locking value). Default True.
        n_jobs        : parallel workers for per-channel extraction.

        Returns
        -------
        Flat dict with per-channel keys prefixed by channel name, plus
        cross-channel keys prefixed by 'cross_CHa_CHb_'.
        """
        signals_2d = validate_signal_matrix(signals_2d, name='signals_2d')
        n_jobs = validate_n_jobs(n_jobs)

        n_channels, N = signals_2d.shape

        if channel_names is None:
            channel_names = [f'ch{i}' for i in range(n_channels)]
        channel_names = validate_unique_names(channel_names, name='channel_names')
        if len(channel_names) != n_channels:
            raise ValueError(
                f"len(channel_names)={len(channel_names)} != n_channels={n_channels}."
            )

        extract_kwargs = dict(
            decomposition_methods=decomposition_methods,
            preprocess_signal=preprocess_signal,
            validate=validate,
            check_consistency=False,
            **preprocess_kwargs,
        )

        channel_features: Dict[str, Dict[str, float]] = {}

        if n_jobs == 1:
            for ch_idx, ch_name in enumerate(channel_names):
                agg = FeatureAggregator(fs=self.fs)
                channel_features[ch_name] = agg.extract_all_features(
                    signals_2d[ch_idx], **extract_kwargs)
        else:
            import os
            workers = (os.cpu_count() or 1) if n_jobs == -1 else n_jobs
            task_results = _run_parallel_extract(
                {channel_names[i]: signals_2d[i] for i in range(n_channels)},
                fs=self.fs,
                extract_kwargs=extract_kwargs,
                max_workers=workers,
            )
            for ch_name, outcome in task_results.items():
                if isinstance(outcome, Exception):
                    raise outcome
                channel_features[ch_name] = outcome

        all_features: Dict[str, float] = {}
        for ch_name, feats in channel_features.items():
            for k, v in feats.items():
                all_features[f'{ch_name}_{k}'] = v

        if include_cross and n_channels > 1:
            cross_feats = self._extract_cross_channel_features(signals_2d, channel_names)
            all_features.update(cross_feats)

        return all_features

    def _extract_cross_channel_features(
        self,
        signals_2d: np.ndarray,
        channel_names: List[str],
    ) -> Dict[str, float]:
        """
        Pairwise cross-channel features for all channel pairs.

        Per pair (a, b):
          coherence_mean : mean magnitude-squared coherence
          xcorr_peak     : peak of normalised cross-correlation
          xcorr_lag      : lag (samples) at peak
          plv            : Phase-Locking Value from Hilbert phases
        """
        from scipy.signal import coherence as scipy_coherence
        from scipy.signal import hilbert

        n_channels = len(channel_names)
        features: Dict[str, float] = {}

        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                prefix = f'cross_{channel_names[i]}_{channel_names[j]}'
                a = signals_2d[i]
                b = signals_2d[j]

                # Coherence
                try:
                    _, Cxy = scipy_coherence(a, b, fs=self.fs)
                    features[f'{prefix}_coherence_mean'] = float(np.mean(Cxy))
                except Exception:
                    features[f'{prefix}_coherence_mean'] = float('nan')

                # Cross-correlation
                try:
                    a_n  = a - np.mean(a)
                    b_n  = b - np.mean(b)
                    denom = (np.sqrt(np.sum(a_n**2)) * np.sqrt(np.sum(b_n**2)) + 1e-10)
                    xcorr = np.correlate(a_n, b_n, mode='full') / denom
                    lags  = np.arange(-(len(a)-1), len(a))
                    peak_idx = int(np.argmax(np.abs(xcorr)))
                    features[f'{prefix}_xcorr_peak'] = float(xcorr[peak_idx])
                    features[f'{prefix}_xcorr_lag']  = int(lags[peak_idx])
                except Exception:
                    features[f'{prefix}_xcorr_peak'] = float('nan')
                    features[f'{prefix}_xcorr_lag']  = float('nan')

                # Phase-Locking Value
                try:
                    phase_a = np.angle(hilbert(a))
                    phase_b = np.angle(hilbert(b))
                    features[f'{prefix}_plv'] = float(
                        np.abs(np.mean(np.exp(1j * (phase_a - phase_b)))))
                except Exception:
                    features[f'{prefix}_plv'] = float('nan')

        return features

    # ------------------------------------------------------------------
    # Pluggable pipeline (unchanged)
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        signal: np.ndarray,
        preprocess_params: Optional[Dict] = None,
        decomposition_methods: Optional[List[str]] = None,
        validate: bool = True,
    ) -> Tuple[Dict[str, float], PipelineMetadata]:
        preprocess_params     = preprocess_params or {}
        decomposition_methods = decomposition_methods or ["fourier", "dwt"]
        signal = validate_signal_1d(signal, name='signal')

        meta = PipelineMetadata(
            fs=self.fs,
            window=self.stft.window,
            nperseg=self.stft.nperseg,
            noverlap=self.stft.noverlap,
        )

        in_shape = signal.shape
        sig = self.preprocess(signal, **preprocess_params)
        meta.stages.append(StageRecord(
            name="preprocess",
            params={
                "detrend_method":   preprocess_params.get("detrend_method", "linear"),
                "denoise_method":   preprocess_params.get("denoise_method", "wavelet"),
                "normalize_method": preprocess_params.get("normalize_method", "zscore"),
            },
            input_shape=in_shape,
            output_shape=sig.shape,
        ))

        raw = self._extract_raw_features(sig)
        meta.stages.append(StageRecord(
            name="raw_features",
            params={"fs": self.fs},
            input_shape=sig.shape,
            output_shape=(len(raw),),
        ))

        all_features = self._add_prefix(raw, "raw")

        for method in decomposition_methods:
            if method == "stft" and self.stft.fs != self.fs:
                warnings.warn(
                    f"[SigFeatX] Metadata mismatch: STFT fs={self.stft.fs} "
                    f"but FeatureAggregator fs={self.fs}.",
                    UserWarning, stacklevel=2,
                )

            decomp_feats, quality_report = self._extract_decomposition_features(
                sig, method, validate=validate)
            all_features.update(self._add_prefix(decomp_feats, method))

            stage_params: Dict[str, Any] = {"method": method, "fs": self.fs}
            if method == "emd":
                stage_params["max_imf"] = self.emd.max_imf
            if method == "vmd":
                stage_params["K"]     = self.vmd.K
                stage_params["alpha"] = self.vmd.alpha
            if method == "svmd":
                stage_params["K_max"] = self.svmd.K_max
                stage_params["alpha"] = self.svmd.alpha
            if method == "lmd":
                stage_params["max_pf"] = self.lmd.max_pf
            if method == "jmd":
                stage_params["K"]    = self.jmd.K
                stage_params["beta"] = self.jmd.beta

            if quality_report is not None:
                all_features.update(self._add_prefix(quality_report.to_dict(), method))
                snr_val = quality_report.snr_db
                stage_params["decomp_snr_db"] = (
                    999.0 if not np.isfinite(snr_val) else snr_val)
                stage_params["decomp_passed"] = quality_report.passed

            meta.stages.append(StageRecord(
                name=f"decompose+features [{method}]",
                params=stage_params,
                input_shape=sig.shape,
                output_shape=(len(decomp_feats),),
            ))

        return all_features, meta

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_raw_features(self, sig: np.ndarray) -> Dict[str, float]:
        features = {}
        features.update(self.time_features.extract(sig))
        features.update(self.freq_features.extract(sig, self.fs))
        features.update(self.entropy_features.extract(sig))
        features.update(self.nonlinear_features.extract(sig))
        violations = validate_feature_dict(features, method="raw")
        for v in violations:
            warnings.warn(str(v), RuntimeWarning, stacklevel=3)
        return features

    def _extract_decomposition_features(
        self,
        sig: np.ndarray,
        method: str,
        validate: bool = True,
    ) -> Tuple[Dict[str, float], Optional[DecompositionReport]]:
        features: Dict[str, float] = {}
        quality_report: Optional[DecompositionReport] = None

        if method == 'fourier':
            freqs, magnitude = self.ft.transform(sig)
            features['dominant_freq']  = freqs[np.argmax(magnitude)]
            features['mean_magnitude'] = np.mean(magnitude)
            features['std_magnitude']  = np.std(magnitude)
        elif method == 'stft':
            f, t, Zxx = self.stft.transform(sig)
            features['mean_power'] = np.mean(Zxx ** 2)
            features['std_power']  = np.std(Zxx ** 2)
            features['max_power']  = np.max(Zxx ** 2)
        elif method == 'dwt':
            coeffs = self.wavelet.dwt(sig)
            features.update(self.decomp_features.extract_from_components(coeffs, 'dwt'))
            if validate:
                quality_report = DecompositionValidator.evaluate(sig, coeffs, method="DWT")
        elif method == 'wpd':
            wpd_dict   = self.wavelet.wpd(sig)
            wpd_coeffs = list(wpd_dict.values())
            features.update(self.decomp_features.extract_from_components(wpd_coeffs, 'wpd'))
            if validate:
                quality_report = DecompositionValidator.evaluate(sig, wpd_coeffs, method="WPD")
        elif method == 'emd':
            imfs = self.emd.decompose(sig)
            features.update(self.decomp_features.extract_from_components(imfs, 'emd'))
            if validate:
                quality_report = DecompositionValidator.evaluate(sig, imfs, method="EMD")
        elif method == 'vmd':
            modes      = self.vmd.decompose(sig)
            modes_list = [modes[i] for i in range(len(modes))]
            features.update(self.decomp_features.extract_from_components(modes_list, 'vmd'))
            if validate:
                quality_report = DecompositionValidator.evaluate(sig, modes, method="VMD")
        elif method == 'svmd':
            modes      = self.svmd.decompose(sig)
            modes_list = [modes[i] for i in range(len(modes))]
            features.update(self.decomp_features.extract_from_components(modes_list, 'svmd'))
            if validate:
                quality_report = DecompositionValidator.evaluate(sig, modes, method="SVMD")
        elif method == 'efd':
            modes      = self.efd.decompose(sig)
            modes_list = [modes[i] for i in range(len(modes))]
            features.update(self.decomp_features.extract_from_components(modes_list, 'efd'))
            if validate:
                quality_report = DecompositionValidator.evaluate(sig, modes, method="EFD")
        elif method == 'lmd':
            pfs = self.lmd.decompose(sig)
            features.update(self.decomp_features.extract_from_components(pfs, 'lmd'))
            if validate:
                quality_report = DecompositionValidator.evaluate(sig, pfs, method="LMD")
 
        elif method == 'jmd':
            modes, jump = self.jmd.decompose(sig)
            modes_list  = [modes[i] for i in range(len(modes))]
            # Treat the jump as an extra component so feature extraction covers it.
            all_components = modes_list + [jump]
            features.update(self.decomp_features.extract_from_components(all_components, 'jmd'))
            # Jump-specific summary features
            features['jmd_jump_energy']  = float(np.sum(jump ** 2))
            features['jmd_jump_rms']     = float(np.sqrt(np.mean(jump ** 2)))
            features['jmd_jump_n_steps'] = float(_count_jump_steps(jump))
            if validate:
                quality_report = DecompositionValidator.evaluate(sig, modes, method="JMD")
 
        else:
            raise ValueError(
                f"Unknown decomposition method '{method}'. "
                "Valid: 'fourier','stft','dwt','wpd','emd','vmd','svmd','efd','lmd','jmd'."
            )
        
        violations = validate_feature_dict(features, method=method)
        for v in violations:
            warnings.warn(str(v), RuntimeWarning, stacklevel=3)

        return features, quality_report

    @staticmethod
    def _add_prefix(features: Dict[str, float], prefix: str) -> Dict[str, float]:
        return {f'{prefix}_{k}': v for k, v in features.items()}

    def get_feature_names(
        self, decomposition_methods: Optional[List[str]] = None
    ) -> List[str]:
        if decomposition_methods is None:
            decomposition_methods = ['fourier', 'dwt']
        dummy = np.random.randn(1000)
        features = self.extract_all_features(
            dummy,
            decomposition_methods=decomposition_methods,
            preprocess_signal=False,
            validate=False,
            check_consistency=False,
        )
        return list(features.keys())


# ---------------------------------------------------------------------------
# Top-level worker (must be module-level to be picklable by ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _worker_extract(
    sig: np.ndarray,
    fs: float,
    extract_kwargs: dict,
) -> Dict[str, float]:
    """Worker for parallel batch and multi-channel extraction."""
    agg = FeatureAggregator(fs=fs)
    return agg.extract_all_features(sig, **extract_kwargs)


def _run_parallel_extract(
    tasks: Dict[Any, np.ndarray],
    *,
    fs: float,
    extract_kwargs: dict,
    max_workers: int,
    show_progress: bool = False,
    total: Optional[int] = None,
) -> Dict[Any, Union[Dict[str, float], Exception]]:
    """Run extraction tasks with a process pool, falling back to threads if needed."""
    total = len(tasks) if total is None else total

    def _execute(executor_cls):
        outputs: Dict[Any, Union[Dict[str, float], Exception]] = {}
        with executor_cls(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_worker_extract, sig, fs, extract_kwargs): key
                for key, sig in tasks.items()
            }
            completed = 0
            for future in as_completed(futures):
                key = futures[future]
                completed += 1
                if show_progress:
                    print(f"\r  Batch: {completed}/{total}", end="", flush=True)
                try:
                    outputs[key] = future.result()
                except Exception as exc:
                    outputs[key] = exc
        return outputs

    try:
        return _execute(ProcessPoolExecutor)
    except (OSError, PermissionError) as exc:
        warnings.warn(
            "[SigFeatX] Process-based parallelism is unavailable in this environment; "
            "falling back to threads.",
            RuntimeWarning,
            stacklevel=3,
        )
        return _execute(ThreadPoolExecutor)

def _count_jump_steps(jump: np.ndarray, threshold_ratio: float = 0.1) -> int:
    """      
      Count significant step-changes in the extracted jump component
        A step is any point where |jump'[i]| > threshold_ratio * max|jump'|.
        Returns an integer count stored as float for compatibility with the
        feature dict (all values are float).
    """
    if len(jump) < 2:
        return 0
    deriv     = np.diff(jump)
    max_deriv = np.max(np.abs(deriv))
    if max_deriv < 1e-10:
        return 0
    return int(np.sum(np.abs(deriv) > threshold_ratio * max_deriv))