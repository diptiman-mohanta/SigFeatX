import warnings
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .preprocess import SignalPreprocessor
from .decompose import (
    FourierTransform, ShortTimeFourierTransform, WaveletDecomposer,
    EMD, VMD, SVMD, EFD
)
from .features.features import (
    TimeDomainFeatures, FrequencyDomainFeatures,
    EntropyFeatures, NonlinearFeatures, DecompositionFeatures
)
from .decomposition_validator import DecompositionValidator, DecompositionReport
from .feature_consistency import validate_feature_dict, CrossMethodChecker


# ---------------------------------------------------------------------------
# Pipeline metadata container (Review #3)
# ---------------------------------------------------------------------------

@dataclass
class StageRecord:
    name: str
    params: Dict[str, Any]
    input_shape: tuple
    output_shape: tuple

    def __str__(self):
        return (
            f"  [{self.name}]  {self.input_shape} → {self.output_shape}  "
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
# Main class
# ---------------------------------------------------------------------------

class FeatureAggregator:
    """Main class for feature extraction pipeline."""

    def __init__(self, fs: float = 1.0):
        self.fs = fs
        self.preprocessor = SignalPreprocessor()

        # Decomposers
        self.ft    = FourierTransform(fs=fs)
        self.stft  = ShortTimeFourierTransform(fs=fs)
        self.wavelet = WaveletDecomposer()
        self.emd   = EMD()
        self.vmd   = VMD()
        self.svmd  = SVMD()
        self.efd   = EFD()

        # Feature extractors
        self.time_features     = TimeDomainFeatures()
        self.freq_features     = FrequencyDomainFeatures()
        self.entropy_features  = EntropyFeatures()
        self.nonlinear_features= NonlinearFeatures()
        self.decomp_features   = DecompositionFeatures()

        # Populated after extract_all_features() when validate=True
        self.last_quality_reports: Dict[str, DecompositionReport] = {}
        # Populated when ≥2 decomposition methods are used
        self.last_consistency_report: Optional[str] = None

    # ------------------------------------------------------------------
    # Preprocessing (unchanged from original)
    # ------------------------------------------------------------------

    def preprocess(self, signal: np.ndarray,
                   denoise: bool = True,
                   normalize: bool = True,
                   detrend: bool = True,
                   **kwargs) -> np.ndarray:
        processed = signal.copy()
        if detrend:
            processed = self.preprocessor.detrend(
                processed, method=kwargs.get('detrend_method', 'linear'))
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
    # Main extraction entry point (extended with validate + consistency)
    # ------------------------------------------------------------------

    def extract_all_features(
        self,
        signal: np.ndarray,
        decomposition_methods: Optional[List[str]] = None,
        preprocess_signal: bool = True,
        extract_raw: bool = True,
        validate: bool = True,              # NEW — Review #1
        check_consistency: bool = True,     # NEW — Review #2
        **preprocess_kwargs,
    ) -> Dict[str, float]:
        """
        Extract all features from signal.

        New parameters
        --------------
        validate         : run DecompositionValidator after each decompose() call.
                           Quality metrics are added to the returned dict and a warning
                           is issued if any method fails its quality thresholds.
        check_consistency: when two or more decomposition methods are requested,
                           run CrossMethodChecker and store the report in
                           self.last_consistency_report.
        """
        if decomposition_methods is None:
            decomposition_methods = ['fourier', 'dwt']

        all_features: Dict[str, float] = {}
        self.last_quality_reports = {}

        sig = self.preprocess(signal, **preprocess_kwargs) if preprocess_signal else signal.copy()

        if extract_raw:
            raw_features = self._extract_raw_features(sig)
            all_features.update(self._add_prefix(raw_features, 'raw'))

        for method in decomposition_methods:
            decomp_features, quality_report = self._extract_decomposition_features(
                sig, method, validate=validate
            )
            all_features.update(self._add_prefix(decomp_features, method))

            if quality_report is not None:
                self.last_quality_reports[method] = quality_report
                # Embed quality metrics directly into the feature dict (Review #1)
                all_features.update(
                    self._add_prefix(quality_report.to_dict(), method)
                )
                if not quality_report.passed:
                    warnings.warn(
                        f"[SigFeatX] Decomposition quality check FAILED for '{method}':\n"
                        + quality_report.summary(),
                        RuntimeWarning,
                        stacklevel=2,
                    )

        # Cross-method consistency check (Review #2)
        component_methods = [
            m for m in decomposition_methods
            if m in ('emd', 'vmd', 'svmd', 'efd', 'dwt', 'wpd')
        ]
        if check_consistency and len(component_methods) >= 2:
            checker = CrossMethodChecker(tolerance=0.3)
            for m in component_methods:
                checker.add_from_aggregator_output(m, all_features, prefix=m)
            comparisons = checker.compare(
                features=["rms", "energy", "std", "kurtosis", "entropy"]
            )
            inconsistent = [c for c in comparisons if not c.consistent]
            if inconsistent:
                feat_names = ", ".join(c.feature for c in inconsistent)
                warnings.warn(
                    f"[SigFeatX] Cross-method consistency warning — features diverge "
                    f"across {component_methods}: {feat_names}. "
                    "Inspect self.last_consistency_report for details.",
                    UserWarning,
                    stacklevel=2,
                )
            self.last_consistency_report = checker.report(
                features=["rms", "energy", "std", "kurtosis", "entropy"]
            )

        return all_features

    # ------------------------------------------------------------------
    # Pluggable pipeline (Review #3)
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        signal: np.ndarray,
        preprocess_params: Optional[Dict] = None,
        decomposition_methods: Optional[List[str]] = None,
        validate: bool = True,
    ):
        """
        Explicit modular pipeline that records metadata for every stage.

        Returns
        -------
        (features_dict, PipelineMetadata)

        The PipelineMetadata records fs, window settings, and every stage's
        input/output shape + parameters so downstream code can detect silent
        mismatches (e.g. STFT window vs fs used in FrequencyDomainFeatures).

        Example
        -------
        agg = FeatureAggregator(fs=1000)
        features, meta = agg.run_pipeline(
            signal,
            preprocess_params={"denoise_method": "wavelet", "normalize_method": "zscore"},
            decomposition_methods=["emd", "vmd"],
            validate=True,
        )
        print(meta)
        """
        preprocess_params = preprocess_params or {}
        decomposition_methods = decomposition_methods or ["fourier", "dwt"]

        meta = PipelineMetadata(
            fs=self.fs,
            window=self.stft.window,
            nperseg=self.stft.nperseg,
            noverlap=self.stft.noverlap,
        )

        # ── Stage 1: Preprocess ──────────────────────────────────────────
        in_shape = signal.shape
        sig = self.preprocess(signal, **preprocess_params)
        meta.stages.append(StageRecord(
            name="preprocess",
            params={
                "detrend_method":    preprocess_params.get("detrend_method", "linear"),
                "denoise_method":    preprocess_params.get("denoise_method", "wavelet"),
                "normalize_method":  preprocess_params.get("normalize_method", "zscore"),
            },
            input_shape=in_shape,
            output_shape=sig.shape,
        ))

        # ── Stage 2: Raw features ────────────────────────────────────────
        raw = self._extract_raw_features(sig)
        meta.stages.append(StageRecord(
            name="raw_features",
            params={"fs": self.fs},
            input_shape=sig.shape,
            output_shape=(len(raw),),
        ))

        # ── Stage 3+: Per-method decompose → feature extract ─────────────
        all_features = self._add_prefix(raw, "raw")

        for method in decomposition_methods:
            # Detect fs mismatch between STFT and FrequencyDomainFeatures (Review #3)
            if method == "stft" and self.stft.fs != self.fs:
                warnings.warn(
                    f"[SigFeatX] Metadata mismatch: STFT is initialised with "
                    f"fs={self.stft.fs} but FeatureAggregator.fs={self.fs}. "
                    "Frequency features will be inconsistent.",
                    UserWarning,
                    stacklevel=2,
                )

            decomp_feats, quality_report = self._extract_decomposition_features(
                sig, method, validate=validate
            )
            all_features.update(self._add_prefix(decomp_feats, method))

            stage_params: Dict[str, Any] = {"method": method, "fs": self.fs}
            if method == "stft":
                stage_params.update({
                    "window": self.stft.window,
                    "nperseg": self.stft.nperseg,
                    "noverlap": self.stft.noverlap,
                })
            if method in ("emd",):
                stage_params["max_imf"] = self.emd.max_imf
            if method in ("vmd",):
                stage_params["K"] = self.vmd.K
                stage_params["alpha"] = self.vmd.alpha
            if method in ("svmd",):
                stage_params["K_max"] = self.svmd.K_max
                stage_params["alpha"] = self.svmd.alpha

            if quality_report is not None:
                all_features.update(
                    self._add_prefix(quality_report.to_dict(), method)
                )
                stage_params["decomp_snr_db"] = quality_report.snr_db
                stage_params["decomp_passed"]  = quality_report.passed

            meta.stages.append(StageRecord(
                name=f"decompose+features [{method}]",
                params=stage_params,
                input_shape=sig.shape,
                output_shape=(len(decomp_feats),),
            ))

        return all_features, meta

    # ------------------------------------------------------------------
    # Internal helpers (unchanged logic, now returns quality report too)
    # ------------------------------------------------------------------

    def _extract_raw_features(self, sig: np.ndarray) -> Dict[str, float]:
        features = {}
        features.update(self.time_features.extract(sig))
        features.update(self.freq_features.extract(sig, self.fs))
        features.update(self.entropy_features.extract(sig))
        features.update(self.nonlinear_features.extract(sig))

        # Review #2 — contract check on raw features
        violations = validate_feature_dict(features, method="raw")
        for v in violations:
            warnings.warn(str(v), RuntimeWarning, stacklevel=3)

        return features

    def _extract_decomposition_features(
        self,
        sig: np.ndarray,
        method: str,
        validate: bool = True,
    ):
        """
        Returns (feature_dict, DecompositionReport | None).
        DecompositionReport is None for methods that don't produce components
        (fourier, stft).
        """
        features: Dict[str, float] = {}
        quality_report: Optional[DecompositionReport] = None

        if method == 'fourier':
            freqs, magnitude = self.ft.transform(sig)
            features['dominant_freq']   = freqs[np.argmax(magnitude)]
            features['mean_magnitude']  = np.mean(magnitude)
            features['std_magnitude']   = np.std(magnitude)

        elif method == 'stft':
            f, t, Zxx = self.stft.transform(sig)
            features['mean_power'] = np.mean(Zxx ** 2)
            features['std_power']  = np.std(Zxx ** 2)
            features['max_power']  = np.max(Zxx ** 2)

        elif method == 'dwt':
            coeffs = self.wavelet.dwt(sig)
            features.update(self.decomp_features.extract_from_components(coeffs, 'dwt'))
            if validate:
                quality_report = DecompositionValidator.evaluate(
                    sig, coeffs, method="DWT"
                )

        elif method == 'wpd':
            wpd_dict = self.wavelet.wpd(sig)
            wpd_coeffs = list(wpd_dict.values())
            features.update(self.decomp_features.extract_from_components(wpd_coeffs, 'wpd'))
            if validate:
                quality_report = DecompositionValidator.evaluate(
                    sig, wpd_coeffs, method="WPD"
                )

        elif method == 'emd':
            imfs = self.emd.decompose(sig)
            features.update(self.decomp_features.extract_from_components(imfs, 'emd'))
            if validate:
                quality_report = DecompositionValidator.evaluate(
                    sig, imfs, method="EMD"
                )

        elif method == 'vmd':
            modes = self.vmd.decompose(sig)
            modes_list = [modes[i] for i in range(len(modes))]
            features.update(self.decomp_features.extract_from_components(modes_list, 'vmd'))
            if validate:
                quality_report = DecompositionValidator.evaluate(
                    sig, modes, method="VMD"
                )

        elif method == 'svmd':
            modes = self.svmd.decompose(sig)
            modes_list = [modes[i] for i in range(len(modes))]
            features.update(self.decomp_features.extract_from_components(modes_list, 'svmd'))
            if validate:
                quality_report = DecompositionValidator.evaluate(
                    sig, modes, method="SVMD"
                )

        elif method == 'efd':
            modes = self.efd.decompose(sig)
            modes_list = [modes[i] for i in range(len(modes))]
            features.update(self.decomp_features.extract_from_components(modes_list, 'efd'))
            if validate:
                quality_report = DecompositionValidator.evaluate(
                    sig, modes, method="EFD"
                )

        # Review #2 — contract check on decomposition features
        violations = validate_feature_dict(features, method=method)
        for v in violations:
            warnings.warn(str(v), RuntimeWarning, stacklevel=3)

        return features, quality_report

    @staticmethod
    def _add_prefix(features: Dict[str, float], prefix: str) -> Dict[str, float]:
        return {f'{prefix}_{k}': v for k, v in features.items()}

    def get_feature_names(self, decomposition_methods: Optional[List[str]] = None) -> List[str]:
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