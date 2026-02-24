"""
SigFeatX - decomposition_validator.py
======================================
Addresses Reddit review #1:
"add SNR/reconstruction error outputs so users know when a method worked vs just ran"

Drop this file into your SigFeatX/ package directory.

HOW IT INTEGRATES WITH YOUR EXISTING CODE
------------------------------------------
In aggregator.py, _extract_decomposition_features() already calls each decomposer
and gets back modes/IMFs but throws away quality information.

Replace those calls like this:

    # OLD
    imfs = self.emd.decompose(sig)

    # NEW
    from .decomposition_validator import DecompositionValidator
    imfs = self.emd.decompose(sig)
    report = DecompositionValidator.evaluate(sig, imfs, method="EMD")
    if not report.passed:
        import warnings
        warnings.warn(f"EMD decomposition quality check failed:\n{report.summary()}")

Or attach it to FeatureAggregator.extract_all_features() to get a report dict
alongside every feature dict.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DecompositionReport:
    method: str
    snr_db: float
    reconstruction_error: float   # normalised RMSE — 0.0 is perfect
    orthogonality_index: float    # Huang et al. index — 0.0 is perfect
    energy_preservation: float    # should be 1.0
    n_components: int
    passed: bool
    _warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        snr_str = "     inf (perfect)" if not np.isfinite(self.snr_db) else f"{self.snr_db:>8.2f} dB"
        lines = [
            f"[{self.method}] Decomposition Quality",
            f"  SNR                 : {snr_str}",
            f"  Reconstruction RMSE : {self.reconstruction_error:>10.6f}  (normalised)",
            f"  Energy preservation : {self.energy_preservation:>10.4f}  (ideal = 1.0)",
            f"  Orthogonality index : {self.orthogonality_index:>10.6f}  (ideal = 0.0)",
            f"  Components          : {self.n_components}",
            f"  Result              : {'PASSED ✓' if self.passed else 'FAILED ✗'}",
        ]
        for w in self._warnings:
            lines.append(f"  ⚠  {w}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """
        Returns flat dict so you can merge it into your feature dict if needed.
        inf SNR (perfect reconstruction) is stored as 999.0 so all values are
        finite and safe to pass to sklearn, pandas, etc.
        """
        snr_safe = 999.0 if not np.isfinite(self.snr_db) else self.snr_db
        return {
            f"decomp_quality_{self.method.lower()}_snr_db": snr_safe,
            f"decomp_quality_{self.method.lower()}_reconstruction_error": self.reconstruction_error,
            f"decomp_quality_{self.method.lower()}_energy_preservation": self.energy_preservation,
            f"decomp_quality_{self.method.lower()}_orthogonality_index": self.orthogonality_index,
        }


class DecompositionValidator:
    # Defaults — override per-call if your application has different tolerances
    SNR_THRESHOLD_DB: float = 20.0
    MAX_RECONSTRUCTION_ERROR: float = 0.01
    MAX_ENERGY_DEVIATION: float = 0.05

    @staticmethod
    def _snr(original: np.ndarray, reconstructed: np.ndarray) -> float:
        signal_power = np.mean(original ** 2)
        noise_power = np.mean((original - reconstructed) ** 2)
        if noise_power < 1e-30:
            return float("inf")
        return 10.0 * np.log10(signal_power / noise_power)

    @staticmethod
    def _normalised_rmse(original: np.ndarray, reconstructed: np.ndarray) -> float:
        rms = np.sqrt(np.mean(original ** 2))
        if rms < 1e-30:
            return 0.0
        return float(np.sqrt(np.mean((original - reconstructed) ** 2)) / rms)

    @staticmethod
    def _orthogonality_index(components: np.ndarray) -> float:
        """
        Huang et al. (1998) orthogonality index.
        Sum of pairwise inner products normalised by total component energy.
        """
        cross = 0.0
        n = len(components)
        for i in range(n):
            for j in range(i + 1, n):
                cross += float(np.abs(np.dot(components[i], components[j])))
        total_energy = float(np.sum(components ** 2))
        return cross / (total_energy + 1e-30)

    @staticmethod
    def _energy_preservation(original: np.ndarray, components: np.ndarray) -> float:
        orig_energy = float(np.sum(original ** 2))
        comp_energy = float(np.sum(components ** 2))
        if orig_energy < 1e-30:
            return 1.0
        return comp_energy / orig_energy

    @classmethod
    def evaluate(
        cls,
        original: np.ndarray,
        components,                     # list of arrays OR 2-D array (n_comp x N)
        method: str = "unknown",
        snr_threshold_db: Optional[float] = None,
        max_reconstruction_error: Optional[float] = None,
        max_energy_deviation: Optional[float] = None,
    ) -> DecompositionReport:
        """
        Evaluate quality of a decomposition.

        Parameters
        ----------
        original   : the signal passed into .decompose()
        components : what .decompose() returned — works with EMD list-of-arrays,
                     VMD/EFD/SVMD 2-D numpy arrays, and wavelet coefficient lists
        method     : label used in the report ("EMD", "VMD", "DWT", etc.)
        """
        snr_thresh = snr_threshold_db   or cls.SNR_THRESHOLD_DB
        max_err    = max_reconstruction_error or cls.MAX_RECONSTRUCTION_ERROR
        max_ediff  = max_energy_deviation     or cls.MAX_ENERGY_DEVIATION

        original   = np.asarray(original,    dtype=float).ravel()

        # Normalise components to 2-D array regardless of input type
        # Handles: list of 1-D arrays (EMD), 2-D array (VMD/EFD/SVMD),
        # list of varying-length wavelet coeffs (DWT — pad to match original length)
        comp_list = list(components)
        max_len   = len(original)
        padded    = []
        for c in comp_list:
            c = np.asarray(c, dtype=float).ravel()
            if len(c) < max_len:
                c = np.pad(c, (0, max_len - len(c)), mode="constant")
            elif len(c) > max_len:
                c = c[:max_len]
            padded.append(c)
        components_2d = np.array(padded)   # shape (n_comp, N)

        reconstructed = np.sum(components_2d, axis=0)

        snr_val  = cls._snr(original, reconstructed)
        rec_err  = cls._normalised_rmse(original, reconstructed)
        orth_idx = cls._orthogonality_index(components_2d)
        ep       = cls._energy_preservation(original, components_2d)

        warns  = []
        passed = True

        if snr_val < snr_thresh:
            passed = False
            warns.append(
                f"SNR {snr_val:.1f} dB < threshold {snr_thresh} dB. "
                "Components do not reconstruct the original accurately."
            )
        if rec_err > max_err:
            passed = False
            warns.append(
                f"Reconstruction RMSE {rec_err:.6f} > threshold {max_err}."
            )
        if abs(ep - 1.0) > max_ediff:
            warns.append(
                f"Energy preservation {ep:.4f} deviates from 1.0 by more than {max_ediff}. "
                "Energy is not conserved across components."
            )
        if orth_idx > 0.2:
            warns.append(
                f"Orthogonality index {orth_idx:.4f} is high — components may be correlated."
            )

        return DecompositionReport(
            method=method,
            snr_db=round(snr_val, 4),
            reconstruction_error=round(rec_err, 8),
            orthogonality_index=round(orth_idx, 6),
            energy_preservation=round(ep, 6),
            n_components=len(comp_list),
            passed=passed,
            _warnings=warns,
        )