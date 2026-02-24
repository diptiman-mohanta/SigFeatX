import warnings
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1.  Feature contracts
#     Derived from the actual formulas in your features/features.py
# ---------------------------------------------------------------------------

@dataclass
class ContractViolation:
    feature: str
    method: str
    value: float
    reason: str

    def __str__(self):
        return f"[{self.method}] '{self.feature}' = {self.value:.6g} — {self.reason}"


# Each entry: (min, max, warn_only)
# min/max of None means no bound in that direction.
# warn_only=True  → issue a warning but don't mark as a hard violation
# warn_only=False → hard violation (something is definitely wrong)
_CONTRACTS: Dict[str, Tuple] = {
    # time domain — your TimeDomainFeatures.extract()
    "rms":                  (0.0,  None,  False),
    "energy":               (0.0,  None,  False),
    "power":                (0.0,  None,  False),
    "variance":             (0.0,  None,  False),
    "std":                  (0.0,  None,  False),
    "mean_absolute":        (0.0,  None,  False),
    "sum_absolute":         (0.0,  None,  False),
    "zero_crossings":       (0.0,  None,  False),
    "zero_crossing_rate":   (0.0,  1.0,   False),
    "crest_factor":         (0.0,  None,  True),   # very high values possible but suspicious
    "shape_factor":         (0.0,  None,  True),
    "impulse_factor":       (0.0,  None,  True),
    "iqr":                  (0.0,  None,  False),
    "kurtosis":             (-10.0, 100.0, True),  # scipy kurtosis is excess (Gaussian=0)
    "skewness":             (-20.0, 20.0,  True),

    # frequency domain — your FrequencyDomainFeatures.extract()
    "spectral_centroid":    (0.0,  None,  False),  # must be positive Hz
    "spectral_spread":      (0.0,  None,  False),
    "spectral_bandwidth":   (0.0,  None,  False),
    "spectral_rolloff":     (0.0,  None,  False),
    "dominant_frequency":   (0.0,  None,  False),
    "max_magnitude":        (0.0,  None,  False),
    "spectral_flatness":    (0.0,  1.0,   False),  # geometric/arithmetic mean ≤ 1 by AM-GM
    # spectral_entropy is in bits (log2), not normalised — no fixed upper bound
    "spectral_entropy":     (0.0,  None,  False),

    # entropy — your EntropyFeatures.extract()
    "sample_entropy":       (0.0,  None,  False),
    "approximate_entropy":  (None, None,  True),   # can be negative with this estimator
    "permutation_entropy":  (0.0,  None,  False),
    "shannon_entropy":      (0.0,  None,  False),

    # nonlinear — your NonlinearFeatures.extract()
    "hjorth_activity":      (0.0,  None,  False),
    "hjorth_mobility":      (0.0,  None,  False),
    "hjorth_complexity":    (0.0,  None,  True),
    "hurst_exponent":       (0.0,  2.0,   True),
    "dfa_alpha":            (0.0,  3.0,   True),
}


def validate_feature_dict(
    features: Dict[str, float],
    method: str = "unknown",
    raise_on_violation: bool = False,
) -> List[ContractViolation]:
    """
    Check every feature value in `features` against known mathematical contracts.

    Parameters
    ----------
    features          : dict returned by any of your *Features.extract() methods
    method            : human label used in violation messages
    raise_on_violation: if True, raises ValueError on the first hard violation

    Returns
    -------
    List of ContractViolation (empty list = all clear)
    """
    violations = []

    for name, value in features.items():
        # Strip any prefix added by _add_prefix(), e.g. "raw_rms" -> "rms"
        bare_name = name.split("_", 1)[-1] if "_" in name else name
        contract = _CONTRACTS.get(bare_name) or _CONTRACTS.get(name)
        if contract is None:
            continue

        lo, hi, warn_only = contract

        if lo is not None and value < lo:
            v = ContractViolation(
                feature=name, method=method, value=value,
                reason=f"Expected ≥ {lo}. Possible NaN propagation or implementation bug."
            )
            violations.append(v)
            if not warn_only:
                if raise_on_violation:
                    raise ValueError(str(v))
                warnings.warn(str(v), RuntimeWarning, stacklevel=2)

        if hi is not None and value > hi:
            v = ContractViolation(
                feature=name, method=method, value=value,
                reason=f"Expected ≤ {hi}. Check for division-by-zero or overflow."
            )
            violations.append(v)
            if not warn_only:
                if raise_on_violation:
                    raise ValueError(str(v))
                warnings.warn(str(v), RuntimeWarning, stacklevel=2)

    return violations


# ---------------------------------------------------------------------------
# 2.  Cross-method consistency checker
# ---------------------------------------------------------------------------

@dataclass
class MethodComparison:
    feature: str
    values: Dict[str, float]      # method -> value
    relative_spread: float        # std(values) / |mean(values)|  — 0 = identical
    consistent: bool
    note: str

    def __str__(self):
        vals = "  ".join(f"{m}={v:.4g}" for m, v in self.values.items())
        status = "✓" if self.consistent else "✗"
        return (
            f"{status} {self.feature:<35s} spread={self.relative_spread:.3f}  |  "
            f"{vals}\n    {self.note}"
        )


class CrossMethodChecker:
    """
    Compare the same statistical features extracted via different decomposition
    methods to detect cases where methods silently produce incompatible numbers.

    Example
    -------
    checker = CrossMethodChecker(tolerance=0.3)
    checker.add_from_aggregator_output("EMD", all_features_dict, prefix="emd")
    checker.add_from_aggregator_output("VMD", all_features_dict, prefix="vmd")
    for line in checker.compare(["rms", "kurtosis", "energy"]):
        print(line)
    """

    def __init__(self, tolerance: float = 0.3):
        """
        Parameters
        ----------
        tolerance : relative spread threshold above which methods are flagged
                    as inconsistent. Default 0.3 = 30% spread.
        """
        self.tolerance = tolerance
        # method_tag -> {bare_feature_name: value}
        self._registry: Dict[str, Dict[str, float]] = {}

    def add_from_aggregator_output(
        self,
        method_tag: str,
        all_features: Dict[str, float],
        prefix: str,
    ):
        """
        Register features from FeatureAggregator.extract_all_features().

        Parameters
        ----------
        method_tag   : label, e.g. "EMD", "VMD"
        all_features : the full dict returned by extract_all_features()
        prefix       : the decomposition prefix used, e.g. "emd", "vmd"
                       Used to pull out only the relevant features.
        """
        # Your _add_prefix() uses f"{prefix}_{k}", e.g. "emd_comp_0_rms"
        # We strip the top-level method prefix so we can align across methods
        subset = {}
        for k, v in all_features.items():
            if k.startswith(f"{prefix}_"):
                bare = k[len(prefix) + 1:]    # "comp_0_rms", etc.
                subset[bare] = v
        self._registry[method_tag] = subset

    def add_raw(self, method_tag: str, features: Dict[str, float]):
        """Register a flat feature dict directly (no prefix stripping)."""
        self._registry[method_tag] = features

    def compare(
        self,
        features: Optional[List[str]] = None,
    ) -> List[MethodComparison]:
        """
        Compare registered methods on the requested features.

        Parameters
        ----------
        features : list of bare feature names to compare, e.g. ["rms", "kurtosis"].
                   If None, compares all features that exist in every registered method.

        Returns
        -------
        List of MethodComparison objects (one per feature)
        """
        if len(self._registry) < 2:
            raise ValueError("Register at least two methods before calling compare().")

        methods = list(self._registry.keys())

        # Determine features to compare
        if features is None:
            # Intersection of keys across all methods
            key_sets = [set(v.keys()) for v in self._registry.values()]
            features = list(key_sets[0].intersection(*key_sets[1:]))

        results = []
        for feat in sorted(features):
            vals = {}
            for m in methods:
                # Support both exact match and suffix match (e.g. "rms" matches "comp_0_rms")
                exact = self._registry[m].get(feat)
                if exact is not None:
                    vals[m] = exact
                else:
                    # Find first key that ends with the feature name
                    for k, v in self._registry[m].items():
                        if k == feat or k.endswith(f"_{feat}"):
                            vals[m] = v
                            break

            if len(vals) < 2:
                continue

            value_arr = np.array(list(vals.values()))
            grand_mean = np.mean(np.abs(value_arr))
            spread = float(np.std(value_arr) / (grand_mean + 1e-30))
            consistent = spread <= self.tolerance

            if consistent:
                note = f"Methods agree within {spread*100:.1f}%."
            else:
                note = (
                    f"Methods diverge by {spread*100:.1f}% (threshold {self.tolerance*100:.0f}%). "
                    "Check that all methods process the same signal length and that "
                    "component energy is comparable before feature extraction."
                )

            results.append(MethodComparison(
                feature=feat,
                values=vals,
                relative_spread=round(spread, 4),
                consistent=consistent,
                note=note,
            ))

        return results

    def report(self, features: Optional[List[str]] = None) -> str:
        comparisons = self.compare(features)
        n_fail = sum(1 for c in comparisons if not c.consistent)
        header = (
            f"Cross-Method Feature Consistency Report\n"
            f"Methods: {', '.join(self._registry.keys())}\n"
            f"{'='*72}\n"
            f"Checked {len(comparisons)} features — "
            f"{len(comparisons)-n_fail} consistent, {n_fail} inconsistent\n"
            f"{'='*72}"
        )
        body = "\n".join(str(c) for c in comparisons)
        return f"{header}\n{body}"