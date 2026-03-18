"""Internal validation helpers for public SigFeatX APIs."""

from typing import Iterable, List

import numpy as np


def validate_signal_1d(
    signal,
    *,
    name: str = "signal",
    min_length: int = 1,
    require_finite: bool = True,
) -> np.ndarray:
    """Coerce a signal to a finite 1D float array with a minimum length."""
    arr = np.asarray(signal, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array; got shape {arr.shape}.")
    if arr.size < min_length:
        raise ValueError(
            f"{name} must contain at least {min_length} sample(s); got {arr.size}."
        )
    if require_finite and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def validate_signal_batch(
    signals,
    *,
    name: str = "signals",
    require_finite: bool = True,
) -> List[np.ndarray]:
    """Normalize a batch input to a list of 1D arrays."""
    if isinstance(signals, np.ndarray):
        if signals.ndim == 1:
            return [validate_signal_1d(signals, name=f"{name}[0]", require_finite=require_finite)]
        if signals.ndim == 2:
            return [
                validate_signal_1d(signals[i], name=f"{name}[{i}]", require_finite=require_finite)
                for i in range(signals.shape[0])
            ]
        raise ValueError(
            f"{name} must be a list of 1D arrays, a 1D array, or a 2D array; "
            f"got shape {signals.shape}."
        )

    try:
        seq = list(signals)
    except TypeError as exc:
        raise ValueError(
            f"{name} must be an iterable of 1D signals or a numpy array."
        ) from exc

    return [
        validate_signal_1d(sig, name=f"{name}[{i}]", require_finite=require_finite)
        for i, sig in enumerate(seq)
    ]


def validate_signal_matrix(
    signals_2d,
    *,
    name: str = "signals_2d",
    require_finite: bool = True,
) -> np.ndarray:
    """Coerce a multichannel signal matrix to a finite 2D float array."""
    arr = np.asarray(signals_2d, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got shape {arr.shape}.")
    if arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(
            f"{name} must have at least 1 channel and 1 sample; got shape {arr.shape}."
        )
    if require_finite and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def validate_sampling_rate(fs: float, *, name: str = "fs") -> float:
    """Validate a positive finite sampling rate."""
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError(f"{name} must be a positive finite number; got {fs}.")
    return fs


def validate_n_jobs(n_jobs: int) -> int:
    """Validate the n_jobs convention used across the package."""
    n_jobs = int(n_jobs)
    if n_jobs == -1 or n_jobs >= 1:
        return n_jobs
    raise ValueError(f"n_jobs must be -1 or a positive integer; got {n_jobs}.")


def validate_unique_names(names: Iterable[str], *, name: str = "names") -> List[str]:
    """Validate that user-provided names are unique and non-empty."""
    out = [str(item) for item in names]
    if any(item == "" for item in out):
        raise ValueError(f"{name} must not contain empty strings.")
    if len(set(out)) != len(out):
        raise ValueError(f"{name} must contain unique values.")
    return out
