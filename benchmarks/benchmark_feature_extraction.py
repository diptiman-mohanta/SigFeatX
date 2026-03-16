#!/usr/bin/env python3
"""Simple local benchmark for common SigFeatX extraction workflows."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import tracemalloc
import warnings
from dataclasses import asdict, dataclass
from typing import Callable, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from SigFeatX import FeatureAggregator


@dataclass
class BenchmarkResult:
    name: str
    median_ms: float
    mean_ms: float
    best_ms: float
    stdev_ms: float
    peak_mem_mib: float
    notes: str = ""


def _make_signal(n_samples: int, fs: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fs
    signal = (
        0.9 * np.sin(2.0 * np.pi * 8.0 * t)
        + 0.45 * np.sin(2.0 * np.pi * 24.0 * t)
        + 0.2 * np.sin(2.0 * np.pi * 55.0 * t)
    )
    signal += 0.03 * rng.standard_normal(n_samples)
    signal += 0.1 * np.sin(2.0 * np.pi * 0.4 * t)
    return signal


def _time_call(fn: Callable[[], object], repeats: int, warmup: int) -> tuple[list[float], list[float], List[str]]:
    timings_s: list[float] = []
    peak_mem_mib: list[float] = []
    notes: List[str] = []

    for iteration in range(warmup + repeats):
        tracemalloc.start()
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fn()
        elapsed_s = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if iteration >= warmup:
            timings_s.append(elapsed_s)
            peak_mem_mib.append(peak / (1024.0 ** 2))
            for warning_record in caught:
                message = str(warning_record.message)
                if message not in notes:
                    notes.append(message)

    return timings_s, peak_mem_mib, notes


def _benchmark(name: str, fn: Callable[[], object], repeats: int, warmup: int) -> BenchmarkResult:
    timings_s, peak_mem_mib, notes = _time_call(fn, repeats=repeats, warmup=warmup)
    timings_ms = [value * 1000.0 for value in timings_s]
    return BenchmarkResult(
        name=name,
        median_ms=statistics.median(timings_ms),
        mean_ms=statistics.mean(timings_ms),
        best_ms=min(timings_ms),
        stdev_ms=statistics.pstdev(timings_ms),
        peak_mem_mib=max(peak_mem_mib),
        notes=" | ".join(notes),
    )


def _format_table(results: List[BenchmarkResult]) -> str:
    headers = ("Benchmark", "Median ms", "Mean ms", "Best ms", "Stdev ms", "Peak MiB", "Notes")
    rows = [
        (
            result.name,
            f"{result.median_ms:9.2f}",
            f"{result.mean_ms:8.2f}",
            f"{result.best_ms:8.2f}",
            f"{result.stdev_ms:8.2f}",
            f"{result.peak_mem_mib:8.2f}",
            result.notes or "-",
        )
        for result in results
    ]

    col_widths = [
        max(len(headers[idx]), *(len(str(row[idx])) for row in rows))
        for idx in range(len(headers))
    ]

    def _fmt(row) -> str:
        return "  ".join(str(value).ljust(col_widths[idx]) for idx, value in enumerate(row))

    lines = [_fmt(headers), _fmt(tuple("-" * width for width in col_widths))]
    lines.extend(_fmt(row) for row in rows)
    return "\n".join(lines)


def _parse_n_jobs_list(raw: str) -> List[int]:
    values = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    if not values:
        raise ValueError("--n-jobs-list must contain at least one value.")

    parsed: List[int] = []
    for value in values:
        n_jobs = int(value)
        if n_jobs != -1 and n_jobs < 1:
            raise ValueError(f"n_jobs must be -1 or >=1; got {n_jobs}.")
        if n_jobs not in parsed:
            parsed.append(n_jobs)
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1500, help="Samples in the base benchmark signal.")
    parser.add_argument("--fs", type=float, default=1000.0, help="Sampling rate in Hz.")
    parser.add_argument("--batch-size", type=int, default=8, help="Signals per batch benchmark.")
    parser.add_argument("--window-size", type=int, default=512, help="Window size for the windowed benchmark.")
    parser.add_argument("--step-size", type=int, default=256, help="Step size for the windowed benchmark.")
    parser.add_argument("--repeats", type=int, default=5, help="Measured repetitions per benchmark.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations per benchmark.")
    parser.add_argument("--include-slow", action="store_true", help="Include a slower EMD benchmark.")
    parser.add_argument(
        "--n-jobs-list",
        type=str,
        default="1",
        help="Comma-separated n_jobs values for batch benchmark sweep (e.g. '1,2,-1').",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON after the table.")
    args = parser.parse_args()

    n_jobs_list = _parse_n_jobs_list(args.n_jobs_list)

    signal = _make_signal(args.samples, args.fs, seed=7)
    long_signal = _make_signal(args.samples * 2, args.fs, seed=11)
    batch = [_make_signal(args.samples, args.fs, seed=100 + idx) for idx in range(args.batch_size)]

    aggregator = FeatureAggregator(fs=args.fs)

    workloads: list[tuple[str, Callable[[], object]]] = [
        (
            "raw_only",
            lambda: aggregator.extract_all_features(
                signal,
                decomposition_methods=[],
                preprocess_signal=False,
                validate=False,
                check_consistency=False,
            ),
        ),
        (
            "raw_plus_fourier_dwt",
            lambda: aggregator.extract_all_features(
                signal,
                decomposition_methods=["fourier", "dwt"],
                preprocess_signal=False,
                validate=False,
                check_consistency=False,
            ),
        ),
        (
            "windowed_fourier",
            lambda: aggregator.extract_windowed(
                long_signal,
                window_size=args.window_size,
                step_size=args.step_size,
                decomposition_methods=["fourier"],
                preprocess_signal=False,
                validate=False,
                check_consistency=False,
            ),
        ),
        (
            "batch_njobs_1_fourier",
            lambda: aggregator.extract_batch(
                batch,
                decomposition_methods=["fourier"],
                preprocess_signal=False,
                validate=False,
                check_consistency=False,
                n_jobs=1,
            ),
        ),
    ]

    # Add user-requested n_jobs sweep workloads, skipping the baseline already present.
    for n_jobs in n_jobs_list:
        if n_jobs == 1:
            continue
        workloads.append(
            (
                f"batch_njobs_{n_jobs}_fourier",
                lambda n_jobs=n_jobs: aggregator.extract_batch(
                    batch,
                    decomposition_methods=["fourier"],
                    preprocess_signal=False,
                    validate=False,
                    check_consistency=False,
                    n_jobs=n_jobs,
                ),
            )
        )

    if args.include_slow:
        workloads.append(
            (
                "raw_plus_fourier_dwt_emd",
                lambda: aggregator.extract_all_features(
                    signal,
                    decomposition_methods=["fourier", "dwt", "emd"],
                    preprocess_signal=False,
                    validate=False,
                    check_consistency=False,
                ),
            )
        )

    results = [
        _benchmark(name, fn, repeats=args.repeats, warmup=args.warmup)
        for name, fn in workloads
    ]

    print(_format_table(results))

    batch_results = {
        result.name: result
        for result in results
        if result.name.startswith("batch_njobs_")
    }
    baseline = batch_results.get("batch_njobs_1_fourier")
    if baseline is not None:
        print("\nBatch Speedup Summary (baseline: n_jobs=1)")
        for name, result in batch_results.items():
            speedup = baseline.median_ms / result.median_ms if result.median_ms > 0 else float("inf")
            n_jobs_label = name.replace("batch_njobs_", "").replace("_fourier", "")
            print(f"  n_jobs={n_jobs_label:>2} | median={result.median_ms/1000.0:.4f}s | speedup={speedup:.2f}x")

    if args.json:
        print()
        print(json.dumps([asdict(result) for result in results], indent=2))


if __name__ == "__main__":
    main()
