import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from SigFeatX import FeatureAggregator
from SigFeatX.decompose import FourierTransform, ShortTimeFourierTransform
from SigFeatX.features.features import (
    EntropyFeatures,
    NonlinearFeatures,
    TimeDomainFeatures,
)
from SigFeatX.preprocess import SignalPreprocessor


def _make_sine(frequency_hz: float = 12.0, fs: int = 200, duration_s: float = 2.0) -> np.ndarray:
    n = int(fs * duration_s)
    t = np.arange(n) / fs
    return np.sin(2.0 * np.pi * frequency_hz * t)


def test_extract_all_features_rejects_empty_signal():
    aggregator = FeatureAggregator(fs=200)

    with pytest.raises(ValueError, match='at least 1 sample'):
        aggregator.extract_all_features(np.array([]), preprocess_signal=False)


def test_feature_extractors_reject_non_finite_signal():
    bad = np.array([0.0, np.nan, 1.0])

    with pytest.raises(ValueError, match='finite'):
        TimeDomainFeatures.extract(bad)


def test_short_signal_entropy_and_nonlinear_features_stay_finite():
    sig = np.array([0.2, -0.1, 0.4])

    entropy = EntropyFeatures.extract(sig)
    nonlinear = NonlinearFeatures.extract(sig)

    assert all(np.isfinite(value) for value in entropy.values())
    assert all(np.isfinite(value) for value in nonlinear.values())


def test_preprocessor_rejects_signal_too_short_for_zero_phase_filter():
    preprocessor = SignalPreprocessor()

    with pytest.raises(ValueError, match='zero-phase filtering'):
        preprocessor.denoise(np.ones(8), method='lowpass', cutoff=10.0, fs=100.0)


def test_extract_batch_accepts_single_signal_array():
    aggregator = FeatureAggregator(fs=200)
    sig = _make_sine()

    result = aggregator.extract_batch(
        sig,
        decomposition_methods=['fourier'],
        preprocess_signal=False,
        validate=False,
    )

    assert result.dataframe.shape[0] == 1
    assert np.isclose(result.dataframe.loc[0, 'fourier_dominant_freq'], 12.0, atol=1e-6)


def test_extract_batch_parallel_matches_sequential():
    aggregator = FeatureAggregator(fs=200)
    signals = [_make_sine(10.0), _make_sine(20.0), _make_sine(35.0)]

    sequential = aggregator.extract_batch(
        signals,
        decomposition_methods=['fourier'],
        preprocess_signal=False,
        validate=False,
        n_jobs=1,
    )
    parallel = aggregator.extract_batch(
        signals,
        decomposition_methods=['fourier'],
        preprocess_signal=False,
        validate=False,
        n_jobs=2,
    )

    assert list(sequential.dataframe.columns) == list(parallel.dataframe.columns)
    assert np.allclose(
        sequential.dataframe['fourier_dominant_freq'],
        parallel.dataframe['fourier_dominant_freq'],
    )
    assert np.allclose(sequential.dataframe['raw_rms'], parallel.dataframe['raw_rms'])


def test_extract_batch_warn_mode_records_bad_signal():
    aggregator = FeatureAggregator(fs=200)
    signals = [_make_sine(10.0), np.array([0.0, np.nan, 1.0]), _make_sine(20.0)]

    with pytest.warns(RuntimeWarning, match='signal 1 failed'):
        result = aggregator.extract_batch(
            signals,
            decomposition_methods=['fourier'],
            preprocess_signal=False,
            validate=False,
            on_error='warn',
        )

    assert result.n_success == 2
    assert result.n_failed == 1
    assert 1 in result.errors
    assert result.dataframe.loc[1].isna().all()


def test_extract_batch_raise_mode_propagates_bad_signal():
    aggregator = FeatureAggregator(fs=200)
    signals = [_make_sine(10.0), np.array([0.0, np.nan, 1.0])]

    with pytest.raises(ValueError, match='finite'):
        aggregator.extract_batch(
            signals,
            decomposition_methods=['fourier'],
            preprocess_signal=False,
            validate=False,
            on_error='raise',
        )


def test_extract_multichannel_parallel_matches_sequential():
    aggregator = FeatureAggregator(fs=200)
    signals = np.vstack([_make_sine(15.0), _make_sine(15.0)])

    sequential = aggregator.extract_multichannel(
        signals,
        channel_names=['left', 'right'],
        decomposition_methods=['fourier'],
        preprocess_signal=False,
        validate=False,
        n_jobs=1,
    )
    parallel = aggregator.extract_multichannel(
        signals,
        channel_names=['left', 'right'],
        decomposition_methods=['fourier'],
        preprocess_signal=False,
        validate=False,
        n_jobs=2,
    )

    assert sequential.keys() == parallel.keys()
    for key in sequential:
        assert np.isclose(sequential[key], parallel[key], atol=1e-9)


def test_extract_multichannel_rejects_duplicate_channel_names():
    aggregator = FeatureAggregator(fs=200)
    signals = np.vstack([_make_sine(12.0), _make_sine(18.0)])

    with pytest.raises(ValueError, match='unique'):
        aggregator.extract_multichannel(
            signals,
            channel_names=['dup', 'dup'],
            decomposition_methods=['fourier'],
            preprocess_signal=False,
            validate=False,
        )


def test_fourier_and_stft_validate_public_configuration():
    with pytest.raises(ValueError, match='positive finite'):
        FourierTransform(fs=0.0)

    with pytest.raises(ValueError, match='noverlap'):
        ShortTimeFourierTransform(fs=100.0, nperseg=16, noverlap=16)
