import numpy as np
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from SigFeatX import FeatureAggregator
from SigFeatX.aggregator import BatchResult


def _make_sine(frequency_hz: float, fs: int = 200, duration_s: float = 2.0) -> np.ndarray:
    n = int(fs * duration_s)
    t = np.arange(n) / fs
    return np.sin(2.0 * np.pi * frequency_hz * t)


def test_extract_batch_returns_dataframe_with_expected_features():
    """Batch extraction should preserve row order and produce feature columns."""
    aggregator = FeatureAggregator(fs=200)
    signals = [_make_sine(10.0), _make_sine(25.0)]

    result = aggregator.extract_batch(
        signals,
        decomposition_methods=['fourier'],
        preprocess_signal=False,
        validate=False,
        check_consistency=False,
    )

    assert result.n_success == 2
    assert result.n_failed == 0
    assert result.dataframe.shape[0] == 2
    assert 'raw_rms' in result.feature_names
    assert 'fourier_dominant_freq' in result.dataframe.columns
    assert np.isclose(result.dataframe.loc[0, 'fourier_dominant_freq'], 10.0, atol=1e-6)
    assert np.isclose(result.dataframe.loc[1, 'fourier_dominant_freq'], 25.0, atol=1e-6)


def test_extract_multichannel_adds_cross_channel_features():
    """Multi-channel extraction should add coherence, xcorr, and PLV summaries."""
    aggregator = FeatureAggregator(fs=200)
    signals = np.vstack([_make_sine(12.0), _make_sine(12.0)])

    features = aggregator.extract_multichannel(
        signals,
        channel_names=['left', 'right'],
        decomposition_methods=['fourier'],
        preprocess_signal=False,
        validate=False,
        include_cross=True,
    )

    assert 'left_raw_rms' in features
    assert 'right_fourier_dominant_freq' in features
    assert features['cross_left_right_coherence_mean'] > 0.9
    assert np.isclose(features['cross_left_right_xcorr_peak'], 1.0, atol=1e-3)
    assert features['cross_left_right_xcorr_lag'] == 0
    assert features['cross_left_right_plv'] > 0.99


def test_extract_windowed_tracks_frequency_changes():
    """Windowed extraction should return one row per window with metadata columns."""
    aggregator = FeatureAggregator(fs=200)
    sig = np.concatenate([_make_sine(10.0, duration_s=1.0), _make_sine(30.0, duration_s=1.0)])

    result = aggregator.extract_windowed(
        sig,
        window_size=200,
        step_size=200,
        decomposition_methods=['fourier'],
        preprocess_signal=False,
        validate=False,
        check_consistency=False,
    )

    assert result.n_success == 2
    assert 'window_idx' in result.dataframe.columns
    assert 'start_time_s' in result.dataframe.columns
    assert 'window_idx' not in result.feature_names
    assert list(result.dataframe['window_idx']) == [0, 1]
    assert np.allclose(result.dataframe['start_time_s'], [0.0, 1.0])
    assert np.isclose(result.dataframe.loc[0, 'fourier_dominant_freq'], 10.0, atol=1e-6)
    assert np.isclose(result.dataframe.loc[1, 'fourier_dominant_freq'], 30.0, atol=1e-6)


def test_extract_windowed_metadata_aligns_to_dataframe_index(monkeypatch):
    """Window metadata should align with dataframe index when rows are sparse/subset."""
    aggregator = FeatureAggregator(fs=200)
    sig = np.concatenate([_make_sine(10.0, duration_s=1.0), _make_sine(30.0, duration_s=1.0)])

    fake_df = pd.DataFrame({'raw_rms': [0.5]}, index=[1])
    fake_df.index.name = 'signal_idx'
    fake_result = BatchResult(
        dataframe=fake_df,
        errors={},
        n_success=1,
        n_failed=0,
        feature_names=['raw_rms'],
    )

    monkeypatch.setattr(
        FeatureAggregator,
        'extract_batch',
        lambda self, *args, **kwargs: fake_result,
    )

    result = aggregator.extract_windowed(
        sig,
        window_size=200,
        step_size=200,
        decomposition_methods=['fourier'],
        preprocess_signal=False,
        validate=False,
        check_consistency=False,
    )

    assert list(result.dataframe['window_idx']) == [1]
    assert list(result.dataframe['start_sample']) == [200]
    assert list(result.dataframe['end_sample']) == [400]
    assert np.allclose(result.dataframe['start_time_s'], [1.0])
    assert np.allclose(result.dataframe['end_time_s'], [2.0])
