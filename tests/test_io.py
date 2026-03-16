import numpy as np
import os
import pytest
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from SigFeatX.io import SignalIO


def test_signal_io_auto_round_trip(tmp_path):
    """Signal save/load should infer format from the path extension."""
    signal = np.linspace(-1.0, 1.0, 32)
    path = tmp_path / 'signal.npy'

    SignalIO.save_signal(signal, str(path))
    loaded = SignalIO.load_signal(str(path))

    assert np.array_equal(loaded, signal)


def test_feature_io_csv_round_trip(tmp_path):
    """CSV feature exports should load back into the original mapping."""
    features = {
        'mean': np.float64(0.5),
        'std': np.float64(1.25),
    }
    path = tmp_path / 'features.csv'

    SignalIO.save_features(features, str(path))
    loaded = SignalIO.load_features(str(path))

    assert loaded == {'mean': 0.5, 'std': 1.25}


def test_feature_io_rejects_unsupported_format(tmp_path):
    """Unsupported feature formats should raise a clear error."""
    path = tmp_path / 'features.yaml'

    with pytest.raises(ValueError, match='Unsupported format'):
        SignalIO.save_features({'mean': 1.0}, str(path), file_format='yaml')
