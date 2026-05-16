"""
tests/test_phase1.py
=====================
Tests for the 0.2.0 phase-1 upgrades:
  - Pipeline (fluent builder)
  - SigFeatXTransformer (sklearn)
  - BatchIO (parquet/hdf5/feather)
  - ProgressBar (tqdm helper)
"""

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from SigFeatX import FeatureAggregator, Pipeline, BatchIO
from SigFeatX._progress import progress_iter, ProgressBar


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fs():
    return 500


@pytest.fixture
def signals(fs):
    rng = np.random.default_rng(0)
    t = np.arange(fs * 2) / fs
    return [
        np.sin(2 * np.pi * f * t) + 0.05 * rng.standard_normal(len(t))
        for f in (10.0, 20.0, 40.0, 60.0)
    ]


@pytest.fixture
def single_signal(fs):
    t = np.arange(fs * 2) / fs
    return np.sin(2 * np.pi * 15 * t)


# ===========================================================================
# Pipeline
# ===========================================================================

class TestPipeline:

    def test_repr(self, fs):
        p = Pipeline(fs=fs).detrend().denoise().normalize().decompose(['fourier'])
        text = repr(p)
        assert 'Pipeline' in text and 'detrend' in text and 'fourier' in text

    def test_extract_single(self, fs, single_signal):
        feats = (
            Pipeline(fs=fs)
            .decompose(['fourier'])
            .extract(single_signal)
        )
        assert isinstance(feats, dict) and len(feats) > 0
        assert 'fourier_dominant_freq' in feats

    def test_extract_batch(self, fs, signals):
        result = (
            Pipeline(fs=fs)
            .decompose('fourier')
            .extract_batch(signals)
        )
        assert result.n_success == len(signals)
        assert result.dataframe.shape[0] == len(signals)

    def test_extract_windowed(self, fs, single_signal):
        result = (
            Pipeline(fs=fs)
            .decompose('fourier')
            .extract_windowed(single_signal, window_size=200, step_size=100)
        )
        assert 'window_idx' in result.dataframe.columns
        assert result.dataframe.shape[0] >= 5

    def test_preprocess_enabled_when_step_added(self, fs, single_signal):
        p = Pipeline(fs=fs).detrend('linear').decompose(['fourier'])
        kw = p._preprocess_kwargs()
        assert kw['detrend'] is True and kw['detrend_method'] == 'linear'
        feats = p.extract(single_signal)
        assert len(feats) > 0

    def test_no_preprocess_explicit(self, fs):
        p = Pipeline(fs=fs).detrend().no_preprocess()
        kw = p._preprocess_kwargs()
        assert kw == {}

    def test_on_error_validation(self, fs):
        p = Pipeline(fs=fs)
        with pytest.raises(ValueError, match="must be 'warn' or 'raise'"):
            p.on_error('bogus')

    def test_clone_independent(self, fs):
        p1 = Pipeline(fs=fs).decompose(['fourier'])
        p2 = p1.clone().decompose(['dwt'])
        assert p1._decomposition_methods == ['fourier']
        assert p2._decomposition_methods == ['dwt']

    def test_to_dict_round_trip(self, fs):
        p = Pipeline(fs=fs).detrend('linear').denoise('wavelet').decompose('emd')
        cfg = p.to_dict()
        assert cfg['fs'] == fs
        assert cfg['detrend']['method'] == 'linear'
        assert cfg['decomposition_methods'] == ['emd']

    def test_pipeline_matches_aggregator(self, fs, single_signal):
        """Pipeline must produce same features as direct FeatureAggregator call."""
        pipe_feats = (
            Pipeline(fs=fs)
            .decompose(['fourier'])
            .include_raw()
            .extract(single_signal)
        )
        agg = FeatureAggregator(fs=fs)
        direct = agg.extract_all_features(
            single_signal,
            decomposition_methods=['fourier'],
            preprocess_signal=False,
            validate=False,
            check_consistency=False,
        )
        assert set(pipe_feats.keys()) == set(direct.keys())
        for k in pipe_feats:
            assert np.isclose(pipe_feats[k], direct[k], equal_nan=True)


# ===========================================================================
# sklearn wrapper
# ===========================================================================

sklearn = pytest.importorskip('sklearn')   # skip whole class if not installed


class TestSklearnWrapper:

    def test_import(self):
        from SigFeatX import SigFeatXTransformer
        assert SigFeatXTransformer is not None

    def test_fit_transform_2d(self, fs, signals):
        from SigFeatX import SigFeatXTransformer
        X = np.array([s[:800] for s in signals])
        tr = SigFeatXTransformer(fs=fs, decomposition_methods=['fourier'],
                                  preprocess_signal=False)
        Xt = tr.fit_transform(X)
        assert Xt.shape[0] == len(signals)
        assert Xt.shape[1] > 0
        assert len(tr.feature_names_) == Xt.shape[1]

    def test_get_feature_names_out(self, fs, signals):
        from SigFeatX import SigFeatXTransformer
        X = np.array([s[:800] for s in signals])
        tr = SigFeatXTransformer(fs=fs, decomposition_methods=['fourier'],
                                  preprocess_signal=False)
        tr.fit_transform(X)
        names = tr.get_feature_names_out()
        assert len(names) == tr.n_features_out_
        assert all(isinstance(n, str) for n in names)

    def test_list_input(self, fs, signals):
        from SigFeatX import SigFeatXTransformer
        tr = SigFeatXTransformer(fs=fs, decomposition_methods=['fourier'],
                                  preprocess_signal=False)
        Xt = tr.fit_transform(signals)
        assert Xt.shape[0] == len(signals)

    def test_3d_multichannel(self, fs, signals):
        from SigFeatX import SigFeatXTransformer
        # 4 signals, 2 channels, 800 samples each
        X = np.stack([np.stack([s[:800], s[:800] + 0.1]) for s in signals])
        assert X.shape == (4, 2, 800)
        tr = SigFeatXTransformer(
            fs=fs,
            decomposition_methods=['fourier'],
            preprocess_signal=False,
            channel_names=['ch0', 'ch1'],
            include_cross_channel=True,
        )
        Xt = tr.fit_transform(X)
        assert Xt.shape == (4, tr.n_features_out_)
        # Cross-channel features should appear
        assert any('cross_' in n for n in tr.feature_names_)

    def test_in_sklearn_pipeline(self, fs, signals):
        from SigFeatX import SigFeatXTransformer
        from sklearn.pipeline import Pipeline as SkPipeline
        from sklearn.preprocessing import StandardScaler

        X = np.array([s[:800] for s in signals])
        sk = SkPipeline([
            ('feat', SigFeatXTransformer(
                fs=fs, decomposition_methods=['fourier'], preprocess_signal=False)),
            ('scaler', StandardScaler()),
        ])
        Xt = sk.fit_transform(X)
        assert Xt.shape[0] == len(signals)
        assert np.isfinite(Xt).any()


# ===========================================================================
# BatchIO
# ===========================================================================

class TestBatchIO:

    def _make_df(self, fs, signals):
        agg = FeatureAggregator(fs=fs)
        return agg.extract_batch(
            signals,
            decomposition_methods=['fourier'],
            preprocess_signal=False,
            validate=False,
        ).dataframe

    def test_parquet_round_trip(self, tmp_path, fs, signals):
        try:
            import pyarrow  # noqa
        except ImportError:
            pytest.skip("pyarrow not installed")

        df = self._make_df(fs, signals)
        path = tmp_path / 'features.parquet'
        BatchIO.save_parquet(df, str(path), metadata={'fs': str(fs)})
        df_back = BatchIO.load_parquet(str(path))
        assert df_back.shape == df.shape
        md = BatchIO.load_parquet_metadata(str(path))
        assert md.get('fs') == str(fs)

    def test_hdf5_round_trip(self, tmp_path, fs, signals):
        try:
            import tables  # noqa
        except ImportError:
            pytest.skip("pytables not installed")

        df = self._make_df(fs, signals)
        path = tmp_path / 'features.h5'
        BatchIO.save_hdf5(df, str(path), key='features')
        df_back = BatchIO.load_hdf5(str(path), key='features')
        assert df_back.shape == df.shape

    def test_feather_round_trip(self, tmp_path, fs, signals):
        try:
            import pyarrow  # noqa
        except ImportError:
            pytest.skip("pyarrow not installed")

        df = self._make_df(fs, signals)
        path = tmp_path / 'features.feather'
        BatchIO.save_feather(df, str(path))
        df_back = BatchIO.load_feather(str(path))
        assert df_back.shape[0] == df.shape[0]

    def test_auto_dispatch_unknown_extension(self, tmp_path, fs, signals):
        df = self._make_df(fs, signals)
        with pytest.raises(ValueError, match="Unsupported extension"):
            BatchIO.save(df, str(tmp_path / 'data.xyz'))

    def test_csv_round_trip(self, tmp_path, fs, signals):
        df = self._make_df(fs, signals)
        path = tmp_path / 'features.csv'
        BatchIO.save(df, str(path))
        df_back = BatchIO.load(str(path))
        assert df_back.shape == df.shape

    def test_append_parquet_partitioned(self, tmp_path, fs, signals):
        try:
            import pyarrow  # noqa
        except ImportError:
            pytest.skip("pyarrow not installed")

        df = self._make_df(fs, signals)
        out_dir = str(tmp_path / 'dataset')
        p1 = BatchIO.append_parquet(df, out_dir)
        p2 = BatchIO.append_parquet(df, out_dir)
        assert os.path.exists(p1) and os.path.exists(p2)
        assert p1 != p2

        import pandas as pd
        full = pd.read_parquet(out_dir)
        assert full.shape[0] == 2 * df.shape[0]


# ===========================================================================
# Progress
# ===========================================================================

class TestProgress:

    def test_progress_iter_disabled_returns_iterable(self):
        items = list(progress_iter(range(5), total=5, enabled=False))
        assert items == [0, 1, 2, 3, 4]

    def test_progress_iter_enabled_runs(self):
        items = list(progress_iter(range(3), total=3, enabled=True))
        assert items == [0, 1, 2]

    def test_progressbar_context_manager(self):
        with ProgressBar(total=3, desc="t", enabled=False) as bar:
            for _ in range(3):
                bar.update(1)

    def test_progressbar_enabled_no_crash(self, capsys):
        with ProgressBar(total=3, desc="t", enabled=True) as bar:
            for _ in range(3):
                bar.update(1)