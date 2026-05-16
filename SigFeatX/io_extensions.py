"""
SigFeatX - io_extensions.py
============================
Big-data writers for batch feature tables.

  - Parquet: fast columnar format, excellent compression, pandas/dask/polars compatible.
  - HDF5:    hierarchical, supports per-key metadata, append mode.
  - Feather: ultra-fast read/write for short-lived caches.

All formats round-trip pandas DataFrames produced by ``BatchResult.dataframe``.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class BatchIO:
    """Read/write large batch feature tables."""

    # ------------------------------------------------------------------
    # Parquet
    # ------------------------------------------------------------------

    @staticmethod
    def save_parquet(
        dataframe,
        filepath: str,
        compression: str = 'snappy',
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Save a feature DataFrame to Parquet.

        Parameters
        ----------
        dataframe   : pandas DataFrame (typically BatchResult.dataframe).
        filepath    : output path; '.parquet' suffix recommended.
        compression : 'snappy' (default, fast), 'gzip', 'brotli', 'zstd', or None.
        metadata    : optional key/value strings stored alongside the table.

        Requires
        --------
        pyarrow (preferred) or fastparquet. Install with:
            pip install pyarrow
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required to save Parquet files.")

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            # Try fastparquet via pandas as a fallback
            try:
                dataframe.to_parquet(filepath, compression=compression,
                                     engine='fastparquet')
                return
            except (ImportError, ValueError) as exc:
                raise ImportError(
                    "Parquet support requires pyarrow (preferred) or fastparquet. "
                    "Install with:  pip install pyarrow"
                ) from exc

        table = pa.Table.from_pandas(dataframe)
        if metadata:
            existing = dict(table.schema.metadata or {})
            existing.update({
                k.encode('utf-8'): str(v).encode('utf-8')
                for k, v in metadata.items()
            })
            table = table.replace_schema_metadata(existing)
        pq.write_table(table, filepath, compression=compression)

    @staticmethod
    def load_parquet(filepath: str):
        """Load a Parquet file back into a pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required to load Parquet files.")
        return pd.read_parquet(filepath)

    @staticmethod
    def load_parquet_metadata(filepath: str) -> Dict[str, str]:
        """Read user metadata attached via ``save_parquet(metadata=...)``."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            return {}
        md = pq.read_metadata(filepath).metadata or {}
        # Skip pyarrow's internal schema/pandas keys; return only user metadata
        internal = (b'pandas', b'ARROW:')
        return {
            k.decode('utf-8'): v.decode('utf-8')
            for k, v in md.items()
            if not any(k.startswith(prefix) for prefix in internal)
        }

    # ------------------------------------------------------------------
    # HDF5
    # ------------------------------------------------------------------

    @staticmethod
    def save_hdf5(
        dataframe,
        filepath: str,
        key: str = 'features',
        mode: str = 'w',
        complib: str = 'blosc',
        complevel: int = 5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save a feature DataFrame to HDF5.

        Parameters
        ----------
        dataframe : pandas DataFrame.
        filepath  : output path; '.h5' or '.hdf5' suffix recommended.
        key       : HDF5 group path; lets you store multiple tables per file.
        mode      : 'w' overwrite, 'a' append.
        complib   : compression library ('blosc', 'zlib', 'lzo', 'bzip2').
        complevel : 0-9 compression level.
        metadata  : optional dict stored as JSON in the table's user_block.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required to save HDF5 files.")

        dataframe.to_hdf(
            filepath,
            key=key,
            mode=mode,
            format='table',
            complib=complib,
            complevel=complevel,
        )

        if metadata:
            try:
                import h5py
                with h5py.File(filepath, 'a') as f:
                    f.attrs[f'{key}_metadata'] = json.dumps(metadata)
            except ImportError:
                # h5py not present; metadata silently dropped
                pass

    @staticmethod
    def load_hdf5(filepath: str, key: str = 'features'):
        """Load a feature DataFrame from HDF5."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required to load HDF5 files.")
        return pd.read_hdf(filepath, key=key)

    @staticmethod
    def load_hdf5_metadata(filepath: str, key: str = 'features') -> Dict[str, Any]:
        """Read user metadata attached via ``save_hdf5(metadata=...)``."""
        try:
            import h5py
        except ImportError:
            return {}
        try:
            with h5py.File(filepath, 'r') as f:
                raw = f.attrs.get(f'{key}_metadata')
                if raw is None:
                    return {}
                if isinstance(raw, bytes):
                    raw = raw.decode('utf-8')
                return json.loads(raw)
        except (OSError, KeyError):
            return {}

    # ------------------------------------------------------------------
    # Feather (cache format)
    # ------------------------------------------------------------------

    @staticmethod
    def save_feather(dataframe, filepath: str, compression: str = 'lz4') -> None:
        """Save to Feather for fast short-lived caching."""
        dataframe.reset_index(drop=False).to_feather(filepath, compression=compression)

    @staticmethod
    def load_feather(filepath: str):
        """Load a Feather file."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required to load Feather files.")
        df = pd.read_feather(filepath)
        # Reset original index if it was preserved on save
        if 'signal_idx' in df.columns:
            df = df.set_index('signal_idx')
        return df

    # ------------------------------------------------------------------
    # Streaming writer (append-friendly)
    # ------------------------------------------------------------------

    @staticmethod
    def append_parquet(
        dataframe,
        directory: str,
        partition_name: Optional[str] = None,
    ) -> str:
        """
        Append a feature batch to a partitioned Parquet dataset.

        Each call writes one file under ``directory/``. Read the whole dataset
        back with ``pd.read_parquet(directory)``.

        Parameters
        ----------
        dataframe      : pandas DataFrame to append.
        directory      : root directory of the dataset.
        partition_name : optional partition filename stub (uses incremental
                         id if not provided).

        Returns
        -------
        Path of the file written.
        """
        os.makedirs(directory, exist_ok=True)
        if partition_name is None:
            existing = sorted(Path(directory).glob('part-*.parquet'))
            partition_name = f'part-{len(existing):06d}'
        out_path = os.path.join(directory, f'{partition_name}.parquet')
        BatchIO.save_parquet(dataframe, out_path)
        return out_path

    # ------------------------------------------------------------------
    # Auto-dispatch by extension
    # ------------------------------------------------------------------

    @staticmethod
    def save(
        dataframe,
        filepath: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Dispatch save by file extension.

        Supports: .parquet, .pq, .h5, .hdf5, .feather, .csv, .json
        """
        suffix = Path(filepath).suffix.lower()
        if suffix in ('.parquet', '.pq'):
            BatchIO.save_parquet(
                dataframe, filepath,
                metadata={k: str(v) for k, v in metadata.items()} if metadata else None,
            )
        elif suffix in ('.h5', '.hdf5'):
            BatchIO.save_hdf5(dataframe, filepath, metadata=metadata)
        elif suffix == '.feather':
            BatchIO.save_feather(dataframe, filepath)
        elif suffix == '.csv':
            dataframe.to_csv(filepath, index=True)
        elif suffix == '.json':
            dataframe.to_json(filepath, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported extension '{suffix}' for '{filepath}'.")

    @staticmethod
    def load(filepath: str):
        """Dispatch load by file extension."""
        suffix = Path(filepath).suffix.lower()
        if suffix in ('.parquet', '.pq'):
            return BatchIO.load_parquet(filepath)
        if suffix in ('.h5', '.hdf5'):
            return BatchIO.load_hdf5(filepath)
        if suffix == '.feather':
            return BatchIO.load_feather(filepath)
        if suffix == '.csv':
            try:
                import pandas as pd
            except ImportError:
                raise ImportError("pandas is required to load CSV files.")
            return pd.read_csv(filepath, index_col=0)
        if suffix == '.json':
            try:
                import pandas as pd
            except ImportError:
                raise ImportError("pandas is required to load JSON files.")
            return pd.read_json(filepath, orient='records')
        raise ValueError(f"Unsupported extension '{suffix}' for '{filepath}'.")