# SigFeatX

Comprehensive signal feature extraction with decomposition, batch processing,
and sklearn integration.

SigFeatX turns raw 1D signals (EEG, ECG, vibration, audio, or any other time
series) into feature tables ready for machine learning: time-domain,
frequency-domain, entropy, nonlinear-dynamics, and decomposition-based
features (EMD, VMD, CEEMDAN, MODWT, HHT, SST, and more), plus batch
processing, an sklearn-compatible transformer, and a fluent pipeline builder.

```{toctree}
:maxdepth: 2
:caption: Contents

quickstart
api
```

## Installation

```bash
pip install SigFeatX
```

See the [README](https://github.com/diptiman-mohanta/SigFeatX#readme) for the
full list of optional extras (`sklearn`, `parquet`, `hdf5`, `viz`).

## Where to start

- New to the library? Start with {doc}`quickstart`.
- Looking for a specific class or function? See the {doc}`api` reference.
- Found a bug or have a feature request? Open an issue on
  [GitHub](https://github.com/diptiman-mohanta/SigFeatX/issues).
