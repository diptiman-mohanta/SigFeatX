# Feature glossary

Every scalar feature SigFeatX can emit, grouped by the class that computes
it, with a one-line definition and the primary literature reference where
the measure is defined. Feature *keys* below are the exact dictionary keys
returned by each `extract()` call (before any aggregator prefix such as
`raw_`).

## Time domain — `TimeDomainFeatures`

Basic statistics and shape descriptors of the raw waveform. Crest / shape /
impulse / clearance factors are standard rotating-machinery condition
indicators (Randall, *Vibration-based Condition Monitoring*, 2011). TKEO is
the Teager–Kaiser energy operator (Kaiser 1990).

| Key | Meaning |
|-----|---------|
| `mean`, `median`, `mode` | Central tendency of the samples |
| `std`, `variance` | Dispersion |
| `max`, `min`, `range`, `peak_to_peak` | Extremes and their span |
| `skewness`, `kurtosis` | 3rd / 4th standardised moments (asymmetry, tailedness) |
| `rms` | Root-mean-square amplitude |
| `energy`, `power` | Sum / mean of squared samples |
| `mean_absolute`, `sum_absolute` | Mean / sum of absolute samples |
| `zero_crossings`, `zero_crossing_rate` | Count / rate of sign changes |
| `line_length` | Sum of absolute successive differences (waveform "length") |
| `autocorrelation_peak_lag`, `autocorrelation_peak_value` | Lag and height of the first non-zero autocorrelation peak |
| `tkeo_mean`, `tkeo_std`, `tkeo_max` | Teager–Kaiser energy operator statistics (Kaiser 1990) |
| `crest_factor` | peak / RMS |
| `shape_factor` | RMS / mean-absolute |
| `impulse_factor` | peak / mean-absolute |
| `clearance_factor` | peak / (mean √\|x\|)² |
| `q25`, `q75`, `iqr` | Quartiles and interquartile range |
| `coeff_variation` | std / \|mean\| |

## Frequency domain — `FrequencyDomainFeatures`

Descriptors of the one-sided power spectrum. Spectral shape features follow
standard audio/MIR conventions (Peeters, *A large set of audio features*,
2004). EEG band powers use canonical clinical bands.

| Key | Meaning |
|-----|---------|
| `spectral_centroid` | Power-weighted mean frequency ("brightness") |
| `spectral_spread`, `spectral_bandwidth` | Power-weighted std around the centroid |
| `spectral_bandwidth_90` | Width of the 5–95% cumulative-power band |
| `spectral_rolloff` | Frequency below which 95% of power lies |
| `dominant_frequency` | Frequency of maximum power |
| `max_magnitude` | Peak spectral magnitude |
| `spectral_flatness` | Geometric/arithmetic mean of the power spectrum — Wiener SFM, tonal (→0) vs noise-like (→1) |
| `spectral_entropy` | Shannon entropy of the normalised power spectrum |
| `spectral_slope` | Slope of log-power vs log-frequency |
| `spectral_kurtosis`, `spectral_skewness` | 4th / 3rd standardised spectral moments |
| `instantaneous_freq_mean`, `instantaneous_freq_std` | Mean / std of the Hilbert instantaneous frequency |
| `energy_{very_low,low,medium,high,very_high}` (+ `_ratio_`) | Energy in five Nyquist-relative bands, absolute and as a fraction |
| `bandpower_{delta,theta,alpha,beta,gamma}` (+ `_rel`) | Classical EEG band powers (δ 0.5–4, θ 4–8, α 8–13, β 13–30, γ 30–100 Hz), absolute and relative |
| `spectral_flux` | Sum of squared frame-to-frame magnitude change |

## Entropy — `EntropyFeatures`

| Key | Meaning | Reference |
|-----|---------|-----------|
| `shannon_entropy` | Entropy of the amplitude histogram | Shannon 1948 |
| `sample_entropy` | Regularity via template matching, self-matches excluded | Richman & Moorman 2000 |
| `permutation_entropy` | Entropy of ordinal (rank) pattern distribution | Bandt & Pompe 2002 |
| `approximate_entropy` | Regularity via template matching, self-matches included | Pincus 1991 |

## Nonlinear dynamics — `NonlinearFeatures`

| Key | Meaning | Reference |
|-----|---------|-----------|
| `hjorth_activity`, `hjorth_mobility`, `hjorth_complexity` | Variance, mean-frequency, and bandwidth descriptors | Hjorth 1970 |
| `higuchi_fractal_dimension` | Fractal dimension from curve-length scaling | Higuchi 1988 |
| `petrosian_fractal_dimension` | Fractal dimension from sign-change density | Petrosian 1995 |
| `hurst_exponent` | Long-range dependence via rescaled-range (R/S) analysis | Hurst 1951 |
| `lyapunov_exponent` | Largest Lyapunov exponent (sensitivity to initial conditions) | Rosenstein et al. 1993 |
| `dfa_alpha` | Detrended fluctuation analysis scaling exponent | Peng et al. 1994 |

## Recurrence quantification — `RQAFeatures`

Quantifies the recurrence plot of a time-delay embedding.
Refs: Eckmann, Kamphorst & Ruelle 1987; Marwan et al. 2007.

| Key | Meaning |
|-----|---------|
| `rqa_rr` | Recurrence rate (density of recurrence points) |
| `rqa_det` | Determinism (fraction of recurrence points on diagonal lines) |
| `rqa_lam` | Laminarity (fraction on vertical lines) |
| `rqa_l_avg`, `rqa_l_max` | Mean / maximum diagonal line length |
| `rqa_tt` | Trapping time (mean vertical line length) |
| `rqa_v_max` | Maximum vertical line length |
| `rqa_entr` | Shannon entropy of the diagonal-line-length distribution |
| `rqa_div` | Divergence (1 / longest diagonal line) |
| `rqa_eps` | Recurrence threshold used (returned for reproducibility) |

## Multifractal DFA — `MFDFAFeatures`

Multifractal detrended fluctuation analysis (Kantelhardt et al. 2002).

| Key | Meaning |
|-----|---------|
| `mfdfa_width` | Width of the singularity spectrum f(α) — degree of multifractality |
| `mfdfa_alpha0` | Position of the spectrum peak |
| `mfdfa_asymmetry` | Left/right asymmetry of f(α) |
| `mfdfa_h_q{n5,n3,n1,p0,p1,p3,p5}` | Generalised Hurst exponent h(q) at q = −5…+5 (`n`/`p` = negative/positive) |

## Advanced entropy & complexity — `AdvancedEntropyFeatures`

| Key | Meaning | Reference |
|-----|---------|-----------|
| `dispersion_entropy` | Entropy of dispersion-pattern distribution (normalised 0–1) | Rostaghi & Azami 2016 |
| `fuzzy_entropy` | SampEn with a smooth exponential similarity kernel | Chen et al. 2007 |
| `lz_complexity` | Lempel–Ziv complexity of the binarised signal | Lempel & Ziv 1976 |
| `bubble_entropy` | Rényi-2 entropy of bubble-sort swap counts, near parameter-free | Manis et al. 2017 |

## Decomposition-derived features — `DecompositionFeatures`

For any decomposition (`comp_i_...`), per-component energy, RMS, mean, std,
peak, entropy, and energy ratio, plus cross-component correlation, energy
ratios, and KL divergence between components.
