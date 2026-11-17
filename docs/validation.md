# Numerical validation scope and limitations

This repository packages research utilities and provides regression tests. **It is not a certified medical device, a clinically validated analysis package, or a substitute for laboratory-specific signal-processing standard operating procedures.**

## Regression test coverage

| Area | Tested cases | Remaining cautions |
|---|---|---|
| PSD | Matches SciPy one-sided periodogram for 99/100/101/102-sample real inputs; density sum follows mean-square Parseval relation | Nonstationary signal handling, multitaper/Welch estimators and device noise standards are out of scope |
| Frequency metrics | Multichannel axis handling, percentile for flat/zero spectra, known dominant sinusoidal frequencies | Continuous trapezoidal percentile convention differs from discrete bin quantile; DC effects require deliberate preprocessing |
| Derivative/resampling | Irregular timestamps, multi-axis channels, out-of-bounds interpolation, time validation | Numerical differentiation amplifies noise; downsampling lacks built-in anti-alias filtering |
| Missing-value fill | Short-gaps-only behavior, endpoint modes, copy semantics, all-missing channels | Long-gap interpolations may change peak magnitudes, load rates and event timing |
| Delay estimation | Recovered known synthetic timestamp offset, insufficient overlap handling | Real sensor sampling jitter, modality mismatch and periodic ambiguity need application-specific work |
| Filtering | Low- and band-pass rejection of test frequencies, channel preservation, invalid cutoff rejection | Edge effects remain possible; zero-phase offline filtering is not causal; filter specification is not automatically optimized for biomechanics |
| Orientation | Axis-angle rotation, inverse of nonunit quaternion, norm preservation, matrix/Euler roundtrips, scalar-first broadcasting | Coordinate systems, handedness, intrinsic/extrinsic Euler definitions and static/dynamic calibrations must be verified against instrumentation |
| Quaternion time series | SLERP shortest-path interpolation, sign invariance, gap fill, Markley averaging, cumulative increments | Jitter, drift and actual sensor fusion estimation are beyond package scope |

## Known limits

- `interpolate_quaternions` currently accepts a **single** `(n_samples,4)` orientation stream. For many streams, call it per channel/trial; `fill_missing_quaternions` and `mean_quaternion` do support batch dimensions.
- `estimate_time_delay` uses RMSE of overlapping samples and is vulnerable to amplitude scaling, drift, missingness, sensor-specific response lags, and periodic minima. Inspect aligned plots and examine plausible bounds.
- `moving_pearson_correlation` is defined only for finite 1-D inputs and returns NaN for degenerate lag overlaps.
- `power_spectral_density` uses a rectangular window by default and will exhibit spectral leakage. Consider appropriate segmentation, tapering and Welch's method for practical EMG or noisy accelerometer analysis.
- `flat_top_window` is a heuristic smoothing kernel, not a frequency-calibrated substitute for a Butterworth filter.
- The `Rotation` object is a convenience wrapper around a **public, mutable** NumPy array; avoid mutating `.data` unexpectedly.

## Recommended validation for a publication

1. Record Python, NumPy, SciPy, and package versions (`uv.lock` or explicit environment capture).
2. Establish coordinate frame conventions (global vs local axes, active vs passive, right- vs left-handed, Euler order, quaternion layout).
3. Confirm physical units (N, m, m/s, N/s, Hz, seconds, rad) and track any normalization.
4. Test each signal-processing step against a reference waveform or benchmark and inspect boundary handling.
5. Test frequency-domain results against expected analytic frequencies and experiment-specific reference code.
6. Quantify robustness to sampling rate, padding, missing segments, artifact rejection and sensor clocks.
7. Keep the original unprocessed data; document missing-data interpolation and filter choice in manuscripts.
8. For hypothesis testing, predefine the relevant features and consider uncertainty, effect size, multiplicity and participant-level independence.

## Test commands

```bash
uv sync
uv run pytest -q
uv run python examples/quickstart.py
uv build
```

The automated regression suite is intended to detect code regressions. It cannot establish validity for arbitrary biomechanical populations, sensor platforms, clinical endpoints or downstream scientific claims.
