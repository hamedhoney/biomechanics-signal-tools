# Public API reference

**Imports:**

```python
from biomechanics_signal_tools.signal import processing, filtering, frequency_analysis
from biomechanics_signal_tools.rotation import quaternions, geometry, Rotation
```

`ndarray` means NumPy-compatible array-like input, converted to floating point for signal and quaternion processing. All sample timestamps are in **seconds**, frequency and cutoff arguments in **Hz**, angular inputs in **radians** unless `degrees=True`. Scalar-first quaternion convention is `[w, x, y, z]`.

## Signal processing (`signal.processing`)

| Function | Input and output | Semantics |
|---|---|---|
| `differentiate(time, values, *, axis=-1)` | `time: (N,)`, values sample length N; returns `(time_mid: (N-1,), dy_dt)` | Forward finite difference with *actual* dt, including irregular sampling |
| `differentiate_interpolated(time, values, *, axis=-1, timeout=None)` | derivative array of `len(timeout)` or original N | Derivative interpolated to requested times, endpoint derivative held |
| `interpolate_along_axis(time_out, time, values, *, axis=-1, left=np.nan, right=np.nan)` | Original non-time dimensions preserved, new sample length | Linear interpolation; no extrapolation unless boundary values explicitly provided |
| `interpolate_rows(time_out, time, rows, *, left=np.nan, right=np.nan)` | `rows: (channels,N)` → `(channels, len(time_out))` | Convenience for two-dimensional arrays |
| `fill_missing(values, *, axis=-1, max_gap=None, extrapolate='nearest')` | Same shape, new array | Linear interpolation of NaNs, optional gap length limit, optional preservation of end NaNs |
| `fill_missing_rows(rows, *, max_gap=None, extrapolate='nearest')` | 2D channels x samples | Convenience for rows |
| `resample_signal((time, values), time_out, *, axis=-1)` | Returns `(time_out, values_resampled)` | Returns NaN outside original time support; **no anti-alias filter** |
| `estimate_time_delay(signal_1, signal_2, bounds=None, method='rmse', *, min_overlap=3, grid_points=301)` | `scipy.optimize.OptimizeResult` | Coarse-grid plus local scalar RMSE search; `.x` is timestamp adjustment for signal 2 |
| `evaluate_time_delay(signal_1, signal_2, delay, function=None, *, min_overlap=3)` | scalar score | Shift signal-2 timestamps, score overlap with signal 1 |
| `moving_pearson_correlation(y1, y2, *, mode='full')` | array of r values for each sample lag | Mean-subtracted Pearson coefficient on *overlapping* windows; short/constant overlap yields NaN |
| `correlation_lags(time1, time2, *, mode='full', delay_type='start')` | array of sample lags | Follows SciPy cross-correlation lag convention; despite input name, returns **indices**, not seconds |
| `first_true_index(mask)` | integer | First true position or length when no true position |
| `nan_root_mean_square(values)` | float | RMS ignoring non-finite entries; all non-finite → NaN |
| `plot_time_delay(signal_1, signal_2, delay=0)` | `(matplotlib.figure.Figure, axes)` | Requires optional plotting dependency, does not show window |


## Filtering (`signal.filtering`)

| Function | Semantics |
|---|---|
| `butterworth_lowpass(values, cutoff_hz, fs, *, order=4, axis=-1)` | Offline zero-phase SOS forward/backward low-pass Butterworth; requires sufficiently long finite input |
| `butterworth_bandpass(values, low_hz, high_hz, fs, *, order=4, axis=-1)` | Offline zero-phase SOS forward/backward band-pass Butterworth |
| `flat_top_window(fc, fs=1.0)` | Normalized flat-top smoothing kernel; cutoff is only a heuristic design parameter |
| `apply_window_filter(values, window, *, axis=-1)` | Convolution along selected sample axis with boundary-weight correction |


## Frequency analysis (`signal.frequency_analysis`)

| Function | Semantics |
|---|---|
| `power_spectral_density(values, fs=1.0, *, axis=-1, detrend=None)` | Returns `(frequency_hz: (F,), psd: (..., F, ...))`; one-sided density periodogram for real uniformly sampled data, no detrending by default |
| `spectral_percentile_frequency(frequencies, psd, percentile, *, axis=-1)` | Integrates piecewise-linear PSD using trapezoids, returns quantile frequency; `percentile` **0..1**, not 0..100 |
| `median_frequency(values, fs=1.0, *, axis=-1, detrend=None)` | 0.5 spectral percentile per channel |
| `dominant_frequency(values, fs=1.0, *, axis=-1, detrend='constant', include_dc=False)` | Frequency at maximum PSD bin; DC excluded by default |


## Quaternion and orientation (`rotation.quaternions`)

All modern orientation functions use **active, right-handed rotation** and **scalar-first** `[w, x, y, z]` with components on the final axis. Batch dimensions are broadcast where feasible. Unless explicitly called out, zero or nonfinite quaternions raise `ValueError`.

| Function | Returns / notes |
|---|---|
| `identity_quaternion(shape=())` | Scalar-first identity; `(4,)` or `(*shape,4)` |
| `quaternion_norm(quaternion)` | Euclidean norm per orientation |
| `normalize_quaternion(quaternion)` | Unit quaternion; zero norm rejected |
| `quaternion_inverse(quaternion)` | Algebraic inverse, correct for nonunit inputs |
| `quaternion_conjugate(quaternion)` | Negates vector part |
| `quaternion_from_axis_angle(angle, axis)` | Angle radians; right-hand rotation around axis vector `(...,3)` |
| `quaternion_to_axis_angle(quaternion)` | `(angle, axis)` with shortest angle in [0, pi]; identity axis [1,0,0] |
| `quaternion_angle(quaternion, degrees=False)` | Shortest rotation magnitude |
| `quaternion_from_rotation_vector(vector)` | Axis×angle rotvec → quaternion |
| `quaternion_to_rotation_vector(quaternion)` | Quaternion → shortest rotvec |
| `quaternion_from_euler(angles, seq='xyz', degrees=False)` | Intrinsic (ordered) 3-axis Euler angles; array shape `(...,3)` |
| `quaternion_to_euler(quaternion, seq='xyz', degrees=False)` | Euler angles; beware gimbal-lock ambiguity |
| `quaternion_from_matrix(matrix)` | Orthonormal *active* matrix `(...,3,3)` → quaternion; reflection rejected |
| `quaternion_to_matrix(quaternion)` | Active rotation matrix `(...,3,3)` |
| `rotation_matrix_from_unit_vectors(*, x=None, y=None, z=None, strict=None)` | Gram–Schmidt using exactly two supplied basis axes; columns are local axes |
| `quaternion_from_unit_vectors(xunit, yunit, zunit=None, *, strict='x')` | Quaternion from right-handed x/y basis (third axis optional consistency check) |
| `multiply_quaternions(q1, q2)` | Hamilton product: apply q2 first, then q1 |
| `left_divide_quaternions(q1, q2)` | inverse(q1) × q2 |
| `right_divide_quaternions(q1, q2)` | q1 × inverse(q2) |
| `rotate_vectors(quaternion, vectors)` | Active rotation, final axis 3 |
| `rotate_reference_frame(quaternion, vectors)` | Passive coordinate transform (inverse active rotation) |
| `positive_scalar_quaternion(quaternion)` | Picks representation with w ≥ 0 |
| `quaternion_power(quaternion, exponent)` | Raises unit rotation to real power |
| `quaternion_differences(quaternions, n=1, *, axis=-2, real_positive=True)` | Per-frame relative orientation changes; n=1 supported |
| `cumulative_quaternion_product(quaternions, n=1, *, axis=-2, initial=None, real_positive=True)` | Cumulative composition of increments, one result per increment |
| `random_unit_quaternion(shape=(), *, angle=None, rng=None)` | Random unit quaternion; optional fixed angle and seeded NumPy RNG |
| `interpolate_quaternions(sample_times, known_times, known_quaternions)` | SLERP for one time series `(N,4)`, NaN outside support |
| `interpolate_quaternion_sequence(q1, q2, num)` | SLERP of two endpoints, returns `(num,4)`, includes both |
| `fill_missing_quaternions(quaternions)` | SLERP internal NaN gaps and hold nearest boundary, shape `(...,samples,4)` |
| `mean_quaternion(quaternions, weights=None)` | Markley orientation mean; weights per sample |
| `filter_quaternions(quaternions, weights)` | Moving Markley weighted mean, nonnegative kernel |
| `swing_twist_decomposition(quaternion, axis)` | `(twist,swing)` with `q=twist*swing` |


## Geometry (`rotation.geometry`)

`vector_dot_product(v1, v2)` multiplies vectors on the final axis and sums. `add_leading_dimensions(n, *arrays)` pads singleton leading axes. `broadcast_leading_dimensions(*arrays)` equalizes the number of dimensions but does not broadcast data itself.

`Rotation(data)` normalizes the input quaternion and stores a **copy** as `.data`. Factory constructors include `Rotation.from_axis_angle`, `Rotation.from_euler`, and `Rotation.from_matrix`. Methods: `.apply`, `.inverse`, `.as_matrix`, `.as_euler`; `r1 @ r2` composes rotations in that order (apply r2 first). `.data` is user-editable; changing it can invalidate assumptions.

## Errors and output policies

Most methods raise `ValueError` for invalid shapes, timestamps, nonfinite samples, or ill-defined rotations, rather than silently returning a malformed result. `fill_missing` and SLERP are explicit exceptions that intentionally accept NaNs. No resampling method performs implicit low-pass filtering. See [validation notes](validation.md) for validated scope and remaining gaps.

## Typing reference

The APIs use `ArrayLike` for numerical inputs and `FloatArray` for
`float64` array outputs. Some scalar-valued operations use `FloatResult`.
See [typing.md](typing.md) for the full definitions and annotated examples.
