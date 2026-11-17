# Worked guide to biomechanics signal tools

The examples here are **synthetic demonstrations**, not measured human data. All signals and coordinate systems must be checked against the particular sensor manufacturer's conventions and scientific protocol.

## 1. Installation and imports

```bash
uv sync
uv run python examples/quickstart.py
```

```python
import numpy as np
from biomechanics_signal_tools.signal import processing as proc
from biomechanics_signal_tools.signal import filtering as filt
from biomechanics_signal_tools.signal import frequency_analysis as freq
from biomechanics_signal_tools.rotation import quaternions as quat
```

**Always track units.** With `t` in seconds and force in newtons, `differentiate(t, force)` returns N/s; with position in meters, the same function returns m/s. The API does not add units automatically.

## 2. Understanding arrays and axes

Two common biomechanical layouts are `(n_channels, n_samples)` and `(n_samples, n_channels)`.

```python
fs = 200
seconds = np.arange(0, 2, 1/fs)
fx = 10 * np.sin(2*np.pi*3*seconds)
fy = 12 * np.cos(2*np.pi*3*seconds)
force_sample_first = np.column_stack([fx, fy])       # (400, 2)
force_channel_first = force_sample_first.T             # (2, 400)

_, rate1 = proc.differentiate(seconds, force_sample_first, axis=0)
_, rate2 = proc.differentiate(seconds, force_channel_first, axis=-1)
np.testing.assert_allclose(rate1.T, rate2)
```

`differentiate` gives derivatives at the midpoints between timestamps, resulting in **N-1** output samples. For signals with irregular time steps, it uses each actual step rather than an assumed `1/fs`.

To return a derivative to the original sample grid:

```python
rate_original_grid = proc.differentiate_interpolated(seconds, fx)
```

The end samples are held at their nearest midpoint derivative. This is a finite-difference estimate, not an analytically exact derivative; filtering first may be necessary to reduce differentiation noise.

## 3. Missing marker or sensor samples

```python
marker_z = np.array([0., 0.2, np.nan, 0.6, np.nan, np.nan, 1.2])
limited = proc.fill_missing(marker_z, max_gap=1, extrapolate="nan")
# [0., .2, .4, .6, nan, nan, 1.2]

all_gaps = proc.fill_missing(marker_z)
# long gaps filled, but this may be inappropriate for rapid movement
```

`fill_missing` returns a **copy** and leaves an all-NaN channel unchanged. `max_gap` counts consecutive missing **samples**, not seconds. A boundary gap is filled with the nearest valid value by default, unless `extrapolate='nan'` is specified. Infinite values raise `ValueError`.

For repeated 3-D marker coordinates, process the sample axis explicitly:

```python
markers = np.ones((101, 12, 3))  # 101 frames, 12 markers, xyz
markers[10:12, 3, 1] = np.nan
filled = proc.fill_missing(markers, axis=0, max_gap=2, extrapolate="nan")
```

## 4. Resampling signals measured on different clocks

```python
t_force = np.arange(0, 2, 1/1000)
t_kinematics = np.arange(0, 2, 1/100)
fz = 600 + 100*np.sin(2*np.pi*2*t_force)
requested_times, fz_for_kinematics = proc.resample_signal(
    (t_force, fz), t_kinematics
)
```

Resampling here is **linear interpolation**. It does not perform anti-alias filtering. Before downsampling high-frequency data, appropriately low-pass filter to below the new Nyquist frequency. Outside the original time support, the output becomes NaN. Timestamps must be finite and strictly increasing.

## 5. Filtering force-plate signals

For offline analysis, Butterworth helpers use SciPy SOS-based `sosfiltfilt`, which is **zero phase but noncausal**. It may not be used as-is for live feedback or real-time control. The specified cutoff is the single-pass -3 dB frequency, not the adjusted double-pass cutoff.

```python
fs = 1000
t = np.arange(0, 3, 1/fs)
fz = 700 + 50*np.sin(2*np.pi*2*t) + 8*np.sin(2*np.pi*90*t)
clean_fz = filt.butterworth_lowpass(fz, cutoff_hz=20, fs=fs, order=4)
```

For a demonstration of band-pass filtering EMG-like signals (not a complete EMG-processing protocol):

```python
emg_like = np.sin(2*np.pi*75*t) + 0.6*np.sin(2*np.pi*3*t)
bandpassed = filt.butterworth_bandpass(emg_like, 20, 150, fs=fs)
rectified = np.abs(bandpassed)
envelope = filt.butterworth_lowpass(rectified, 6, fs=fs)
```

No baseline correction, movement artifact detector, amplitude normalization, notch filter, or clinical quality rules are included automatically. Choose filter bandwidths for the actual experiment and reference validated protocols.

`flat_top_window(fc, fs)` creates a **normalized smoothing kernel**. `apply_window_filter(x, window)` performs edge-corrected weighted convolution. This is *not* equivalent to a calibrated Butterworth cutoff.

```python
window = filt.flat_top_window(fc=8, fs=200)
smoothed = filt.apply_window_filter(np.cos(2*np.pi*2*seconds), window)
```

The original `cleanQuantized`, `dequantizePoints`, and `dequantize` heuristics remain available through descriptive names, but have **not been comprehensively validated** against diverse quantization noise and can change signal event timing. Treat them as experimental helpers.

## 6. Frequency-domain analysis

```python
fs = 500
t = np.arange(0, 4, 1/fs)
x = 1.5*np.sin(2*np.pi*30*t) + 0.5*np.sin(2*np.pi*65*t)
f_hz, psd = freq.power_spectral_density(x, fs=fs)
peak = freq.dominant_frequency(x, fs=fs)
median = freq.median_frequency(x, fs=fs)
f_95 = freq.spectral_percentile_frequency(f_hz, psd, 0.95)
```

`power_spectral_density` is a **one-sided periodogram** of a **real-valued uniformly sampled** signal, reported as density in input²/Hz. By default it uses a rectangular window and no detrending; specify `detrend='constant'` or preprocess the signal to remove DC. Even- and odd-length signals correctly handle their single-sided FFT bins.

The percentile helper integrates the PSD **as a continuous piecewise-linear curve** using trapezoids. This is a defined interpolation convention; other literature may use discrete bin-sum percentiles or Welch estimates and obtain slightly different frequencies. The `percentile` argument is a **fraction 0 to 1**, not a percentage between 0 and 100. A zero-energy signal returns NaN for its median frequency.

For multiple EMG channels shaped `(samples, channels)`:

```python
channels = np.column_stack([x, x * 0.5])
f, spectra = freq.power_spectral_density(channels, fs, axis=0)
medians_hz = freq.median_frequency(channels, fs, axis=0)
assert spectra.shape == (f.size, 2)
```

**Do not interpret a peak frequency as an EMG fatigue indicator without appropriate filtering, segmentation, stationarity checks, and physiological context.**

## 7. Estimate time delay between sensors

The objective shifts the **timestamps of signal 2**, not its values. Thus `result.x = -0.1` means the signal-2 timestamps must be moved **earlier** by 100 ms to align with signal 1.

```python
fs = 100
t = np.arange(0, 4, 1/fs)
y = np.sin(2*np.pi*(0.7*t + 0.12*t**2)) + 0.2*np.cos(2*np.pi*1.9*t)
reference = (t, y)
shifted_clock = (t + 0.10, y)

result = proc.estimate_time_delay(reference, shifted_clock, bounds=(-0.2, 0))
print(result.x)  # approximately -0.10 seconds
```

Specify realistic `bounds`, because periodic signals can have several plausible local minima. RMSE assumes the signals should have comparable amplitudes and offsets. It is usually inappropriate to match different sensor modalities without physical preprocessing or a model. At least three overlapping finite samples are needed by default.

If Matplotlib is installed:

```python
fig, axes = proc.plot_time_delay(reference, shifted_clock, delay=result.x)
fig.savefig("alignment.png", dpi=150)
```

## 8. Scalar-first quaternions: conventions

A quaternion is stored as `[w, x, y, z]`, with `w` the scalar component. The representation is **unit length** when encoding orientation. `q` and `-q` encode the **same** physical orientation.

Rotations are active and right-handed: a positive 90° rotation around +Z maps global +X to +Y.

```python
q = quat.quaternion_from_axis_angle(np.pi / 2, [0, 0, 1])
assert np.allclose(quat.rotate_vectors(q, [1, 0, 0]), [0, 1, 0])
```

Quaternions from many IMU libraries and SciPy are **scalar last** (`[x, y, z, w]`). Convert explicitly:

```python
q_wxyz = np.array([0.70710678, 0., 0., 0.70710678])
q_xyzw = np.r_[q_wxyz[1:], q_wxyz[0]]
```

### Euler angles

```python
q = quat.quaternion_from_euler([30, 10, -5], seq="xyz", degrees=True)
angles_deg = quat.quaternion_to_euler(q, seq="xyz", degrees=True)
```

Here `xyz` means **intrinsic xyz** rotation / ordered quaternion product `qx * qy * qz`, which matches the ordering used by the original implementation. Euler angles can be ambiguous near gimbal lock, and IMU/device coordinate systems may use other conventions. Verify against a physical calibration and not merely component-wise angle comparison.

### Orientation matrices

```python
R = quat.quaternion_to_matrix(q)  # 3x3 active matrix
q_again = quat.quaternion_from_matrix(R)
# q_again and q may differ in sign; compare rotations, not components
```

`quaternion_from_matrix` rejects non-orthonormal and left-handed matrices. `rotation_matrix_from_unit_vectors(x=..., y=...)` creates an **active matrix with axis vectors as columns**, with the three coordinate axes stored as columns.

### Composing and inverting rotations

```python
qz = quat.quaternion_from_axis_angle(np.pi / 2, [0, 0, 1])
qx = quat.quaternion_from_axis_angle(np.pi / 4, [1, 0, 0])
combined = quat.multiply_quaternions(qz, qx)  # apply qx, then qz
undone = quat.quaternion_inverse(combined)
restored = quat.rotate_vectors(undone, quat.rotate_vectors(combined, [1, 2, 3]))
```

The modern algebraic inverse correctly divides by the **squared norm**. A zero quaternion raises `ValueError`; no silent division-by-zero warnings are used.

## 9. Interpolation and missing IMU orientations

Use **spherical linear interpolation** (SLERP) rather than interpolating quaternion components. SLERP follows the shortest rotational path and handles the equivalent signs `q` and `-q`.

```python
t = np.array([0., 1.])
known = np.stack([
    quat.quaternion_from_axis_angle(0, [0, 0, 1]),
    quat.quaternion_from_axis_angle(np.pi / 2, [0, 0, 1]),
])
t_new = np.linspace(0, 1, 11)
q_new = quat.interpolate_quaternions(t_new, t, known)  # shape (11, 4)
```

Out-of-range timestamps yield NaN quaternions. Missing quaternion rows are filled with SLERP between neighboring valid samples; at the first or last gap, the nearest valid orientation is held:

```python
raw = np.array([known[0], [np.nan]*4, known[1]])
fixed = quat.fill_missing_quaternions(raw)
```

For **averaging several orientation estimates** at the same instant:

```python
q_estimates = np.stack((known[1], -known[1], known[1]))
mean_orientation = quat.mean_quaternion(q_estimates)
```

Use a **Markley mean** (accounting for sign symmetry), not `np.mean(q, axis=0)`.

For a temporal moving average:

```python
smoothed = quat.filter_quaternions(q_new, weights=np.ones(5))
```

## 10. Rotation convenience object

```python
from biomechanics_signal_tools.rotation import Rotation

pelvis_to_lab = Rotation.from_euler([0., 0., 90.], degrees=True)
lab_velocity = pelvis_to_lab.apply([1., 0., 0.])
pelvis_velocity = pelvis_to_lab.inverse().apply(lab_velocity)
```

`Rotation` offers `.data`, `.apply`, `.inverse`, `.as_matrix`, `.as_euler` and `@` composition. Its internal array is a normalized copy but remains user-editable; do not mutate `.data` in a way that violates quaternion normalization.

