# Biomechanics Signal Tools

Numerical utilities for biomechanics time series, force-plate and EMG-style
signals, motion capture, and 3-D orientation analysis.

**Version 0.4.0 | Python 3.9+ | NumPy + SciPy | uv**

This package provides a compact collection of typed, tested numerical functions
for filtering, interpolation, frequency analysis, signal synchronization, and
quaternion-based rotations. It uses only the current implementation and a
single, descriptive public API.

> **Research use:** Passing regression tests does not establish clinical
> validity. Validate the processing pipeline against your sampling system,
> coordinate frames, physical units, and study-specific reference data.

## Installation and setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then
run these commands from the extracted project root:

```bash
uv sync
uv run python examples/quickstart.py
uv run pytest -q
```

To add the package to another uv-based project:

```bash
uv add /absolute/path/to/biomechanics-signal-tools
```

For optional plotting:

```bash
uv sync --extra plot
uv run python examples/force_plate.py --plot
```

Run the package's build step with `uv build`. To install from a published
Git repository, use `uv add git+https://github.com/OWNER/REPOSITORY.git`
after replacing the URL with the actual repository.

## A quick biomechanics workflow

```python
import numpy as np
from biomechanics_signal_tools.signal import filtering, frequency_analysis, processing
from biomechanics_signal_tools.rotation import Rotation, quaternions

fs = 200.0
seconds = np.arange(0, 3, 1 / fs)
force_n = 700 + 50 * np.sin(2 * np.pi * 3 * seconds)
force_n += 5 * np.sin(2 * np.pi * 35 * seconds)

# Filter force and compute the derivative in N/s.
clean = filtering.butterworth_lowpass(force_n, cutoff_hz=10, fs=fs)
time_mid, loading_rate = processing.differentiate(seconds, clean)

# The peak frequency is returned in Hz.
peak_hz = frequency_analysis.dominant_frequency(clean, fs)

# Scalar-first [w, x, y, z] quaternions describe active rotations.
q = quaternions.quaternion_from_axis_angle(np.pi / 2, [0, 0, 1])
rotated_x = quaternions.rotate_vectors(q, [1, 0, 0])

# Or use the orientation object.
orientation = Rotation.from_axis_angle(np.pi / 2, [0, 0, 1])
assert np.allclose(orientation.apply([1, 0, 0]), rotated_x)
```

## Modules

| Module | Main capabilities |
|---|---|
| `signal.processing` | Derivatives, interpolation, NaN gap filling, resampling, time-delay estimation, lagged Pearson correlation |
| `signal.filtering` | Zero-phase Butterworth low-pass/band-pass, weighted window smoothing |
| `signal.frequency_analysis` | One-sided PSD, spectral percentiles, median and dominant frequency |
| `rotation.quaternions` | Quaternion arithmetic, active rotations, axis/angle and Euler conversions, SLERP, averaging and smoothing |
| `rotation.geometry` | Vector dot products, broadcast helpers, and the `Rotation` convenience class |
| `types` | Shared NumPy input/output type aliases |

### Input and output conventions

| Item | Convention |
|---|---|
| Time | Seconds; one-dimensional, finite, strictly increasing |
| Frequency and sampling rate | Hertz (`Hz`) |
| Signal samples | Final axis by default; select `axis=0` for `(samples, channels)` |
| Angle | Radians unless `degrees=True` |
| Quaternion | `[w, x, y, z]`, scalar-first, final axis of size 4 |
| 3-D vector | `[x, y, z]`, final axis of size 3 |
| Rotations | Right-handed **active** rotation; `r1 @ r2` applies `r2` first |
| Missing samples | `NaN` handled explicitly in gap-filling methods |

### Multi-channel signal example

```python
signals = np.column_stack((np.sin(seconds), np.cos(seconds)))
time_mid, derivative = processing.differentiate(seconds, signals, axis=0)
freqs, psd = frequency_analysis.power_spectral_density(signals, fs, axis=0)
```

### Fill only short missing-data gaps

```python
markers = np.array([1., np.nan, 3., np.nan, np.nan, np.nan, 7.])
filled = processing.fill_missing(markers, max_gap=1, extrapolate="nan")
# [1., 2., 3., nan, nan, nan, 7.]
```

### Type annotations

```python
from biomechanics_signal_tools.types import ArrayLike, FloatArray


def filtered_force(samples: ArrayLike, sample_rate: float) -> FloatArray:
    return filtering.butterworth_lowpass(samples, cutoff_hz=10, fs=sample_rate)
```

All array-output annotations use NumPy's `numpy.typing`; shape requirements are
validated at runtime. See [typing guide](docs/typing.md).

## Documentation and runnable examples

- [Usage guide](docs/usage.md): worked pipelines for multi-channel force,
  marker trajectories, frequency analysis, signal alignment, and orientations.
- [API reference](docs/api.md): public functions, parameters, units, and outputs.
- [Typing guide](docs/typing.md): NumPy aliases, array/scalar return types, and examples.
- [Validation notes](docs/validation.md): tested numerical properties and limitations.

```bash
uv run python examples/quickstart.py
uv run python examples/force_plate.py
uv run python examples/emg_spectrum.py
uv run python examples/imu_orientation.py
uv run python examples/typed_usage.py
```

## Package layout

```text
biomechanics-signal-tools/
|-- pyproject.toml
|-- README.md
|-- CONTRIBUTING.md
|-- docs/
|   |-- api.md
|   |-- typing.md
|   |-- usage.md
|   `-- validation.md
|-- examples/
|   |-- quickstart.py
|   |-- force_plate.py
|   |-- emg_spectrum.py
|   |-- imu_orientation.py
|   `-- typed_usage.py
|-- src/biomechanics_signal_tools/
|   |-- __init__.py
|   |-- types.py
|   |-- signal/
|   |   |-- __init__.py
|   |   |-- filtering.py
|   |   |-- frequency_analysis.py
|   |   `-- processing.py
|   `-- rotation/
|       |-- __init__.py
|       |-- geometry.py
|       `-- quaternions.py
`-- tests/
```

Development commands:

```bash
uv sync
uv run pytest -q
uv run python -m compileall -q src
uv build
```

Tests cover time- and frequency-domain processing, filtering, orientation
arithmetic and conversion, and type annotations. The tests are also run in
GitHub Actions using `.github/workflows/tests.yml`.
