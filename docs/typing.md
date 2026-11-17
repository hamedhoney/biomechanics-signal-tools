# Type hints and NumPy arrays

Shared type aliases live in
`biomechanics_signal_tools.types`. These aliases are used across filtering, processing, frequency analysis,
geometry, and quaternions.

## Available aliases

| Alias | Meaning | Example use |
|---|---|---|
| `ArrayLike` | `numpy.typing.ArrayLike`; accepts NumPy-compatible inputs (lists, tuples, scalars, arrays) | Parameters processed by `np.asarray` |
| `FloatArray` | `numpy.typing.NDArray[np.float64]` | Arrays explicitly converted to float64 |
| `FloatingArray` | `numpy.typing.NDArray[np.floating[Any]]` | Data with floating dtype, including float32 |
| `IntArray` | `numpy.typing.NDArray[np.integer[Any]]` | Integer index arrays |
| `BoolArray` | `numpy.typing.NDArray[np.bool_]` | Boolean masks |
| `AnyNDArray` | `numpy.typing.NDArray[Any]` | Array operations that may preserve or vary dtype |
| `FloatResult` | `Union[FloatArray, np.float64]` | Scalar or batched output, e.g. quaternion angle |
| `SignalPair` | `tuple[ArrayLike, ArrayLike]` | `(time, values)` input to alignment/resampling |

`Any` in generic NumPy dtype parameters means **any dtype**; use `FloatArray`
when the function actually guarantees `float64`.

## How to annotate your code

```python
import numpy as np
from biomechanics_signal_tools.types import ArrayLike, FloatArray, SignalPair
from biomechanics_signal_tools.signal import filtering, processing


def clean_force(values: ArrayLike, sampling_rate: float) -> FloatArray:
    # filter internally normalizes to float64
    return filtering.butterworth_lowpass(values, 15.0, sampling_rate)


def resampled(signal: SignalPair, times: ArrayLike) -> tuple[FloatArray, FloatArray]:
    return processing.resample_signal(signal, times)

print(clean_force([1, 2, 3] * 40, 200).dtype)  # float64
```

To retain an input floating precision in your own function, prefer
`FloatingArray` and **do not** pass `dtype=np.float64` into `np.asarray`:

```python
from biomechanics_signal_tools.types import FloatingArray


def preserve_floating_dtype(samples: FloatingArray) -> FloatingArray:
    return samples.copy()
```

## Scalar, shapes and runtime checks

`NDArray[np.float64]` expresses the *dtype*, not the dimensions or number of
samples. A signal `(samples,)`, a motion capture trajectory `(samples, 3)` and
quaternions `(samples, 4)` share the same array dtype annotation. The package
checks required shapes at runtime (e.g. quaternions require final axis length 4).

Scalar results can be NumPy scalars. For instance, `dominant_frequency` returns
`np.float64` for a single channel and a float64 array for multiple channels;
`FloatResult` covers both. Numeric scalar-returning functions like
`nan_root_mean_square` explicitly return a Python `float`.

Type hints are *not* runtime validation, and `ArrayLike` can include inputs that
will still fail numerical validation (e.g. object or complex arrays). Public
functions generally call `np.asarray(..., dtype=float)`, then check time,
shape, sampling frequency and finite-value assumptions.

`FloatArray` is not a runtime class; do not call `isinstance(x, FloatArray)`.
Check `isinstance(x, np.ndarray)` and `x.dtype` when runtime checks are needed.

## Using the annotations in a development environment

Editors such as VS Code/Pylance can use function annotations for completion and
warnings. Run the optional type checker configured for your own projects. The
package tests verify annotations are present and resolve with
`typing.get_type_hints`; **this is not a full mypy or pyright type check**.

The package remains compatible with Python **3.9+**. `TypeAlias` comes from
`typing_extensions` rather than `typing` because Python 3.9 lacks
`typing.TypeAlias`. This dependency is declared in `pyproject.toml`.

See [`examples/typed_usage.py`](../examples/typed_usage.py) for a runnable
force-plate and quaternion example.
