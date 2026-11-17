"""Validated numerical helpers for sampled biomechanics signals.

Signal axis is configurable; time is one-dimensional and strictly increasing.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
from scipy.optimize import OptimizeResult, minimize_scalar
from scipy.signal import correlate, correlation_lags as scipy_correlation_lags

from ..types import AnyNDArray, ArrayLike, FloatArray, SignalPair

def _time(t: ArrayLike, length: Optional[int] = None) -> FloatArray:
    time = np.asarray(t, dtype=float)
    if time.ndim != 1 or time.size == 0 or not np.all(np.isfinite(time)):
        raise ValueError("time must be a nonempty, finite 1-D array")
    if length is not None and time.size != length:
        raise ValueError("time length must match the sample axis")
    if np.any(np.diff(time) <= 0):
        raise ValueError("time must be strictly increasing with no duplicates")
    return time


def _axis(values: AnyNDArray, axis: int) -> int:
    if not isinstance(axis, (int, np.integer)) or not -values.ndim <= axis < values.ndim:
        raise ValueError(f"axis {axis} is out of bounds for {values.ndim}-D input")
    return axis % values.ndim


def differentiate(
    time: ArrayLike,
    values: ArrayLike,
    *,
    axis: int = -1,
) -> tuple[FloatArray, FloatArray]:
    """Return midpoint times and dy/dt, retaining every nonsample dimension.

    ``time`` is measured in seconds; derivative units equal values/second.
    At least two samples are needed. Output has one fewer sample than input.
    """
    y = np.asarray(values, dtype=float)
    ax = _axis(y, axis)
    t = _time(time, y.shape[ax])
    if t.size < 2:
        raise ValueError("differentiation requires at least two samples")
    dt = np.diff(t)
    shape = [1] * y.ndim
    shape[ax] = dt.size
    return (t[:-1] + t[1:]) / 2, np.diff(y, axis=ax) / dt.reshape(shape)


def interpolate_along_axis(
    time_out: ArrayLike,
    time: ArrayLike,
    values: ArrayLike,
    *,
    axis: int = -1,
    left: Optional[float] = np.nan,
    right: Optional[float] = np.nan,
) -> FloatArray:
    """Linear interpolation along ``axis``; outside range uses left/right.

    Samples containing NaN are not implicitly removed. Use ``fill_missing``
    explicitly if gap filling is appropriate for the sensor data.
    """
    y = np.asarray(values, dtype=float)
    ax = _axis(y, axis)
    xp = _time(time, y.shape[ax])
    x = np.asarray(time_out, dtype=float)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("time_out must be a finite 1-D array")
    moved = np.moveaxis(y, ax, -1)
    flat = moved.reshape(-1, xp.size)
    result = np.array([np.interp(x, xp, row, left=left, right=right) for row in flat])
    return np.moveaxis(result.reshape(moved.shape[:-1] + (x.size,)), -1, ax)


def interpolate_rows(
    time_out: ArrayLike,
    time: ArrayLike,
    rows: ArrayLike,
    *,
    left: Optional[float] = np.nan,
    right: Optional[float] = np.nan,
) -> FloatArray:
    """Interpolate a two-dimensional array with one channel per row."""
    data = np.asarray(rows)
    if data.ndim != 2:
        raise ValueError("rows must have shape (channels, samples)")
    return interpolate_along_axis(time_out, time, data, left=left, right=right)


def differentiate_interpolated(
    time: ArrayLike,
    values: ArrayLike,
    *,
    axis: int = -1,
    timeout: Optional[ArrayLike] = None,
) -> FloatArray:
    """Finite-difference derivative interpolated at the requested timestamps.

    By default the output times equal ``time``. Endpoints are held at the
    first/last finite-difference derivative (not extrapolated).
    """
    mid, gradient = differentiate(time, values, axis=axis)
    return interpolate_along_axis(time if timeout is None else timeout, mid,
                                  gradient, axis=axis, left=None, right=None)


def fill_missing(
    values: ArrayLike,
    *,
    axis: int = -1,
    max_gap: Optional[int] = None,
    extrapolate: str = "nearest",
) -> FloatArray:
    """Fill NaNs by linear interpolation, returning a float copy.

    ``max_gap`` limits the number of consecutive NaN samples to fill.
    ``extrapolate='nearest'`` fills boundary gaps with the nearest valid value;
    ``extrapolate='nan'`` preserves missing boundaries. All-NaN channels stay NaN.
    """
    data = np.asarray(values, dtype=float)
    ax = _axis(data, axis)
    if extrapolate not in ("nearest", "nan"):
        raise ValueError("extrapolate must be 'nearest' or 'nan'")
    if max_gap is not None and (not isinstance(max_gap, (int, np.integer)) or max_gap < 1):
        raise ValueError("max_gap must be a positive integer or None")
    if np.any(np.isinf(data)):
        raise ValueError("signal cannot contain infinity")
    moved = np.moveaxis(data, ax, -1)
    out = moved.copy().reshape(-1, moved.shape[-1])
    for row in out:
        good = np.isfinite(row)
        if not np.any(good):
            continue
        missing = ~good
        idx = np.arange(row.size)
        estimate = np.interp(idx, idx[good], row[good])
        if extrapolate == "nan":
            missing &= (idx >= idx[good][0]) & (idx <= idx[good][-1])
        if max_gap is not None:
            padded = np.pad(~good, (1, 1), constant_values=False)
            starts = np.flatnonzero(np.diff(padded.astype(int)) == 1)
            ends = np.flatnonzero(np.diff(padded.astype(int)) == -1)
            eligible = np.zeros(row.size, dtype=bool)
            for start, end in zip(starts, ends):
                if end - start <= max_gap:
                    eligible[start:end] = True
            missing &= eligible
        row[missing] = estimate[missing]
    return np.moveaxis(out.reshape(moved.shape), -1, ax)


def fill_missing_rows(
    rows: ArrayLike,
    *,
    max_gap: Optional[int] = None,
    extrapolate: str = "nearest",
) -> FloatArray:
    """Fill NaNs independently in rows; input shape is (channels, samples)."""
    x = np.asarray(rows)
    if x.ndim != 2:
        raise ValueError("rows must be two-dimensional")
    return fill_missing(x, max_gap=max_gap, extrapolate=extrapolate)


def resample_signal(
    signal: SignalPair,
    time_out: ArrayLike,
    *,
    axis: int = -1,
) -> tuple[FloatArray, FloatArray]:
    """Resample a ``(time, values)`` pair; samples outside support are NaN."""
    t, y = signal
    requested = np.asarray(time_out, dtype=float)
    return requested, interpolate_along_axis(requested, t, y, axis=axis)


def nan_root_mean_square(values: ArrayLike) -> float:
    """RMS of finite values. Return nan when there are no finite samples."""
    x = np.asarray(values, dtype=float)
    valid = np.isfinite(x)
    return float(np.sqrt(np.mean(x[valid] ** 2))) if np.any(valid) else float("nan")


def first_true_index(mask: ArrayLike) -> int:
    """First true index of a 1-D mask, or mask length if all false."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 1:
        raise ValueError("mask must be one-dimensional")
    return int(np.flatnonzero(m)[0]) if np.any(m) else len(m)


def evaluate_time_delay(
    signal_1: SignalPair,
    signal_2: SignalPair,
    delay: float,
    function: Optional[Callable[[FloatArray, FloatArray], float]] = None,
    *,
    min_overlap: int = 3,
) -> float:
    """Score shifting timestamps of signal_2 by ``delay`` (seconds).

    A positive returned delay means signal_2 must be shifted later to align
    with signal_1. Uses reference timestamps from signal_1, not extrapolation.
    """
    t1, y1 = signal_1
    t2, y2 = signal_2
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    t1 = _time(t1, y1.shape[-1])
    t2 = _time(t2, y2.shape[-1])
    if y1.shape[:-1] != y2.shape[:-1]:
        raise ValueError("signals must have matching channel dimensions")
    if min_overlap < 2:
        raise ValueError("min_overlap must be at least 2")
    shifted = t2 + delay
    inside = (t1 >= shifted[0]) & (t1 <= shifted[-1])
    if np.count_nonzero(inside) < min_overlap:
        return float("inf")
    y1_overlap = y1[..., inside]
    y2_overlap = interpolate_along_axis(t1[inside], shifted, y2)
    finite = np.isfinite(y1_overlap) & np.isfinite(y2_overlap)
    if np.count_nonzero(finite) < min_overlap:
        return float("inf")
    if function is None:
        return nan_root_mean_square((y1_overlap - y2_overlap)[finite])
    return float(function(y1_overlap[finite], y2_overlap[finite]))


def estimate_time_delay(
    signal_1: SignalPair,
    signal_2: SignalPair,
    bounds: Optional[tuple[Optional[float], Optional[float]]] = None,
    method: str = "rmse",
    *,
    min_overlap: int = 3,
    grid_points: int = 301,
) -> OptimizeResult:
    """Estimate timestamp shift needed to align signal_2 to signal_1.

    First checks a coarse search grid, then refines a local minimum using
    bounded scalar optimization. ``result.x`` is the shift in seconds.
    Only 'rmse' is currently supported. Periodic signals can have multiple
    minima; specify tight bounds and examine alignment.
    """
    if method != "rmse":
        raise ValueError("Only method='rmse' is supported")
    if grid_points < 3:
        raise ValueError("grid_points must be at least 3")
    t1 = _time(signal_1[0])
    t2 = _time(signal_2[0])
    span2 = t2[-1] - t2[0]
    inferred = (t1[0] - t2[0] - span2 / 2, t1[-1] - t2[0] - span2 / 2)
    if bounds is None:
        low, high = inferred
    else:
        low = inferred[0] if bounds[0] is None else float(bounds[0])
        high = inferred[1] if bounds[1] is None else float(bounds[1])
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        raise ValueError("bounds must be finite and strictly increasing")
    cost = lambda d: evaluate_time_delay(signal_1, signal_2, d, min_overlap=min_overlap)
    grid = np.linspace(low, high, grid_points)
    scores = np.array([cost(d) for d in grid])
    if not np.any(np.isfinite(scores)):
        raise ValueError("no valid overlap for requested delay bounds")
    best = int(np.argmin(scores))
    a = grid[max(best - 1, 0)]
    b = grid[min(best + 1, grid.size - 1)]
    res = minimize_scalar(cost, bounds=(a, b), method="bounded", options={"xatol": 1e-11})
    if scores[best] <= res.fun:
        res.x = grid[best]
        res.fun = scores[best]
    return res


def moving_pearson_correlation(y1: ArrayLike, y2: ArrayLike, *, mode: str = "full") -> FloatArray:
    """Pearson r for each lag, using means of the *overlapping* samples.

    Input is two finite 1-D arrays. Lags follow ``scipy.signal.correlation_lags``.
    Where overlap has <2 samples or zero variance, returns NaN.
    """
    a = np.asarray(y1, dtype=float)
    b = np.asarray(y2, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.size == 0 or b.size == 0:
        raise ValueError("both signals must be nonempty one-dimensional arrays")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("correlation requires finite values")
    if mode not in ("full", "same", "valid"):
        raise ValueError("mode must be full, same, or valid")
    ones_a = np.ones_like(a)
    ones_b = np.ones_like(b)
    n = correlate(ones_a, ones_b, mode=mode, method="direct")
    a_sum = correlate(a, ones_b, mode=mode, method="direct")
    b_sum = correlate(ones_a, b, mode=mode, method="direct")
    a_sq = correlate(a * a, ones_b, mode=mode, method="direct")
    b_sq = correlate(ones_a, b * b, mode=mode, method="direct")
    cross = correlate(a, b, mode=mode, method="direct")
    with np.errstate(divide="ignore", invalid="ignore"):
        covariance = cross - a_sum * b_sum / n
        variance_a = a_sq - a_sum**2 / n
        variance_b = b_sq - b_sum**2 / n
        result = covariance / np.sqrt(np.maximum(variance_a, 0) * np.maximum(variance_b, 0))
    result[(n < 2) | (variance_a <= 1e-12) | (variance_b <= 1e-12)] = np.nan
    return np.clip(result, -1.0, 1.0)


def correlation_lags(
    time1: ArrayLike,
    time2: ArrayLike,
    *,
    mode: str = "full",
    delay_type: str = "start",
) -> FloatArray:
    """Return sample index lags for a correlation of sequences with those lengths."""
    if delay_type not in ("start", "middle"):
        raise ValueError("delay_type must be 'start' or 'middle'")
    lags = scipy_correlation_lags(len(time1), len(time2), mode=mode)
    return lags.astype(float) + ((len(time2) - 1) / 2 if delay_type == "middle" else 0)



def plot_time_delay(
    signal_1: SignalPair,
    signal_2: SignalPair,
    delay: float = 0.0,
) -> tuple[Any, AnyNDArray]:
    """Plot original and aligned signals; requires optional ``matplotlib``.

    Returns the matplotlib ``(figure, axes)`` and does not call ``show()``.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_time_delay requires matplotlib; use 'uv sync --extra plot'") from exc
    t1, v1 = signal_1
    t2, v2 = signal_2
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    ax[0].plot(t1, np.moveaxis(np.asarray(v1), -1, 0), label="signal 1")
    ax[0].plot(t2, np.moveaxis(np.asarray(v2), -1, 0), label="signal 2")
    ax[0].set_title("Original samples")
    ax[1].plot(t1, np.moveaxis(np.asarray(v1), -1, 0), label="signal 1")
    ax[1].plot(np.asarray(t2) + delay, np.moveaxis(np.asarray(v2), -1, 0), label="signal 2 (shifted)")
    ax[1].set_title(f"Signal 2 timestamps shifted by {delay:.4g} s")
    for axes in ax:
        axes.set_ylabel("Signal value")
        axes.legend()
    ax[1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig, ax
