"""Smoothing and stable zero-phase IIR filtering for uniformly sampled signals."""
from __future__ import annotations

from typing import Literal, Union

import numpy as np
from scipy import signal as scipy_signal

from ..types import ArrayLike, FloatArray


def _validate_fs(fs: float) -> None:
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be positive and finite")


def flat_top_window(fc: float, fs: float = 1.0) -> FloatArray:
    """Legacy-design normalized flat-top smoothing window; fs and fc in Hz.

    This is a *smoothing kernel*, not a precise cutoff or drop-in Butterworth
    substitute. Use butterworth_lowpass for a defined IIR cutoff.
    """
    _validate_fs(fs)
    if not np.isfinite(fc) or fc <= 0 or fc >= fs / 2:
        raise ValueError("fc must be strictly between zero and Nyquist")
    length = max(3, int(np.ceil(4.6 * (fs / 2) / fc)))
    weights = scipy_signal.windows.flattop(length)
    total = np.sum(weights)
    if not np.isfinite(total) or abs(total) <= 1e-14:
        raise ValueError("invalid flat-top window normalization")
    return weights / total


def apply_window_filter(values: ArrayLike, window: ArrayLike, *, axis: int = -1) -> FloatArray:
    """Normalized windowed convolution with edge-weight compensation.

    Both input arrays must be finite and the (1-D) window must not exceed
    the sample length. Works with arbitrarily shaped channel arrays.
    """
    x = np.asarray(values, dtype=float)
    w = np.asarray(window, dtype=float)
    if x.ndim == 0 or w.ndim != 1 or w.size == 0 or w.size > x.shape[axis]:
        raise ValueError("window must be 1-D, nonempty and no longer than the sample axis")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(w)):
        raise ValueError("window and signal must contain finite values")
    if abs(np.sum(w)) <= 1e-14:
        raise ValueError("window weights must have a nonzero sum")
    xm = np.moveaxis(x, axis, -1)
    flattened = xm.reshape(-1, xm.shape[-1])
    denom = np.convolve(np.ones(xm.shape[-1]), w, mode="same")
    if np.any(np.abs(denom) < 1e-14):
        raise ValueError("window normalization is zero at some samples")
    out = np.array([np.convolve(r, w, mode="same") / denom for r in flattened])
    return np.moveaxis(out.reshape(xm.shape), -1, axis)


def butterworth_lowpass(
    values: ArrayLike,
    cutoff_hz: float,
    fs: float,
    *,
    order: int = 4,
    axis: int = -1,
) -> FloatArray:
    """Zero-phase Butterworth low-pass filtering via SOS and sosfiltfilt.

    The supplied cutoff is the *single-pass* -3 dB cutoff, not an adjusted
    forward-backward effective cutoff. This filter is noncausal.
    """
    return _butter(values, cutoff_hz, fs, kind="lowpass", order=order, axis=axis)


def butterworth_bandpass(
    values: ArrayLike,
    low_hz: float,
    high_hz: float,
    fs: float,
    *,
    order: int = 4,
    axis: int = -1,
) -> FloatArray:
    """Zero-phase Butterworth band-pass; cutoff frequencies are in hertz."""
    return _butter(values, (low_hz, high_hz), fs, kind="bandpass", order=order, axis=axis)


def _butter(
    values: ArrayLike,
    cutoff: Union[float, tuple[float, float]],
    fs: float,
    *,
    kind: Literal["lowpass", "bandpass"],
    order: int,
    axis: int,
) -> FloatArray:
    _validate_fs(fs)
    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("order must be a positive integer")
    cutoff_array = np.atleast_1d(np.asarray(cutoff, dtype=float))
    if not np.all(np.isfinite(cutoff_array)) or np.any(cutoff_array <= 0) or np.any(cutoff_array >= fs / 2):
        raise ValueError("cutoff frequencies must be strictly within (0, Nyquist)")
    if cutoff_array.size == 2 and cutoff_array[0] >= cutoff_array[1]:
        raise ValueError("low_hz must be less than high_hz")
    data = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(data)):
        raise ValueError("signal must be finite; clean missing data before filtering")
    if data.ndim == 0:
        raise ValueError("signal must have a sample axis")
    sos = scipy_signal.butter(order, cutoff, btype=kind, fs=fs, output="sos")
    try:
        return scipy_signal.sosfiltfilt(sos, data, axis=axis)
    except ValueError as exc:
        raise ValueError("signal is too short for zero-phase padding; add samples or use a different filter") from exc
