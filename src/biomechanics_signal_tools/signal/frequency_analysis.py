"""One-sided power spectra and interpretable summary frequencies."""
from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
from scipy import signal

from ..types import ArrayLike, FloatArray, FloatResult

Detrend = Optional[Union[str, Callable[[FloatArray], FloatArray]]]


def power_spectral_density(
    values: ArrayLike,
    fs: float = 1.0,
    *,
    axis: int = -1,
    detrend: Detrend = None,
) -> tuple[FloatArray, FloatArray]:
    """Return ``(frequency_hz, PSD)`` for a real uniformly sampled series.

    Uses SciPy's one-sided *density*-scaled periodogram. Even and odd lengths
    are handled correctly (only an even-length FFT has a Nyquist singleton).
    Output units are input_units^2/Hz. NaNs are not supported; remove artifacts
    and detrend or window signals as appropriate before spectral inference.
    """
    data = np.asarray(values, dtype=float)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be positive and finite")
    if data.ndim == 0 or data.shape[axis] < 2:
        raise ValueError("PSD requires at least two samples")
    if not np.all(np.isfinite(data)):
        raise ValueError("signal must contain only finite values")
    return signal.periodogram(data, fs=fs, axis=axis, detrend=detrend,
                              window="boxcar", scaling="density",
                              return_onesided=True)


def spectral_percentile_frequency(
    frequencies: ArrayLike,
    psd: ArrayLike,
    percentile: float,
    *,
    axis: int = -1,
) -> FloatArray:
    """Frequency below which ``percentile`` (fraction in [0, 1]) of power lies.

    Integrates the piecewise-linear PSD using trapezoids. Accepts any number
    of independent channels, with the frequency dimension selected by axis.
    A zero-power channel returns NaN.
    """
    f = np.asarray(frequencies, dtype=float)
    p = np.asarray(psd, dtype=float)
    if f.ndim != 1 or f.size < 2 or not np.all(np.isfinite(f)) or np.any(np.diff(f) <= 0):
        raise ValueError("frequencies must be finite and strictly increasing")
    if not 0 <= percentile <= 1:
        raise ValueError("percentile must be a fraction between 0 and 1")
    if p.ndim == 0 or p.shape[axis] != f.size or not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("PSD must be finite, nonnegative and match frequency axis")
    matrix = np.moveaxis(p, axis, -1)
    widths = np.diff(f)
    areas = (matrix[..., :-1] + matrix[..., 1:]) * widths / 2
    cdf = np.concatenate([np.zeros_like(matrix[..., :1]), np.cumsum(areas, axis=-1)], axis=-1)
    flat = cdf.reshape(-1, f.size)
    result = np.full(flat.shape[0], np.nan)
    for idx, row in enumerate(flat):
        total = row[-1]
        if total > 0:
            # np.interp accepts repeated cdf values; plateau locations don't
            # affect interior percentiles, and endpoints are explicitly set.
            if percentile == 0:
                result[idx] = f[0]
            elif percentile == 1:
                result[idx] = f[-1]
            else:
                target = percentile * total
                k = min(max(int(np.searchsorted(row, target, side="right")) - 1, 0), f.size - 2)
                while k < f.size - 2 and row[k + 1] <= row[k]:
                    k += 1
                result[idx] = f[k] + (f[k + 1] - f[k]) * (target - row[k]) / (row[k + 1] - row[k])
    return result.reshape(matrix.shape[:-1])


def median_frequency(
    values: ArrayLike,
    fs: float = 1.0,
    *,
    axis: int = -1,
    detrend: Detrend = None,
) -> FloatArray:
    """50th-percentile frequency of a one-sided PSD, in hertz."""
    frequencies, density = power_spectral_density(values, fs, axis=axis, detrend=detrend)
    return spectral_percentile_frequency(frequencies, density, 0.5, axis=axis)


def dominant_frequency(
    values: ArrayLike,
    fs: float = 1.0,
    *,
    axis: int = -1,
    detrend: Detrend = "constant",
    include_dc: bool = False,
) -> FloatResult:
    """Frequency of the PSD maximum (Hz), excluding DC unless requested."""
    f, density = power_spectral_density(values, fs, axis=axis, detrend=detrend)
    moved = np.moveaxis(density, axis, -1).copy()
    if not include_dc:
        moved[..., 0] = -np.inf
    return f[np.argmax(moved, axis=-1)]
