"""Type-annotated usage example, runnable with ``uv run python examples/typed_usage.py``."""
import numpy as np

from biomechanics_signal_tools.types import ArrayLike, FloatArray
from biomechanics_signal_tools.signal import filtering, processing
from biomechanics_signal_tools.rotation import quaternions


def filter_force_plate(force: ArrayLike, sampling_rate: float) -> FloatArray:
    """Accept lists/tuples/arrays; consistently return float64 force samples."""
    return filtering.butterworth_lowpass(force, cutoff_hz=12.0, fs=sampling_rate)


def compute_force_rate(time: ArrayLike, force: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Return (time midpoints in seconds, force derivative in N/s)."""
    return processing.differentiate(time, force)


if __name__ == '__main__':
    time = np.arange(0, 1, 1 / 200)
    force = [500.0 + 40.0 * np.sin(2 * np.pi * 3 * t) for t in time]
    filtered = filter_force_plate(force, 200)
    midpoints, rate = compute_force_rate(time, filtered)
    quat = quaternions.quaternion_from_axis_angle(np.pi / 2, [0, 0, 1])
    rotated = quaternions.rotate_vectors(quat, [1, 0, 0])
    print(f"Filtered force dtype: {filtered.dtype}, force rate shape: {rate.shape}")
    print(f"Time midpoints shape: {midpoints.shape}, rotated vector: {rotated}")
