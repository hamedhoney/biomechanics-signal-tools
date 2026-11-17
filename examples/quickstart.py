"""Quick start: uv run python examples/quickstart.py"""
import numpy as np
from biomechanics_signal_tools.signal import filtering, frequency_analysis, processing
from biomechanics_signal_tools.rotation import quaternions

fs = 200
seconds = np.arange(0, 2, 1 / fs)
force = 100 + 20*np.sin(2*np.pi*5*seconds) + 4*np.sin(2*np.pi*45*seconds)
clean = filtering.butterworth_lowpass(force, 12, fs)
print("Dominant frequency (Hz):", frequency_analysis.dominant_frequency(clean, fs))

midpoints, derivative = processing.differentiate(seconds, clean)
print("Force rate (N/s), first 3:", derivative[:3])

q = quaternions.quaternion_from_axis_angle(np.pi/2, [0, 0, 1])
print("Rotated x unit vector:", quaternions.rotate_vectors(q, [1, 0, 0]))
