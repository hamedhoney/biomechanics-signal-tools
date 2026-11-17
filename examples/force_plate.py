"""Synthetic vertical GRF analysis: uv run python examples/force_plate.py [--plot]"""
import sys
import numpy as np
from biomechanics_signal_tools.signal import filtering, frequency_analysis, processing

fs = 1000
seconds = np.arange(0, 3, 1/fs)
# Demo waveform only: not a measured ground-reaction force from a participant.
fz = 700 + 80*np.sin(2*np.pi*2*seconds) + 12*np.sin(2*np.pi*75*seconds)
filtered = filtering.butterworth_lowpass(fz, cutoff_hz=20, fs=fs)
time_rate, loading_rate = processing.differentiate(seconds, filtered)
print(f"Synthetic force RMS: {np.sqrt(np.mean(filtered**2)):.1f} N")
print(f"Maximum synthetic |loading rate|: {np.max(np.abs(loading_rate)):.1f} N/s")
print(f"Dominant non-DC component: {frequency_analysis.dominant_frequency(filtered, fs):.1f} Hz")

if "--plot" in sys.argv:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install optional plot extra: uv sync --extra plot") from exc
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(seconds, fz, label="synthetic raw")
    axes[0].plot(seconds, filtered, label="low-pass filtered")
    axes[0].set_ylabel("Vertical force (N)")
    axes[0].legend()
    axes[1].plot(time_rate, loading_rate)
    axes[1].set_ylabel("Force rate (N/s)")
    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig("force_plate_demo.png", dpi=150)
    print("Saved force_plate_demo.png")
