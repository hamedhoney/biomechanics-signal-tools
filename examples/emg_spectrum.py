"""Synthetic EMG-style demonstration: uv run python examples/emg_spectrum.py"""
import numpy as np
from biomechanics_signal_tools.signal import filtering, frequency_analysis

fs = 1000
rng = np.random.default_rng(2)
seconds = np.arange(0, 5, 1/fs)
# This is deliberately synthetic and does not model a biologically plausible EMG spectrum.
raw = (np.sin(2*np.pi*80*seconds) + 0.3*np.sin(2*np.pi*140*seconds)
       + 0.3*np.sin(2*np.pi*3*seconds) + 0.1*rng.standard_normal(seconds.size))
bandpassed = filtering.butterworth_bandpass(raw, 20, 250, fs)
rectified = np.abs(bandpassed)
envelope = filtering.butterworth_lowpass(rectified, 6, fs)
f, psd = frequency_analysis.power_spectral_density(bandpassed, fs, detrend="constant")
print("Peak frequency (Hz):", frequency_analysis.dominant_frequency(bandpassed, fs))
print("Median frequency (Hz):", round(float(frequency_analysis.spectral_percentile_frequency(f, psd, 0.5)), 1))
print("Rectified envelope mean (arbitrary units):", round(float(np.mean(envelope)), 3))
