import numpy as np
import pytest

from biomechanics_signal_tools.signal import filtering as f


def test_butterworth_lowpass_attenuates_high_frequency():
    fs = 200
    t = np.arange(0, 3, 1/fs)
    low = np.sin(2 * np.pi * 3 * t)
    high = np.sin(2 * np.pi * 45 * t)
    result = f.butterworth_lowpass(low + high, 10, fs)
    region = slice(100, -100)
    assert np.sqrt(np.mean((result[region] - low[region])**2)) < 0.08


def test_bandpass_detects_expected_frequency():
    fs = 500
    t = np.arange(0, 4, 1 / fs)
    wanted = np.sin(2 * np.pi * 65 * t)
    contaminant = np.sin(2 * np.pi * 5 * t)
    result = f.butterworth_bandpass(wanted + contaminant, 40, 100, fs)
    assert np.std(result[200:-200] - wanted[200:-200]) < 0.1


def test_sos_filter_preserves_multichannel_shape():
    x = np.ones((2, 400))
    assert f.butterworth_lowpass(x, 12, 100).shape == x.shape


@pytest.mark.parametrize('cutoff', [0, -1, 50, 90])
def test_filter_rejects_bad_cutoff(cutoff):
    with pytest.raises(ValueError):
        f.butterworth_lowpass(np.arange(100), cutoff, 100)


def test_window_filter_preserves_constant_even_at_edges():
    data = np.ones((2, 100)) * np.array([[3], [5]])
    result = f.apply_window_filter(data, [1, 2, 1], axis=1)
    np.testing.assert_allclose(result, data)


def test_flat_top_window_normalized():
    w = f.flat_top_window(fc=5, fs=100)
    assert w.ndim == 1
    assert np.isclose(np.sum(w), 1)


def test_window_rejects_oversized_kernel():
    with pytest.raises(ValueError):
        f.apply_window_filter([1, 2], [1, 2, 3])
