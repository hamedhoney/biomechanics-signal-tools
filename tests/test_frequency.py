import numpy as np
import pytest
from scipy.signal import periodogram

from biomechanics_signal_tools.signal import frequency_analysis as f


@pytest.mark.parametrize('samples', [99, 100, 101, 102])
def test_psd_matches_scipy_and_variance(samples):
    fs = 200
    t = np.arange(samples) / fs
    x = np.sin(2*np.pi*8*t) + 0.17*np.cos(2*np.pi*33*t)
    freqs, density = f.power_spectral_density(x, fs)
    scipy_f, scipy_d = periodogram(x, fs=fs, detrend=None)
    np.testing.assert_allclose(freqs, scipy_f)
    np.testing.assert_allclose(density, scipy_d, rtol=1e-12)
    np.testing.assert_allclose(np.sum(density) * fs / samples, np.mean(x*x), rtol=1e-12)


def test_multiple_channels_and_axis():
    fs = 200
    time = np.arange(200) / fs
    y = np.stack((np.sin(2*np.pi*8*time), np.sin(2*np.pi*29*time)), axis=-1)
    freq, psd = f.power_spectral_density(y, fs, axis=0)
    assert psd.shape == (freq.size, 2)
    np.testing.assert_allclose(f.dominant_frequency(y, fs, axis=0), [8., 29.])
    medians = f.median_frequency(y, fs, axis=0)
    assert medians.shape == (2,)
    np.testing.assert_allclose(medians, [8., 29.], atol=1)


def test_percentile_works_with_nd_and_zero_power():
    frequencies = np.arange(5., dtype=float)
    density = np.array([[1., 1., 1., 1., 1.], [0, 0, 0, 0, 0]])
    result = f.spectral_percentile_frequency(frequencies, density, 0.5)
    assert result[0] == pytest.approx(2.)
    assert np.isnan(result[1])


@pytest.mark.parametrize('invalid_p', [-0.1, 1.1])
def test_percentile_validates_input(invalid_p):
    with pytest.raises(ValueError):
        f.spectral_percentile_frequency([0, 1, 2], [0, 1, 0], invalid_p)


def test_psd_rejects_nonfinite_signal_and_invalid_fs():
    with pytest.raises(ValueError):
        f.power_spectral_density([1, np.nan, 2], 100)
    with pytest.raises(ValueError):
        f.power_spectral_density([1, 2, 3], -100)


def test_median_frequency_zero_signal_is_nan():
    assert np.isnan(f.median_frequency(np.zeros(40), fs=100))
