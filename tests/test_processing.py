import numpy as np
import pytest

from biomechanics_signal_tools.signal import processing as p


def test_differentiate_multi_axis_irregular_time():
    time = np.array([0., 0.1, 0.4, 1.0])
    data = np.stack((2 * time, 3 * time), axis=-1)  # (samples, channels)
    mid, gradient = p.differentiate(time, data, axis=0)
    np.testing.assert_allclose(mid, [0.05, 0.25, 0.7])
    np.testing.assert_allclose(gradient, np.array([[2., 3.]] * 3))


@pytest.mark.parametrize('bad_time', [[0, 1, 1], [0, 2, 1], [0, np.nan, 3]])
def test_differentiate_rejects_invalid_time(bad_time):
    with pytest.raises(ValueError):
        p.differentiate(bad_time, [1, 2, 3])


def test_differentiation_interpolated_time_and_shape():
    t = np.linspace(0, 1, 11)
    y = np.stack((t**2, 3 * t), axis=0)
    out = p.differentiate_interpolated(t, y)
    assert out.shape == y.shape
    np.testing.assert_allclose(out[1], 3, atol=1e-14)
    assert np.isfinite(out).all()


def test_interpolation_multidimensional_and_nan_outside():
    t = [0, 1, 2]
    data = np.array([[0., 10.], [1., 20.], [2., 30.]])
    result = p.interpolate_along_axis([-1, 0.5, 3], t, data, axis=0)
    assert result.shape == (3, 2)
    assert np.isnan(result[0]).all() and np.isnan(result[-1]).all()
    np.testing.assert_allclose(result[1], [0.5, 15.])


def test_fill_missing_copy_max_gap_and_edges():
    raw = np.array([np.nan, 1., np.nan, 3., np.nan, np.nan, 6., np.nan])
    saved = raw.copy()
    result = p.fill_missing(raw, max_gap=1, extrapolate="nan")
    assert np.isnan(result[0]) and np.isnan(result[-1])
    assert result[2] == 2
    assert np.isnan(result[4]) and np.isnan(result[5])
    np.testing.assert_array_equal(np.isnan(raw), np.isnan(saved))
    np.testing.assert_allclose(raw[~np.isnan(raw)], saved[~np.isnan(saved)])


def test_fill_missing_all_missing_and_single_valid():
    assert np.isnan(p.fill_missing([np.nan, np.nan])).all()
    np.testing.assert_allclose(p.fill_missing([np.nan, 2., np.nan]), [2., 2., 2.])


def test_fill_missing_rejects_infinity():
    with pytest.raises(ValueError):
        p.fill_missing([np.nan, np.inf])


def test_resample_signal_and_outside():
    t, y = p.resample_signal(([0, 1, 2], [0, 2, 4]), [-1, 0.5, 1.5])
    assert t.shape == y.shape
    assert np.isnan(y[0])
    np.testing.assert_allclose(y[1:], [1, 3])


def test_delay_estimation_of_known_timestamp_shift():
    fs = 100
    t = np.arange(0, 4, 1/fs)
    waveform = np.sin(2 * np.pi * (0.7 * t + 0.12*t*t)) + 0.2 * np.cos(2*np.pi*1.9*t)
    signal1 = (t, waveform)
    # signal_2 values have identical sampling amplitudes, but timestamps are 0.12 s late.
    signal2 = (t + 0.12, waveform)
    estimated = p.estimate_time_delay(signal1, signal2, bounds=(-0.3, 0.1))
    assert estimated.success
    assert abs(estimated.x + 0.12) < 0.005
    assert estimated.fun < 1e-4


def test_delay_with_no_overlap():
    with pytest.raises(ValueError, match="no valid overlap"):
        p.estimate_time_delay(([0, 1, 2], [1, 2, 3]),
                              ([10, 11, 12], [1, 2, 3]), bounds=(0, 1))


def test_pearson_removes_offsets_and_returns_nan_at_short_overlap():
    a = np.array([2., 3., 5., 4., 9.])
    b = 10 + 3 * a
    corr = p.moving_pearson_correlation(a, b)
    lags = p.correlation_lags(a, b)
    assert len(corr) == len(lags)
    assert corr[lags == 0] == pytest.approx(1, abs=1e-12)
    assert np.isnan(corr[0]) and np.isnan(corr[-1])


def test_first_true_index():
    assert p.first_true_index([False, False, True]) == 2
    assert p.first_true_index([False, False]) == 2
    assert p.first_true_index([]) == 0
