import numpy as np
from biomechanics_signal_tools.signal import filtering, frequency_analysis, processing
from biomechanics_signal_tools.rotation import quaternions, geometry

def test_psd_peak():
    fs = 200
    t = np.arange(200) / fs
    f, psd = frequency_analysis.power_spectral_density(np.sin(2*np.pi*10*t), fs)
    assert f[np.argmax(psd)] == 10

def test_differentiation():
    t, dy = processing.differentiate(np.array([0., 1., 2.]), np.array([0., 1., 4.]))
    np.testing.assert_allclose(dy, [1, 3])

def test_rotation():
    q = quaternions.quaternion_from_axis_angle(np.pi/2, [0, 0, 1])
    np.testing.assert_allclose(quaternions.rotate_vectors(q, np.array([1., 0., 0.])), [0, 1, 0], atol=1e-12)
