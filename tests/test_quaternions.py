import numpy as np
import pytest
from scipy.spatial.transform import Rotation as SciPyRotation

from biomechanics_signal_tools.rotation import Rotation, quaternions as q


def assert_same_rotation(a, b, atol=1e-10):
    aa = q.normalize_quaternion(a)
    bb = q.normalize_quaternion(b)
    np.testing.assert_allclose(np.abs(np.sum(aa * bb, axis=-1)), 1., atol=atol)


def test_quaternion_norm_and_inverse_for_nonunit_quaternion():
    raw = np.array([2., 1., 2., 0.])
    inverse = q.quaternion_inverse(raw)
    np.testing.assert_allclose(q.multiply_quaternions(raw, inverse), [1, 0, 0, 0], atol=1e-12)
    assert q.quaternion_norm(raw) == pytest.approx(3.)


def test_normalize_rejects_zero_quaternion():
    with pytest.raises(ValueError):
        q.normalize_quaternion([0, 0, 0, 0])


def test_quaternion_from_axis_angle_rotates_vector_and_preserves_norm():
    orient = q.quaternion_from_axis_angle(np.pi/2, [0, 0, 5])
    out = q.rotate_vectors(orient, [1, 0, 0])
    np.testing.assert_allclose(out, [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(q.quaternion_to_axis_angle(orient)[0], np.pi/2)


def test_identity_axis_angle_and_rotvec():
    quat = q.quaternion_from_rotation_vector([0, 0, 0])
    np.testing.assert_allclose(quat, [1, 0, 0, 0])
    angle, axis = q.quaternion_to_axis_angle(quat)
    assert angle == 0
    np.testing.assert_allclose(axis, [1, 0, 0])
    np.testing.assert_allclose(q.quaternion_to_rotation_vector(quat), [0, 0, 0])


def test_rotate_vectors_broadcast_batch():
    orient = q.quaternion_from_axis_angle([0, np.pi/2], [[0, 0, 1], [0, 0, 1]])
    vectors = np.array([1., 0, 0])
    np.testing.assert_allclose(q.rotate_vectors(orient, vectors),
                               [[1, 0, 0], [0, 1, 0]], atol=1e-12)


def test_normalizes_nonunit_before_rotation():
    orient = q.quaternion_from_axis_angle(np.pi/2, [0, 0, 1])
    np.testing.assert_allclose(q.rotate_vectors(orient * 3, [1, 0, 0]),
                               [0, 1, 0], atol=1e-12)


def test_matrix_roundtrip_against_scipy_rotation():
    q0 = q.quaternion_from_euler([15, -20, 45], seq="xyz", degrees=True)
    matrix = q.quaternion_to_matrix(q0)
    assert_same_rotation(q.quaternion_from_matrix(matrix), q0)
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(matrix), 1., atol=1e-12)


def test_reject_nonorthogonal_matrix():
    with pytest.raises(ValueError):
        q.quaternion_from_matrix(np.diag([1, 2, 1]))


def test_euler_intrinsic_matches_original_product():
    angle = np.array([0.2, 0.3, 0.4])
    expected = q.multiply_quaternions(
        q.multiply_quaternions(q.quaternion_from_axis_angle(angle[0], [1, 0, 0]),
                               q.quaternion_from_axis_angle(angle[1], [0, 1, 0])),
        q.quaternion_from_axis_angle(angle[2], [0, 0, 1]))
    assert_same_rotation(q.quaternion_from_euler(angle, "xyz"), expected)
    recovered = q.quaternion_to_euler(expected, "xyz")
    np.testing.assert_allclose(recovered, angle, atol=1e-12)


def test_shortest_path_slerp_and_outside_nan():
    start = q.identity_quaternion()
    end = q.quaternion_from_axis_angle(np.pi, [0, 0, 1])
    interpolated = q.interpolate_quaternions([-1, 0, 0.5, 1, 2], [0, 1], [start, -end])
    assert np.isnan(interpolated[0]).all() and np.isnan(interpolated[-1]).all()
    assert_same_rotation(interpolated[2], q.quaternion_from_axis_angle(np.pi/2, [0, 0, 1]))


def test_fill_missing_quaternion_gaps_and_copy():
    start = q.identity_quaternion()
    end = q.quaternion_from_axis_angle(np.pi/2, [0, 0, 1])
    original = np.array([[np.nan] * 4, start, [np.nan]*4, end, [np.nan]*4])
    restored = q.fill_missing_quaternions(original)
    assert np.isnan(original[0]).all()
    assert_same_rotation(restored[2], q.quaternion_from_axis_angle(np.pi/4, [0, 0, 1]))
    assert_same_rotation(restored[0], start)
    assert_same_rotation(restored[-1], end)


def test_mean_quaternion_invariant_to_sign():
    orient = q.quaternion_from_axis_angle(np.pi/4, [0, 1, 0])
    result = q.mean_quaternion(np.stack((orient, -orient, orient)), [1, 1, 2])
    assert_same_rotation(result, orient)


def test_quaternion_differences_and_cumulative_product():
    frame = np.stack([q.quaternion_from_axis_angle(a, [0, 0, 1])
                      for a in [0, 0.1, 0.2, 0.3]])
    increment = q.quaternion_differences(frame)
    reconstructed = q.cumulative_quaternion_product(increment, initial=frame[0])
    assert reconstructed.shape == (3, 4)
    for a, b in zip(reconstructed, frame[1:]):
        assert_same_rotation(a, b)


def test_random_quaternions_unit_and_fixed_angle():
    generator = np.random.default_rng(4)
    random = q.random_unit_quaternion((2, 3), rng=generator)
    assert random.shape == (2, 3, 4)
    np.testing.assert_allclose(q.quaternion_norm(random), 1, atol=1e-12)
    fixed = q.random_unit_quaternion((4,), angle=0.4, rng=generator)
    np.testing.assert_allclose(q.quaternion_angle(fixed), 0.4, atol=1e-12)


def test_basis_matrix_columns_and_handedness():
    matrix = q.rotation_matrix_from_unit_vectors(x=[1, 0, 0], y=[0, 1, 0])
    np.testing.assert_allclose(matrix, np.eye(3))
    assert_same_rotation(q.quaternion_from_unit_vectors([1, 0, 0], [0, 1, 0], [0, 0, 1]),
                         q.identity_quaternion())
    with pytest.raises(ValueError):
        q.rotation_matrix_from_unit_vectors(x=[1, 0, 0], y=[2, 0, 0])


def test_swing_twist_recombines():
    orient = q.quaternion_from_euler([0.2, 0.3, 0.4])
    twist, swing = q.swing_twist_decomposition(orient, [0, 0, 1])
    assert_same_rotation(q.multiply_quaternions(twist, swing), orient)


def test_quaternion_window_smoothing_ignores_sign_flips():
    orient = q.quaternion_from_axis_angle(0.4, [1, 0, 0])
    timeline = np.stack((orient, -orient, orient, orient, -orient))
    smooth = q.filter_quaternions(timeline, np.ones(3))
    for sample in smooth:
        assert_same_rotation(sample, orient)


def test_rotation_class_compose_and_inverse():
    r = Rotation.from_axis_angle(np.pi/2, [0, 0, 1])
    transformed = r.apply([1, 0, 0])
    np.testing.assert_allclose(transformed, [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(r.inverse().apply(transformed), [1, 0, 0], atol=1e-12)
    composed = r @ r
    np.testing.assert_allclose(composed.apply([1, 0, 0]), [-1, 0, 0], atol=1e-12)
    assert isinstance(repr(r), str)
