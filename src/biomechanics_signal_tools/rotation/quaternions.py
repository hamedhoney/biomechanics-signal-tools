"""Validated scalar-first quaternion and active 3-D orientation operations.

The public quaternion format is always [w, x, y, z]; scipy stores [x, y, z, w].
Array operations broadcast along batch dimensions, leaving the last axis for
quaternion/vector components. Active rotations map vectors between coordinates.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as SciPyRotation, Slerp

from ..types import ArrayLike, FloatArray, FloatResult


def _quaternion(values: ArrayLike) -> FloatArray:
    q = np.asarray(values, dtype=float)
    if q.ndim == 0 or q.shape[-1] != 4:
        raise ValueError("quaternion must have shape (..., 4), scalar first [w,x,y,z]")
    if not np.all(np.isfinite(q)):
        raise ValueError("quaternions must contain finite values")
    return q


def _vector(values: ArrayLike) -> FloatArray:
    x = np.asarray(values, dtype=float)
    if x.ndim == 0 or x.shape[-1] != 3 or not np.all(np.isfinite(x)):
        raise ValueError("vectors must have shape (..., 3) with finite values")
    return x


def _to_xyzw(q: FloatArray) -> FloatArray:
    return np.concatenate((q[..., 1:], q[..., :1]), axis=-1)


def _from_xyzw(q: FloatArray) -> FloatArray:
    return np.concatenate((q[..., -1:], q[..., :-1]), axis=-1)


def _rotation(q: ArrayLike) -> SciPyRotation:
    q = normalize_quaternion(q)
    return SciPyRotation.from_quat(_to_xyzw(q))


def normalize_quaternion(quaternion: ArrayLike) -> FloatArray:
    """Normalize one or more scalar-first quaternions; reject zero length."""
    q = _quaternion(quaternion)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("cannot normalize a zero quaternion")
    return q / norm


def quaternion_norm(quaternion: ArrayLike) -> FloatResult:
    """Euclidean norm of one or more quaternions."""
    return np.linalg.norm(_quaternion(quaternion), axis=-1)


def quaternion_conjugate(quaternion: ArrayLike) -> FloatArray:
    q = _quaternion(quaternion).copy()
    q[..., 1:] *= -1
    return q


def quaternion_inverse(quaternion: ArrayLike) -> FloatArray:
    """Algebraic inverse (conjugate / squared norm), including nonunit inputs."""
    q = _quaternion(quaternion)
    squared = np.sum(q * q, axis=-1, keepdims=True)
    if np.any(squared == 0):
        raise ValueError("cannot invert a zero quaternion")
    return quaternion_conjugate(q) / squared


def identity_quaternion(shape: tuple[int, ...] = ()) -> FloatArray:
    """Identity orientation, shape ``(*shape, 4)`` (scalar-first)."""
    out = np.zeros(tuple(shape) + (4,))
    out[..., 0] = 1
    return out


def quaternion_from_axis_angle(angle: ArrayLike, axis: ArrayLike) -> FloatArray:
    """Quaternion for a right-handed rotation by angle (radians) about axis."""
    a = np.asarray(angle, dtype=float)
    v = _vector(axis)
    if not np.all(np.isfinite(a)):
        raise ValueError("angles must be finite")
    norm = np.linalg.norm(v, axis=-1)
    a, norm = np.broadcast_arrays(a, norm)
    v = np.broadcast_to(v, a.shape + (3,))
    if np.any((norm == 0) & (a != 0)):
        raise ValueError("axis cannot be zero when angle is nonzero")
    safe_norm = np.where(norm == 0, 1.0, norm)
    xyz = (v / safe_norm[..., None]) * np.sin(a / 2)[..., None]
    return np.concatenate((np.cos(a / 2)[..., None], xyz), axis=-1)


def quaternion_to_axis_angle(quaternion: ArrayLike) -> tuple[FloatResult, FloatArray]:
    """Shortest nonnegative rotation angle in [0, pi] and unit axis.

    Identity has an arbitrary conventional axis [1,0,0].
    """
    q = normalize_quaternion(quaternion)
    q = np.where(q[..., :1] < 0, -q, q)
    vector_norm = np.linalg.norm(q[..., 1:], axis=-1)
    angle = 2 * np.arctan2(vector_norm, q[..., 0])
    axis = np.zeros_like(q[..., 1:])
    np.divide(q[..., 1:], vector_norm[..., None], out=axis,
              where=vector_norm[..., None] > 1e-14)
    axis[..., 0] = np.where(vector_norm <= 1e-14, 1.0, axis[..., 0])
    return angle, axis


def quaternion_angle(quaternion: ArrayLike, degrees: bool = False) -> FloatResult:
    angle, _ = quaternion_to_axis_angle(quaternion)
    return np.degrees(angle) if degrees else angle


def quaternion_from_rotation_vector(vector: ArrayLike) -> FloatArray:
    """Convert axis*angle rotation vector (radians) to scalar-first quaternion."""
    v = _vector(vector)
    return _from_xyzw(SciPyRotation.from_rotvec(v).as_quat())


def quaternion_to_rotation_vector(quaternion: ArrayLike) -> FloatArray:
    """Convert quaternion into shortest rotation vector in radians."""
    return _rotation(quaternion).as_rotvec()


def _euler_sequence(seq: str) -> str:
    if not isinstance(seq, str) or len(seq) != 3:
        raise ValueError("Euler sequence must have three axis letters, e.g. 'xyz'")
    seq = seq.replace("1", "x").replace("2", "y").replace("3", "z")
    if any(ch.lower() not in "xyz" for ch in seq):
        raise ValueError("Euler sequence may only contain x, y and z axes")
    # Legacy code multiplies axis rotations in their given order: qx * qy * qz.
    # This is scipy's intrinsic, uppercase Euler convention.
    return seq.upper()


def quaternion_from_euler(angles: ArrayLike, seq: str = "xyz", degrees: bool = False) -> FloatArray:
    """Intrinsic Euler rotations in specified sequence, e.g. 'xyz' (radians)."""
    a = np.asarray(angles, dtype=float)
    if a.ndim == 0 or a.shape[-1] != 3 or not np.all(np.isfinite(a)):
        raise ValueError("angles must have shape (..., 3) and be finite")
    return _from_xyzw(SciPyRotation.from_euler(_euler_sequence(seq), a, degrees=degrees).as_quat())


def quaternion_to_euler(
    quaternion: ArrayLike,
    seq: str = "xyz",
    degrees: bool = False,
) -> FloatArray:
    """Extract intrinsic Euler angles; gimbal lock may make angles ambiguous."""
    return _rotation(quaternion).as_euler(_euler_sequence(seq), degrees=degrees)


def quaternion_from_matrix(matrix: ArrayLike) -> FloatArray:
    """Convert active 3x3 orientation matrix/matrices to scalar-first quaternion."""
    m = np.asarray(matrix, dtype=float)
    if m.shape[-2:] != (3, 3) or not np.all(np.isfinite(m)):
        raise ValueError("matrix must have shape (..., 3, 3) with finite entries")
    gram = np.matmul(np.swapaxes(m, -1, -2), m)
    if not np.allclose(gram, np.eye(3), atol=1e-5) or not np.allclose(np.linalg.det(m), 1, atol=1e-5):
        raise ValueError("matrix must be orthonormal with determinant +1")
    return _from_xyzw(SciPyRotation.from_matrix(m).as_quat())


def quaternion_to_matrix(quaternion: ArrayLike) -> FloatArray:
    """Active rotation matrix mapping v to R @ v."""
    return _rotation(quaternion).as_matrix()


def multiply_quaternions(q1: ArrayLike, q2: ArrayLike) -> FloatArray:
    """Hamilton product q1*q2: apply q2 first, then q1."""
    a = _quaternion(q1)
    b = _quaternion(q2)
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack((aw*bw - ax*bx - ay*by - az*bz,
                     aw*bx + ax*bw + ay*bz - az*by,
                     aw*by - ax*bz + ay*bw + az*bx,
                     aw*bz + ax*by - ay*bx + az*bw), axis=-1)


def right_divide_quaternions(q1: ArrayLike, q2: ArrayLike) -> FloatArray:
    return multiply_quaternions(q1, quaternion_inverse(q2))


def left_divide_quaternions(q1: ArrayLike, q2: ArrayLike) -> FloatArray:
    return multiply_quaternions(quaternion_inverse(q1), q2)


def rotate_vectors(quaternion: ArrayLike, vectors: ArrayLike) -> FloatArray:
    """Active right-handed rotation of vectors, supporting numpy batch broadcasting."""
    q = normalize_quaternion(quaternion)
    v = _vector(vectors)
    t = 2 * np.cross(q[..., 1:], v)
    return v + q[..., :1] * t + np.cross(q[..., 1:], t)


def rotate_reference_frame(quaternion: ArrayLike, vectors: ArrayLike) -> FloatArray:
    """Passive coordinate change (inverse active rotation)."""
    return rotate_vectors(quaternion_conjugate(normalize_quaternion(quaternion)), vectors)


def positive_scalar_quaternion(quaternion: ArrayLike) -> FloatArray:
    """Canonicalize quaternion sign (nonnegative scalar component)."""
    q = _quaternion(quaternion)
    return np.where(q[..., :1] < 0, -q, q)


def quaternion_power(quaternion: ArrayLike, exponent: ArrayLike) -> FloatArray:
    """Raise a unit rotation quaternion to a real power via rotation vectors."""
    v = quaternion_to_rotation_vector(quaternion)
    p = np.asarray(exponent, dtype=float)
    if not np.all(np.isfinite(p)):
        raise ValueError("exponent must be finite")
    return quaternion_from_rotation_vector(v * p[..., None])


def interpolate_quaternions(
    sample_times: ArrayLike,
    known_times: ArrayLike,
    known_quaternions: ArrayLike,
) -> FloatArray:
    """Shortest-path SLERP over time; outside known interval returns NaNs.

    ``known_quaternions`` must have shape (n_known, 4). Interpolated samples
    have shape (n_requested, 4). Data are expected in scalar-first format.
    """
    samples = np.asarray(sample_times, dtype=float)
    t = np.asarray(known_times, dtype=float)
    q = _quaternion(known_quaternions)
    if (samples.ndim != 1 or t.ndim != 1 or t.size != q.shape[0] or
            q.ndim != 2 or not np.all(np.isfinite(samples)) or
            not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0)):
        raise ValueError("expected sorted distinct 1-D times and (n_known,4) quaternions")
    if t.size < 2:
        raise ValueError("SLERP needs at least two known orientations")
    out = np.full((samples.size, 4), np.nan)
    inside = (samples >= t[0]) & (samples <= t[-1])
    if np.any(inside):
        rot = SciPyRotation.from_quat(_to_xyzw(normalize_quaternion(q)))
        out[inside] = _from_xyzw(Slerp(t, rot)(samples[inside]).as_quat())
    return out


def interpolate_quaternion_sequence(q1: ArrayLike, q2: ArrayLike, num: int) -> FloatArray:
    """Generate ``num`` shortest-path SLERP orientations including endpoints."""
    if not isinstance(num, (int, np.integer)) or num < 2:
        raise ValueError("num must be an integer >= 2")
    a = normalize_quaternion(q1)
    b = normalize_quaternion(q2)
    if a.shape != (4,) or b.shape != (4,):
        raise ValueError("both endpoint quaternions must have shape (4,)")
    result = interpolate_quaternions(np.linspace(0, 1, num),
                                    np.array([0., 1.]), np.stack((a, b)))
    result[0], result[-1] = a, b
    return result


def fill_missing_quaternions(quaternions: ArrayLike) -> FloatArray:
    """SLERP internal NaN gaps; hold nearest endpoint on leading/trailing gaps.

    Supports shape (..., n_samples, 4). All-missing series remain NaN and the
    input is never changed.
    """
    q = np.asarray(quaternions, dtype=float)
    if q.ndim < 2 or q.shape[-1] != 4:
        raise ValueError("expected quaternion time series of shape (..., n, 4)")
    if np.any(np.isinf(q)):
        raise ValueError("quaternions cannot contain infinities")
    out = q.copy().reshape((-1, q.shape[-2], 4))
    for row in out:
        good = np.all(np.isfinite(row), axis=-1)
        if not np.any(good):
            continue
        if np.count_nonzero(good) == 1:
            row[:] = normalize_quaternion(row[good][0])
            continue
        idx = np.arange(row.shape[0])
        start, stop = idx[good][0], idx[good][-1]
        original = row[good].copy()
        row[start:stop + 1] = interpolate_quaternions(idx[start:stop + 1], idx[good], original)
        row[:start] = row[start]
        row[stop + 1:] = row[stop]
    return out.reshape(q.shape)


def mean_quaternion(quaternions: ArrayLike, weights: Optional[ArrayLike] = None) -> FloatArray:
    """Weighted Markley quaternion average, invariant to q vs -q.

    Supports ``(..., n_samples, 4)`` input. ``weights`` can be 1-D with one
    weight per sample; they must be nonnegative and not all zero.
    """
    q = normalize_quaternion(quaternions)
    if q.ndim < 2:
        raise ValueError("expected an array with n_samples and four components")
    if q.shape[-2] == 0:
        raise ValueError("cannot average an empty quaternion series")
    w = np.ones(q.shape[-2]) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (q.shape[-2],) or not np.all(np.isfinite(w)) or np.any(w < 0) or not np.any(w > 0):
        raise ValueError("weights must be finite nonnegative vector of length n_samples")
    matrix = np.einsum("...ni,...nj,n->...ij", q, q, w)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    result = eigenvectors[..., -1]
    return positive_scalar_quaternion(result)


def rotation_matrix_from_unit_vectors(
    *,
    x: Optional[ArrayLike] = None,
    y: Optional[ArrayLike] = None,
    z: Optional[ArrayLike] = None,
    strict: Optional[str] = None,
) -> FloatArray:
    """Right-handed ACTIVE rotation matrix with local axis vectors as columns.

    Specify any two of x, y, z as arrays with last dimension 3. ``strict``
    optionally selects which of the two axes must be preserved exactly; the
    other is orthogonalized via Gram-Schmidt. Nearly collinear axes are rejected.
    """
    known = {key: _vector(v) for key, v in (("x", x), ("y", y), ("z", z)) if v is not None}
    if len(known) != 2:
        raise ValueError("provide exactly two of x, y, z")
    first = strict if strict is not None else next(iter(known))
    if first not in known:
        raise ValueError("strict must name one of the provided axes")
    second = next(key for key in known if key != first)
    a, b = np.broadcast_arrays(known[first], known[second])
    na = np.linalg.norm(a, axis=-1, keepdims=True)
    if np.any(na <= 1e-12):
        raise ValueError("axis vector must be nonzero")
    a = a / na
    b = b - np.sum(a * b, axis=-1, keepdims=True) * a
    nb = np.linalg.norm(b, axis=-1, keepdims=True)
    if np.any(nb <= 1e-12):
        raise ValueError("axis vectors must not be collinear")
    b = b / nb
    axes = {first: a, second: b}
    missing = next(key for key in "xyz" if key not in axes)
    if missing == "z":
        axes["z"] = np.cross(axes["x"], axes["y"])
    elif missing == "x":
        axes["x"] = np.cross(axes["y"], axes["z"])
    else:
        axes["y"] = np.cross(axes["z"], axes["x"])
    return np.stack((axes["x"], axes["y"], axes["z"]), axis=-1)


def quaternion_differences(
    quaternions: ArrayLike,
    n: int = 1,
    *,
    axis: int = -2,
    real_positive: bool = True,
) -> FloatArray:
    """Relative sample-to-sample rotation inv(q[t])*q[t+1]."""
    q = normalize_quaternion(quaternions)
    if n != 1:
        raise ValueError("only first quaternion differences are supported")
    ax = axis % q.ndim
    if ax == q.ndim - 1:
        raise ValueError("time axis cannot be the quaternion component axis")
    first = np.take(q, range(q.shape[ax] - 1), axis=ax)
    second = np.take(q, range(1, q.shape[ax]), axis=ax)
    result = left_divide_quaternions(first, second)
    return positive_scalar_quaternion(result) if real_positive else result


def cumulative_quaternion_product(
    quaternions: ArrayLike,
    n: int = 1,
    *,
    axis: int = -2,
    initial: Optional[ArrayLike] = None,
    real_positive: bool = True,
) -> FloatArray:
    """Cumulative Hamilton products of sample increments, including first once.

    If ``initial`` is supplied, each cumulative product is prefixed by that
    orientation and the output retains the original number of samples.
    """
    q = normalize_quaternion(quaternions)
    if n != 1:
        raise ValueError("only first cumulative products are supported")
    ax = axis % q.ndim
    if ax == q.ndim - 1:
        raise ValueError("time axis cannot be the quaternion component axis")
    moved = np.moveaxis(q, ax, -2)
    out = np.empty_like(moved)
    running = identity_quaternion(moved.shape[:-2]) if initial is None else normalize_quaternion(initial)
    for index in range(moved.shape[-2]):
        running = multiply_quaternions(running, moved[..., index, :])
        if real_positive:
            running = positive_scalar_quaternion(running)
        out[..., index, :] = running
    return np.moveaxis(out, -2, ax)


def random_unit_quaternion(
    shape: tuple[int, ...] = (),
    *,
    angle: Optional[ArrayLike] = None,
    rng: Optional[np.random.Generator] = None,
) -> FloatArray:
    """Sample random unit orientations; optional fixed rotation angle (rad)."""
    generator = np.random.default_rng() if rng is None else rng
    if angle is None:
        return _from_xyzw(SciPyRotation.random(num=int(np.prod(shape)) if shape else 1,
                                               random_state=generator).as_quat().reshape(tuple(shape) + (4,)))
    vectors = generator.normal(size=tuple(shape) + (3,))
    vectors = vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)
    return quaternion_from_axis_angle(angle, vectors)


def quaternion_from_unit_vectors(
    xunit: ArrayLike,
    yunit: ArrayLike,
    zunit: Optional[ArrayLike] = None,
    *,
    strict: str = "x",
) -> FloatArray:
    """Create quaternion from orthogonalized active basis vectors.

    The third vector is optional and, when supplied, must agree with the
    right-handed basis computed from x/y. For a basis with other two axes,
    use ``rotation_matrix_from_unit_vectors`` then ``quaternion_from_matrix``.
    """
    matrix = rotation_matrix_from_unit_vectors(x=xunit, y=yunit, strict=strict)
    if zunit is not None:
        z = _vector(zunit)
        znorm = np.linalg.norm(z, axis=-1, keepdims=True)
        if np.any(znorm <= 1e-12) or not np.allclose(z / znorm, matrix[..., :, 2], atol=1e-5):
            raise ValueError("zunit is inconsistent with the right-handed x/y basis")
    return quaternion_from_matrix(matrix)


def filter_quaternions(quaternions: ArrayLike, weights: ArrayLike) -> FloatArray:
    """Centered moving Markley average of orientations on the time axis (-2).

    Nonnegative 1-D weights define the finite-window smoothing kernel.
    Boundary windows are truncated and weights renormalized implicitly.
    This is a noncausal smoother; not a frequency-calibrated low-pass filter.
    """
    q = normalize_quaternion(quaternions)
    w = np.asarray(weights, dtype=float)
    if q.ndim < 2 or w.ndim != 1 or w.size == 0 or w.size > q.shape[-2]:
        raise ValueError("expected (..., samples, 4) and a 1-D window no longer than samples")
    if not np.all(np.isfinite(w)) or np.any(w < 0) or np.sum(w) <= 0:
        raise ValueError("window weights must be finite, nonnegative and sum to >0")
    result = np.empty_like(q)
    half = w.size // 2
    for i in range(q.shape[-2]):
        left = max(0, i - half)
        right = min(q.shape[-2], i - half + w.size)
        from_w = left - (i - half)
        result[..., i, :] = mean_quaternion(q[..., left:right, :],
                                           w[from_w:from_w + (right - left)])
    return result


def swing_twist_decomposition(
    quaternion: ArrayLike,
    axis: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Return (twist, swing) where q = twist * swing.

    Twist is about ``axis``. Undefined at the exact singularity where the
    twist's real and axial components both vanish.
    """
    q = normalize_quaternion(quaternion)
    direction = _vector(axis)
    norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("twist axis cannot be zero")
    direction = direction / norm
    projection = np.sum(q[..., 1:] * direction, axis=-1, keepdims=True)
    tw = np.concatenate((q[..., :1], direction * projection), axis=-1)
    twist = normalize_quaternion(tw)
    return twist, left_divide_quaternions(twist, q)


__all__ = [
    "normalize_quaternion", "quaternion_norm", "quaternion_conjugate",
    "quaternion_inverse", "identity_quaternion", "quaternion_from_axis_angle",
    "quaternion_to_axis_angle", "quaternion_angle", "quaternion_from_rotation_vector",
    "quaternion_to_rotation_vector", "quaternion_from_euler", "quaternion_to_euler",
    "quaternion_from_matrix", "quaternion_to_matrix", "multiply_quaternions",
    "right_divide_quaternions", "left_divide_quaternions", "rotate_vectors",
    "rotate_reference_frame", "positive_scalar_quaternion", "quaternion_power",
    "interpolate_quaternions", "interpolate_quaternion_sequence",
    "fill_missing_quaternions", "mean_quaternion", "rotation_matrix_from_unit_vectors",
    "quaternion_differences", "cumulative_quaternion_product", "random_unit_quaternion",
    "quaternion_from_unit_vectors", "filter_quaternions", "swing_twist_decomposition",
]
