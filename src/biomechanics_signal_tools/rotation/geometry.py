"""3-D vector operations and a lightweight, Rotation convenience API."""
from __future__ import annotations

from typing import Union

from ..types import AnyNDArray, ArrayLike, FloatArray

import numpy as np


def add_leading_dimensions(n: int, *arrays: ArrayLike) -> tuple[AnyNDArray, ...]:
    """Pad leading singleton axes until every input has ``ndim >= n``."""
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    return tuple(np.reshape(x, (1,) * max(0, n - np.ndim(x)) + np.shape(x)) for x in arrays)


def broadcast_leading_dimensions(*arrays: ArrayLike) -> tuple[AnyNDArray, ...]:
    """Add singleton leading dimensions to match input array ranks.

    Does *not* broadcast actual values; numpy broadcast rules apply later.
    """
    if not arrays:
        return ()
    return add_leading_dimensions(max(np.ndim(x) for x in arrays), *arrays)


def vector_dot_product(v1: ArrayLike, v2: ArrayLike) -> Union[AnyNDArray, np.generic]:
    """Vector dot product along the final coordinate dimension."""
    a, b = np.asarray(v1), np.asarray(v2)
    if a.ndim == 0 or b.ndim == 0 or a.shape[-1] != b.shape[-1]:
        raise ValueError("vectors must have matching final dimensions")
    return np.sum(a * b, axis=-1)


class Rotation:
    """Simple orientation object backed by a normalized scalar-first quaternion.

    The object stores ``data`` as an independent array. Composition ``a @ b``
    applies ``b`` first and then ``a``. Use ``apply`` for vector rotation.
    """

    def __init__(self, data: ArrayLike) -> None:
        from .quaternions import normalize_quaternion
        self.data = normalize_quaternion(data).copy()

    @classmethod
    def from_euler(cls, angles: ArrayLike, seq: str = "xyz", degrees: bool = False) -> Rotation:
        from .quaternions import quaternion_from_euler
        return cls(quaternion_from_euler(angles, seq=seq, degrees=degrees))

    @classmethod
    def from_matrix(cls, matrix: ArrayLike) -> Rotation:
        from .quaternions import quaternion_from_matrix
        return cls(quaternion_from_matrix(matrix))

    @classmethod
    def from_axis_angle(cls, angle: ArrayLike, axis: ArrayLike) -> Rotation:
        from .quaternions import quaternion_from_axis_angle
        return cls(quaternion_from_axis_angle(angle, axis))

    def apply(self, vector: ArrayLike) -> FloatArray:
        from .quaternions import rotate_vectors
        return rotate_vectors(self.data, vector)

    def inverse(self) -> Rotation:
        from .quaternions import quaternion_inverse
        return type(self)(quaternion_inverse(self.data))

    def as_matrix(self) -> FloatArray:
        from .quaternions import quaternion_to_matrix
        return quaternion_to_matrix(self.data)

    def as_euler(self, seq: str = "xyz", degrees: bool = False) -> FloatArray:
        from .quaternions import quaternion_to_euler
        return quaternion_to_euler(self.data, seq=seq, degrees=degrees)

    def __matmul__(self, other: object) -> Rotation:
        from .quaternions import multiply_quaternions
        if not isinstance(other, Rotation):
            return NotImplemented
        return type(self)(multiply_quaternions(self.data, other.data))

    def __repr__(self) -> str:
        return f"Rotation(data={self.data!r})"
