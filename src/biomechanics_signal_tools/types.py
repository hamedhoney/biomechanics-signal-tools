"""Shared NumPy-friendly type aliases for the public API.

Use ``ArrayLike`` for arguments normalized via ``np.asarray`` and
``FloatArray`` for arrays guaranteed to have ``float64`` dtype. Dimension
conventions (e.g. ``(samples, 3)`` vectors, ``(..., 4)`` quaternions) are
validated at runtime: standard NumPy annotations do not enforce shape.

``AnyNDArray`` is intentionally reserved for dtype-preserving helpers and
operations whose result dtype cannot be guaranteed.
"""
from typing import Any, Union

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

ArrayLike: TypeAlias = npt.ArrayLike
FloatArray: TypeAlias = npt.NDArray[np.float64]
FloatingArray: TypeAlias = npt.NDArray[np.floating[Any]]
IntArray: TypeAlias = npt.NDArray[np.integer[Any]]
BoolArray: TypeAlias = npt.NDArray[np.bool_]
AnyNDArray: TypeAlias = npt.NDArray[Any]
FloatResult: TypeAlias = Union[FloatArray, np.float64]
SignalPair: TypeAlias = tuple[ArrayLike, ArrayLike]

__all__ = [
    "ArrayLike", "FloatArray", "FloatingArray", "IntArray", "BoolArray",
    "AnyNDArray", "FloatResult", "SignalPair",
]
