"""The typing surface is consistent and resolves at runtime."""

import inspect
from typing import get_args, get_origin, get_type_hints

import numpy as np
import numpy.typing as npt
import pytest

from biomechanics_signal_tools import __version__, types
from biomechanics_signal_tools.signal import filtering, frequency_analysis, processing
from biomechanics_signal_tools.rotation import Rotation, geometry, quaternions


def test_alias_definitions_are_numpy_native():
    assert types.ArrayLike == npt.ArrayLike
    assert types.FloatArray == npt.NDArray[np.float64]
    assert types.FloatingArray == npt.NDArray[np.floating[__import__('typing').Any]]
    assert types.IntArray == npt.NDArray[np.integer[__import__('typing').Any]]
    assert types.BoolArray == npt.NDArray[np.bool_]
    assert types.AnyNDArray == npt.NDArray[__import__('typing').Any]
    assert __version__ == "0.4.0"
    assert get_origin(types.SignalPair) is tuple
    assert get_args(types.SignalPair) == (types.ArrayLike, types.ArrayLike)


@pytest.mark.parametrize('module', [
    filtering, frequency_analysis, processing, geometry, quaternions,
])
def test_public_functions_are_fully_annotated(module):
    for name, fun in inspect.getmembers(module, inspect.isfunction):
        if fun.__module__ != module.__name__:
            continue
        sig = inspect.signature(fun)
        assert sig.return_annotation is not inspect.Signature.empty, name
        for param in sig.parameters.values():
            assert param.annotation is not inspect.Parameter.empty, (name, param.name)
        assert 'return' in get_type_hints(fun), name


def test_class_methods_have_resolvable_annotations():
    for name in ('__init__', 'from_euler', 'from_matrix', 'from_axis_angle',
                 'apply', 'inverse', 'as_matrix', 'as_euler', '__matmul__', '__repr__'):
        assert 'return' in get_type_hints(getattr(Rotation, name))


def test_list_tuple_inputs_produce_float64_arrays():
    mid, dydt = processing.differentiate((0, 1, 2), [1, 3, 7])
    assert mid.dtype == np.float64
    assert dydt.dtype == np.float64
    assert np.allclose(dydt, [2, 4])

    filt = filtering.butterworth_lowpass([0] * 50 + [1] * 50, 3, 50)
    assert filt.dtype == np.float64
    frequencies, density = frequency_analysis.power_spectral_density((0, 1, 0, -1), 4)
    assert frequencies.dtype == np.float64
    assert density.dtype == np.float64

    q = quaternions.quaternion_from_axis_angle(np.pi / 2, (0, 0, 1))
    assert q.dtype == np.float64
    assert quaternions.rotate_vectors(q, [1, 0, 0]).dtype == np.float64
    assert Rotation([1, 0, 0, 0]).data.dtype == np.float64



def test_bool_mask_and_integer_series():
    assert processing.first_true_index([False, False, True]) == 2
    assert processing.fill_missing([1, 2, 3]).dtype == np.float64
