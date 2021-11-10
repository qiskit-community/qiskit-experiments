# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=method-hidden,too-many-return-statements,c-extension-no-member

"""Experiment serialization methods."""

import json
import math
import dataclasses
import importlib
import inspect
from types import FunctionType
from typing import Any, Tuple, Dict, Type, Optional

import numpy as np
from qiskit.quantum_info.operators import Operator, Choi
from qiskit.quantum_info.states import Statevector, DensityMatrix


def serialize_safe_float(obj: any):
    """Recursively serialize basic types safely handing inf and NaN"""
    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        else:
            value = obj
            if math.isnan(obj):
                value = "NaN"
            elif obj == math.inf:
                value = "Infinity"
            elif obj == -math.inf:
                value = "-Infinity"
            return {"__type__": "safe_float", "__value__": value}
    elif isinstance(obj, (list, tuple)):
        return [serialize_safe_float(i) for i in obj]
    elif isinstance(obj, dict):
        return {key: serialize_safe_float(val) for key, val in obj.items()}
    elif isinstance(obj, complex):
        return {"__type__": "complex", "__value__": serialize_safe_float([obj.real, obj.imag])}
    elif isinstance(obj, np.ndarray):
        value = obj.tolist()
        if issubclass(obj.dtype.type, np.inexact) and not np.isfinite(obj).all():
            value = serialize_safe_float(value)
        return {"__type__": "array", "__value__": value}
    return obj


def serialize_object(
    cls: Type, args: Optional[Tuple] = None, kwargs: Optional[Dict] = None, safe_float: bool = True
) -> Dict:
    """Serialize a class object from its init args and kwargs.

    Args:
        cls: The object to be serialized.
        args: the class init arg values for reconstruction.
        kwargs: the class init kwarg values for reconstruction.
        safe_float: if True check float values for NaN, inf and -inf
                    and cast to strings during serialization.

    Returns:
        Dict for serialization.
    """
    value = {
        "__name__": cls.__name__,
        "__module__": cls.__module__,
    }
    if safe_float:
        args = serialize_safe_float(args)
        kwargs = serialize_safe_float(kwargs)
    if args:
        value["__args__"] = args
    if kwargs:
        value["__kwargs__"] = kwargs
    return {"__type__": "__object__", "__value__": value}


def deserialize_object(mod_name: str, class_name: str, args: Tuple, kwargs: Dict) -> Any:
    """Deserialize a class object from its init args and kwargs.

    Args:
        mod_name: Name of the module.
        class_name: Name of the class.
        args: args for class init method.
        kwargs: kwargs for class init method.

    Returns:
        Deserialized object.

    Raises:
        ValueError: If unable to find the class.
    """
    mod = importlib.import_module(mod_name)
    for name, cls in inspect.getmembers(mod, inspect.isclass):
        if name == class_name:
            return cls(*args, **kwargs)
    raise ValueError(f"Unable to find class {class_name} in module {mod_name}")


class ExperimentEncoder(json.JSONEncoder):
    """JSON Encoder for Numpy arrays and complex numbers."""

    def default(self, obj: Any) -> Any:  # pylint: disable=arguments-differ
        if isinstance(obj, (np.ndarray, complex)):
            return serialize_safe_float(obj)
        if dataclasses.is_dataclass(obj):
            return serialize_object(type(obj), kwargs=dataclasses.asdict(obj), safe_float=True)
        if isinstance(obj, (Operator, Choi)):
            return serialize_object(
                type(obj),
                args=(obj.data,),
                kwargs={"input_dims": obj.input_dims(), "output_dims": obj.output_dims()},
                safe_float=False,
            )
        if isinstance(obj, (Statevector, DensityMatrix)):
            return serialize_object(
                type(obj), args=(obj.data,), kwargs={"dims": obj.dims()}, safe_float=False
            )
        if isinstance(obj, FunctionType):
            return {"__type__": "function", "__value__": obj.__name__}
        try:
            return super().default(obj)
        except TypeError:
            return {"__type__": "__class_name__", "__value__": type(obj).__name__}


class ExperimentDecoder(json.JSONDecoder):
    """JSON Decoder for Numpy arrays and complex numbers."""

    _NaNs = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}

    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        """Object hook."""
        if "__type__" in obj:
            obj_type = obj["__type__"]
            if obj_type == "complex":
                val = obj["__value__"]
                return val[0] + 1j * val[1]
            if obj_type == "array":
                return np.array(obj["__value__"])
            if obj_type == "function":
                return obj["__value__"]
            if obj_type == "__object__":
                value = obj["__value__"]
                class_name = value["__name__"]
                mod_name = value["__module__"]
                args = value.get("__args__", tuple())
                kwargs = value.get("__kwargs__", dict())
                return deserialize_object(mod_name, class_name, args, kwargs)
            if obj_type == "safe_float":
                value = obj["__value__"]
                return self._NaNs.get(value, value)
            if obj_type == "__class_name__":
                return obj["__value__"]
        return obj
