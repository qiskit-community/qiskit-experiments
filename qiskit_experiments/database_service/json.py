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
# pylint: disable=method-hidden,too-many-return-statements

"""Experiment serialization methods."""

import json
import dataclasses
import importlib
import inspect
from types import FunctionType
from typing import Any, Tuple, Dict, Type, Optional

import numpy as np
from qiskit.quantum_info.operators import Operator, Choi
from qiskit.quantum_info.states import Statevector, DensityMatrix


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


def serialize_object(
    cls: Type, args: Optional[Tuple] = None, kwargs: Optional[Dict] = None
) -> Dict:
    """Serialize a class object from its init args and kwargs.

    Args:
        cls: The object to be serialized.
        args: the class init arg values for reconstruction.
        kwargs: the class init kwarg values for reconstruction.

    Returns:
        Dict for serialization.
    """
    value = {
        "__name__": cls.__name__,
        "__module__": cls.__module__,
    }
    if args:
        value["__args__"] = args
    if kwargs:
        value["__kwargs__"] = kwargs
    return {"__type__": "__object__", "__value__": value}


class ExperimentEncoder(json.JSONEncoder):
    """JSON Encoder for Numpy arrays and complex numbers."""

    def default(self, obj: Any) -> Any:  # pylint: disable=arguments-differ
        if isinstance(obj, np.ndarray):
            return {"__type__": "array", "__value__": obj.tolist()}
        if isinstance(obj, complex):
            return {"__type__": "complex", "__value__": [obj.real, obj.imag]}
        if dataclasses.is_dataclass(obj):
            return serialize_object(type(obj), kwargs=dataclasses.asdict(obj))
        if isinstance(obj, (Operator, Choi)):
            return serialize_object(
                type(obj),
                args=(obj.data,),
                kwargs={"input_dims": obj.input_dims(), "output_dims": obj.output_dims()},
            )
        if isinstance(obj, (Statevector, DensityMatrix)):
            return serialize_object(type(obj), args=(obj.data,), kwargs={"dims": obj.dims()})
        if isinstance(obj, FunctionType):
            return {"__type__": "function", "__value__": obj.__name__}
        try:
            return super().default(obj)
        except TypeError:
            return {"__type__": "__class_name__", "__value__": type(obj).__name__}


class ExperimentDecoder(json.JSONDecoder):
    """JSON Decoder for Numpy arrays and complex numbers."""

    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        """Object hook."""
        if "__type__" in obj:
            if obj["__type__"] == "complex":
                val = obj["__value__"]
                return val[0] + 1j * val[1]
            if obj["__type__"] == "array":
                return np.array(obj["__value__"])
            if obj["__type__"] == "function":
                return obj["__value__"]
            if obj["__type__"] == "__object__":
                value = obj["__value__"]
                class_name = value["__name__"]
                mod_name = value["__module__"]
                args = value.get("__args__", tuple())
                kwargs = value.get("__kwargs__", dict())
                return deserialize_object(mod_name, class_name, args, kwargs)
            if obj["__type__"] == "__class_name__":
                return obj["__value__"]
        return obj
