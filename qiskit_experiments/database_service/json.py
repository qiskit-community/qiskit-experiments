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

import base64
import io
import json
import dataclasses
import importlib
import inspect
from types import FunctionType
from typing import Any, Tuple, Dict, Optional

import numpy as np
from qiskit.circuit import qpy_serialization, QuantumCircuit
from qiskit.quantum_info.operators import Operator, Choi
from qiskit.quantum_info.states import Statevector, DensityMatrix


def deserialize_object(value: Dict) -> Any:
    """Deserialize a class object from its init args and kwargs.

    Args:
        value: serialized value of the object.

    Returns:
        Deserialized object.

    Raises:
        ValueError: If unable to find the class.
    """
    mod_name = value["__module__"]
    mod = importlib.import_module(mod_name)
    class_name = value["__name__"]
    for name, cls in inspect.getmembers(mod, inspect.isclass):
        if name == class_name:
            if hasattr(cls, "_deserialize"):
                return cls._deserialize(value)
            else:
                args = value.get("__args__", tuple())
                kwargs = value.get("__kwargs__", dict())
                return cls(*args, **kwargs)
    raise ValueError(f"Unable to find class {class_name} in module {mod_name}")


def serialize_object(obj: any, args: Optional[Tuple] = None, kwargs: Optional[Dict] = None) -> Dict:
    """Serialize a class object.

    Args:
        obj: The object to be serialized.
        args: Optional class init arg values for reconstruction.
        kwargs: Optional class init kwarg values for reconstruction.

    Returns:
        Dict for serialization.
    """
    if hasattr(obj, "_serialize"):
        value = obj._serialize()
    else:
        value = {}
    if "__name__" not in value:
        value["__name__"] = type(obj).__name__
    if "__module__" not in value:
        value["__module__"] = type(obj).__module__
    if args is not None:
        value["__args__"] = args
    if kwargs is not None:
        value["__kwargs__"] = kwargs
    return {"__type__": "__object__", "__value__": value}


class ExperimentEncoder(json.JSONEncoder):
    """JSON Encoder for Numpy arrays and complex numbers."""

    def default(self, obj: Any) -> Any:  # pylint: disable=arguments-differ
        if isinstance(obj, np.ndarray):
            return {"__type__": "array", "__value__": obj.tolist()}
        if isinstance(obj, complex):
            return {"__type__": "complex", "__value__": [obj.real, obj.imag]}
        if isinstance(obj, QuantumCircuit):
            buf = io.BytesIO()
            qpy_serialization.dump(obj, buf)
            qpy_str = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
            return {"__type__": "qpy_circuit", "__value__": qpy_str}
        if hasattr(obj, "_serialize"):
            return serialize_object(obj)
        if dataclasses.is_dataclass(obj):
            return serialize_object(obj, kwargs=dataclasses.asdict(obj))
        if isinstance(obj, (Operator, Choi)):
            return serialize_object(
                obj,
                args=(obj.data,),
                kwargs={"input_dims": obj.input_dims(), "output_dims": obj.output_dims()},
            )
        if isinstance(obj, (Statevector, DensityMatrix)):
            return serialize_object(obj, args=(obj.data,), kwargs={"dims": obj.dims()})
        if isinstance(obj, FunctionType):
            return {"__type__": "function", "__value__": obj.__name__}
        try:
            return super().default(obj)
        except TypeError:
            return serialize_object(obj)


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
            if obj["__type__"] == "qpy_circuit":
                buf =  io.BytesIO(base64.standard_b64decode(obj["__value__"]))
                return qpy_serialization.load(buf)[0]
            if obj["__type__"] == "__class_name__":
                return obj["__value__"]
            if obj["__type__"] == "__object__":
                try:
                    return deserialize_object(obj["__value__"])
                except ValueError:
                    return obj["__value__"]
        return obj
