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
import warnings
import io
import base64
from types import FunctionType, MethodType
from typing import Any, Tuple, Dict, Type, Optional, Union

import numpy as np
from qiskit.circuit import qpy_serialization, QuantumCircuit
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


def deserialize_warning(name: str, module: str):
    """Display a warning for missing type in module"""
    warnings.warn(f"Cannot deserialize {name}. The type could not be found in module {module}")


def serialize_object(obj: Any, settings: Optional[Dict] = None, safe_float: bool = True) -> Dict:
    """Serialize a class instance from its init args and kwargs.

    Args:
        obj: The object to be serialized.
        settings: settings for reconstructing the object from args and kwargs
        safe_float: if True check float values for NaN, inf and -inf
                    and cast to strings during serialization.

    Returns:
        Dict serialized class instance.
    """
    value = {"name": type(obj).__name__, "module": type(obj).__module__}
    if settings is None:
        if hasattr(obj, "__json_encode__"):
            settings = obj.__json_encode__
        else:
            settings = {}
    if safe_float:
        settings = serialize_safe_float(settings)
    value["settings"] = settings
    return {"__type__": "object", "__value__": value}


def deserialize_object(value: Dict) -> Any:
    """Deserialize class instance saved as settings"""
    name = value["name"]
    mod = value["module"]
    settings = value.get("settings", {})

    cls = None
    if mod == "__main__":
        cls = globals().get(name, None)
    else:
        scope = importlib.import_module(mod)
        for name_, obj in inspect.getmembers(scope, inspect.isclass):
            if name_ == name:
                cls = obj
                break
    if cls is None:
        deserialize_warning(name, mod)
        return value

    try:
        # Check if class defines a __json_decode__ method
        if hasattr(cls, "__json_decode__"):
            return cls.__json_decode__(settings)
        else:
            args = settings.get("args", tuple())
            kwargs = settings.get("kwargs", dict())
            return cls(*args, **kwargs)
    except Exception:  # pylint: disable=broad-except
        warnings.warn(
            f"Could not deserialize instance of class {name} from settings {settings}",
        )
    return value


def serialize_type(type_name: Union[Type, FunctionType, MethodType]):
    """Serialize a type, function, or class method"""
    value = {"name": type_name.__qualname__, "module": type_name.__module__}
    return {"__type__": "type", "__value__": value}


def deserialize_type(value: Dict):
    """Deserialize a type, function, or class method"""
    qualname = value["name"].split(".", maxsplit=1)
    if len(qualname) == 2:
        method_cls, name = qualname
    else:
        method_cls = None
        name = qualname[0]
    mod = value["module"]

    scope = None
    if mod == "__main__":
        if method_cls is None:
            if name in globals():
                return globals()[name]
            else:
                deserialize_warning(name, mod)
                return value
        else:
            scope = globals().get(method_cls, None)
    else:
        mod_scope = importlib.import_module(mod)
        if method_cls is None:
            scope = mod_scope
        else:
            for name_, obj in inspect.getmembers(mod_scope, inspect.isclass):
                if name_ == method_cls:
                    scope = obj
    if scope is None:
        deserialize_warning(name, mod)
        return value

    def predicate(x):
        return inspect.isfunction(x) or inspect.ismethod(x) or inspect.isclass(x)

    for name_, obj in inspect.getmembers(scope, predicate):
        if name_ == name:
            return obj

    deserialize_warning(name, mod)
    return value


def serialize_circuit(circuit: QuantumCircuit):
    """Serialize a QuantumCircuit"""
    buf = io.BytesIO()
    qpy_serialization.dump(circuit, buf)
    qpy_str = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return {"__type__": "qpy_circuit", "__value__": qpy_str}


def deserialize_circuit(value: bytes):
    """Deserialize a QuantumCircuit"""
    buf = io.BytesIO(base64.standard_b64decode(value))
    return qpy_serialization.load(buf)[0]


def serialize_object_legacy(
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


def deserialize_object_legacy(mod_name: str, class_name: str, args: Tuple, kwargs: Dict) -> Any:
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
        if hasattr(obj, "__json_encode__"):
            return serialize_object(obj)
        if dataclasses.is_dataclass(obj):
            return serialize_object(obj, settings={"kwargs": dataclasses.asdict(obj)})
        if isinstance(obj, QuantumCircuit):
            return serialize_circuit(obj)
        if isinstance(obj, (Operator, Choi)):
            return serialize_object(
                obj,
                settings={
                    "args": (obj.data,),
                    "kwargs": {"input_dims": obj.input_dims(), "output_dims": obj.output_dims()},
                },
                safe_float=False,
            )
        if isinstance(obj, (Statevector, DensityMatrix)):
            return serialize_object(
                obj,
                settings={"args": (obj.data,), "kwargs": {"dims": obj.dims()}},
                safe_float=False,
            )
        if isinstance(obj, (type, FunctionType, MethodType)):
            return serialize_type(obj)
        try:
            return super().default(obj)
        except TypeError:
            return serialize_type(type(obj))


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
            if obj_type == "safe_float":
                value = obj["__value__"]
                return self._NaNs.get(value, value)
            if obj_type == "qpy_circuit":
                return deserialize_circuit(obj["__value__"])
            if obj_type == "object":
                return deserialize_object(obj["__value__"])
            if obj_type == "type":
                return deserialize_type(obj["__value__"])
            # Legacy: Remove
            if obj_type == "function":
                return obj["__value__"]
            if obj_type == "__object__":
                value = obj["__value__"]
                class_name = value["__name__"]
                mod_name = value["__module__"]
                args = value.get("__args__", tuple())
                kwargs = value.get("__kwargs__", dict())
                return deserialize_object_legacy(mod_name, class_name, args, kwargs)
            if obj_type == "__class_name__":
                return obj["__value__"]
        return obj
