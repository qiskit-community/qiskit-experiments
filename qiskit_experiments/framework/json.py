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

import base64
import dataclasses
import importlib
import inspect
import io
import json
import math
import traceback
import warnings
import zlib
from datetime import datetime
from functools import lru_cache
from types import FunctionType, MethodType
from typing import Any, Dict, Type, Optional, Union, Callable

import lmfit
import numpy as np
import scipy.sparse as sps
import uncertainties
from qiskit import qpy
from qiskit.circuit import ParameterExpression, QuantumCircuit, Instruction
from qiskit_experiments.version import __version__


@lru_cache()
def get_module_version(mod_name: str) -> str:
    """Return the __version__ of a module if defined.

    Args:
        mod_name: The module to extract the version from.

    Returns:
        The module version. If the module is `__main__` the
        qiskit-experiments version will be returned.
    """
    if "." in mod_name:
        return get_module_version(mod_name.split(".", maxsplit=1)[0])

    # Return qiskit experiments version for classes in this
    # module or defined in main
    if mod_name in ["qiskit_experiments", "__main__"]:
        return __version__

    # For other classes attempt to use their module version
    # if it is defined
    try:
        mod = importlib.import_module(mod_name)
        return getattr(mod, "__version__", None)
    except Exception:  # pylint: disable=broad-except
        return None


@lru_cache()
def get_object_version(obj: Any) -> str:
    """Return the module version of an object, class, or function.

    Note that if the object is defined in `__main__` instead
    of a module the current qiskit-experiments version will be used.

    Args:
        obj: A type or function to extract the module version for.

    Returns:
        The module version for the object. If the object is defined
        in `__main__` the qiskit-experiments version will be returned.
    """
    if not istype(obj):
        return get_object_version(type(obj))
    base_mod = obj.__module__.split(".", maxsplit=1)[0]
    return get_module_version(base_mod)


def _show_warning(
    msg: Optional[str] = None,
    traceback_msg: Optional[str] = None,
    mod_name: Optional[str] = None,
    save_version: Optional[str] = None,
    load_version: Optional[str] = None,
):
    """Show warning for partial deserialization"""
    warning_msg = f"{msg} " if msg else ""
    if mod_name != "__main__":
        mod_name = mod_name.split(".", maxsplit=1)[0]
    if save_version != load_version:
        warning_msg += (
            f"\nNOTE: The current version of module '{mod_name}' ({load_version})"
            f" differs from the version used for serialization ({save_version})."
        )
    if traceback_msg:
        warning_msg += f"\nThe following exception was raised:\n{traceback_msg}"
    warnings.warn(warning_msg, stacklevel=3)


def _deprecation_warning(name: str, version: str):
    """Show warning for deprecated serialization"""
    warnings.warn(
        f"Deserializated data for <{name}> stored in a deprecated serialization format."
        " Re-serialize or re-save the data to update the serialization format otherwise"
        f" loading this data may fail in qiskit-experiments version {version}. ",
        DeprecationWarning,
    )


def _serialize_bytes(data: bytes, compress: bool = True) -> Dict[str, Any]:
    """Serialize binary data.

    Args:
        data: Data to be serialized.
        compress: Whether to compress the serialized data.

    Returns:
        The serialized object value as a dict.
    """
    if compress:
        data = zlib.compress(data)
    value = {
        "encoded": base64.standard_b64encode(data).decode("utf-8"),
        "compressed": compress,
    }
    return {"__type__": "b64encoded", "__value__": value}


def _deserialize_bytes(value: Dict) -> str:
    """Deserialize binary encoded data.

    Args:
        value: value to be deserialized.

    Returns:
        Deserialized string representation.

    Raises:
        ValueError: If encoded data cannot be deserialized.
    """
    try:
        encoded = value["encoded"]
        compressed = value["compressed"]
        decoded = base64.standard_b64decode(encoded)
        if compressed:
            decoded = zlib.decompress(decoded)
        return decoded
    except Exception as ex:  # pylint: disable=broad-except
        raise ValueError("Could not deserialize binary encoded data.") from ex


def _serialize_and_encode(
    data: Any, serializer: Callable, compress: bool = True, **kwargs: Any
) -> str:
    """Serialize the input data and return the encoded string.

    Args:
        data: Data to be serialized.
        serializer: Function used to serialize data.
        compress: Whether to compress the serialized data.
        kwargs: Keyword arguments to pass to the serializer.

    Returns:
        String representation.
    """
    with io.BytesIO() as buff:
        serializer(buff, data, **kwargs)
        buff.seek(0)
        serialized_data = buff.read()
    return _serialize_bytes(serialized_data, compress=compress)


def _decode_and_deserialize(value: Dict, deserializer: Callable, name: Optional[str] = None) -> Any:
    """Decode and deserialize input data.

    Args:
        value: The binary encoded serialized data value.
        deserializer: Function used to deserialize data.
        name: Object type name for warning message if deserialization fails.

    Returns:
        Deserialized data.

    Raises:
        ValueError: If deserialization fails.
    """
    try:
        with io.BytesIO() as buff:
            buff.write(value)
            buff.seek(0)
            orig = deserializer(buff)
        return orig
    except Exception as ex:  # pylint: disable=broad-except
        raise ValueError(f"Could not deserialize <{name}> data.") from ex


def _serialize_safe_float(obj: any):
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
        return [_serialize_safe_float(i) for i in obj]
    elif isinstance(obj, dict):
        return {key: _serialize_safe_float(val) for key, val in obj.items()}
    elif isinstance(obj, complex):
        return {"__type__": "complex", "__value__": _serialize_safe_float([obj.real, obj.imag])}
    return obj


def istype(obj: Any) -> bool:
    """Return True if object is a class, function, or method type"""
    return inspect.isclass(obj) or inspect.isfunction(obj) or inspect.ismethod(obj)


def _serialize_type(type_name: Union[Type, FunctionType, MethodType]):
    """Serialize a type, function, or class method"""
    mod = type_name.__module__
    value = {
        "name": type_name.__qualname__,
        "module": mod,
        "version": get_module_version(mod),
    }
    return {"__type__": "type", "__value__": value}


def _deserialize_type(value: Dict):
    """Deserialize a type, function, or class method"""
    traceback_msg = None
    load_version = None
    try:
        qualname = value["name"].split(".", maxsplit=1)
        if len(qualname) == 2:
            method_cls, name = qualname
        else:
            method_cls = None
            name = qualname[0]
        mod = value["module"]
        mod_scope = importlib.import_module(mod)
        scope = None
        if method_cls is None:
            scope = mod_scope
        else:
            for name_, obj in inspect.getmembers(mod_scope, inspect.isclass):
                if name_ == method_cls:
                    scope = obj
        if scope is not None:
            for name_, obj in inspect.getmembers(scope, istype):
                if name_ == name:
                    return obj
    except Exception as ex:  # pylint: disable=broad-except
        traceback_msg = "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))

    # Show warning
    warning_msg = f"Cannot deserialize {name}. The type could not be found in module {mod}."
    save_version = value.get("version", None)
    load_version = get_module_version(mod)
    _show_warning(
        warning_msg,
        traceback_msg=traceback_msg,
        mod_name=mod,
        save_version=save_version,
        load_version=load_version,
    )

    # Return partially deserialized value
    return value


def _serialize_object(obj: Any, settings: Optional[Dict] = None, safe_float: bool = True) -> Dict:
    """Serialize a class instance from its init args and kwargs.

    Args:
        obj: The object to be serialized.
        settings: Optional, settings for reconstructing the object from kwargs.
        safe_float: If True check float values for NaN, inf and -inf
                    and cast to strings during serialization.

    Returns:
        Dict serialized class instance.
    """
    if settings is None:
        if hasattr(obj, "__json_encode__"):
            settings = obj.__json_encode__()
        elif hasattr(obj, "settings"):
            settings = obj.settings
        else:
            settings = {}
    if safe_float:
        settings = _serialize_safe_float(settings)
    cls = type(obj)
    value = {
        "class": _serialize_type(cls),
        "settings": settings,
        "version": get_object_version(cls),
    }
    return {"__type__": "object", "__value__": value}


def _deserialize_object(value: Dict) -> Any:
    """Deserialize class instance saved as settings"""
    cls = value.get("class", {})
    if isinstance(cls, dict):
        # Deserialization of class type failed.
        return value

    settings = value.get("settings", {})
    if hasattr(cls, "__json_decode__"):
        try:
            return cls.__json_decode__(settings)
        except Exception as ex:  # pylint: disable=broad-except
            traceback_msg = "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
            warning_msg = (
                f"Could not deserialize instance of class {cls} from value {settings} "
                "using __json_decode__ method."
            )
    else:
        try:
            return cls(**settings)
        except Exception as ex:  # pylint: disable=broad-except
            traceback_msg = "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
            warning_msg = f"Could not deserialize instance of class {cls} from settings {settings}."

    # Display warning msg if deserialization failed
    mod_name = cls.__module__
    load_version = get_object_version(cls)
    save_version = value.get("version")
    _show_warning(
        warning_msg,
        traceback_msg=traceback_msg,
        mod_name=mod_name,
        save_version=save_version,
        load_version=load_version,
    )

    # Return partially deserialized value
    return value


def _deserialize_object_legacy(value: Dict) -> Any:
    """Deserialize a class object from its init args and kwargs."""
    try:
        class_name = value["__name__"]
        mod_name = value["__module__"]
        args = value.get("__args__", ())
        kwargs = value.get("__kwargs__", {})
        mod = importlib.import_module(mod_name)
        for name, cls in inspect.getmembers(mod, inspect.isclass):
            if name == class_name:
                return cls(*args, **kwargs)

        raise Exception(  # pylint: disable=broad-exception-raised
            f"Unable to find class {class_name} in module {mod_name}"
        )

    except Exception as ex:  # pylint: disable=broad-except
        traceback_msg = "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        warning_msg = f"Unable to initialize {class_name}."
        _show_warning(warning_msg, traceback_msg=traceback_msg)
        return value


class ExperimentEncoder(json.JSONEncoder):
    """JSON Encoder for Qiskit Experiments.

    This class extends the default Python JSONEncoder by including built-in
    support for

    * complex numbers, inf and NaN floats, sets, and dataclasses.
    * NumPy ndarrays and SciPy sparse matrices.
    * Qiskit ``QuantumCircuit``.
    * Any class that implements a ``__json_encode__`` method or a
      ``settings`` property.

    Generic classes can be serialized by this encoder. This is done
    by attempting the following methods in order:

    1.  The object has a ``__json_encode__`` method. This should have signature

        .. code-block:: python

            def __json_encode__(self) -> Any:
                # return a JSON serializable object value

        The value returned by ``__json_encode__`` must be an object that can be
        serialized by the JSON encoder (for example a ``dict`` containing
        other JSON serializable objects).

        To deserialize this object using the :class:`ExperimentDecoder` the
        class must also provide a ``__json_decode__`` class method that can
        convert the value returned by ``__json_encode__`` back to the object.
        This method should have signature

        .. code-block:: python

            @classmethod
            def __json_decode__(cls, value: Any) -> cls:
                # recover the object from the `value` returned by __json_encode__

    2.  The object has a ``settings`` property. This should have signature

        .. code-block:: python

            @property
            def settings(self) -> Dict[str, Any]:
                # Return settings value for reconstructing the instance

        Deserialization of objects from the ``value`` dictionary returned by
        ``settings`` is done by calling the class `__init__` method
        ``cls(**settings)``.

    3.  In all other cases only the object class is saved. Deserialization
        will attempt to recover the object from default initialization of
        its class as ``cls()``.

    .. note::

        Serialization of custom classes works for user-defined classes in
        Python scripts, notebooks, or third party modules. Note however
        that these will only be able to be de-serialized if that class
        can be imported form the same scope at the time the
        :class:`ExperimentDecoder` is invoked.
    """

    def default(self, obj: Any) -> Any:  # pylint: disable=arguments-renamed
        if istype(obj):
            return _serialize_type(obj)
        if hasattr(obj, "__json_encode__"):
            return _serialize_object(obj)
        if isinstance(obj, complex):
            return _serialize_safe_float(obj)
        if isinstance(obj, set):
            return {"__type__": "set", "__value__": list(obj)}
        if isinstance(obj, np.ndarray):
            value = _serialize_and_encode(obj, np.save, allow_pickle=False)
            return {"__type__": "ndarray", "__value__": value}
        if isinstance(obj, sps.spmatrix):
            value = _serialize_and_encode(obj, sps.save_npz, compress=False)
            return {"__type__": "spmatrix", "__value__": value}
        if isinstance(obj, bytes):
            return _serialize_bytes(obj)
        if isinstance(obj, datetime):
            return {"__type__": "datetime", "__value__": obj.isoformat()}
        if isinstance(obj, np.number):
            return obj.item()
        if dataclasses.is_dataclass(obj):
            # Note that dataclass.asdict recursively convert nested dataclass into dictionary.
            # Thus inter dataclass is unintentionally decoded as dictionary.
            # obj.__dict__ doesn't convert inter dataclass thus serialization
            # is offloaded to usual json serialization mechanism.
            return _serialize_object(obj, settings=obj.__dict__)
        if isinstance(obj, uncertainties.UFloat):
            # This could be UFloat (AffineScalarFunc) or Variable.
            # UFloat is a base class of Variable that contains parameter correlation.
            # i.e. Variable is special subclass for single number.
            # Since this object is not serializable, we will drop correlation information
            # during serialization. Then both can be serialized as Variable.
            # Note that UFloat doesn't have a tag.
            settings = {
                "value": _serialize_safe_float(obj.nominal_value),
                "std_dev": _serialize_safe_float(obj.std_dev),
                "tag": getattr(obj, "tag", None),
            }
            cls = uncertainties.core.Variable
            return {
                "__type__": "object",
                "__value__": {
                    "class": _serialize_type(cls),
                    "settings": settings,
                    "version": get_object_version(cls),
                },
            }
        if isinstance(obj, lmfit.Model):
            # LMFIT Model object. Delegate serialization to LMFIT.
            return {
                "__type__": "LMFIT.Model",
                "__value__": obj.dumps(),
            }
        if isinstance(obj, Instruction):
            # Serialize gate by storing it in a circuit.
            circuit = QuantumCircuit(obj.num_qubits, obj.num_clbits)
            circuit.append(obj, range(obj.num_qubits), range(obj.num_clbits))
            value = _serialize_and_encode(
                data=circuit, serializer=lambda buff, data: qpy.dump(data, buff)
            )
            return {"__type__": "Instruction", "__value__": value}
        if isinstance(obj, QuantumCircuit):
            value = _serialize_and_encode(
                data=obj, serializer=lambda buff, data: qpy.dump(data, buff)
            )
            return {"__type__": "QuantumCircuit", "__value__": value}
        if isinstance(obj, ParameterExpression):
            value = _serialize_and_encode(
                data=obj,
                serializer=qpy._write_parameter_expression,
                compress=False,
            )
            return {"__type__": "ParameterExpression", "__value__": value}
        if istype(obj):
            return _serialize_type(obj)
        try:
            return super().default(obj)
        except TypeError:
            return _serialize_object(obj)


class ExperimentDecoder(json.JSONDecoder):
    """JSON Decoder for Qiskit Experiments.

    This class extends the default Python JSONDecoder by including built-in
    support for all objects that that can be serialized using the
    :class:`ExperimentEncoder`.

    See :class:`ExperimentEncoder` class documentation for details.
    """

    _NaNs = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}

    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        """Object hook."""
        if "__type__" in obj:
            obj_type = obj["__type__"]
            obj_val = obj["__value__"]
            if obj_type == "complex":
                return obj_val[0] + 1j * obj_val[1]
            if obj_type == "ndarray":
                return _decode_and_deserialize(obj_val, np.load, name=obj_type)
            if obj_type == "spmatrix":
                return _decode_and_deserialize(obj_val, sps.load_npz, name=obj_type)
            if obj_type == "b64encoded":
                return _deserialize_bytes(obj_val)
            if obj_type == "set":
                return set(obj_val)
            if obj_type == "datetime":
                return datetime.fromisoformat(obj_val)
            if obj_type == "LMFIT.Model":
                tmp = lmfit.Model(func=None)
                load_obj = tmp.loads(s=obj_val)
                return load_obj
            if obj_type == "Instruction":
                circuit = _decode_and_deserialize(obj_val, qpy.load, name="QuantumCircuit")[0]
                return circuit.data[0].operation
            if obj_type == "QuantumCircuit":
                return _decode_and_deserialize(obj_val, qpy.load, name=obj_type)[0]
            if obj_type == "ParameterExpression":
                return _decode_and_deserialize(
                    obj_val, qpy._read_parameter_expression, name=obj_type
                )
            if obj_type == "safe_float":
                return self._NaNs.get(obj_val, obj_val)
            if obj_type == "object":
                return _deserialize_object(obj_val)
            if obj_type == "type":
                return _deserialize_type(obj_val)

            # Deprecated formats
            if obj_type == "array":
                _deprecation_warning(obj_type, "0.3.0")
                return np.array(obj_val)
            if obj_type == "function":
                _deprecation_warning(obj_type, "0.3.0")
                return obj_val
            if obj_type == "__object__":
                _deprecation_warning(obj_type, "0.3.0")
                return _deserialize_object_legacy(obj_val)
            if obj_type == "__class_name__":
                _deprecation_warning(obj_type, "0.3.0")
                return obj_val
        return obj
