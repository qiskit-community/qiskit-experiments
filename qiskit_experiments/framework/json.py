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
import importlib
import inspect
import io
import json
import math
import traceback
import warnings
import zlib
from collections.abc import Callable
from datetime import datetime
from functools import lru_cache
from importlib.metadata import entry_points
from types import FunctionType, MethodType
from typing import Any

import lmfit
import numpy as np
import scipy.sparse as sps
import uncertainties
from qiskit import qpy, quantum_info
from qiskit.circuit import ParameterExpression, QuantumCircuit, Instruction
from qiskit.exceptions import QiskitError
from qiskit.providers import Backend
from qiskit_experiments.version import __version__


# This can be set to true in testing to raise an exception if a class is
# falling through to default serialization
_strict_serialization = False  # pylint: disable=invalid-name
# Set of Python packages that are allowed for deserialization. The set is
# loaded and set here once when first needed.
_allowed_packages = None  # pylint: disable=invalid-name


@lru_cache
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


@lru_cache
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
    msg: str | None = None,
    traceback_msg: str | None = None,
    mod_name: str | None = None,
    save_version: str | None = None,
    load_version: str | None = None,
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


def _serialize_bytes(data: bytes, compress: bool = True) -> dict[str, Any]:
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


def _deserialize_bytes(value: dict) -> str:
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


def _decode_and_deserialize(value: dict, deserializer: Callable, name: str | None = None) -> Any:
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


def _serialize_safe_float(obj: Any):
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


def _serialize_type(type_name: type | FunctionType | MethodType):
    """Serialize a type, function, or class method"""
    mod = type_name.__module__
    value = {
        "name": type_name.__qualname__,
        "module": mod,
        "version": get_module_version(mod),
    }
    return {"__type__": "type", "__value__": value}


def _load_allowed_packages():
    global _allowed_packages  # pylint: disable=global-statement
    if _allowed_packages is None:
        ep_pkgs = [
            e.module for e in entry_points(group="qiskit_experiments.deserialization_packages")
        ]
        _allowed_packages = frozenset(ep_pkgs)


def _check_quantum_info_class(cls) -> bool:
    """Check cls is a quantum_info type to be deserialized"""
    mod = getattr(cls, "__module__", "")
    settings = getattr(cls, "settings", None)
    # Class comes from qiskit.quantum_info and has a settings property
    return (
        mod.startswith("qiskit.quantum_info")
        and hasattr(quantum_info, cls.__name__)
        and isinstance(settings, property)
    )


def _load_quantum_info_type(name: str, module: str) -> Any:
    """Attempt to load a type from the qiskit quantum_info package"""
    cls = getattr(quantum_info, name, None)
    if cls is None:
        raise QiskitError(f"Could not load type {name} from module {module}!")
    if not hasattr(cls, "__module__"):
        raise QiskitError(
            f"'{name}' specified to load from {module} does not appear to be a class!"
        )
    if not cls.__module__.startswith("qiskit.quantum_info"):
        raise QiskitError(
            f"Could not load type {name} from module {module}. It appears to come "
            "from {cls.__module__} instead."
        )
    settings = getattr(cls, "settings", None)
    if settings is None or not isinstance(settings, property):
        raise QiskitError(
            f'Class {name} from qiskit.quantum_info does not have a "settings" property. '
            f'Only class from qiskit.quantum_info with a "settings" property can be loaded.'
        )

    return cls


def _deserialize_ufloat(value: dict[str, Any]) -> uncertainties.UFloat:
    settings = value.get("settings", {})

    if "value" in settings:
        return uncertainties.ufloat(settings["value"], settings.get("std_dev"))
    raise QiskitError(f"Bad ufloat settings: {settings}")


def _deserialize_type(value: dict):
    """Deserialize a Python type"""
    traceback_msg = None
    load_version = None

    if "." in value["name"]:
        raise QiskitError(f"Deserializing class members is no longer supported: {value}")
    try:
        name = value["name"]
        mod = value["module"]

        # These two conditionals handle previously serialized data before
        # dedicated ufloat and qiskit.quantum_info serializers were added.
        #
        # Perhaps they can be removed in the future
        if mod == "uncertainties.core" and name == "Variable":
            return uncertainties.ufloat
        if mod.startswith("qiskit.quantum_info"):
            return _load_quantum_info_type(value["name"], value["module"])

        _load_allowed_packages()
        package = mod.partition(".")[0]
        if package not in _allowed_packages:
            raise QiskitError(
                f"Import of {package} denied. It must be registered with the "
                "'qiskit_experiments.deserialization_packages' entry point to "
                "allow loading objects from it. See the documentation for "
                "'qiskit_experiments.framework.ExperimentEncoder'."
            )
        scope = importlib.import_module(mod)
        if not hasattr(scope, name):
            raise QiskitError(f"Requested object '{name}' not foudn in '{mod}'!")
        obj = getattr(scope, name)
        if not inspect.isclass(obj):
            raise QiskitError(f"Requested object '{name}' of '{mod}' is not a class!")
        if obj.__module__.partition(".")[0] != package:
            raise QiskitError(
                f"Object '{name}' of '{mod}' appears to come from {obj.__module__} instead!"
            )
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


def _serialize_object(obj: Any) -> dict:
    """Serialize a class instance from its init args and kwargs.

    Args:
        obj: The object to be serialized.

    Returns:
        Dict serialized class instance.
    """
    if hasattr(obj, "__json_encode__"):
        settings = obj.__json_encode__()
        has_json_encode = True
    else:
        settings = {}
        has_json_encode = False
    settings = _serialize_safe_float(settings)
    cls = type(obj)
    value = {
        "class": _serialize_type(cls),
        "settings": settings,
        "version": get_object_version(cls),
    }
    if _strict_serialization and not has_json_encode:
        # We do not expect to use _serialize_object except for cases where
        # __json_encode__ is defined
        raise ValueError(f"Unexpected default serialization for {value}")
    return {"__type__": "object", "__value__": value}


def _deserialize_object(value: dict) -> Any:
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
        traceback_msg = None
        warning_msg = (
            f"Could not deserialize instance of class {cls} from settings {settings}. "
            f"{cls}.__json_decode__ does not exist."
        )

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


class ExperimentEncoder(json.JSONEncoder):
    """JSON Encoder for Qiskit Experiments.

    .. warning::

        It is recommended only to deserialize data with
        :class:`ExperimentDecoder` from trusted sources. For custom classes,
        the deserialization procedure involves dynamic execution of code based
        on the content of the serialized data. The deserialization code
        includes some safeguards:

        1. Only modules registered with the
           ``qiskit_experiments.deserialization_packages`` entry point are
           imported dynamically.

        2. Only classes (as determined by Python's ``inspect.isclass``
           function) are referenced from the imported modules for further
           processing.

        3. These classes are checked to ensure that they were
           defined by the registered modules they were loaded from.

        4. For the referenced classes, only the ``__json_decode__`` method
           is called with the serialized data.

        Even with these safeguards, loading a payload involves instantiating
        registered classes with arbitrary inputs. These classes were not
        written assuming malicious input.

        Note that versions of Qiskit Experiments older than 0.14 could load
        arbitrary functions like ``subprocess.run`` and pass them data from
        the deserialization payload.

    This class extends the default Python JSONEncoder by including built-in
    support for

    * complex numbers, inf and NaN floats, sets, and dataclasses.
    * NumPy ndarrays and SciPy sparse matrices.
    * Qiskit ``QuantumCircuit``.
    * Any class that implements a ``__json_encode__`` method

    Custom classes can be serialized by this encoder by implementing a
    ``__json_encode__`` method. The serialization procedure is as follows:

    The ``__json_encode__`` method should have signature

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
        def __json_decode__(cls, value: Any) -> Self:
            # recover the object from the `value` returned by __json_encode__

    Additionally, the custom class's package metadata must register the top
    level import package as a ``qiskit_experiments.def`` Python entry point. In
    ``pyproject.toml`` the entry point registration would like this:

    .. code-block:: toml

        [project.entry-points."qiskit_experiments.deserialization_packages"]
        custom-package-name = "custom_package"

    where ``custom_package`` is the Python import module that the custom class
    is below (the import path before the first ``.``; ``custom_package`` for
    class ``MyClass`` if it is normally imported as ``from
    custom_package.subpackage import MyClass`` for example). The entry point
    name, ``custom-package-name`` in the example, is not used and can be set to
    any descriptive name.

    If the object has no ``__json_encode__`` method and all other special cases
    (numpy arrays, Qiskit quantum info classes, etc.) do not apply, the
    object is serialized as though ``__json_encode__`` returned an empty dict.
    Without a ``__json_decode__`` method, the object will be loaded by
    :class:`ExperimentDecoder` as a dictionary containing the name of the
    object's class and module. This incomplete loading of the object may lead
    to other code execution problems.

    .. note::

        Serialization of custom classes works for user-defined classes in
        Python scripts, notebooks, or third party modules. Note however
        that these will only be able to be de-serialized if that class
        can be imported form the same scope at the time the
        :class:`ExperimentDecoder` is invoked. For scripts and notebook, the
        scope is named ``__main__`` which is registered by default with the
        ``qiskit_experiments.deserialization_packages`` entry point.
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
        if isinstance(obj, uncertainties.UFloat):
            # This could be UFloat (AffineScalarFunc) or Variable.
            # UFloat is a base class of Variable that contains parameter correlation.
            # i.e. Variable is special subclass for single number.
            # Since this object is not serializable, we will drop correlation information
            # during serialization. Then both can be serialized as Variable.
            # Note that UFloat doesn't have a tag.
            return {
                "__type__": "ufloat",
                "__value__": {
                    "value": _serialize_safe_float(obj.nominal_value),
                    "std_dev": _serialize_safe_float(obj.std_dev),
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
        if _check_quantum_info_class(obj.__class__):
            return {
                "__type__": "qiskit.quantum_info",
                "__value__": {
                    "class": obj.__class__.__name__,
                    "settings": _serialize_safe_float(obj.settings),
                },
            }
        if isinstance(obj, Backend):
            return None
        try:
            return super().default(obj)
        except TypeError:
            return _serialize_object(obj)


class ExperimentDecoder(json.JSONDecoder):
    """JSON Decoder for Qiskit Experiments.

    .. warning::

        It is recommended to use this class only on trusted data. See the
        warning in the :class:`ExperimentEncoder` documentation for more
        details.

    This class extends the default Python JSONDecoder by including built-in
    support for all objects that that can be serialized using the
    :class:`ExperimentEncoder`.

    See :class:`ExperimentEncoder` class documentation for further details.
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
            if obj_type == "ufloat":
                return uncertainties.ufloat(obj_val["value"], obj_val["std_dev"])
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
            if obj_type == "qiskit.quantum_info" and hasattr(quantum_info, obj_val["class"]):
                cls = getattr(quantum_info, obj_val["class"])
                return cls(**obj_val["settings"])
            if obj_type == "safe_float":
                return self._NaNs.get(obj_val, obj_val)
            if _check_quantum_info_class(obj_val.get("class")):
                return obj_val["class"](**obj_val["settings"])
            if obj_val.get("class") is uncertainties.ufloat:
                return _deserialize_ufloat(obj_val)
            if obj_type == "object":
                return _deserialize_object(obj_val)
            if obj_type == "type":
                return _deserialize_type(obj_val)
        return obj
