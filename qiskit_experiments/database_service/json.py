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
# pylint: disable=method-hidden

"""Experiment serialization methods."""

import json
from typing import Any

import numpy as np

from .utils import FitVal


class ExperimentEncoder(json.JSONEncoder):
    """JSON Encoder for Numpy arrays and complex numbers."""

    def default(self, obj: Any) -> Any:  # pylint: disable=arguments-differ
        if hasattr(obj, "tolist"):
            return {"__type__": "array", "__value__": obj.tolist()}
        if isinstance(obj, complex):
            return {"__type__": "complex", "__value__": [obj.real, obj.imag]}
        if isinstance(obj, FitVal):
            return {
                "__type__": obj.__class__.__name__,
                "__value__": obj.value,
                "__stderr__": obj.stderr,
                "__unit__": obj.unit,
            }
        if callable(obj):
            return {"__type__": "callable", "__value__": obj.__name__}
        try:
            return super().default(obj)
        except TypeError:
            return {"__type__": "__class_name__", "__value__": obj.__class__.__name__}


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
            if obj["__type__"] == FitVal.__name__:
                return FitVal(
                    value=obj["__value__"], stderr=obj["__stderr__"], unit=obj["__unit__"]
                )
            if obj["__type__"] == "callable":
                return obj["__value__"]
            if obj["__type__"] == "__class_name__":
                return obj["__value__"]
        return obj
