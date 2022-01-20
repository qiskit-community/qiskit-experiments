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
"""
Qiskit Experiments test case class
"""

import dataclasses
import json
from typing import Any, Callable, Optional

import numpy as np
from qiskit.test import QiskitTestCase
from qiskit_experiments.framework import (
    ExperimentDecoder,
    ExperimentEncoder,
    BaseExperiment,
    BaseAnalysis,
)


class QiskitExperimentsTestCase(QiskitTestCase):
    """Qiskit Experiments specific extra functionality for test cases."""

    def assertRoundTripSerializable(self, obj: Any, check_func: Optional[Callable] = None):
        """Assert that an object is round trip serializable.

        Args:
            obj: the object to be serialized.
            check_func: Optional, a custom function ``check_func(a, b) -> bool``
                        to check equality of the original object with the decoded
                        object. If None the ``__eq__`` method of the original
                        object will be used.
        """
        try:
            encoded = json.dumps(obj, cls=ExperimentEncoder)
        except TypeError:
            self.fail("JSON serialization raised unexpectedly.")
        try:
            decoded = json.loads(encoded, cls=ExperimentDecoder)
        except TypeError:
            self.fail("JSON deserialization raised unexpectedly.")
        if check_func is None:
            self.assertEqual(obj, decoded)
        else:
            self.assertTrue(check_func(obj, decoded), msg=f"{obj} != {decoded}")

    @staticmethod
    def experiments_equiv(exp1, exp2) -> bool:
        """Check if two experiments are equivalent by comparing their configs"""
        config1 = exp1.config()
        config2 = exp2.config()
        try:
            if config1 == config2:
                return True
        except ValueError:
            pass

        return _test_all_elements_equiv(exp1.config(), exp2.config())


def _test_all_elements_equiv(data1, data2) -> bool:
    """A helper function to check if two data are equivalent."""
    # pylint: disable = too-many-return-statements
    configrable_type = (BaseExperiment, BaseAnalysis)
    list_type = (list, tuple)

    if isinstance(data1, configrable_type) and isinstance(data2, configrable_type):
        return _test_all_elements_equiv(data1.config(), data2.config())
    elif dataclasses.is_dataclass(data1) and dataclasses.is_dataclass(data2):
        return _test_all_elements_equiv(dataclasses.asdict(data1), dataclasses.asdict(data2))
    elif isinstance(data1, dict) and isinstance(data2, dict):
        if set(data1) != set(data2):
            return False
        return all(_test_all_elements_equiv(data1[k], data2[k]) for k in data1.keys())
    elif isinstance(data1, np.ndarray) or isinstance(data2, np.ndarray):
        return np.allclose(data1, data2)
    elif isinstance(data1, list_type) and isinstance(data2, list_type):
        return all(_test_all_elements_equiv(e1, e2) for e1, e2 in zip(data1, data2))

    return data1 == data2
