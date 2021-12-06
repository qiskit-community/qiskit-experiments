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

from typing import Any, Callable, Optional
import json
import numpy as np

from qiskit.test import QiskitTestCase
from qiskit_experiments.framework import ExperimentDecoder, ExperimentEncoder


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
        # pylint: disable = too-many-boolean-expressions, too-many-return-statements
        config1 = exp1.config()
        config2 = exp2.config()
        try:
            if config1 == config2:
                return True
        except ValueError:
            pass
        if (
            config1.cls != config2.cls
            or len(config1.args) != len(config2.args)
            or len(config1.kwargs) != len(config2.kwargs)
            or len(config1.experiment_options) != len(config2.experiment_options)
            or len(config1.transpile_options) != len(config2.transpile_options)
            or len(config1.run_options) != len(config2.run_options)
        ):
            return False

        # Check each entry
        for arg1, arg2 in zip(config1.args, config2.args):
            if isinstance(arg1, np.ndarray) or isinstance(arg2, np.ndarray):
                if not np.all(np.asarray(arg1) == np.asarray(arg2)):
                    return False
            elif isinstance(arg1, tuple) or isinstance(arg2, tuple):
                # JSON serialization converts tuples to lists
                if list(arg1) != list(arg2):
                    return False
            elif arg1 != arg2:
                return False
        for attr in ["kwargs", "experiment_options", "transpile_options", "run_options"]:
            dict1 = getattr(config1, attr)
            dict2 = getattr(config2, attr)
            for key1, val1 in dict1.items():
                val2 = dict2[key1]
                if isinstance(val1, np.ndarray) or isinstance(val2, np.ndarray):
                    if not np.allclose(val1, val2):
                        return False
                elif isinstance(val1, tuple) or isinstance(val2, tuple):
                    if list(val1) != list(val2):
                        return False
                elif val1 != val2:
                    return False
        return True
