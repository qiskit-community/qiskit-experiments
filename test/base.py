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
import warnings
from typing import Any, Callable, Optional

import numpy as np
from qiskit.test import QiskitTestCase
from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.database_service.db_experiment_data import ExperimentStatus
from qiskit_experiments.framework import (
    ExperimentDecoder,
    ExperimentEncoder,
    ExperimentData,
    BaseExperiment,
    BaseAnalysis,
)


class QiskitExperimentsTestCase(QiskitTestCase):
    """Qiskit Experiments specific extra functionality for test cases."""

    def assertExperimentDone(self, experiment_data: ExperimentData):
        """Blocking execution of next line until all threads are completed then
        checks if status returns Done.

        Args:
            experiment_data: Experiment data to evaluate.
        """
        experiment_data.block_for_results()

        self.assertEqual(
            experiment_data.status(),
            ExperimentStatus.DONE,
            msg="All threads are executed but status is not DONE. " + experiment_data.errors(),
        )

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
    def json_equiv(data1, data2) -> bool:
        """Check if two experiments are equivalent by comparing their configs"""
        # pylint: disable = too-many-return-statements
        configrable_type = (BaseExperiment, BaseAnalysis)
        list_type = (list, tuple, set)
        skipped = (Calibrations,)

        if isinstance(data1, skipped) and isinstance(data2, skipped):
            warnings.warn(f"Equivalence check for data {data1.__class__.__name__} is skipped.")
            return True
        elif isinstance(data1, configrable_type) and isinstance(data2, configrable_type):
            return QiskitExperimentsTestCase.json_equiv(data1.config(), data2.config())
        elif dataclasses.is_dataclass(data1) and dataclasses.is_dataclass(data2):
            # not using asdict. this copies all objects.
            return QiskitExperimentsTestCase.json_equiv(data1.__dict__, data2.__dict__)
        elif isinstance(data1, dict) and isinstance(data2, dict):
            if set(data1) != set(data2):
                return False
            return all(
                QiskitExperimentsTestCase.json_equiv(data1[k], data2[k]) for k in data1.keys()
            )
        elif isinstance(data1, np.ndarray) or isinstance(data2, np.ndarray):
            return np.allclose(data1, data2)
        elif isinstance(data1, list_type) and isinstance(data2, list_type):
            return all(QiskitExperimentsTestCase.json_equiv(e1, e2) for e1, e2 in zip(data1, data2))

        return data1 == data2
