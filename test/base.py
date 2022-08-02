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
import pickle
import warnings
from typing import Any, Callable, Optional

import numpy as np
import uncertainties
from lmfit import Model
from qiskit.test import QiskitTestCase
from qiskit_experiments.data_processing import DataAction, DataProcessor
from qiskit_experiments.framework.experiment_data import ExperimentStatus
from qiskit_experiments.framework import (
    ExperimentDecoder,
    ExperimentEncoder,
    ExperimentData,
    BaseExperiment,
    BaseAnalysis,
)
from qiskit_experiments.curve_analysis.visualization.base_drawer import BaseCurveDrawer
from qiskit_experiments.curve_analysis.curve_data import CurveFitResult


class QiskitExperimentsTestCase(QiskitTestCase):
    """Qiskit Experiments specific extra functionality for test cases."""

    def assertExperimentDone(
        self,
        experiment_data: ExperimentData,
        timeout: float = 120,
    ):
        """Blocking execution of next line until all threads are completed then
        checks if status returns Done.

        Args:
            experiment_data: Experiment data to evaluate.
            timeout: The maximum time in seconds to wait for executor to complete.
        """
        experiment_data.block_for_results(timeout=timeout)

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

    def assertRoundTripPickle(self, obj: Any, check_func: Optional[Callable] = None):
        """Assert that an object is round trip serializable using pickle module.

        Args:
            obj: the object to be serialized.
            check_func: Optional, a custom function ``check_func(a, b) -> bool``
                        to check equality of the original object with the decoded
                        object. If None the ``__eq__`` method of the original
                        object will be used.
        """
        try:
            encoded = pickle.dumps(obj)
        except TypeError:
            self.fail("pickle raised unexpectedly.")
        try:
            decoded = pickle.loads(encoded)
        except TypeError:
            self.fail("pickle deserialization raised unexpectedly.")
        if check_func is None:
            self.assertEqual(obj, decoded)
        else:
            self.assertTrue(check_func(obj, decoded), msg=f"{obj} != {decoded}")

    @classmethod
    def json_equiv(cls, data1, data2) -> bool:
        """Check if two experiments are equivalent by comparing their configs"""
        # pylint: disable = too-many-return-statements
        configrable_type = (BaseExperiment, BaseAnalysis, BaseCurveDrawer)
        compare_repr = (DataAction, DataProcessor)
        list_type = (list, tuple, set)
        skipped = tuple()

        if isinstance(data1, skipped) and isinstance(data2, skipped):
            warnings.warn(f"Equivalence check for data {data1.__class__.__name__} is skipped.")
            return True
        elif isinstance(data1, configrable_type) and isinstance(data2, configrable_type):
            return cls.json_equiv(data1.config(), data2.config())
        elif dataclasses.is_dataclass(data1) and dataclasses.is_dataclass(data2):
            # not using asdict. this copies all objects.
            return cls.json_equiv(data1.__dict__, data2.__dict__)
        elif isinstance(data1, dict) and isinstance(data2, dict):
            if set(data1) != set(data2):
                return False
            return all(cls.json_equiv(data1[k], data2[k]) for k in data1.keys())
        elif isinstance(data1, np.ndarray) or isinstance(data2, np.ndarray):
            return np.allclose(data1, data2)
        elif isinstance(data1, list_type) and isinstance(data2, list_type):
            return all(cls.json_equiv(e1, e2) for e1, e2 in zip(data1, data2))
        elif isinstance(data1, uncertainties.UFloat) and isinstance(data2, uncertainties.UFloat):
            return cls.ufloat_equiv(data1, data2)
        elif isinstance(data1, Model) and isinstance(data2, Model):
            return cls.json_equiv(data1.dumps(), data2.dumps())
        elif isinstance(data1, CurveFitResult) and isinstance(data2, CurveFitResult):
            return cls.curve_fit_data_equiv(data1, data2)
        elif isinstance(data1, compare_repr) and isinstance(data2, compare_repr):
            # otherwise compare instance representation
            return repr(data1) == repr(data2)

        return data1 == data2

    @staticmethod
    def ufloat_equiv(data1: uncertainties.UFloat, data2: uncertainties.UFloat) -> bool:
        """Check if two values with uncertainties are equal. No correlation is considered."""
        return data1.n == data2.n and data1.s == data2.s

    @classmethod
    def analysis_result_equiv(cls, result1, result2):
        """Test two analysis results are equivalent"""
        # Check basic attributes skipping service which is not serializable
        for att in [
            "name",
            "value",
            "extra",
            "device_components",
            "result_id",
            "experiment_id",
            "chisq",
            "quality",
            "verified",
            "tags",
            "auto_save",
            "source",
        ]:
            if not cls.json_equiv(getattr(result1, att), getattr(result2, att)):
                return False
        return True

    @classmethod
    def curve_fit_data_equiv(cls, data1, data2):
        """Test two curve fit result are equivalent."""
        for att in [
            "method",
            "model_repr",
            "success",
            "nfev",
            "message",
            "dof",
            "init_params",
            "chisq",
            "reduced_chisq",
            "aic",
            "bic",
            "params",
            "var_names",
            "x_data",
            "y_data",
            "covar",
        ]:
            if not cls.json_equiv(getattr(data1, att), getattr(data2, att)):
                return False
        return True

    @classmethod
    def experiment_data_equiv(cls, data1, data2):
        """Check two experiment data containers are equivalent"""

        # Check basic attrbiutes
        # Skip non-compatible backend
        for att in [
            "experiment_id",
            "experiment_type",
            "parent_id",
            "tags",
            "job_ids",
            "figure_names",
            "share_level",
            "metadata",
        ]:
            if not cls.json_equiv(getattr(data1, att), getattr(data2, att)):
                return False

        # Check length of data, results, child_data
        # check for child data attribute so this method still works for
        # DbExperimentData
        if hasattr(data1, "child_data"):
            child_data1 = data1.child_data()
        else:
            child_data1 = []
        if hasattr(data2, "child_data"):
            child_data2 = data2.child_data()
        else:
            child_data2 = []

        if (
            len(data1.data()) != len(data2.data())
            or len(data1.analysis_results()) != len(data2.analysis_results())
            or len(child_data1) != len(child_data2)
        ):
            return False

        # Check data
        if not cls.json_equiv(data1.data(), data2.data()):
            return False

        # Check analysis resultsx
        for result1, result2 in zip(data1.analysis_results(), data2.analysis_results()):
            if not cls.analysis_result_equiv(result1, result2):
                return False

        # Check child data
        for child1, child2 in zip(child_data1, child_data2):
            if not cls.experiment_data_equiv(child1, child2):
                return False

        return True
