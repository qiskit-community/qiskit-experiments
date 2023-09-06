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

import os
import json
import pickle
import warnings
from typing import Any, Callable, Optional

import fixtures
import uncertainties
from qiskit.test import QiskitTestCase
from qiskit.utils.deprecation import deprecate_func

from qiskit_experiments.framework import (
    ExperimentDecoder,
    ExperimentEncoder,
    ExperimentData,
)
from qiskit_experiments.framework.experiment_data import ExperimentStatus
from .extended_equality import is_equivalent

# Fail tests that take longer than this
TEST_TIMEOUT = os.environ.get("TEST_TIMEOUT", 60)


class QiskitExperimentsTestCase(QiskitTestCase):
    """Qiskit Experiments specific extra functionality for test cases."""

    def setUp(self):
        super().setUp()
        self.useFixture(fixtures.Timeout(TEST_TIMEOUT, gentle=False))

    @classmethod
    def setUpClass(cls):
        """Set-up test class."""
        super().setUpClass()

        # Some functionality may be deprecated in Qiskit Experiments. If the deprecation warnings aren't
        # filtered, the tests will fail as ``QiskitTestCase`` sets all warnings to be treated as an error
        # by default.
        # pylint: disable=invalid-name
        allow_deprecationwarning_message = [
            # TODO: Remove in 0.6, when submodule `.curve_analysis.visualization` is removed.
            r".*Plotting and drawing functionality has been moved",
            r".*Legacy drawers from `.curve_analysis.visualization are deprecated",
        ]
        for msg in allow_deprecationwarning_message:
            warnings.filterwarnings("default", category=DeprecationWarning, message=msg)

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

    def assertEqualExtended(
        self,
        first: Any,
        second: Any,
        *,
        msg: Optional[str] = None,
        strict_type: bool = False,
    ):
        """Extended equality assertion which covers Qiskit Experiments classes.

        .. note::
            Some Qiskit Experiments class may intentionally avoid implementing
            the equality dunder method, or may be used in some unusual situations.
            These are mainly caused by to JSON round trip situation, and some custom classes
            doesn't guarantee object equality after round trip.
            This assertion function forcibly compares input two objects with
            the custom equality checker, which is implemented for unittest purpose.

        Args:
            first: First object to compare.
            second: Second object to compare.
            msg: Optional. Custom error message issued when first and second object are not equal.
            strict_type: Set True to enforce type check before comparison.
        """
        default_msg = f"{first} != {second}"

        self.assertTrue(
            is_equivalent(first, second, strict_type=strict_type),
            msg=msg or default_msg,
        )

    def assertRoundTripSerializable(
        self,
        obj: Any,
        *,
        check_func: Optional[Callable] = None,
        strict_type: bool = False,
    ):
        """Assert that an object is round trip serializable.

        Args:
            obj: the object to be serialized.
            check_func: Optional, a custom function ``check_func(a, b) -> bool``
                to check equality of the original object with the decoded
                object. If None :meth:`.assertEqualExtended` is called.
            strict_type: Set True to enforce type check before comparison.
        """
        try:
            encoded = json.dumps(obj, cls=ExperimentEncoder)
        except TypeError:
            self.fail("JSON serialization raised unexpectedly.")
        try:
            decoded = json.loads(encoded, cls=ExperimentDecoder)
        except TypeError:
            self.fail("JSON deserialization raised unexpectedly.")

        if check_func is not None:
            self.assertTrue(check_func(obj, decoded), msg=f"{obj} != {decoded}")
        else:
            self.assertEqualExtended(obj, decoded, strict_type=strict_type)

    def assertRoundTripPickle(
        self,
        obj: Any,
        *,
        check_func: Optional[Callable] = None,
        strict_type: bool = False,
    ):
        """Assert that an object is round trip serializable using pickle module.

        Args:
            obj: the object to be serialized.
            check_func: Optional, a custom function ``check_func(a, b) -> bool``
                to check equality of the original object with the decoded
                object. If None :meth:`.assertEqualExtended` is called.
            strict_type: Set True to enforce type check before comparison.
        """
        try:
            encoded = pickle.dumps(obj)
        except TypeError:
            self.fail("pickle raised unexpectedly.")
        try:
            decoded = pickle.loads(encoded)
        except TypeError:
            self.fail("pickle deserialization raised unexpectedly.")

        if check_func is not None:
            self.assertTrue(check_func(obj, decoded), msg=f"{obj} != {decoded}")
        else:
            self.assertEqualExtended(obj, decoded, strict_type=strict_type)

    @classmethod
    @deprecate_func(
        since="0.6",
        additional_msg="Use test.extended_equality.is_equivalent instead.",
        pending=True,
        package_name="qiskit-experiments",
    )
    def json_equiv(cls, data1, data2) -> bool:
        """Check if two experiments are equivalent by comparing their configs"""
        return is_equivalent(data1, data2)

    @staticmethod
    @deprecate_func(
        since="0.6",
        additional_msg="Use test.extended_equality.is_equivalent instead.",
        pending=True,
        package_name="qiskit-experiments",
    )
    def ufloat_equiv(data1: uncertainties.UFloat, data2: uncertainties.UFloat) -> bool:
        """Check if two values with uncertainties are equal. No correlation is considered."""
        return is_equivalent(data1, data2)

    @classmethod
    @deprecate_func(
        since="0.6",
        additional_msg="Use test.extended_equality.is_equivalent instead.",
        pending=True,
        package_name="qiskit-experiments",
    )
    def analysis_result_equiv(cls, result1, result2):
        """Test two analysis results are equivalent"""
        return is_equivalent(result1, result2)

    @classmethod
    @deprecate_func(
        since="0.6",
        additional_msg="Use test.extended_equality.is_equivalent instead.",
        pending=True,
        package_name="qiskit-experiments",
    )
    def curve_fit_data_equiv(cls, data1, data2):
        """Test two curve fit result are equivalent."""
        return is_equivalent(data1, data2)

    @classmethod
    @deprecate_func(
        since="0.6",
        additional_msg="Use test.extended_equality.is_equivalent instead.",
        pending=True,
        package_name="qiskit-experiments",
    )
    def experiment_data_equiv(cls, data1, data2):
        """Check two experiment data containers are equivalent"""
        return is_equivalent(data1, data2)
