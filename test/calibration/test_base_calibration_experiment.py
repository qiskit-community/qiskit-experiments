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

"""Tests for the base class for calibration-type experiments."""

from test.base import QiskitExperimentsTestCase

from qiskit_experiments.library import QubitSpectroscopy
from qiskit_experiments.calibration_management.calibrations import Calibrations
from qiskit_experiments.calibration_management.base_calibration_experiment import (
    BaseCalibrationExperiment,
)


class TestBaseCalibrationClass(QiskitExperimentsTestCase):
    """Tests for base calibration experiment classes."""

    def test_class_order(self):
        """Test warnings when the BaseCalibrationExperiment is not the first parent."""

        class CorrectOrder(BaseCalibrationExperiment, QubitSpectroscopy):
            """A class with the correct order should not produce warnings.."""

            def __init__(self):
                """A dummy class for parent order testing."""
                super().__init__(Calibrations(), 0, [0, 1, 2])

        CorrectOrder()

        with self.assertWarns(Warning):

            # pylint: disable=unused-variable
            class WrongOrder(QubitSpectroscopy, BaseCalibrationExperiment):
                """Merely defining this class is enough to raise the warning."""

                def __init__(self):
                    """A dummy class for parent order testing."""
                    super().__init__(Calibrations(), 0, [0, 1, 2])
