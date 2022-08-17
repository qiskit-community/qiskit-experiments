# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the utility functions."""

from test.base import QiskitExperimentsTestCase

from qiskit.circuit import Parameter
import qiskit.pulse as pulse

from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.calibration_management import EchoCrossResonance
from qiskit_experiments.calibration_management.calibration_utils import validate_channels


class TestValidateChannels(QiskitExperimentsTestCase):
    """Test validate channels."""

    def test_ecr_lib(self):
        """Test channel validation."""

        # Test schedules with references.
        lib = EchoCrossResonance()

        self.assertEqual(validate_channels(lib["ecr"]), set())

        # Has a drive channel and a control channel that should be valid
        self.assertEqual(len(validate_channels(lib["cr45p"])), 2)

    def test_raise_on_multiple_parameters(self):
        """Test that an error is raised on a sum of parameters"""
        p1, p2 = Parameter("p1"), Parameter("p2")

        with pulse.build() as sched:
            pulse.play(pulse.Drag(160, 0.5, 40, 0), pulse.DriveChannel(p1 + p2))

        with self.assertRaises(CalibrationError):
            validate_channels(sched)

    def test_invalid_name(self):
        """Test that an error is raised on an invalid channel name."""
        with pulse.build() as sched:
            pulse.play(pulse.Drag(160, 0.5, 40, 0), pulse.DriveChannel(Parameter("0&1")))

        with self.assertRaises(CalibrationError):
            validate_channels(sched)
