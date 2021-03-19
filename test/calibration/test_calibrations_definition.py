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

"""Test the class that holds the calibrations."""

from datetime import datetime

from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeAlmaden
from qiskit.circuit import Parameter
import qiskit.pulse as pulse
from qiskit.pulse import Drag, DriveChannel
from qiskit_experiments.calibration import CalibrationsDefinition
from qiskit_experiments.calibration import ParameterValue


class TestCalibrationsDefinition(QiskitTestCase):
    """Class to test the calibration definitions."""

    def test_simple_schedule(self):
        """Test that we can add a schedule."""

        sigma = Parameter('σ')
        d0 = Parameter('0')
        amp = Parameter('A')
        beta = Parameter('β')

        with pulse.build(name='xp') as xp:
            pulse.play(Drag(160, amp, sigma, beta), DriveChannel(d0))

        cals = CalibrationsDefinition(FakeAlmaden())
        cals.add_schedules([xp])

        # Check that there is a schedule in cals
        self.assertEqual(len(cals.schedules()), 1)

        # Check that we can get the schedule for different drive parameters.
        sched = cals.get_schedule('xp', (0, ), ['σ', 'A', 'β'])
        self.assertEqual(sched.instructions[0][1].channel, DriveChannel(0))

        sched = cals.get_schedule('xp', (3, ), ['σ', 'A', 'β'])
        self.assertEqual(sched.instructions[0][1].channel, DriveChannel(3))

        for param in [sigma, amp, beta]:
            self.assertTrue(param in sched.parameters)

        # Add a parameter
        cals.add_parameter_value('A', ParameterValue(0.123, datetime.now()), chs=[DriveChannel(0)])

        sched = cals.get_schedule('xp', (0, ), ['σ', 'β'])
        self.assertEqual(sched.instructions[0][1].pulse.amp, 0.123)
        self.assertTrue(amp not in sched.parameters)
