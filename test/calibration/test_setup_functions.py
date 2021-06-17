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

"""Class to test the calibrations setup methods."""

from qiskit.pulse import Drag, DriveChannel
import qiskit.pulse as pulse
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeAthens

from qiskit_experiments.calibration.calibrations_setup import standard_single_qubit_gates


class TestCalibrationsSetup(QiskitTestCase):
    """Test the various setup methods."""

    def test_standard_single_qubit_gates(self):
        """Test the setup of single-qubit gates."""

        # Test non-linked parameters
        cals = standard_single_qubit_gates(FakeAthens())

        tests = [
            ("xp", 0.2),
            ("xm", -0.2),
            ("x90p", 0.1),
            ("x90m", -0.1),
            ("y90p", 0.1j),
            ("y90m", -0.1j),
        ]

        for name, amp in tests:
            for qubit in range(5):
                with pulse.build(name=name) as sched:
                    pulse.play(Drag(160, amp, 40, 0.0), DriveChannel(qubit))

                self.assertEqual(cals.get_schedule(name, (qubit,)), sched)

        xp = cals.get_template("xp", (0,))
        xm = cals.get_template("xm", (0,))

        self.assertNotEqual(xp.parameters, xm.parameters)

        # Test linked parameters
        cals = standard_single_qubit_gates(FakeAthens(), link_amplitudes=True, link_drag=True)

        xp = cals.get_template("xp", (0,))
        xm = cals.get_template("xm", (0,))

        self.assertEqual(xp.parameters, xm.parameters)
