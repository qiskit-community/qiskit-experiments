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

import qiskit.pulse as pulse
from qiskit.test import QiskitTestCase

from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon
from qiskit_experiments.exceptions import CalibrationError


class TestFixedFrequencyTransmon(QiskitTestCase):
    """Test the various setup methods."""

    def test_standard_single_qubit_gates(self):
        """Test the setup of single-qubit gates."""

        library = FixedFrequencyTransmon(default_values={"duration": 320})

        for gate in ["x", "sx"]:
            sched = library[gate]
            self.assertTrue(isinstance(sched, pulse.ScheduleBlock))
            self.assertEqual(len(sched.parameters), 5)

        sched_x = library["x"]
        sched_sx = library["sx"]

        self.assertEqual(sched_x.blocks[0].pulse.duration, sched_sx.blocks[0].pulse.duration)
        self.assertEqual(sched_x.blocks[0].pulse.sigma, sched_sx.blocks[0].pulse.sigma)

        expected = [
            (0.5, "amp", (), "x"),
            (0.0, "β", (), "x"),
            (320, "duration", (), "x"),
            (80, "σ", (), "x"),
            (320, "duration", (), "sx"),
            (0.0, "β", (), "sx"),
            (0.25, "amp", (), "sx"),
            (80, "σ", (), "sx"),
        ]

        for param_conf in library.default_values():
            self.assertTrue(param_conf in expected)

        # Check that an error gets raise if the gate is not in the library.
        with self.assertRaises(CalibrationError):
            print(library["bswap"])

        # Test the basis gates of the library.
        self.assertListEqual(library.basis_gates, ["x", "sx"])

    def test_turn_off_drag(self):
        """Test the use_drag parameter."""

        library = FixedFrequencyTransmon(use_drag=False)
        self.assertTrue(isinstance(library["x"].blocks[0].pulse, pulse.Gaussian))

        library = FixedFrequencyTransmon()
        self.assertTrue(isinstance(library["x"].blocks[0].pulse, pulse.Drag))
