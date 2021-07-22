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
        sched_y = library["y"]
        sched_sx = library["sx"]
        sched_sy = library["sy"]

        self.assertEqual(sched_x.blocks[0].pulse.duration, sched_sx.blocks[0].pulse.duration)
        self.assertEqual(sched_x.blocks[0].pulse.sigma, sched_sx.blocks[0].pulse.sigma)

        self.assertEqual(len(sched_x.parameters & sched_y.parameters), 4)
        self.assertEqual(len(sched_sx.parameters & sched_sy.parameters), 4)

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
        self.assertListEqual(library.basis_gates, ["x", "y", "sx", "sy"])

    def test_turn_off_drag(self):
        """Test the use_drag parameter."""

        library = FixedFrequencyTransmon(use_drag=False)
        self.assertTrue(isinstance(library["x"].blocks[0].pulse, pulse.Gaussian))

        library = FixedFrequencyTransmon()
        self.assertTrue(isinstance(library["x"].blocks[0].pulse, pulse.Drag))

    def test_unlinked_parameters(self):
        """Test the we get schedules with unlinked parameters."""

        library = FixedFrequencyTransmon(link_parameters=False)

        sched_x = library["x"]
        sched_y = library["y"]
        sched_sx = library["sx"]
        sched_sy = library["sy"]

        # Test the number of parameters.
        self.assertEqual(len(sched_x.parameters & sched_y.parameters), 2)
        self.assertEqual(len(sched_sx.parameters & sched_sy.parameters), 2)

        expected = [
            (0.5, "amp", (), "x"),
            (0.0, "β", (), "x"),
            (160, "duration", (), "x"),
            (40, "σ", (), "x"),
            (160, "duration", (), "sx"),
            (0.0, "β", (), "sx"),
            (0.25, "amp", (), "sx"),
            (40, "σ", (), "sx"),
            (0.5j, "amp", (), "y"),
            (0.0, "β", (), "y"),
            (160, "duration", (), "y"),
            (40, "σ", (), "y"),
            (160, "duration", (), "sy"),
            (0.0, "β", (), "sy"),
            (0.25j, "amp", (), "sy"),
            (40, "σ", (), "sy"),
        ]

        self.assertSetEqual(set(library.default_values()), set(expected))

    def setup_partial_gates(self):
        """Check that we do not setup all gates if not required."""

        library = FixedFrequencyTransmon(basis_gates=["x", "sy"])

        self.assertTrue("x" in library)
        self.assertTrue("sy" in library)
        self.assertTrue("y" not in library)
        self.assertTrue("sx" not in library)

        with self.assertRaises(CalibrationError):
            FixedFrequencyTransmon(basis_gates=["x", "bswap"])
