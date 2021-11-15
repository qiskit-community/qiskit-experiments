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

from test.base import QiskitExperimentsTestCase
import qiskit.pulse as pulse

from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon
from qiskit_experiments.exceptions import CalibrationError


class TestFixedFrequencyTransmon(QiskitExperimentsTestCase):
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

    def test_setup_partial_gates(self):
        """Check that we do not setup all gates if not required."""

        library = FixedFrequencyTransmon(basis_gates=["x", "sy"])

        self.assertTrue("x" in library)
        self.assertTrue("sy" in library)
        self.assertTrue("y" not in library)
        self.assertTrue("sx" not in library)

        with self.assertRaises(CalibrationError):
            FixedFrequencyTransmon(basis_gates=["x", "bswap"])

    def test_serialization(self):
        """Test the serialization of the object."""

        lib1 = FixedFrequencyTransmon(
            basis_gates=["x", "sy"],
            default_values={"duration": 320},
            use_drag=False,
            link_parameters=False,
        )

        lib2 = FixedFrequencyTransmon.from_config(lib1.config)

        self.assertEqual(lib2.basis_gates, lib1.basis_gates)

        # Note: we convert to string since the parameters prevent a direct comparison.
        self.assertTrue(self._test_library_equivalence(lib1, lib2))

        # Test that the extra args are properly accounted for.
        lib3 = FixedFrequencyTransmon(
            basis_gates=["x", "sy"],
            default_values={"duration": 320},
            use_drag=True,
            link_parameters=False,
        )

        self.assertFalse(self._test_library_equivalence(lib1, lib3))

    def _test_library_equivalence(self, lib1, lib2) -> bool:
        """Test if libraries are equivalent."""

        if len(set(lib1.basis_gates)) != len(set(lib2.basis_gates)):
            return False

        for gate in lib1.basis_gates:
            if str(lib1[gate]) != str(lib2[gate]):
                return False

        return True
