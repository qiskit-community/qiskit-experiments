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

"""Class to test the backend calibrations."""

import unittest
from test.base import QiskitExperimentsTestCase

from qiskit import transpile, QuantumCircuit
from qiskit.circuit import Parameter, Gate
import qiskit.pulse as pulse
from qiskit.test.mock import FakeArmonk, FakeBelem

from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.calibration_management import BackendCalibrations
from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon


class TestBackendCalibrations(QiskitExperimentsTestCase):
    """Class to test the functionality of a BackendCalibrations"""

    def test_run_options(self):
        """Test that we can get run options."""
        cals = BackendCalibrations(FakeArmonk())

        self.assertEqual(cals.get_meas_frequencies(), [6993370669.000001])
        self.assertEqual(cals.get_qubit_frequencies(), [4971852852.405576])

    def test_setup_withLibrary(self):
        """Test that we can setup with a library."""

        cals = BackendCalibrations(
            FakeArmonk(),
            library=FixedFrequencyTransmon(
                basis_gates=["x", "sx"], default_values={"duration": 320}
            ),
        )

        # Check the x gate
        with pulse.build(name="x") as expected:
            pulse.play(pulse.Drag(duration=320, amp=0.5, sigma=80, beta=0), pulse.DriveChannel(0))

        self.assertEqual(cals.get_schedule("x", (0,)), expected)

        # Check the sx gate
        with pulse.build(name="sx") as expected:
            pulse.play(pulse.Drag(duration=320, amp=0.25, sigma=80, beta=0), pulse.DriveChannel(0))

        self.assertEqual(cals.get_schedule("sx", (0,)), expected)

    def test_instruction_schedule_map_export(self):
        """Test that exporting the inst map works as planned."""

        backend = FakeBelem()

        cals = BackendCalibrations(
            backend,
            library=FixedFrequencyTransmon(basis_gates=["sx"]),
        )

        u_chan = pulse.ControlChannel(Parameter("ch0.1"))
        with pulse.build(name="cr") as cr:
            pulse.play(pulse.GaussianSquare(640, 0.5, 64, 384), u_chan)

        cals.add_schedule(cr, num_qubits=2)
        cals.update_inst_map({"cr"})

        for qubit in range(backend.configuration().num_qubits):
            self.assertTrue(cals.default_inst_map.has("sx", (qubit,)))

        # based on coupling map of Belem to keep the test robust.
        expected_pairs = [(0, 1), (1, 0), (1, 2), (2, 1), (1, 3), (3, 1), (3, 4), (4, 3)]
        coupling_map = set(tuple(pair) for pair in backend.configuration().coupling_map)

        for pair in expected_pairs:
            self.assertTrue(pair in coupling_map)
            self.assertTrue(cals.default_inst_map.has("cr", pair), pair)

    @unittest.skip("Requires qiskit terra >= 0.19.0")
    def test_inst_map_transpilation(self):
        """Test that we can use the inst_map to inject the cals into the circuit."""

        cals = BackendCalibrations(
            FakeArmonk(),
            library=FixedFrequencyTransmon(basis_gates=["x"]),
        )

        param = Parameter("amp")
        cals.inst_map_add("Rabi", (0,), "x", assign_params={"amp": param})

        circ = QuantumCircuit(1)
        circ.x(0)
        circ.append(Gate("Rabi", num_qubits=1, params=[param]), (0,))

        circs, amps = [], [0.12, 0.25]

        for amp in amps:
            new_circ = circ.assign_parameters({param: amp}, inplace=False)
            circs.append(new_circ)

        # Check that calibrations are absent
        for circ in circs:
            self.assertEqual(len(circ.calibrations), 0)

        # Transpile to inject the cals.
        # TODO Enable this test once terra 0.19.0 is live.
        circs = transpile(circs)  # TODO add: inst_map=cals.instruction_schedule_map

        # Check that we have the expected schedules.
        with pulse.build() as x_expected:
            pulse.play(pulse.Drag(160, 0.5, 40, 0), pulse.DriveChannel(0))

        for idx, circ in enumerate(circs):
            amp = amps[idx]

            with pulse.build() as rabi_expected:
                pulse.play(pulse.Drag(160, amp, 40, 0), pulse.DriveChannel(0))

            self.assertEqual(circ.calibrations["x"][((0,), ())], x_expected)

            circ_rabi = next(iter(circ.calibrations["Rabi"].values()))
            self.assertEqual(circ_rabi, rabi_expected)

        # Test the removal of the Rabi instruction
        self.assertTrue(cals.default_inst_map.has("Rabi", (0,)))

        cals.default_inst_map.remove("Rabi", (0,))

        self.assertFalse(cals.default_inst_map.has("Rabi", (0,)))

    def test_inst_map_updates(self):
        """Test that updating a parameter will force an inst map update."""

        cals = BackendCalibrations(
            FakeBelem(),
            library=FixedFrequencyTransmon(basis_gates=["sx", "x"]),
        )

        # Test the schedules before the update.
        for qubit in range(5):
            for gate, amp in [("x", 0.5), ("sx", 0.25)]:
                with pulse.build() as expected:
                    pulse.play(pulse.Drag(160, amp, 40, 0), pulse.DriveChannel(qubit))

                self.assertEqual(cals.default_inst_map.get(gate, qubit), expected)

        # Update the duration, this should impact all gates.
        cals.add_parameter_value(200, "duration", schedule="sx")

        # Test that all schedules now have an updated duration in the inst_map
        for qubit in range(5):
            for gate, amp in [("x", 0.5), ("sx", 0.25)]:
                with pulse.build() as expected:
                    pulse.play(pulse.Drag(200, amp, 40, 0), pulse.DriveChannel(qubit))

                self.assertEqual(cals.default_inst_map.get(gate, qubit), expected)

        # Update the amp on a single qubit, this should only update one gate in the inst_map
        cals.add_parameter_value(0.8, "amp", qubits=(4,), schedule="sx")

        # Test that all schedules now have an updated duration in the inst_map
        for qubit in range(5):
            for gate, amp in [("x", 0.5), ("sx", 0.25)]:

                if gate == "sx" and qubit == 4:
                    amp = 0.8

                with pulse.build() as expected:
                    pulse.play(pulse.Drag(200, amp, 40, 0), pulse.DriveChannel(qubit))

                self.assertEqual(cals.default_inst_map.get(gate, qubit), expected)

    def test_cx_cz_case(self):
        """Test the case where the coupling map has CX and CZ on different qubits.

        We use FakeBelem which has a linear coupling map and will restrict ourselves to
        qubits 0, 1, and 2. The Cals will define a template schedule for CX and CZ. We will
        mock this with GaussianSquare and Gaussian pulses since the nature of the schedules
        is irrelevant here. The parameters for CX will only have values for qubis 0 and 1 while
        the parameters for CZ will only have values for qubis 1 and 2. We therefore will have
        a CX on qubits 0, 1 in the inst. map and a CZ on qubits 1, 2.
        """

        cals = BackendCalibrations(FakeBelem())

        sig = Parameter("σ")
        dur = Parameter("duration")
        width = Parameter("width")
        amp_cx = Parameter("amp")
        amp_cz = Parameter("amp")
        uchan = Parameter("ch1.0")

        with pulse.build(name="cx") as cx:
            pulse.play(
                pulse.GaussianSquare(duration=dur, amp=amp_cx, sigma=sig, width=width),
                pulse.ControlChannel(uchan),
            )

        with pulse.build(name="cz") as cz:
            pulse.play(
                pulse.Gaussian(duration=dur, amp=amp_cz, sigma=sig), pulse.ControlChannel(uchan)
            )

        cals.add_schedule(cx, num_qubits=2)
        cals.add_schedule(cz, num_qubits=2)

        cals.add_parameter_value(640, "duration", schedule="cx")
        cals.add_parameter_value(64, "σ", schedule="cx")
        cals.add_parameter_value(320, "width", qubits=(0, 1), schedule="cx")
        cals.add_parameter_value(320, "width", qubits=(1, 0), schedule="cx")
        cals.add_parameter_value(0.1, "amp", qubits=(0, 1), schedule="cx")
        cals.add_parameter_value(0.8, "amp", qubits=(1, 0), schedule="cx")
        cals.add_parameter_value(0.1, "amp", qubits=(2, 1), schedule="cz")
        cals.add_parameter_value(0.8, "amp", qubits=(1, 2), schedule="cz")

        # CX only defined for qubits (0, 1) and (1,0)?
        self.assertTrue(cals.default_inst_map.has("cx", (0, 1)))
        self.assertTrue(cals.default_inst_map.has("cx", (1, 0)))
        self.assertFalse(cals.default_inst_map.has("cx", (2, 1)))
        self.assertFalse(cals.default_inst_map.has("cx", (1, 2)))

        # CZ only defined for qubits (2, 1) and (1,2)?
        self.assertTrue(cals.default_inst_map.has("cz", (2, 1)))
        self.assertTrue(cals.default_inst_map.has("cz", (1, 2)))
        self.assertFalse(cals.default_inst_map.has("cz", (0, 1)))
        self.assertFalse(cals.default_inst_map.has("cz", (1, 0)))

    def test_alternate_initialization(self):
        """Test that we can initialize without a backend object."""

        backend = FakeBelem()
        library = FixedFrequencyTransmon(basis_gates=["sx", "x"])

        cals1 = BackendCalibrations(backend, library)
        cals2 = BackendCalibrations(
            library=library,
            control_config=backend.configuration().control_channels,
            coupling_map=backend.configuration().coupling_map,
            num_qubits=backend.configuration().num_qubits,
        )

        self.assertEqual(str(cals1.get_schedule("x", 1)), str(cals2.get_schedule("x", 1)))

        with self.assertRaises(CalibrationError):
            BackendCalibrations(
                coupling_map=backend.configuration().coupling_map,
                num_qubits=backend.configuration().num_qubits,
            )
