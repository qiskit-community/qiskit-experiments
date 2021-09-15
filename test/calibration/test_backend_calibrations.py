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

from qiskit import transpile, QuantumCircuit
from qiskit.circuit import Parameter, Gate
import qiskit.pulse as pulse
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeArmonk, FakeBelem

from qiskit_experiments.calibration_management import BackendCalibrations
from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon


class TestBackendCalibrations(QiskitTestCase):
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

        cals.add_schedule(cr, n_qubits=2)
        cals.complete_inst_map_update({"cr"})

        for qubit in range(backend.configuration().num_qubits):
            self.assertTrue(cals.instruction_schedule_map.has("sx", (qubit,)))

        # based on coupling map of Belem to keep the test robust.
        expected_pairs = [(0, 1), (1, 0), (1, 2), (2, 1), (1, 3), (3, 1), (3, 4), (4, 3)]
        coupling_map = set(tuple(pair) for pair in backend.configuration().coupling_map)

        for pair in expected_pairs:
            self.assertTrue(pair in coupling_map)
            self.assertTrue(cals.instruction_schedule_map.has("cr", pair), pair)

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
        circs = transpile(circs, inst_map=cals.instruction_schedule_map)

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
