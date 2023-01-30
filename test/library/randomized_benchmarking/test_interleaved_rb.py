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

"""Test for interleaved randomized benchmarking experiments."""

from test.base import QiskitExperimentsTestCase
from test.library.randomized_benchmarking.mixin import RBTestMixin
from ddt import ddt, data, unpack

from qiskit.circuit import Delay, QuantumCircuit, Parameter, Gate
from qiskit.circuit.library import SXGate, CXGate, TGate
from qiskit.exceptions import QiskitError
from qiskit.providers.fake_provider import FakeManila, FakeManilaV2, FakeWashington
from qiskit_experiments.library import randomized_benchmarking as rb


@ddt
class TestInterleavedRB(QiskitExperimentsTestCase, RBTestMixin):
    """Test for InterleavedRB without running the experiments."""

    def setUp(self):
        """Setup the tests."""
        super().setUp()
        self.backend = FakeManila()
        self.backend_with_timing_constraint = FakeWashington()

    # ### Tests for configuration ###
    def test_non_clifford_interleaved_element(self):
        """Verifies trying to run interleaved RB with non Clifford element throws an exception"""
        with self.assertRaises(QiskitError):
            rb.InterleavedRB(
                interleaved_element=TGate(),  # T gate is not Clifford, this should fail
                physical_qubits=[0],
                lengths=[1, 2, 3, 5, 8, 13],
            )

    @data([5, "dt"], [1e-7, "s"], [32, "ns"])
    @unpack
    def test_interleaving_delay_with_invalid_duration(self, duration, unit):
        """Raise if delay with invalid duration is given as interleaved_element"""
        with self.assertRaises(QiskitError):
            rb.InterleavedRB(
                interleaved_element=Delay(duration, unit=unit),
                physical_qubits=[0],
                lengths=[1, 2, 3],
                backend=self.backend_with_timing_constraint,
            )

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = rb.InterleavedRB(
            interleaved_element=SXGate(),
            physical_qubits=(0,),
            lengths=[10, 20, 30],
            seed=123,
        )
        loaded_exp = rb.InterleavedRB.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.json_equiv(exp, loaded_exp))

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = rb.InterleavedRB(
            interleaved_element=SXGate(), physical_qubits=(0,), lengths=[10, 20, 30], seed=123
        )
        self.assertRoundTripSerializable(exp, self.json_equiv)

    def test_analysis_config(self):
        """ "Test converting analysis to and from config works"""
        analysis = rb.InterleavedRBAnalysis()
        loaded = rb.InterleavedRBAnalysis.from_config(analysis.config())
        self.assertNotEqual(analysis, loaded)
        self.assertEqual(analysis.config(), loaded.config())

    # ### Tests for circuit generation ###
    class ThreeQubitGate(Gate):
        """A 3-qubit Clifford gate for tests"""

        def __init__(self):
            super().__init__("3q-gate", 3, [])

        def _define(self):
            qc = QuantumCircuit(3, name=self.name)
            qc.cx(0, 1)
            qc.x(2)
            self.definition = qc

    @data([SXGate(), [3], 4], [CXGate(), [4, 7], 5], [ThreeQubitGate(), [0, 1, 2], 3])
    @unpack
    def test_generate_interleaved_circuits(self, interleaved_element, qubits, length):
        """Test interleaved circuit generation"""
        exp = rb.InterleavedRB(
            interleaved_element=interleaved_element,
            physical_qubits=qubits,
            lengths=[length],
            num_samples=1,
        )
        circuits = exp.circuits()
        self.assertAllIdentity(circuits)

    @data([SXGate(), [3], 4], [CXGate(), [4, 7], 5])
    @unpack
    def test_interleaved_structure(self, interleaved_element, qubits, length):
        """Verifies that when generating an interleaved circuit, it will be
        identical to the original circuit up to additions of
        barrier and interleaved element between any two Cliffords.
        """
        exp = rb.InterleavedRB(
            interleaved_element=interleaved_element,
            physical_qubits=qubits,
            lengths=[length],
            num_samples=1,
        )

        circuits = exp.circuits()
        c_std = circuits[0]
        c_int = circuits[1]
        if c_std.metadata["interleaved"]:
            c_std, c_int = c_int, c_std
        num_cliffords = c_std.metadata["xval"]
        std_idx = 0
        int_idx = 0
        for _ in range(num_cliffords):
            # clifford
            self.assertEqual(c_std[std_idx], c_int[int_idx])
            # barrier
            self.assertEqual(c_std[std_idx + 1][0].name, "barrier")
            self.assertEqual(c_int[std_idx + 1][0].name, "barrier")
            # for interleaved circuit: interleaved element + barrier
            self.assertEqual(c_int[int_idx + 2][0].name, interleaved_element.name)
            self.assertEqual(c_int[int_idx + 3][0].name, "barrier")
            std_idx += 2
            int_idx += 4

    def test_preserve_interleaved_circuit_element(self):
        """Interleaved RB should not change a given interleaved circuit during RB circuit generation."""
        interleaved_circ = QuantumCircuit(2, name="bell_with_delay")
        interleaved_circ.h(0)
        interleaved_circ.delay(1.0e-7, 0, unit="s")
        interleaved_circ.cx(0, 1)

        exp = rb.InterleavedRB(
            interleaved_element=interleaved_circ, physical_qubits=[2, 1], lengths=[1], num_samples=1
        )
        circuits = exp.circuits()
        # Get the first interleaved operation in the interleaved RB sequence:
        # 0: clifford, 1: barrier, 2: interleaved
        actual = circuits[1][2].operation
        self.assertEqual(interleaved_circ.count_ops(), actual.definition.count_ops())

    def test_interleaving_delay(self):
        """Test delay instruction can be interleaved."""
        # See qiskit-experiments/#727 for details
        from qiskit_experiments.framework.backend_timing import BackendTiming

        timing = BackendTiming(self.backend)
        exp = rb.InterleavedRB(
            interleaved_element=Delay(timing.round_delay(time=1.0e-7)),
            physical_qubits=[0],
            lengths=[1],
            num_samples=1,
            seed=1234,  # This seed gives a 2-gate clifford
            backend=self.backend,
        )
        int_circs = exp.circuits()[1]
        self.assertEqual(int_circs.count_ops().get("delay", 0), 1)
        self.assertAllIdentity([int_circs])

    def test_interleaving_circuit_with_delay(self):
        """Test circuit with delay can be interleaved."""
        delay_qc = QuantumCircuit(2)
        delay_qc.delay(160, [0])
        delay_qc.x(1)

        exp = rb.InterleavedRB(
            interleaved_element=delay_qc,
            physical_qubits=[1, 2],
            lengths=[1],
            num_samples=1,
            seed=1234,
            backend=self.backend,
        )
        int_circ = exp.circuits()[1]
        self.assertAllIdentity([int_circ])

    def test_interleaving_parameterized_circuit(self):
        """Fail if parameterized circuit is interleaved but after assigned it may be interleaved."""
        qubits = (2,)
        theta = Parameter("theta")
        phi = Parameter("phi")
        lam = Parameter("lambda")
        cliff_circ_with_param = QuantumCircuit(1)
        cliff_circ_with_param.rz(theta, 0)
        cliff_circ_with_param.sx(0)
        cliff_circ_with_param.rz(phi, 0)
        cliff_circ_with_param.sx(0)
        cliff_circ_with_param.rz(lam, 0)

        with self.assertRaises(QiskitError):
            rb.InterleavedRB(
                interleaved_element=cliff_circ_with_param,
                physical_qubits=qubits,
                lengths=[3],
                num_samples=4,
                backend=self.backend,
            )

        # # TODO: Enable after Clifford supports creation from circuits with rz
        # # parameters must be assigned before initializing InterleavedRB
        # param_map = {theta: np.pi / 2, phi: -np.pi / 2, lam: np.pi / 2}
        # cliff_circ_with_param.assign_parameters(param_map, inplace=True)
        #
        # exp = rb.InterleavedRB(
        #     interleaved_element=cliff_circ_with_param,
        #     physical_qubits=qubits,
        #     lengths=[3],
        #     num_samples=4,
        #     backend=self.backend,
        # )
        # circuits = exp.circuits()
        # for qc in circuits:
        #     self.assertEqual(qc.num_parameters, 0)

    # ### Tests for transpiled circuit generation ###
    def test_interleaved_circuit_is_decomposed(self):
        """Test if interleaved circuit is decomposed in transpiled circuits."""
        delay_qc = QuantumCircuit(2)
        delay_qc.delay(160, [0])
        delay_qc.x(1)

        exp = rb.InterleavedRB(
            interleaved_element=delay_qc,
            physical_qubits=[1, 2],
            lengths=[3],
            num_samples=1,
            seed=1234,
            backend=self.backend,
        )
        transpiled = exp._transpiled_circuits()
        for qc in transpiled:
            self.assertTrue(all(not inst.operation.name.startswith("circuit") for inst in qc))
            self.assertTrue(all(not inst.operation.name.startswith("Clifford") for inst in qc))

    def test_interleaving_cnot_gate_with_non_supported_direction(self):
        """Test if cx(0, 1) can be interleaved for backend that support only cx(1, 0)."""
        my_backend = FakeManilaV2()
        del my_backend.target["cx"][(0, 1)]  # make support only cx(1, 0)

        exp = rb.InterleavedRB(
            interleaved_element=CXGate(),
            physical_qubits=(0, 1),
            lengths=[3],
            num_samples=4,
            backend=my_backend,
        )
        transpiled = exp._transpiled_circuits()
        for qc in transpiled:
            self.assertTrue(qc.count_ops().get("cx", 0) > 0)
            expected_qubits = (qc.qubits[1], qc.qubits[0])
            for inst in qc:
                if inst.operation.name == "cx":
                    self.assertEqual(inst.qubits, expected_qubits)
