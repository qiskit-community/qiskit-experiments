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

"""Test for randomized benchmarking experiments."""
from test.base import QiskitExperimentsTestCase

import copy

import numpy as np
from ddt import ddt, data, unpack

from qiskit.circuit import Delay, QuantumCircuit, Parameter, Gate
from qiskit.circuit.library import SXGate, CXGate, TGate, CZGate, ECRGate
from qiskit.exceptions import QiskitError
from qiskit.providers.fake_provider import FakeManila, FakeManilaV2, FakeWashington, FakeParis
from qiskit.pulse import Schedule, InstructionScheduleMap
from qiskit.quantum_info import Operator
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler import Layout, PassManager, CouplingMap
from qiskit.transpiler.exceptions import TranspilerError

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from qiskit_experiments.database_service.exceptions import ExperimentEntryNotFound
from qiskit_experiments.framework.composite import ParallelExperiment
from qiskit_experiments.library import randomized_benchmarking as rb
from qiskit_experiments.library.randomized_benchmarking.clifford_utils import (
    compute_target_bitstring,
)
from qiskit_experiments.library.randomized_benchmarking.sampling_utils import EdgeGrabSampler


class RBTestMixin:
    """Mixin for RB tests."""

    def assertAllIdentity(self, circuits):
        """Test if all experiment circuits are identity."""
        for circ in circuits:
            num_qubits = circ.num_qubits
            qc_iden = QuantumCircuit(num_qubits)
            circ.remove_final_measurements()
            self.assertTrue(Operator(circ).equiv(Operator(qc_iden)))


@ddt
class TestStandardRB(QiskitExperimentsTestCase, RBTestMixin):
    """Test for StandardRB without running the experiments."""

    def setUp(self):
        """Setup the tests."""
        super().setUp()
        self.backend = FakeManilaV2()

    # ### Tests for configuration ###
    @data(
        {"physical_qubits": [3, 3], "lengths": [1, 3, 5, 7, 9], "num_samples": 1, "seed": 100},
        {"physical_qubits": [0, 1], "lengths": [1, 3, 5, -7, 9], "num_samples": 1, "seed": 100},
        {"physical_qubits": [0, 1], "lengths": [1, 3, 5, 7, 9], "num_samples": -4, "seed": 100},
        {"physical_qubits": [0, 1], "lengths": [1, 3, 5, 7, 9], "num_samples": 0, "seed": 100},
        {"physical_qubits": [0, 1], "lengths": [1, 5, 5, 5, 9], "num_samples": 2, "seed": 100},
    )
    def test_invalid_configuration(self, configs):
        """Test raise error when creating experiment with invalid configs."""
        self.assertRaises(QiskitError, rb.StandardRB, **configs)

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = rb.StandardRB(physical_qubits=(0,), lengths=[10, 20, 30], seed=123)
        loaded_exp = rb.StandardRB.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.json_equiv(exp, loaded_exp))

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = rb.StandardRB(physical_qubits=(0,), lengths=[10, 20, 30], seed=123)
        self.assertRoundTripSerializable(exp, self.json_equiv)

    def test_analysis_config(self):
        """ "Test converting analysis to and from config works"""
        analysis = rb.RBAnalysis()
        loaded = rb.RBAnalysis.from_config(analysis.config())
        self.assertNotEqual(analysis, loaded)
        self.assertEqual(analysis.config(), loaded.config())

    # ### Tests for circuit generation ###
    @data([[3], 4], [[4, 7], 5], [[0, 1, 2], 3])
    @unpack
    def test_generate_circuits(self, physical_qubits, length):
        """Test RB circuit generation"""
        exp = rb.StandardRB(physical_qubits=physical_qubits, lengths=[length], num_samples=1)
        circuits = exp.circuits()
        self.assertAllIdentity(circuits)

    def test_return_same_circuit(self):
        """Test if setting the same seed returns the same circuits."""
        exp1 = rb.StandardRB(
            physical_qubits=(0, 1),
            lengths=[10, 20, 30],
            seed=123,
            backend=self.backend,
        )

        exp2 = rb.StandardRB(
            physical_qubits=(0, 1),
            lengths=[10, 20, 30],
            seed=123,
            backend=self.backend,
        )

        circs1 = exp1.circuits()
        circs2 = exp2.circuits()

        self.assertEqual(circs1[0].decompose(), circs2[0].decompose())
        self.assertEqual(circs1[1].decompose(), circs2[1].decompose())
        self.assertEqual(circs1[2].decompose(), circs2[2].decompose())

    def test_full_sampling_single_qubit(self):
        """Test if full sampling generates different circuits."""
        exp1 = rb.StandardRB(
            physical_qubits=(0,),
            lengths=[10, 20, 30],
            seed=123,
            backend=self.backend,
            full_sampling=False,
        )
        exp2 = rb.StandardRB(
            physical_qubits=(0,),
            lengths=[10, 20, 30],
            seed=123,
            backend=self.backend,
            full_sampling=True,
        )
        circs1 = exp1.circuits()
        circs2 = exp2.circuits()

        self.assertEqual(circs1[0].decompose(), circs2[0].decompose())

        # fully sampled circuits are regenerated while other is just built on top of previous length
        self.assertNotEqual(circs1[1].decompose(), circs2[1].decompose())
        self.assertNotEqual(circs1[2].decompose(), circs2[2].decompose())

    def test_full_sampling_2_qubits(self):
        """Test if full sampling generates different circuits."""
        exp1 = rb.StandardRB(
            physical_qubits=(0, 1),
            lengths=[10, 20, 30],
            seed=123,
            backend=self.backend,
            full_sampling=False,
        )

        exp2 = rb.StandardRB(
            physical_qubits=(0, 1),
            lengths=[10, 20, 30],
            seed=123,
            backend=self.backend,
            full_sampling=True,
        )

        circs1 = exp1.circuits()
        circs2 = exp2.circuits()

        self.assertEqual(circs1[0].decompose(), circs2[0].decompose())

        # fully sampled circuits are regenerated while other is just built on top of previous length
        self.assertNotEqual(circs1[1].decompose(), circs2[1].decompose())
        self.assertNotEqual(circs1[2].decompose(), circs2[2].decompose())

    # ### Tests for transpiled circuit generation ###
    def test_calibrations_via_transpile_options(self):
        """Test if calibrations given as transpile_options show up in transpiled circuits."""
        physical_qubits = (2,)
        my_sched = Schedule(name="custom_sx_gate")
        my_inst_map = InstructionScheduleMap()
        my_inst_map.add(SXGate(), physical_qubits, my_sched)

        exp = rb.StandardRB(
            physical_qubits=physical_qubits,
            lengths=[3],
            num_samples=4,
            backend=self.backend,
            seed=123,
        )
        exp.set_transpile_options(inst_map=my_inst_map)
        transpiled = exp._transpiled_circuits()
        for qc in transpiled:
            self.assertTrue(qc.calibrations)
            self.assertTrue(
                qc.has_calibration_for((SXGate(), [qc.qubits[q] for q in physical_qubits], []))
            )
            self.assertEqual(qc.calibrations["sx"][(physical_qubits, tuple())], my_sched)

    def test_calibrations_via_custom_backend(self):
        """Test if calibrations given as custom backend show up in transpiled circuits."""
        physical_qubits = (2,)
        my_sched = Schedule(name="custom_sx_gate")
        my_backend = copy.deepcopy(self.backend)
        my_backend.target["sx"][physical_qubits].calibration = my_sched

        exp = rb.StandardRB(
            physical_qubits=physical_qubits, lengths=[3], num_samples=4, backend=my_backend
        )
        transpiled = exp._transpiled_circuits()
        for qc in transpiled:
            self.assertTrue(qc.calibrations)
            self.assertTrue(
                qc.has_calibration_for((SXGate(), [qc.qubits[q] for q in physical_qubits], []))
            )
            self.assertEqual(qc.calibrations["sx"][(physical_qubits, tuple())], my_sched)

    def test_backend_with_directed_basis_gates(self):
        """Test if correct circuits are generated from backend with directed basis gates."""
        my_backend = copy.deepcopy(self.backend)
        del my_backend.target["cx"][(1, 2)]  # make cx on {1, 2} one-sided

        exp = rb.StandardRB(physical_qubits=(1, 2), lengths=[3], num_samples=4, backend=my_backend)
        transpiled = exp._transpiled_circuits()
        for qc in transpiled:
            self.assertTrue(qc.count_ops().get("cx", 0) > 0)
            expected_qubits = (qc.qubits[2], qc.qubits[1])
            for inst in qc:
                if inst.operation.name == "cx":
                    self.assertEqual(inst.qubits, expected_qubits)


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
    def test_generate_interleaved_circuits(self, interleaved_element, physical_qubits, length):
        """Test interleaved circuit generation"""
        exp = rb.InterleavedRB(
            interleaved_element=interleaved_element,
            physical_qubits=physical_qubits,
            lengths=[length],
            num_samples=1,
        )
        circuits = exp.circuits()
        self.assertAllIdentity(circuits)

    @data([SXGate(), [3], 4], [CXGate(), [4, 7], 5])
    @unpack
    def test_interleaved_structure(self, interleaved_element, physical_qubits, length):
        """Verifies that when generating an interleaved circuit, it will be
        identical to the original circuit up to additions of
        barrier and interleaved element between any two Cliffords.
        """
        exp = rb.InterleavedRB(
            interleaved_element=interleaved_element,
            physical_qubits=physical_qubits,
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
        physical_qubits = (2,)
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
                physical_qubits=physical_qubits,
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
        #     physical_qubits=physical_qubits,
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


class RBRunTestCase(QiskitExperimentsTestCase, RBTestMixin):
    """Base test case for running RB experiments defining a common noise model."""

    def setUp(self):
        """Setup the tests."""
        super().setUp()

        # depolarizing error
        self.p1q = 0.004
        self.p2q = 0.04
        self.pvz = 0.0
        self.pcz = 0.06

        # basis gates
        self.basis_gates = ["rz", "sx", "cx"]

        # setup noise model
        sx_error = depolarizing_error(self.p1q, 1)
        rz_error = depolarizing_error(self.pvz, 1)
        cx_error = depolarizing_error(self.p2q, 2)
        cz_error = depolarizing_error(self.pcz, 2)

        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(sx_error, "sx")
        noise_model.add_all_qubit_quantum_error(rz_error, "rz")
        noise_model.add_all_qubit_quantum_error(cx_error, "cx")
        noise_model.add_all_qubit_quantum_error(cz_error, "cz")

        self.noise_model = noise_model

        # Need level1 for consecutive gate cancellation for reference EPC value calculation
        self.transpiler_options = {
            "basis_gates": self.basis_gates,
            "optimization_level": 1,
        }

        # Aer simulator
        self.backend = AerSimulator(noise_model=noise_model, seed_simulator=123)


class TestRunStandardRB(RBRunTestCase):
    """Test for running StandardRB."""

    def test_single_qubit(self):
        """Test single qubit RB."""
        exp = rb.StandardRB(
            physical_qubits=(0,),
            lengths=list(range(1, 300, 30)),
            seed=123,
            backend=self.backend,
        )
        exp.analysis.set_options(gate_error_ratio=None)
        exp.set_transpile_options(**self.transpiler_options)
        self.assertAllIdentity(exp.circuits())

        expdata = exp.run()
        self.assertExperimentDone(expdata)

        # Given we have gate number per Clifford n_gpc, we can compute EPC as
        # EPC = 1 - (1 - r)^n_gpc
        # where r is gate error of SX gate, i.e. dep-parameter divided by 2.
        # We let transpiler use SX and RZ.
        # The number of physical gate per Clifford will distribute
        # from 0 to 2, i.e. arbitrary U gate can be decomposed into up to 2 SX with RZs.
        # We may want to expect the average number of SX is (0 + 1 + 2) / 3 = 1.0.
        epc = expdata.analysis_results("EPC")

        epc_expected = 1 - (1 - 1 / 2 * self.p1q) ** 1.0
        self.assertAlmostEqual(epc.value.n, epc_expected, delta=3 * epc.value.std_dev)

    def test_two_qubit(self):
        """Test two qubit RB. Use default basis gates."""
        exp = rb.StandardRB(
            physical_qubits=(0, 1),
            lengths=list(range(1, 30, 3)),
            seed=123,
            backend=self.backend,
        )
        exp.analysis.set_options(gate_error_ratio=None)
        transpiler_options = {"optimization_level": 1}
        exp.set_transpile_options(**transpiler_options)
        self.assertAllIdentity(exp.circuits())

        expdata = exp.run()
        self.assertExperimentDone(expdata)

        # Given CX error is dominant and 1q error can be negligible.
        # Arbitrary SU(4) can be decomposed with (0, 1, 2, 3) CX gates, the expected
        # average number of CX gate per Clifford is 1.5.
        # Since this is two qubit RB, the dep-parameter is factored by 3/4.
        epc = expdata.analysis_results("EPC")
        # Allow for 30 percent tolerance since we ignore 1q gate contribution
        epc_expected = 1 - (1 - 3 / 4 * self.p2q) ** 1.5
        self.assertAlmostEqual(epc.value.n, epc_expected, delta=0.3 * epc_expected)

    def test_three_qubit(self):
        """Test two qubit RB. Use default basis gates."""
        exp = rb.StandardRB(
            physical_qubits=(0, 1, 2),
            lengths=list(range(1, 30, 3)),
            seed=123,
            backend=self.backend,
        )
        exp.analysis.set_options(gate_error_ratio=None)
        exp.set_transpile_options(**self.transpiler_options)
        self.assertAllIdentity(exp.circuits())

        expdata = exp.run()
        self.assertExperimentDone(expdata)

        # Given CX error is dominant and 1q error can be negligible.
        # Arbitrary SU(8) can be decomposed with [0,...,7] CX gates, the expected
        # average number of CX gate per Clifford is 3.5.
        # Since this is three qubit RB, the dep-parameter is factored by 7/8.
        epc = expdata.analysis_results("EPC")
        # Allow for 50 percent tolerance since we ignore 1q gate contribution
        epc_expected = 1 - (1 - 7 / 8 * self.p2q) ** 3.5
        self.assertAlmostEqual(epc.value.n, epc_expected, delta=0.5 * epc_expected)

    def test_add_more_circuit_yields_lower_variance(self):
        """Test variance reduction with larger number of sampling."""
        exp1 = rb.StandardRB(
            physical_qubits=(0, 1),
            lengths=list(range(1, 30, 3)),
            seed=123,
            backend=self.backend,
            num_samples=3,
        )
        exp1.analysis.set_options(gate_error_ratio=None)
        exp1.set_transpile_options(**self.transpiler_options)
        expdata1 = exp1.run()
        self.assertExperimentDone(expdata1)

        exp2 = rb.StandardRB(
            physical_qubits=(0, 1),
            lengths=list(range(1, 30, 3)),
            seed=456,
            backend=self.backend,
            num_samples=5,
        )
        exp2.analysis.set_options(gate_error_ratio=None)
        exp2.set_transpile_options(**self.transpiler_options)
        expdata2 = exp2.run()
        self.assertExperimentDone(expdata2)

        self.assertLess(
            expdata2.analysis_results("EPC").value.s,
            expdata1.analysis_results("EPC").value.s,
        )

    def test_poor_experiment_result(self):
        """Test edge case that tail of decay is not sampled.

        This is a special case that fit outcome is very sensitive to initial guess.
        Perhaps generated initial guess is close to a local minima.
        """
        from qiskit.providers.fake_provider import FakeVigoV2

        backend = FakeVigoV2()
        backend.set_options(seed_simulator=123)
        # TODO: this test no longer makes sense (yields small reduced_chisq)
        #  after fixing how to call fake backend v2 (by adding the next line)
        # Need to call target before running fake backend v2 to load correct data
        self.assertLess(backend.target["sx"][(0,)].error, 0.001)

        exp = rb.StandardRB(
            physical_qubits=(0,),
            lengths=[100, 200, 300],
            seed=123,
            backend=backend,
            num_samples=5,
        )
        exp.set_transpile_options(basis_gates=["x", "sx", "rz"], optimization_level=1)

        expdata = exp.run()
        self.assertExperimentDone(expdata)
        overview = expdata.analysis_results(0).value
        # This yields bad fit due to poor data points, but still fit is not completely off.
        self.assertLess(overview.reduced_chisq, 14)

    def test_expdata_serialization(self):
        """Test serializing experiment data works."""
        exp = rb.StandardRB(
            physical_qubits=(0,),
            lengths=list(range(1, 200, 50)),
            seed=123,
            backend=self.backend,
        )
        exp.set_transpile_options(**self.transpiler_options)
        expdata = exp.run()
        self.assertExperimentDone(expdata)
        self.assertRoundTripSerializable(expdata, check_func=self.experiment_data_equiv)
        self.assertRoundTripPickle(expdata, check_func=self.experiment_data_equiv)

    def test_single_qubit_parallel(self):
        """Test single qubit RB in parallel."""
        physical_qubits = [0, 2]
        lengths = list(range(1, 300, 30))
        exps = []
        for qubit in physical_qubits:
            exp = rb.StandardRB(
                physical_qubits=[qubit], lengths=lengths, seed=123, backend=self.backend
            )
            exp.analysis.set_options(gate_error_ratio=None, plot_raw_data=False)
            exps.append(exp)

        par_exp = ParallelExperiment(exps)
        par_exp.set_transpile_options(**self.transpiler_options)

        par_expdata = par_exp.run(backend=self.backend)
        self.assertExperimentDone(par_expdata)
        epc_expected = 1 - (1 - 1 / 2 * self.p1q) ** 1.0
        for i in range(2):
            epc = par_expdata.child_data(i).analysis_results("EPC")
            self.assertAlmostEqual(epc.value.n, epc_expected, delta=3 * epc.value.std_dev)

    def test_two_qubit_parallel(self):
        """Test two qubit RB in parallel."""
        qubit_pairs = [[0, 1], [2, 3]]
        lengths = list(range(1, 30, 3))
        exps = []
        for pair in qubit_pairs:
            exp = rb.StandardRB(
                physical_qubits=pair, lengths=lengths, seed=123, backend=self.backend
            )
            exp.analysis.set_options(gate_error_ratio=None, plot_raw_data=False)
            exps.append(exp)

        par_exp = ParallelExperiment(exps)
        par_exp.set_transpile_options(**self.transpiler_options)

        par_expdata = par_exp.run(backend=self.backend)
        self.assertExperimentDone(par_expdata)
        epc_expected = 1 - (1 - 3 / 4 * self.p2q) ** 1.5
        for i in range(2):
            epc = par_expdata.child_data(i).analysis_results("EPC")
            # Allow for 30 percent tolerance since we ignore 1q gate contribution
            self.assertAlmostEqual(epc.value.n, epc_expected, delta=0.3 * epc_expected)

    def test_two_qubit_with_cz(self):
        """Test two qubit RB."""
        transpiler_options = {
            "basis_gates": ["sx", "rz", "cz"],
            "optimization_level": 1,
        }

        exp = rb.StandardRB(
            physical_qubits=(0, 1),
            lengths=list(range(1, 50, 5)),
            seed=123,
            backend=self.backend,
        )
        exp.analysis.set_options(gate_error_ratio=None)
        exp.set_transpile_options(**transpiler_options)

        expdata = exp.run()
        self.assertAllIdentity(exp.circuits())
        self.assertExperimentDone(expdata)

        # Given CX error is dominant and 1q error can be negligible.
        # Arbitrary SU(4) can be decomposed with (0, 1, 2, 3) CZ gates, the expected
        # average number of CZ gate per Clifford is 1.5.
        # Since this is two qubit RB, the dep-parameter is factored by 3/4.
        epc = expdata.analysis_results("EPC")

        # Allow for 30 percent tolerance since we ignore 1q gate contribution
        epc_expected = 1 - (1 - 3 / 4 * self.pcz) ** 1.5
        self.assertAlmostEqual(epc.value.n, epc_expected, delta=0.3 * epc_expected)


class TestRunInterleavedRB(RBRunTestCase):
    """Test for running InterleavedRB."""

    def test_single_qubit(self):
        """Test single qubit IRB."""
        exp = rb.InterleavedRB(
            interleaved_element=SXGate(),
            physical_qubits=(0,),
            lengths=list(range(1, 300, 30)),
            seed=123,
            backend=self.backend,
        )
        exp.set_transpile_options(**self.transpiler_options)
        self.assertAllIdentity(exp.circuits())

        expdata = exp.run()
        self.assertExperimentDone(expdata)

        # Since this is interleaved, we can directly compare values, i.e. n_gpc = 1
        epc = expdata.analysis_results("EPC")
        epc_expected = 1 / 2 * self.p1q
        self.assertAlmostEqual(epc.value.n, epc_expected, delta=3 * epc.value.std_dev)

    def test_two_qubit(self):
        """Test two qubit IRB."""
        exp = rb.InterleavedRB(
            interleaved_element=CXGate(),
            physical_qubits=(0, 1),
            lengths=list(range(1, 30, 3)),
            seed=123,
            backend=self.backend,
        )
        exp.set_transpile_options(**self.transpiler_options)
        self.assertAllIdentity(exp.circuits())

        expdata = exp.run()
        self.assertExperimentDone(expdata)

        # Since this is interleaved, we can directly compare values, i.e. n_gpc = 1
        epc = expdata.analysis_results("EPC")
        epc_expected = 3 / 4 * self.p2q
        self.assertAlmostEqual(epc.value.n, epc_expected, delta=3 * epc.value.std_dev)

    def test_two_qubit_with_cz(self):
        """Test two qubit IRB."""
        transpiler_options = {
            "basis_gates": ["sx", "rz", "cz"],
            "optimization_level": 1,
        }
        exp = rb.InterleavedRB(
            interleaved_element=CZGate(),
            physical_qubits=(0, 1),
            lengths=list(range(1, 30, 3)),
            seed=1234,
            backend=self.backend,
        )
        exp.set_transpile_options(**transpiler_options)
        self.assertAllIdentity(exp.circuits())

        expdata = exp.run()
        self.assertExperimentDone(expdata)

        # Since this is interleaved, we can directly compare values, i.e. n_gpc = 1
        epc = expdata.analysis_results("EPC")
        epc_expected = 3 / 4 * self.pcz
        self.assertAlmostEqual(epc.value.n, epc_expected, delta=3 * epc.value.std_dev)

    def test_expdata_serialization(self):
        """Test serializing experiment data works."""
        exp = rb.InterleavedRB(
            interleaved_element=SXGate(),
            physical_qubits=(0,),
            lengths=list(range(1, 200, 50)),
            seed=123,
            backend=self.backend,
        )
        exp.set_transpile_options(**self.transpiler_options)
        expdata = exp.run()
        self.assertExperimentDone(expdata)
        self.assertRoundTripSerializable(expdata, check_func=self.experiment_data_equiv)
        self.assertRoundTripPickle(expdata, check_func=self.experiment_data_equiv)


def decr_dep_param(q, q_1, q_2, coupling_map):
    """Helper function to generate a one-qubit depolarizing channel whose
    parameter depends on coupling map distance in a backend"""
    d = min(coupling_map.distance(q, q_1), coupling_map.distance(q, q_2))
    return 0.0035 * 0.999**d


class NonlocalCXDepError(TransformationPass):
    """Transpiler pass for simulating nonlocal errors in a quantum device"""

    def __init__(self, coupling_map, initial_layout=None):
        """Maps a DAGCircuit onto a `coupling_map` using swap gates.
        Args:
            coupling_map (CouplingMap): Directed graph represented a coupling map.
            initial_layout (Layout): initial layout of qubits in mapping
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.initial_layout = initial_layout

    def run(self, dag):
        """Runs the NonlocalCXDepError pass on `dag`

        Args:
            dag (DAGCircuit): DAG to map.

        Returns:
            DAGCircuit: A mapped DAG.

        Raises:
            TranspilerError: initial layout and coupling map do not have the
            same size
        """

        if self.initial_layout is None:
            if self.property_set["layout"]:
                self.initial_layout = self.property_set["layout"]
            else:
                self.initial_layout = Layout.generate_trivial_layout(*dag.qregs.values())

        if len(dag.qubits) != len(self.initial_layout):
            raise TranspilerError("The layout does not match the amount of qubits in the DAG")

        if len(self.coupling_map.physical_qubits) != len(self.initial_layout):
            raise TranspilerError(
                "Mappers require to have the layout to be the same size as the coupling map"
            )

        canonical_register = dag.qregs["q"]
        trivial_layout = Layout.generate_trivial_layout(canonical_register)
        current_layout = trivial_layout.copy()

        subdags = []
        for layer in dag.layers():
            graph = layer["graph"]
            cxs = graph.op_nodes(op=CXGate)
            if len(cxs) > 0:
                for cx in cxs:
                    qubit_1 = current_layout[cx.qargs[0]]
                    qubit_2 = current_layout[cx.qargs[1]]
                    for qubit in range(dag.num_qubits()):
                        dep_param = decr_dep_param(qubit, qubit_1, qubit_2, self.coupling_map)
                        graph.apply_operation_back(
                            depolarizing_error(dep_param, 1).to_instruction(),
                            qargs=[canonical_register[qubit]],
                            cargs=[],
                        )
            subdags.append(graph)

        err_dag = dag.copy_empty_like()
        for subdag in subdags:
            err_dag.compose(subdag)

        return err_dag


class NoiseSimulator(AerSimulator):
    """Quantum device simulator that has nonlocal CX errors"""

    def run(self, circuits, validate=False, parameter_binds=None, **run_options):
        """Applies transpiler pass NonlocalCXDepError to circuits run on this backend"""
        pm = PassManager()
        cm = CouplingMap(couplinglist=self.configuration().coupling_map)
        pm.append([NonlocalCXDepError(cm)])
        noise_circuits = pm.run(circuits)
        return super().run(
            noise_circuits, validate=validate, parameter_binds=parameter_binds, **run_options
        )


@ddt
class TestMirrorRB(QiskitExperimentsTestCase, RBTestMixin):
    """Test for mirror RB."""

    seed = 123

    def setUp(self):
        """Setup the tests."""
        super().setUp()
        self.backend = FakeParis()

        self.basis_gates = ["sx", "rz", "cx"]

        self.transpiler_options = {
            "basis_gates": self.basis_gates,
            "optimization_level": 1,
        }

    def test_return_same_circuit(self):
        """Test if setting the same seed returns the same circuits."""
        lengths = [10, 20]
        exp1 = rb.MirrorRB(
            physical_qubits=(0, 1),
            lengths=lengths,
            seed=self.seed,
            backend=self.backend,
        )

        exp2 = rb.MirrorRB(
            physical_qubits=(0, 1),
            lengths=lengths,
            seed=self.seed,
            backend=self.backend,
        )

        circs1 = exp1.circuits()
        circs2 = exp2.circuits()

        for circ1, circ2 in zip(circs1, circs2):
            self.assertEqual(circ1.decompose(), circ2.decompose())

    def test_full_sampling(self):
        """Test if full sampling generates different circuits."""
        exp1 = rb.MirrorRB(
            physical_qubits=(0, 1),
            lengths=[10, 20],
            seed=self.seed,
            backend=self.backend,
            num_samples=1,
            full_sampling=True,
        )

        exp2 = rb.MirrorRB(
            physical_qubits=(0, 1),
            lengths=[10, 20],
            seed=self.seed,
            backend=self.backend,
            num_samples=1,
            full_sampling=False,
        )

        circs1 = exp1.circuits()
        circs2 = exp2.circuits()

        self.assertNotEqual(circs1[0].decompose(), circs2[0].decompose())

        # fully sampled circuits are regenerated while other is just built on
        # top of previous length
        self.assertNotEqual(circs1[1].decompose(), circs2[1].decompose())

    def test_zero_2q_gate_density(self):
        """Test that there are no two-qubit gates when the two-qubit gate
        density is set to 0."""
        exp = rb.MirrorRB(
            physical_qubits=(0, 1),
            lengths=[40],
            seed=self.seed,
            backend=self.backend,
            num_samples=1,
            two_qubit_gate_density=0,
        )
        circ = exp.circuits()[0].decompose()
        for datum in circ.data:
            inst_name = datum[0].name
            self.assertNotEqual("cx", inst_name)

    def test_max_2q_gate_density(self):
        """Test that every intermediate Clifford layer is filled with two-qubit
        gates when the two-qubit gate density is set to 0.5, its maximum value
        (assuming an even number of qubits and a backend coupling map with full
        connectivity)."""
        backend = AerSimulator(coupling_map=CouplingMap.from_full(4).get_edges())
        exp = rb.MirrorRB(
            physical_qubits=(0, 1, 2, 3),
            lengths=[40],
            seed=self.seed,
            backend=backend,
            num_samples=1,
            two_qubit_gate_density=0.5,
        )
        circ = exp.circuits()[0].decompose()
        num_cxs = 0
        for datum in circ.data:
            if datum[0].name == "cx":
                num_cxs += 1
        self.assertEqual(80, num_cxs)

    def test_start_end_clifford(self):
        """Test that the number of layers is correct depending on whether
        start_end_clifford is set to True or False by counting the number of barriers."""
        exp = rb.MirrorRB(
            physical_qubits=(0,),
            lengths=[2],
            seed=self.seed,
            backend=self.backend,
            num_samples=1,
            start_end_clifford=True,
            pauli_randomize=False,
            two_qubit_gate_density=0.2,
            inverting_pauli_layer=False,
        )
        circ = exp.circuits()[0]
        num_barriers = 0
        for datum in circ.data:
            if datum[0].name == "barrier":
                num_barriers += 1
        self.assertEqual(5, num_barriers)

    def test_pauli_randomize(self):
        """Test that the number of layers is correct depending on whether
        pauli_randomize is set to True or False by counting the number of barriers."""
        exp = rb.MirrorRB(
            physical_qubits=(0,),
            lengths=[2],
            seed=self.seed,
            backend=self.backend,
            num_samples=1,
            start_end_clifford=False,
            pauli_randomize=True,
            two_qubit_gate_density=0.2,
            inverting_pauli_layer=False,
        )
        circ = exp.circuits()[0]
        num_barriers = 0
        for datum in circ.data:
            if datum[0].name == "barrier":
                num_barriers += 1
        self.assertEqual(6, num_barriers)

    def test_inverting_pauli_layer(self):
        """Test that a circuit with an inverting Pauli layer at the end generates
        an all-zero output."""
        exp = rb.MirrorRB(
            physical_qubits=(0, 1, 2),
            lengths=[2],
            seed=self.seed,
            backend=self.backend,
            num_samples=3,
            start_end_clifford=True,
            pauli_randomize=True,
            two_qubit_gate_density=0.2,
            inverting_pauli_layer=True,
        )
        self.assertEqual(
            compute_target_bitstring(exp.circuits()[0].remove_final_measurements(inplace=False)),
            "000",
        )

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = rb.MirrorRB([0], lengths=[10, 20, 30], seed=123, backend=self.backend)
        loaded_exp = rb.MirrorRB.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.json_equiv(exp, loaded_exp))

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = rb.MirrorRB([0], lengths=[10, 20, 30], seed=123)
        self.assertRoundTripSerializable(exp, self.json_equiv)

    def test_analysis_config(self):
        """ "Test converting analysis to and from config works"""
        analysis = rb.RBAnalysis()
        loaded = rb.RBAnalysis.from_config(analysis.config())
        self.assertNotEqual(analysis, loaded)
        self.assertEqual(analysis.config(), loaded.config())

    def test_backend_with_directed_basis_gates(self):
        """Test if correct circuits are generated from backend with directed basis gates."""
        my_backend = copy.deepcopy(FakeManilaV2())
        del my_backend.target["cx"][(1, 2)]  # make cx on {1, 2} one-sided

        exp = rb.MirrorRB(
            physical_qubits=(1, 2),
            two_qubit_gate_density=0.5,
            lengths=[4],
            num_samples=4,
            backend=my_backend,
            seed=self.seed,
        )
        transpiled = exp._transpiled_circuits()
        for qc in transpiled:
            self.assertTrue(qc.count_ops().get("cx", 0) > 0)
            expected_qubits = (qc.qubits[2], qc.qubits[1])
            for inst in qc:
                if inst.operation.name == "cx":
                    self.assertEqual(inst.qubits, expected_qubits)

    def test_change_distribution_options(self):
        """Test that changing the distribution option and then resetting works as expected."""

        exp = rb.MirrorRB(
            physical_qubits=range(4),
            two_qubit_gate_density=1,
            lengths=[4],
            num_samples=1,
            backend=self.backend,
            seed=self.seed,
        )

        exp.distribution.gate_distribution = [(0.2, 1, "clifford"), (0.8, 2, ECRGate)]
        gates = exp.circuits()[0].count_ops()
        if "ecr" not in gates or "cx" in gates:
            raise QiskitError("Unexpected output gate distribution.")

        exp.distribution = EdgeGrabSampler

        gates = exp.circuits()[0].count_ops()

        if "cx" not in gates or "ecr" in gates:
            raise QiskitError("Unexpected output gate distribution.")


@ddt
class TestRunMirrorRB(RBRunTestCase):
    """Class for testing execution of mirror RB experiments."""

    seed = 123

    def setUp(self):
        """Setup the tests."""
        super().setUp()

        # depolarizing error
        self.p1q = 0.02
        self.p2q = 0.10
        self.pvz = 0.0

        # basis gates
        self.basis_gates = ["sx", "rz", "cx", "id"]

        # setup noise model
        sx_error = depolarizing_error(self.p1q, 1)
        rz_error = depolarizing_error(self.pvz, 1)
        cx_error = depolarizing_error(self.p2q, 2)

        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(sx_error, "sx")
        noise_model.add_all_qubit_quantum_error(rz_error, "rz")
        noise_model.add_all_qubit_quantum_error(cx_error, "cx")

        self.noise_model = noise_model
        self.basis_gates = noise_model.basis_gates

        # Need level1 for consecutive gate cancellation for reference EPC value calculation
        self.transpiler_options = {
            "basis_gates": self.basis_gates,
            "optimization_level": 1,
        }

        # Aer simulator
        self.backend = AerSimulator(
            noise_model=noise_model,
            seed_simulator=123,
            coupling_map=AerSimulator.from_backend(FakeParis()).configuration().coupling_map,
        )

    def test_single_qubit(self):
        """Test single qubit mirror RB."""
        exp = rb.MirrorRB(
            physical_qubits=(0,),
            lengths=list(range(2, 300, 40)),
            seed=self.seed,
            backend=self.backend,
            num_samples=20,
        )
        exp.analysis.set_options(gate_error_ratio=None)
        exp.set_transpile_options(**self.transpiler_options)

        expdata = exp.run()
        self.assertExperimentDone(expdata)

        # Given we have gate number per Clifford n_gpc, we can compute EPC as
        # EPC = 1 - (1 - r)^n_gpc
        # where r is gate error of SX gate, i.e. dep-parameter divided by 2.
        # We let transpiler use SX and RZ.
        # The number of physical gate per Clifford will distribute
        # from 0 to 2, i.e. arbitrary U gate can be decomposed into up to 2 SX with RZs.
        # We may want to expect the average number of SX is (0 + 1 + 2) / 3 = 1.0.
        # But for mirror RB, we must also add the SX gate number per Pauli n_gpp,
        # which is 2 for X and Y gates and 0 for I and Z gates (average = 1.0). So the
        # formula should be EPC = 1 - (1 - r)^(n_gpc + n_gpp) = 1 - (1 - r)^2
        epc = expdata.analysis_results("EPC")
        epc_expected = 1 - (1 - 1 / 2 * self.p1q) ** 2.0

        self.assertAlmostEqual(epc.value.n, epc_expected, delta=0.1 * epc_expected)

    def test_two_qubit(self):
        """Test two qubit RB."""
        two_qubit_gate_density = 0.2
        exp = rb.MirrorRB(
            physical_qubits=(0, 1),
            lengths=list(range(2, 80, 16)),
            seed=self.seed,
            backend=self.backend,
            num_samples=20,
            two_qubit_gate_density=two_qubit_gate_density,
        )
        exp.analysis.set_options(gate_error_ratio=None)
        exp.set_transpile_options(**self.transpiler_options)

        expdata = exp.run()
        self.assertExperimentDone(expdata)

        # Given a two qubit gate density xi and an n qubit circuit, a Clifford
        # layer has n*xi two-qubit gates. Obviously a Pauli has no two-qubit
        # gates, so on aveage, a Clifford + Pauli layer has n*xi two-qubit gates
        # and 2*n - 2*n*xi one-qubit gates (two layers have 2*n lattice sites,
        # 2*n*xi of which are occupied by two-qubit gates). For two-qubit
        # mirrored RB, the average infidelity is ((2^2 - 1)/2^2 = 3/4) times
        # the two-qubit depolarizing parameter
        epc = expdata.analysis_results("EPC")
        cx_factor = (1 - 3 * self.p2q / 4) ** (2 * two_qubit_gate_density)
        sx_factor = (1 - self.p1q / 2) ** (2 * 2 * (1 - two_qubit_gate_density))
        epc_expected = 1 - cx_factor * sx_factor
        self.assertAlmostEqual(epc.value.n, epc_expected, delta=0.2 * epc_expected)

    def test_two_qubit_nonlocal_noise(self):
        """Test for 2 qubit Mirrored RB with a nonlocal noise model"""
        # depolarizing error
        p1q = 0.0
        p2q = 0.01
        pvz = 0.0

        # setup noise model
        sx_error = depolarizing_error(p1q, 1)
        rz_error = depolarizing_error(pvz, 1)
        cx_error = depolarizing_error(p2q, 2)

        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(sx_error, "sx")
        noise_model.add_all_qubit_quantum_error(rz_error, "rz")
        noise_model.add_all_qubit_quantum_error(cx_error, "cx")

        basis_gates = ["id", "sx", "rz", "cx"]
        # Need level1 for consecutive gate cancellation for reference EPC value calculation
        transpiler_options = {
            "basis_gates": basis_gates,
            "optimization_level": 1,
        }
        # Coupling map is 3 x 3 lattice
        noise_backend = NoiseSimulator(
            noise_model=noise_model,
            seed_simulator=123,
            coupling_map=CouplingMap.from_grid(3, 3).get_edges(),
        )

        two_qubit_gate_density = 0.2
        exp = rb.MirrorRB(
            physical_qubits=(0, 1),
            lengths=list(range(2, 110, 20)),
            seed=self.seed,
            backend=noise_backend,
            num_samples=30,
            two_qubit_gate_density=two_qubit_gate_density,
        )
        exp.analysis.set_options(gate_error_ratio=None)
        exp.set_transpile_options(**transpiler_options)
        expdata = exp.run(noise_backend)
        self.assertExperimentDone(expdata)

        epc = expdata.analysis_results("EPC")
        # Compared to expected EPC in two-qubit test without nonlocal noise above,
        # we include an extra factor for the nonlocal CX error. This nonlocal
        # error is modeled by a one-qubit depolarizing channel on each qubit after
        # each CX, so the expected number of one-qubit depolarizing channels
        # induced by CXs is (number of CXs) * (number of qubits) = (two qubit gate
        # density) * (number of qubits) * (number of qubits).
        num_q = 2
        cx_factor = (1 - 3 * p2q / 4) ** (num_q * two_qubit_gate_density)
        sx_factor = (1 - p1q / 2) ** (2 * num_q * (1 - two_qubit_gate_density))
        cx_nonlocal_factor = (1 - 0.0035 / 2) ** (num_q * num_q * two_qubit_gate_density)
        epc_expected = 1 - cx_factor * sx_factor * cx_nonlocal_factor
        self.assertAlmostEqual(epc.value.n, epc_expected, delta=0.2 * epc_expected)

    def test_three_qubit_nonlocal_noise(self):
        """Test three-qubit mirrored RB on a nonlocal noise model"""
        # depolarizing error
        p1q = 0.001
        p2q = 0.01
        pvz = 0.0

        # setup noise modelle
        sx_error = depolarizing_error(p1q, 1)
        rz_error = depolarizing_error(pvz, 1)
        cx_error = depolarizing_error(p2q, 2)

        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(sx_error, "sx")
        noise_model.add_all_qubit_quantum_error(rz_error, "rz")
        noise_model.add_all_qubit_quantum_error(cx_error, "cx")

        basis_gates = ["id", "sx", "rz", "cx"]
        # Need level1 for consecutive gate cancellation for reference EPC value calculation
        transpiler_options = {
            "basis_gates": basis_gates,
            "optimization_level": 1,
        }
        noise_backend = NoiseSimulator(
            noise_model=noise_model,
            seed_simulator=123,
            coupling_map=CouplingMap.from_grid(3, 3).get_edges(),
        )

        two_qubit_gate_density = 0.2
        exp = rb.MirrorRB(
            physical_qubits=(0, 1, 2),
            lengths=list(range(2, 110, 50)),
            seed=self.seed,
            backend=noise_backend,
            num_samples=20,
            two_qubit_gate_density=two_qubit_gate_density,
        )
        exp.analysis.set_options(gate_error_ratio=None)
        exp.set_transpile_options(**transpiler_options)
        expdata = exp.run(noise_backend)
        self.assertExperimentDone(expdata)

        epc = expdata.analysis_results("EPC")
        # The expected EPC was computed in simulations not presented here.
        # Method:
        # 1. Sample N Clifford layers according to the edgegrab algorithm
        #    in clifford_utils.
        # 2. Transpile these into SX, RZ, and CX gates.
        # 3. Replace each SX and CX with one- and two-qubit depolarizing
        #    channels, respectively, and remove RZ gates.
        # 4. Use qiskit.quantum_info.average_gate_fidelity on these N layers
        #    to compute 1 - EPC for each layer, and average over the N layers.
        epc_expected = 0.0124
        self.assertAlmostEqual(epc.value.n, epc_expected, delta=0.2 * epc_expected)

    def test_add_more_circuit_yields_lower_variance(self):
        """Test variance reduction with larger number of sampling."""
        exp1 = rb.MirrorRB(
            physical_qubits=(0, 1),
            lengths=list(range(2, 30, 4)),
            seed=self.seed,
            backend=self.backend,
            num_samples=3,
            inverting_pauli_layer=False,
        )
        exp1.analysis.set_options(gate_error_ratio=None)
        exp1.set_transpile_options(**self.transpiler_options)
        expdata1 = exp1.run()
        self.assertExperimentDone(expdata1)

        exp2 = rb.MirrorRB(
            physical_qubits=(0, 1),
            lengths=list(range(2, 30, 4)),
            seed=456,
            backend=self.backend,
            num_samples=10,
            inverting_pauli_layer=False,
        )
        exp2.analysis.set_options(gate_error_ratio=None)
        exp2.set_transpile_options(**self.transpiler_options)
        expdata2 = exp2.run()
        self.assertExperimentDone(expdata2)

        self.assertLess(
            expdata2.analysis_results("EPC").value.s,
            expdata1.analysis_results("EPC").value.s,
        )

    @data(
        {
            "physical_qubits": [3, 3],
            "lengths": [2, 4, 6, 8, 10],
            "num_samples": 1,
            "seed": 100,
            "backend": AerSimulator(coupling_map=[[0, 1], [1, 0]]),
        },  # repeated qubits
        {
            "physical_qubits": [0, 1],
            "lengths": [2, 4, 6, -8, 10],
            "num_samples": 1,
            "seed": 100,
            "backend": AerSimulator(coupling_map=[[0, 1], [1, 0]]),
        },  # negative length
        {
            "physical_qubits": [0, 1],
            "lengths": [2, 4, 6, 8, 10],
            "num_samples": -4,
            "seed": 100,
            "backend": AerSimulator(coupling_map=[[0, 1], [1, 0]]),
        },  # negative number of samples
        {
            "physical_qubits": [0, 1],
            "lengths": [2, 4, 6, 8, 10],
            "num_samples": 0,
            "seed": 100,
            "backend": AerSimulator(coupling_map=[[0, 1], [1, 0]]),
        },  # zero samples
        {
            "physical_qubits": [0, 1],
            "lengths": [2, 6, 6, 6, 10],
            "num_samples": 2,
            "seed": 100,
            "backend": AerSimulator(coupling_map=[[0, 1], [1, 0]]),
        },  # repeated lengths
        {
            "physical_qubits": [0, 1],
            "lengths": [2, 4, 5, 8, 10],
            "num_samples": 2,
            "seed": 100,
            "backend": AerSimulator(coupling_map=[[0, 1], [1, 0]]),
        },  # odd length
        {
            "physical_qubits": [0, 1],
            "lengths": [2, 4, 6, 8, 10],
            "num_samples": 1,
            "seed": 100,
            "two_qubit_gate_density": -0.1,
            "backend": AerSimulator(coupling_map=[[0, 1], [1, 0]]),
        },  # negative two-qubit gate density
    )
    def test_invalid_configuration(self, configs):
        """Test raise error when creating experiment with invalid configs."""
        self.assertRaises(QiskitError, rb.MirrorRB, **configs)

    @data(
        {
            "physical_qubits": [0, 1],
            "lengths": [2, 4, 6, 8, 10],
            "num_samples": 1,
            "seed": 100,
            "backend": None,
        },  # no backend
    )
    def test_no_backend(self, configs):
        """Test raise error when no backend is provided for sampling circuits."""
        mirror_exp = rb.MirrorRB(**configs)
        self.assertRaises(QiskitError, mirror_exp.run)

    def test_expdata_serialization(self):
        """Test serializing experiment data works."""
        exp = rb.MirrorRB(
            physical_qubits=(0,),
            lengths=list(range(2, 200, 50)),
            seed=self.seed,
            backend=self.backend,
            inverting_pauli_layer=False,
        )
        exp.set_transpile_options(**self.transpiler_options)
        expdata = exp.run()
        self.assertExperimentDone(expdata)
        self.assertRoundTripSerializable(expdata, check_func=self.experiment_data_equiv)
        self.assertRoundTripPickle(expdata, check_func=self.experiment_data_equiv)


class TestEPGAnalysis(QiskitExperimentsTestCase):
    """Test case for EPG calculation from EPC.

    EPG and depolarizing probability p are assumed to have following relationship

        EPG = (2^n - 1) / 2^n · p

    This p is provided to the Aer noise model, thus we verify EPG computation
    by comparing the value with the depolarizing probability.
    """

    def setUp(self):
        """Setup the tests."""
        super().setUp()

        # Setup noise model, including more gate for complicated EPG computation
        # Note that 1Q channel error is amplified to check 1q channel correction mechanism
        self.p_x = 0.04
        self.p_h = 0.02
        self.p_s = 0.0
        self.p_cx = 0.09
        x_error = depolarizing_error(self.p_x, 1)
        h_error = depolarizing_error(self.p_h, 1)
        s_error = depolarizing_error(self.p_s, 1)
        cx_error = depolarizing_error(self.p_cx, 2)

        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(x_error, "x")
        noise_model.add_all_qubit_quantum_error(h_error, "h")
        noise_model.add_all_qubit_quantum_error(s_error, "s")
        noise_model.add_all_qubit_quantum_error(cx_error, "cx")

        # Need level1 for consecutive gate cancellation for reference EPC value calculation
        transpiler_options = {
            "basis_gates": ["x", "h", "s", "cx"],
            "optimization_level": 1,
        }

        # Aer simulator
        backend = AerSimulator(noise_model=noise_model, seed_simulator=123)

        # Prepare experiment data and cache for analysis
        exp_1qrb_q0 = rb.StandardRB(
            physical_qubits=(0,),
            lengths=[1, 10, 30, 50, 80, 120, 150, 200],
            seed=123,
            backend=backend,
        )
        exp_1qrb_q0.set_transpile_options(**transpiler_options)
        expdata_1qrb_q0 = exp_1qrb_q0.run(analysis=None).block_for_results(timeout=300)

        exp_1qrb_q1 = rb.StandardRB(
            physical_qubits=(1,),
            lengths=[1, 10, 30, 50, 80, 120, 150, 200],
            seed=123,
            backend=backend,
        )
        exp_1qrb_q1.set_transpile_options(**transpiler_options)
        expdata_1qrb_q1 = exp_1qrb_q1.run(analysis=None).block_for_results(timeout=300)

        exp_2qrb = rb.StandardRB(
            physical_qubits=(0, 1),
            lengths=[1, 3, 5, 10, 15, 20, 30, 50],
            seed=123,
            backend=backend,
        )
        exp_2qrb.set_transpile_options(**transpiler_options)
        expdata_2qrb = exp_2qrb.run(analysis=None).block_for_results(timeout=300)

        self.expdata_1qrb_q0 = expdata_1qrb_q0
        self.expdata_1qrb_q1 = expdata_1qrb_q1
        self.expdata_2qrb = expdata_2qrb

    def test_default_epg_ratio(self):
        """Calculate EPG with default ratio dictionary. H and X have the same ratio."""
        analysis = rb.RBAnalysis()
        analysis.set_options(outcome="0")
        result = analysis.run(self.expdata_1qrb_q0, replace_results=False)
        self.assertExperimentDone(result)

        s_epg = result.analysis_results("EPG_s")
        h_epg = result.analysis_results("EPG_h")
        x_epg = result.analysis_results("EPG_x")

        self.assertEqual(s_epg.value.n, 0.0)

        # H and X gate EPG are assumed to be the same, so this underestimate X and overestimate H
        self.assertEqual(h_epg.value.n, x_epg.value.n)
        self.assertLess(x_epg.value.n, self.p_x * 0.5)
        self.assertGreater(h_epg.value.n, self.p_h * 0.5)

    def test_no_epg(self):
        """Calculate no EPGs."""
        analysis = rb.RBAnalysis()
        analysis.set_options(outcome="0", gate_error_ratio=None)
        result = analysis.run(self.expdata_1qrb_q0, replace_results=False)
        self.assertExperimentDone(result)

        with self.assertRaises(ExperimentEntryNotFound):
            result.analysis_results("EPG_s")

        with self.assertRaises(ExperimentEntryNotFound):
            result.analysis_results("EPG_h")

        with self.assertRaises(ExperimentEntryNotFound):
            result.analysis_results("EPG_x")

    def test_with_custom_epg_ratio(self):
        """Calculate no EPGs with custom EPG ratio dictionary."""
        analysis = rb.RBAnalysis()
        analysis.set_options(outcome="0", gate_error_ratio={"x": 2, "h": 1, "s": 0})
        result = analysis.run(self.expdata_1qrb_q0, replace_results=False)
        self.assertExperimentDone(result)

        h_epg = result.analysis_results("EPG_h")
        x_epg = result.analysis_results("EPG_x")

        self.assertAlmostEqual(x_epg.value.n, self.p_x * 0.5, delta=0.005)
        self.assertAlmostEqual(h_epg.value.n, self.p_h * 0.5, delta=0.005)

    def test_2q_epg(self):
        """Compute 2Q EPG without correction.

        Since 1Q gates are designed to have comparable EPG with CX gate,
        this will overestimate the error of CX gate.
        """
        analysis = rb.RBAnalysis()
        analysis.set_options(outcome="00")
        result = analysis.run(self.expdata_2qrb, replace_results=False)
        self.assertExperimentDone(result)

        cx_epg = result.analysis_results("EPG_cx")

        self.assertGreater(cx_epg.value.n, self.p_cx * 0.75)

    def test_2q_epg_with_correction(self):
        """Check that 2Q EPG with 1Q depolarization correction gives a better (smaller) result than
        without the correction."""
        analysis_1qrb_q0 = rb.RBAnalysis()
        analysis_1qrb_q0.set_options(outcome="0", gate_error_ratio={"x": 2, "h": 1, "s": 0})
        result_q0 = analysis_1qrb_q0.run(self.expdata_1qrb_q0, replace_results=False)
        self.assertExperimentDone(result_q0)

        analysis_1qrb_q1 = rb.RBAnalysis()
        analysis_1qrb_q1.set_options(outcome="0", gate_error_ratio={"x": 2, "h": 1, "s": 0})
        result_q1 = analysis_1qrb_q1.run(self.expdata_1qrb_q1, replace_results=False)
        self.assertExperimentDone(result_q1)

        analysis_2qrb = rb.RBAnalysis()
        analysis_2qrb.set_options(
            outcome="00",
        )
        result_2qrb = analysis_2qrb.run(self.expdata_2qrb)
        self.assertExperimentDone(result_2qrb)
        cx_epg_raw = result_2qrb.analysis_results("EPG_cx")

        analysis_2qrb = rb.RBAnalysis()
        analysis_2qrb.set_options(
            outcome="00",
            epg_1_qubit=result_q0.analysis_results() + result_q1.analysis_results(),
        )
        result_2qrb = analysis_2qrb.run(self.expdata_2qrb)
        self.assertExperimentDone(result_2qrb)
        cx_epg_corrected = result_2qrb.analysis_results("EPG_cx")
        self.assertLess(
            np.abs(cx_epg_corrected.value.n - self.p_cx * 0.75),
            np.abs(cx_epg_raw.value.n - self.p_cx * 0.75),
        )
