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
import copy

from test.base import QiskitExperimentsTestCase
from test.library.randomized_benchmarking.mixin import RBTestMixin
from ddt import ddt, data, unpack

from qiskit.circuit.library import SXGate
from qiskit.exceptions import QiskitError
from qiskit.providers.fake_provider import FakeManilaV2
from qiskit.pulse import Schedule, InstructionScheduleMap
from qiskit_experiments.library import randomized_benchmarking as rb


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
    def test_generate_circuits(self, qubits, length):
        """Test RB circuit generation"""
        exp = rb.StandardRB(physical_qubits=qubits, lengths=[length], num_samples=1)
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
        qubits = (2,)
        my_sched = Schedule(name="custom_sx_gate")
        my_inst_map = InstructionScheduleMap()
        my_inst_map.add(SXGate(), qubits, my_sched)

        exp = rb.StandardRB(
            physical_qubits=qubits, lengths=[3], num_samples=4, backend=self.backend, seed=123
        )
        exp.set_transpile_options(inst_map=my_inst_map)
        transpiled = exp._transpiled_circuits()
        for qc in transpiled:
            self.assertTrue(qc.calibrations)
            self.assertTrue(qc.has_calibration_for((SXGate(), [qc.qubits[q] for q in qubits], [])))
            self.assertEqual(qc.calibrations["sx"][(qubits, tuple())], my_sched)

    def test_calibrations_via_custom_backend(self):
        """Test if calibrations given as custom backend show up in transpiled circuits."""
        qubits = (2,)
        my_sched = Schedule(name="custom_sx_gate")
        my_backend = copy.deepcopy(self.backend)
        my_backend.target["sx"][qubits].calibration = my_sched

        exp = rb.StandardRB(physical_qubits=qubits, lengths=[3], num_samples=4, backend=my_backend)
        transpiled = exp._transpiled_circuits()
        for qc in transpiled:
            self.assertTrue(qc.calibrations)
            self.assertTrue(qc.has_calibration_for((SXGate(), [qc.qubits[q] for q in qubits], [])))
            self.assertEqual(qc.calibrations["sx"][(qubits, tuple())], my_sched)

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
