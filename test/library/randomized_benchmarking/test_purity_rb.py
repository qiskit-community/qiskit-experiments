# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test for purity randomized benchmarking experiments."""

from test.base import QiskitExperimentsTestCase
from test.library.randomized_benchmarking.mixin import RBTestMixin
from ddt import ddt, data, unpack

from qiskit.exceptions import QiskitError
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit_experiments.library import randomized_benchmarking as rb


@ddt
class TestPurityRB(QiskitExperimentsTestCase, RBTestMixin):
    """Test for PurityRB without running the experiments."""

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
        self.assertRaises(QiskitError, rb.PurityRB, **configs)

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = rb.PurityRB(physical_qubits=(0,), lengths=[10, 20, 30], seed=123)
        loaded_exp = rb.PurityRB.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertEqualExtended(exp, loaded_exp)

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = rb.PurityRB(physical_qubits=(0,), lengths=[1, 3], seed=123)
        self.assertRoundTripSerializable(exp)

    def test_circuit_roundtrip_serializable(self):
        """Test circuits round trip JSON serialization"""
        exp = rb.PurityRB(physical_qubits=(0,), lengths=[1, 3], seed=123)
        self.assertRoundTripSerializable(exp._transpiled_circuits())

    def test_analysis_config(self):
        """ "Test converting analysis to and from config works"""
        analysis = rb.PurityRBAnalysis()
        loaded = rb.PurityRBAnalysis.from_config(analysis.config())
        self.assertNotEqual(analysis, loaded)
        self.assertEqual(analysis.config(), loaded.config())

    # ### Tests for circuit generation ###
    @data([[3], 4], [[4, 7], 5])
    @unpack
    def test_generate_circuits(self, qubits, length):
        """Test Purity RB circuit generation"""
        exp = rb.PurityRB(physical_qubits=qubits, lengths=[length], num_samples=1)
        circuits = exp.circuits()
        # Purity RB generates 3^n circuits per trial (n = num_qubits)
        # where n is the number of qubits
        expected_num_circuits = 3 ** len(qubits)
        self.assertEqual(len(circuits), expected_num_circuits)
        # Check that all circuits have the same xval (Clifford length)
        for circ in circuits:
            self.assertEqual(circ.metadata["xval"], length)
        # Check that all circuits belong to the same trial
        for circ in circuits:
            self.assertEqual(circ.metadata["trial"], 0)
        # Verify the base circuits (without post-rotations) are identity
        self.assertAllIdentity(circuits)

    def test_return_same_circuit(self):
        """Test if setting the same seed returns the same circuits."""
        exp1 = rb.PurityRB(
            physical_qubits=(0, 1),
            lengths=[10, 20, 30],
            seed=123,
            backend=self.backend,
        )

        exp2 = rb.PurityRB(
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
        exp1 = rb.PurityRB(
            physical_qubits=(0,),
            lengths=[10, 20, 30],
            seed=123,
            backend=self.backend,
            full_sampling=False,
        )
        exp2 = rb.PurityRB(
            physical_qubits=(0,),
            lengths=[10, 20, 30],
            seed=123,
            backend=self.backend,
            full_sampling=True,
        )
        circs1 = exp1.circuits()
        circs2 = exp2.circuits()

        # First circuit should be the same
        self.assertEqual(circs1[0].decompose(), circs2[0].decompose())

        # For purity RB, we need to compare the base circuits (first of each trial)
        # Each trial generates 3 circuits for 1-qubit case
        # fully sampled circuits are regenerated while other is just built on top of previous length
        self.assertNotEqual(circs1[3].decompose(), circs2[3].decompose())
        self.assertNotEqual(circs1[6].decompose(), circs2[6].decompose())

    def test_full_sampling_2_qubits(self):
        """Test if full sampling generates different circuits."""
        exp1 = rb.PurityRB(
            physical_qubits=(0, 1),
            lengths=[10, 20, 30],
            seed=123,
            backend=self.backend,
            full_sampling=False,
        )

        exp2 = rb.PurityRB(
            physical_qubits=(0, 1),
            lengths=[10, 20, 30],
            seed=123,
            backend=self.backend,
            full_sampling=True,
        )

        circs1 = exp1.circuits()
        circs2 = exp2.circuits()

        # First circuit should be the same
        self.assertEqual(circs1[0].decompose(), circs2[0].decompose())

        # For 2-qubit purity RB, each trial generates 9 circuits
        # fully sampled circuits are regenerated while other is just built on top of previous length
        self.assertNotEqual(circs1[9].decompose(), circs2[9].decompose())
        self.assertNotEqual(circs1[18].decompose(), circs2[18].decompose())

    def test_circuit_metadata(self):
        """Test that circuit metadata is correctly set."""
        exp = rb.PurityRB(
            physical_qubits=(0,),
            lengths=[5, 10],
            num_samples=2,
            seed=123,
        )
        circuits = exp.circuits()

        # For 1 qubit: 3 circuits per trial
        # 2 lengths * 2 samples = 4 trials
        # Total: 4 * 3 = 12 circuits
        self.assertEqual(len(circuits), 12)

        # Check trial numbering
        for i in range(4):
            for j in range(3):
                circ_idx = i * 3 + j
                self.assertEqual(circuits[circ_idx].metadata["trial"], i)

        # Check xval (Clifford length)
        # Circuits are ordered by trial:
        # [length1_sample1, length2_sample1, length1_sample2, length2_sample2]
        # For 1 qubit, each trial generates 3 circuits
        for i in range(3):  # Trial 0: length 5
            self.assertEqual(circuits[i].metadata["xval"], 5)
        for i in range(3, 6):  # Trial 1: length 10
            self.assertEqual(circuits[i].metadata["xval"], 10)
        for i in range(6, 9):  # Trial 2: length 5
            self.assertEqual(circuits[i].metadata["xval"], 5)
        for i in range(9, 12):  # Trial 3: length 10
            self.assertEqual(circuits[i].metadata["xval"], 10)


class TestRunPurityRB(QiskitExperimentsTestCase, RBTestMixin):
    """Test for running PurityRB."""

    def setUp(self):
        """Setup the tests."""
        super().setUp()

        # depolarizing error
        self.p1q = 0.004
        self.p2q = 0.04

        # basis gates
        self.basis_gates = ["rz", "sx", "cx"]

        # setup noise model
        sx_error = depolarizing_error(self.p1q, 1)
        cx_error = depolarizing_error(self.p2q, 2)

        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(sx_error, "sx")
        noise_model.add_all_qubit_quantum_error(cx_error, "cx")

        self.noise_model = noise_model

        # Transpiler options
        self.transpiler_options = {
            "basis_gates": self.basis_gates,
        }

        # Aer simulator
        self.backend = AerSimulator(noise_model=noise_model, seed_simulator=123)

    def test_single_qubit(self):
        """Test single qubit Purity RB."""
        exp = rb.PurityRB(
            physical_qubits=(0,),
            lengths=list(range(1, 200, 30)),
            seed=123,
            backend=self.backend,
        )
        exp.analysis.set_options(gate_error_ratio=None)
        exp.set_transpile_options(**self.transpiler_options)
        self.assertAllIdentity(exp.circuits())

        expdata = exp.run()
        self.assertExperimentDone(expdata)

        # Check that EPC_pur is calculated
        epc_pur = expdata.analysis_results("EPC_pur", dataframe=True).iloc[0]

        # For purity RB, the relationship is:
        # EPC_pur = (2^n - 1) / 2^n * (1 - alpha^0.5)
        # where alpha is the decay parameter
        # For 1 qubit: EPC_pur ≈ 0.5 * (1 - alpha^0.5)
        # The expected value should be close to standard RB EPC
        epc_expected = 1 - (1 - 1 / 2 * self.p1q) ** 1.0

        # Allow for larger tolerance due to purity measurement
        # Using 6 sigma to account for statistical fluctuations in purity measurements
        self.assertAlmostEqual(epc_pur.value.n, epc_expected, delta=6 * epc_pur.value.std_dev)

    def test_two_qubit(self):
        """Test two qubit Purity RB."""
        exp = rb.PurityRB(
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

        # Check that EPC_pur is calculated
        epc_pur = expdata.analysis_results("EPC_pur", dataframe=True).iloc[0]

        # For 2-qubit purity RB with CX error dominant
        # Expected EPC similar to standard RB
        epc_expected = 1 - (1 - 3 / 4 * self.p2q) ** 1.5

        # Allow for 40 percent tolerance due to purity measurement and 1q gate contribution
        self.assertAlmostEqual(epc_pur.value.n, epc_expected, delta=0.4 * epc_expected)

    def test_expdata_serialization(self):
        """Test serializing experiment data works."""
        exp = rb.PurityRB(
            physical_qubits=(0,),
            lengths=list(range(1, 100, 30)),
            seed=123,
            backend=self.backend,
        )
        exp.set_transpile_options(**self.transpiler_options)
        expdata = exp.run()
        self.assertExperimentDone(expdata)
        self.assertRoundTripSerializable(expdata)
        self.assertRoundTripPickle(expdata)
