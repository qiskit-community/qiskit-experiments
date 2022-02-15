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

import numpy as np
from ddt import ddt, data, unpack
from qiskit.circuit.library import SXGate, CXGate, TGate, XGate
from qiskit.exceptions import QiskitError
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
from qiskit.quantum_info import Clifford
from qiskit.quantum_info.operators.predicates import matrix_equal

from qiskit_experiments.library import randomized_benchmarking as rb


class RBTestCase(QiskitExperimentsTestCase):
    """Base test case for randomized benchmarking defining a common noise model."""

    def setUp(self):
        """Setup the tests."""
        super().setUp()

        # depolarizing error
        self.p1q = 0.002
        self.p2q = 0.010
        self.pvz = 0.0

        # basis gates
        self.basis_gates = ["sx", "rz", "cx"]

        # setup noise model
        sx_error = depolarizing_error(self.p1q, 1)
        rz_error = depolarizing_error(self.pvz, 1)
        cx_error = depolarizing_error(self.p2q, 2)

        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(sx_error, "sx")
        noise_model.add_all_qubit_quantum_error(rz_error, "rz")
        noise_model.add_all_qubit_quantum_error(cx_error, "cx")

        self.noise_model = noise_model

        # Need level1 for consecutive gate cancellation for reference EPC value calculation
        self.transpiler_options = {
            "basis_gates": self.basis_gates,
            "optimization_level": 1,
        }

        # Aer simulator run options
        self.run_options = {
            "seed_simulator": 123,
            "noise_model": self.noise_model,
        }

        self.backend = AerSimulator()

    def is_identity(self, circuits):
        """Standard randomized benchmarking test - Identity check.
        (assuming all the operator are spanned by clifford group)
        """
        for circ in circuits:
            num_qubits = circ.num_qubits
            circ.remove_final_measurements()
            # Checking if the matrix representation is the identity matrix
            self.assertTrue(
                matrix_equal(Clifford(circ).to_matrix(), np.identity(2**num_qubits)),
                "Clifford sequence doesn't result in the identity matrix.",
            )


@ddt
class TestStandardRB(RBTestCase):
    """Test for standard RB."""

    @data(123, 456)
    def test_single_qubit(self, seed):
        """Test single qubit RB."""
        exp = rb.StandardRB(
            qubits=(0,),
            lengths=list(range(1, 1000, 100)),
            seed=seed,
            backend=self.backend,
        )
        exp.set_run_options(**self.run_options)
        exp.set_transpile_options(**self.transpiler_options)
        exp.analysis.set_options(gate_error_ratio={((0,), "sx"): 1.0, ((0,), "rz"): 0.0})
        self.is_identity(exp.circuits())

        expdata = exp.run()
        self.assertExperimentDone(expdata)

        # Since this is 1Q RB, the gate error is factored by 0.5 := (2^1 - 1) / 2^1
        # We know the depolarizing parameter since we defined the noise model
        sx_epg = expdata.analysis_results("EPG_sx")
        rz_epg = expdata.analysis_results("EPG_rz")

        # Allow for 10 percent tolerance
        self.assertAlmostEqual(sx_epg.value.n, self.p1q / 2, delta=0.1 * self.p1q)
        self.assertAlmostEqual(rz_epg.value.n, self.pvz / 2, delta=0.1 * self.p1q)

        # Given we have gate number per Clifford n_gpc, we can compute EPC as
        # EPC = 1 - (1 - r)^n_gpc
        # where r is gate error of SX gate, i.e. dep-parameter divided by 2.
        # We let transpiler use SX and RZ.
        # The number of physical gate per Clifford will distribute
        # from 0 to 2, i.e. arbitrary U gate can be decomposed into up to 2 SX with RZs.
        # We may want to expect the average number of SX is (0 + 1 + 2) / 3 = 1.0.
        epc = expdata.analysis_results("EPC")

        epc_expected = 1 - (1 - 1 / 2 * self.p1q) ** 1.0
        self.assertAlmostEqual(epc.value.n, epc_expected, delta=0.1 * epc_expected)

    @data(123, 456)
    def test_two_qubit(self, seed):
        """Test two qubit RB."""
        exp = rb.StandardRB(
            qubits=(0, 1),
            lengths=list(range(1, 200, 20)),
            seed=seed,
            backend=self.backend,
        )
        exp.set_run_options(**self.run_options)
        exp.set_transpile_options(**self.transpiler_options)
        self.is_identity(exp.circuits())

        expdata = exp.run()
        self.assertExperimentDone(expdata)

        # Given CX error is dominant and 1q error can be negligible.
        # Arbitrary SU(4) can be decomposed with (0, 1, 2, 3) CX gates, the expected
        # average number of CX gate per Clifford is 1.5.
        # Since this is two qubit RB, the dep-parameter is factored by 3/4.
        epc = expdata.analysis_results("EPC")

        # Allow for 20 percent tolerance since we ignore 1q gate contribution
        epc_expected = 1 - (1 - 3 / 4 * self.p2q) ** 1.5
        self.assertAlmostEqual(epc.value.n, epc_expected, delta=0.3 * epc_expected)

    @data(
        {"qubits": [3, 3], "lengths": [1, 3, 5, 7, 9], "num_samples": 1, "seed": 100},
        {"qubits": [0, 1], "lengths": [1, 3, 5, -7, 9], "num_samples": 1, "seed": 100},
        {"qubits": [0, 1], "lengths": [1, 3, 5, 7, 9], "num_samples": -4, "seed": 100},
        {"qubits": [0, 1], "lengths": [1, 3, 5, 7, 9], "num_samples": 0, "seed": 100},
        {"qubits": [0, 1], "lengths": [1, 5, 5, 5, 9], "num_samples": 2, "seed": 100},
    )
    def test_invalid_configuration(self, configs):
        """Test raise error when creating experiment with invalid configs."""
        self.assertRaises(QiskitError, rb.StandardRB, **configs)

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = rb.StandardRB(qubits=(0,), lengths=[10, 20, 30], seed=123)
        loaded_exp = rb.StandardRB.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.json_equiv(exp, loaded_exp))

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = rb.StandardRB(qubits=(0,), lengths=[10, 20, 30], seed=123)
        self.assertRoundTripSerializable(exp, self.json_equiv)

    def test_analysis_config(self):
        """ "Test converting analysis to and from config works"""
        analysis = rb.RBAnalysis()
        loaded = rb.RBAnalysis.from_config(analysis.config())
        self.assertNotEqual(analysis, loaded)
        self.assertEqual(analysis.config(), loaded.config())


@ddt
class TestInterleavedRB(RBTestCase):
    """Test for interleaved RB. No data driven test due to overhead (roughly 2x)"""

    @data([XGate(), [3], 4], [CXGate(), [4, 7], 5])
    @unpack
    def test_interleaved_structure(self, interleaved_element, qubits, length):
        """Verifies that when generating an interleaved circuit, it will be
        identical to the original circuit up to additions of
        barrier and interleaved element between any two Cliffords.
        """
        exp = rb.InterleavedRB(
            interleaved_element=interleaved_element, qubits=qubits, lengths=[length], num_samples=1
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
            # barrier
            self.assertEqual(c_std[std_idx][0].name, "barrier")
            self.assertEqual(c_int[int_idx][0].name, "barrier")
            # clifford
            self.assertEqual(c_std[std_idx + 1], c_int[int_idx + 1])
            # for interleaved circuit: barrier + interleaved element
            self.assertEqual(c_int[int_idx + 2][0].name, "barrier")
            self.assertEqual(c_int[int_idx + 3][0].name, interleaved_element.name)
            std_idx += 2
            int_idx += 4

    def test_single_qubit(self):
        """Test single qubit IRB."""
        exp = rb.InterleavedRB(
            interleaved_element=SXGate(),
            qubits=(0,),
            lengths=list(range(1, 1000, 100)),
            seed=123,
            backend=self.backend,
        )
        exp.set_run_options(**self.run_options)
        exp.set_transpile_options(**self.transpiler_options)
        self.is_identity(exp.circuits())

        expdata = exp.run()
        self.assertExperimentDone(expdata, timeout=300)

        # Since this is interleaved, we can directly compare values, i.e. n_gpc = 1
        epc = expdata.analysis_results("EPC")
        epc_expected = 1 / 2 * self.p1q
        self.assertAlmostEqual(epc.value.n, epc_expected, delta=0.1 * epc_expected)

    def test_two_qubit(self):
        """Test two qubit IRB."""
        exp = rb.InterleavedRB(
            interleaved_element=CXGate(),
            qubits=(0, 1),
            lengths=list(range(1, 200, 20)),
            seed=123,
            backend=self.backend,
        )
        exp.set_run_options(**self.run_options)
        exp.set_transpile_options(**self.transpiler_options)
        self.is_identity(exp.circuits())

        expdata = exp.run()
        self.assertExperimentDone(expdata)

        # Since this is interleaved, we can directly compare values, i.e. n_gpc = 1
        epc = expdata.analysis_results("EPC")
        epc_expected = 3 / 4 * self.p2q
        self.assertAlmostEqual(epc.value.n, epc_expected, delta=0.1 * epc_expected)

    def test_non_clifford_interleaved_element(self):
        """Verifies trying to run interleaved RB with non Clifford element throws an exception"""
        qubits = 1
        lengths = [1, 4, 6, 9, 13, 16]
        interleaved_element = TGate()  # T gate is not Clifford, this should fail
        self.assertRaises(
            QiskitError,
            rb.InterleavedRB,
            interleaved_element=interleaved_element,
            qubits=qubits,
            lengths=lengths,
        )

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = rb.InterleavedRB(
            interleaved_element=SXGate(), qubits=(0,), lengths=[10, 20, 30], seed=123
        )
        loaded_exp = rb.InterleavedRB.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.json_equiv(exp, loaded_exp))

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = rb.InterleavedRB(
            interleaved_element=SXGate(), qubits=(0,), lengths=[10, 20, 30], seed=123
        )
        self.assertRoundTripSerializable(exp, self.json_equiv)

    def test_analysis_config(self):
        """ "Test converting analysis to and from config works"""
        analysis = rb.InterleavedRBAnalysis()
        loaded = rb.InterleavedRBAnalysis.from_config(analysis.config())
        self.assertNotEqual(analysis, loaded)
        self.assertEqual(analysis.config(), loaded.config())
