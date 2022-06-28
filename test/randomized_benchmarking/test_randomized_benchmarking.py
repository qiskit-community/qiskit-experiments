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
import random
from ddt import ddt, data, unpack
from qiskit.circuit import Delay, QuantumCircuit
from qiskit.circuit.library import SXGate, CXGate, TGate, XGate, YGate, ZGate, HGate, SGate
from qiskit.exceptions import QiskitError
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
from qiskit.quantum_info import Clifford, Operator

from qiskit_experiments.library import randomized_benchmarking as rb
from qiskit_experiments.library.randomized_benchmarking import CliffordUtils
from qiskit_experiments.database_service.exceptions import DbExperimentEntryNotFound
from qiskit_experiments.framework.composite import ParallelExperiment


class RBTestCase(QiskitExperimentsTestCase):
    """Base test case for randomized benchmarking defining a common noise model."""

    def setUp(self):
        """Setup the tests."""
        super().setUp()

        # depolarizing error
        self.p1q = 0.02
        self.p2q = 0.10
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

        # Aer simulator
        self.backend = AerSimulator(noise_model=noise_model, seed_simulator=123)

    def assertAllIdentity(self, circuits):
        """Test if all experiment circuits are identity."""
        for circ in circuits:
            num_qubits = circ.num_qubits
            qc_iden = QuantumCircuit(num_qubits)
            circ.remove_final_measurements()
            assert (Operator(circ).equiv(Operator(qc_iden)))

@ddt
class TestStandardRB(RBTestCase):
    """Test for standard RB."""

    def test_single_qubit(self):
        """Test single qubit RB."""
        exp = rb.StandardRB(
            qubits=(0,),
            lengths=list(range(1, 300, 30)),
            seed=123,
            backend=self.backend,
            transpiled_rb=True
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
        self.assertAlmostEqual(epc.value.n, epc_expected, delta=0.1 * epc_expected)

    def test_two_qubit(self):
        """Test two qubit RB."""
        exp = rb.StandardRB(
            qubits=(0, 1),
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
        # Arbitrary SU(4) can be decomposed with (0, 1, 2, 3) CX gates, the expected
        # average number of CX gate per Clifford is 1.5.
        # Since this is two qubit RB, the dep-parameter is factored by 3/4.
        epc = expdata.analysis_results("EPC")

        # Allow for 50 percent tolerance since we ignore 1q gate contribution
        epc_expected = 1 - (1 - 3 / 4 * self.p2q) ** 1.5
        self.assertAlmostEqual(epc.value.n, epc_expected, delta=0.5 * epc_expected)

    def test_add_more_circuit_yields_lower_variance(self):
        """Test variance reduction with larger number of sampling."""
        exp1 = rb.StandardRB(
            qubits=(0, 1),
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
            qubits=(0, 1),
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

    def test_return_same_circuit(self):
        """Test if setting the same seed returns the same circuits."""
        exp1 = rb.StandardRB(
            qubits=(0, 1),
            lengths=[10, 20, 30],
            seed=123,
            backend=self.backend,
        )

        exp2 = rb.StandardRB(
            qubits=(0, 1),
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
            qubits=(0,),
            lengths=[10, 20, 30],
            seed=123,
            backend=self.backend,
            full_sampling=False,
            transpiled_rb=True
        )
        exp2 = rb.StandardRB(
            qubits=(0,),
            lengths=[10, 20, 30],
            seed=123,
            backend=self.backend,
            full_sampling=True,
            transpiled_rb=True
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
            qubits=(0, 1),
            lengths=[10, 20, 30],
            seed=123,
            backend=self.backend,
            full_sampling=False,
        )

        exp2 = rb.StandardRB(
            qubits=(0, 1),
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
        exp = rb.StandardRB(qubits=(0,), lengths=[10, 20, 30], seed=123, transpiled_rb=True)
        loaded_exp = rb.StandardRB.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.json_equiv(exp, loaded_exp))

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = rb.StandardRB(qubits=(0,), lengths=[10, 20, 30], seed=123, transpiled_rb=True)
        self.assertRoundTripSerializable(exp, self.json_equiv)

    def test_analysis_config(self):
        """ "Test converting analysis to and from config works"""
        analysis = rb.RBAnalysis()
        loaded = rb.RBAnalysis.from_config(analysis.config())
        self.assertNotEqual(analysis, loaded)
        self.assertEqual(analysis.config(), loaded.config())

    def test_expdata_serialization(self):
        """Test serializing experiment data works."""
        exp = rb.StandardRB(
            qubits=(0,),
            lengths=list(range(1, 200, 50)),
            seed=123,
            backend=self.backend,
            transpiled_rb=True
        )
        exp.set_transpile_options(**self.transpiler_options)
        expdata = exp.run()
        self.assertExperimentDone(expdata)
        self.assertRoundTripSerializable(expdata, check_func=self.experiment_data_equiv)
        self.assertRoundTripPickle(expdata, check_func=self.experiment_data_equiv)

    def test_single_qubit_parallel(self):
        """Test single qubit RB in parallel."""
        qubits = [0, 2]
        lengths = list(range(1, 300, 30))
        exps = []
        for qubit in qubits:
            exp = rb.StandardRB(
                qubits=[qubit],
                lengths=lengths,
                seed=123,
                backend=self.backend,
                transpiled_rb=True
            )
            exp.analysis.set_options(gate_error_ratio=None, plot_raw_data=False)
            exps.append(exp)

        par_exp = ParallelExperiment(exps)
        par_exp.set_transpile_options(optimization_level=1)

        par_expdata = par_exp.run(backend=self.backend)
        self.assertExperimentDone(par_expdata)
        epc_expected = 1 - (1 - 1 / 2 * self.p1q) ** 1.0
        for i in range(2):
            epc = par_expdata.child_data(i).analysis_results("EPC")
            self.assertAlmostEqual(epc.value.n, epc_expected, delta=0.1 * epc_expected)

@ddt
class TestInterleavedRB(RBTestCase):
    """Test for interleaved RB."""

    @data([SXGate(), [3], 4], [CXGate(), [4, 7], 5])
    @unpack
    def test_interleaved_structure(self, interleaved_element, qubits, length):
        """Verifies that when generating an interleaved circuit, it will be
        identical to the original circuit up to additions of
        barrier and interleaved element between any two Cliffords.
        """
        full_sampling =[True, False]
        for val in full_sampling:
            exp = rb.InterleavedRB(
                interleaved_element=interleaved_element, qubits=qubits, lengths=[length],
                num_samples=1, full_sampling=val, transpiled_rb=True
            )
            exp.set_transpile_options(**self.transpiler_options)
            exp.set_transpile_options(basis_gates=self.basis_gates)
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
                std_idx += 1
                int_idx += 1
                while c_std[std_idx][0].name != "barrier":
                    self.assertEqual(c_std[std_idx], c_int[int_idx])
                    std_idx += 1
                    int_idx += 1
                # for interleaved circuit: barrier + interleaved element
                self.assertEqual(c_int[int_idx][0].name, "barrier")
                int_idx += 1
                self.assertEqual(c_int[int_idx][0].name, interleaved_element.name)
                int_idx += 1

    def test_single_qubit(self):
        """Test single qubit IRB, once with an interleaved gate, once with an interleaved
        Clifford circuit.
        """
        interleaved_gate = SXGate()
        utils = CliffordUtils()
        random.seed(123)
        num = random.randint(0, 23)
        interleaved_clifford = utils.clifford_1_qubit_circuit(num)
        # The circuit created for interleaved_clifford is:
        # qc = QuantumCircuit(1)
        # qc.rz(np.pi/2, 0)
        # qc.sx(0)
        # qc.rz(np.pi/2, 0)
        # Since there is a single sx per interleaved_element,
        # therefore epc_expected is the same as for when interleaved_element = SXGate()
        for interleaved_element in [interleaved_gate, interleaved_clifford]:
            exp = rb.InterleavedRB(
                interleaved_element=interleaved_element,
                qubits=(0,),
                lengths=list(range(1, 300, 30)),
                seed=123,
                backend=self.backend,
                transpiled_rb=True,
            )
            exp.set_transpile_options(**self.transpiler_options)
            exp.set_transpile_options(basis_gates=self.basis_gates)

            self.assertAllIdentity(exp.circuits())

            expdata = exp.run()
            self.assertExperimentDone(expdata)

        # Since this is interleaved, we can directly compare values, i.e. n_gpc = 1
            epc = expdata.analysis_results("EPC")
            epc_expected = 1 / 2 * self.p1q
            self.assertAlmostEqual(epc.value.n, epc_expected, delta=0.1 * epc_expected)

    def test_two_qubit(self):
        """Test two qubit IRB."""
        exp = rb.InterleavedRB(
            interleaved_element=CXGate(),
            qubits=(0, 1),
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
        self.assertAlmostEqual(epc.value.n, epc_expected, delta=0.1 * epc_expected)

    def test_non_clifford_interleaved_element(self):
        """Verifies trying to run interleaved RB with non Clifford element throws an exception"""
        qubits = [0]
        lengths = [1, 4, 6, 9, 13, 16]
        interleaved_element = TGate()  # T gate is not Clifford, this should fail
        self.assertRaises(
            QiskitError,
            rb.InterleavedRB,
            interleaved_element=interleaved_element,
            qubits=qubits,
            lengths=lengths,
            transpiled_rb=True
        )

    def test_interleaving_delay(self):
        """Test delay instruction can be interleaved."""
        # See qiskit-experiments/#727 for details
        interleaved_element = Delay(10, unit="us")
        exp = rb.InterleavedRB(
            interleaved_element,
            qubits=[0],
            lengths=[1],
            num_samples=1,
            seed=1234,   # This seed gives a 2-gate clifford
            transpiled_rb=True
        )
        exp.set_transpile_options(**self.transpiler_options)
        exp.set_transpile_options(basis_gates=self.basis_gates)
        # Does not raise an error
        _, int_circs = exp.circuits()

        # barrier, 2-gate clifford, barrier, "delay", barrier, ...
        self.assertEqual(int_circs.data[4][0].name, interleaved_element.name)

        # Transpiled delay duration is represented in seconds, so must convert from us
        self.assertEqual(int_circs.data[4][0].unit, "s")
        self.assertAlmostEqual(int_circs.data[4][0].params[0], interleaved_element.params[0] * 1e-6)
        self.assertAllIdentity([int_circs])

    def test_interleaving_circuit_with_delay(self):
        """Test circuit with delay can be interleaved."""
        delay_qc = QuantumCircuit(2)
        delay_qc.delay(10, [0], unit="us")
        delay_qc.x(1)

        exp = rb.InterleavedRB(
            interleaved_element=delay_qc,
            qubits=[1, 2],
            lengths=[1],
            seed=123,
            num_samples=1,
            transpiled_rb=True
        )
        _, int_circ = exp.circuits()

        qc = QuantumCircuit(2)
        qc.x(1)
        expected_inversion = Clifford(int_circ.data[1][0]).compose(qc).adjoint()
        # barrier, clifford, barrier, "interleaved circuit", barrier, inversion, ...
        self.assertEqual(expected_inversion, Clifford(int_circ.data[5][0]))

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = rb.InterleavedRB(
            interleaved_element=SXGate(), qubits=(0,), lengths=[10, 20, 30], seed=123, transpiled_rb=True
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

    def test_expdata_serialization(self):
        """Test serializing experiment data works."""
        exp = rb.InterleavedRB(
            interleaved_element=SXGate(),
            qubits=(0,),
            lengths=list(range(1, 200, 50)),
            seed=123,
            backend=self.backend,
            transpiled_rb=True
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
        x_error = depolarizing_error(0.04, 1)
        h_error = depolarizing_error(0.02, 1)
        s_error = depolarizing_error(0.00, 1)
        cx_error = depolarizing_error(0.08, 2)

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
            qubits=(0,),
            lengths=[1, 10, 30, 50, 80, 120, 150, 200],
            seed=123,
            backend=backend,
            transpiled_rb=True
        )
        exp_1qrb_q0.set_transpile_options(**transpiler_options)
        expdata_1qrb_q0 = exp_1qrb_q0.run(analysis=None).block_for_results(timeout=300)
        exp_1qrb_q1 = rb.StandardRB(
            qubits=(1,),
            lengths=[1, 10, 30, 50, 80, 120, 150, 200],
            seed=123,
            backend=backend,
            transpiled_rb=True
        )
        exp_1qrb_q1.set_transpile_options(**transpiler_options)
        expdata_1qrb_q1 = exp_1qrb_q1.run(analysis=None).block_for_results(timeout=300)

        exp_2qrb = rb.StandardRB(
            qubits=(0, 1),
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
        self.assertLess(x_epg.value.n, 0.04 * 0.5)
        self.assertGreater(h_epg.value.n, 0.02 * 0.5)

    def test_no_epg(self):
        """Calculate no EPGs."""
        analysis = rb.RBAnalysis()
        analysis.set_options(outcome="0", gate_error_ratio=None)
        result = analysis.run(self.expdata_1qrb_q0, replace_results=False)
        self.assertExperimentDone(result)

        with self.assertRaises(DbExperimentEntryNotFound):
            result.analysis_results("EPG_s")

        with self.assertRaises(DbExperimentEntryNotFound):
            result.analysis_results("EPG_h")

        with self.assertRaises(DbExperimentEntryNotFound):
            result.analysis_results("EPG_x")

    def test_with_custom_epg_ratio(self):
        """Calculate no EPGs with custom EPG ratio dictionary."""
        analysis = rb.RBAnalysis()
        analysis.set_options(outcome="0", gate_error_ratio={"x": 2, "h": 1, "s": 0})
        result = analysis.run(self.expdata_1qrb_q0, replace_results=False)
        self.assertExperimentDone(result)

        h_epg = result.analysis_results("EPG_h")
        x_epg = result.analysis_results("EPG_x")

        self.assertAlmostEqual(x_epg.value.n, 0.04 * 0.5, delta=0.005)
        self.assertAlmostEqual(h_epg.value.n, 0.02 * 0.5, delta=0.005)

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

        self.assertGreater(cx_epg.value.n, 0.08 * 0.75)

    def test_correct_1q_depolarization(self):
        """Compute 2Q EPG with 1Q depolarization correction."""
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
            epg_1_qubit=result_q0.analysis_results() + result_q1.analysis_results(),
        )
        result_2qrb = analysis_2qrb.run(self.expdata_2qrb)
        self.assertExperimentDone(result_2qrb)

        cx_epg = result_2qrb.analysis_results("EPG_cx")
        self.assertAlmostEqual(cx_epg.value.n, 0.08 * 0.75, delta=0.006)
