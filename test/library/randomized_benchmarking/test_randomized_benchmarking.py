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
from qiskit.providers.fake_provider import FakeParis
from qiskit.circuit import Delay, QuantumCircuit
from qiskit.circuit.library import SXGate, CXGate, TGate, XGate
from qiskit.exceptions import QiskitError
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
from qiskit.quantum_info import Clifford
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler import Layout, PassManager, CouplingMap
from qiskit.transpiler.exceptions import TranspilerError

from qiskit_experiments.library import randomized_benchmarking as rb
from qiskit_experiments.database_service.exceptions import ExperimentEntryNotFound


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

    def assertAllIdentity(self, circuits):
        """Test if all experiment circuits are identity."""
        for circ in circuits:
            num_qubits = circ.num_qubits
            iden = Clifford(np.eye(2 * num_qubits, dtype=bool))
            circ.remove_final_measurements()
            # In case the final output is |0>^num_qubits up to a phase, we use .table.pauli
            self.assertEqual(
                Clifford(circ).table.pauli,
                iden.table.pauli,
                f"Circuit {circ.name} doesn't result in the identity matrix.",
            )


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
class TestMirrorRB(RBTestCase):
    """Test for mirror RB."""

    def test_single_qubit(self):
        """Test single qubit mirror RB."""
        exp = rb.MirrorRB(
            qubits=(0,),
            lengths=list(range(2, 300, 20)),
            seed=124,
            backend=self.backend,
            num_samples=30,
        )
        # exp.analysis.set_options(gate_error_ratio=None)
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
            qubits=(0, 1),
            lengths=list(range(2, 300, 20)),
            seed=123,
            backend=self.backend,
            num_samples=30,
            two_qubit_gate_density=two_qubit_gate_density,
        )
        exp.analysis.set_options(gate_error_ratio=None)
        exp.set_transpile_options(**self.transpiler_options)
        self.assertAllIdentity(exp.circuits())

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
        self.assertAlmostEqual(epc.value.n, epc_expected, delta=0.1 * epc_expected)

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
            qubits=(0, 1),
            lengths=list(range(2, 110, 20)),
            seed=123,
            backend=noise_backend,
            num_samples=20,
            two_qubit_gate_density=two_qubit_gate_density,
        )
        exp.analysis.set_options(gate_error_ratio=None)
        exp.set_transpile_options(**transpiler_options)
        self.assertAllIdentity(exp.circuits())
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
        self.assertAlmostEqual(epc.value.n, epc_expected, delta=0.1 * epc_expected)

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
            qubits=(0, 1, 2),
            lengths=list(range(2, 110, 50)),
            seed=123,
            backend=noise_backend,
            num_samples=20,
            two_qubit_gate_density=two_qubit_gate_density,
        )
        exp.analysis.set_options(gate_error_ratio=None)
        exp.set_transpile_options(**transpiler_options)
        self.assertAllIdentity(exp.circuits())
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
            qubits=(0, 1),
            lengths=list(range(2, 30, 4)),
            seed=123,
            backend=self.backend,
            num_samples=3,
            inverting_pauli_layer=False,
        )
        exp1.analysis.set_options(gate_error_ratio=None)
        exp1.set_transpile_options(**self.transpiler_options)
        expdata1 = exp1.run()
        self.assertExperimentDone(expdata1)

        exp2 = rb.MirrorRB(
            qubits=(0, 1),
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

    def test_return_same_circuit(self):
        """Test if setting the same seed returns the same circuits."""
        lengths = [10, 20]
        exp1 = rb.MirrorRB(
            qubits=(0, 1),
            lengths=lengths,
            seed=123,
            backend=self.backend,
        )

        exp2 = rb.MirrorRB(
            qubits=(0, 1),
            lengths=lengths,
            seed=123,
            backend=self.backend,
        )

        circs1 = exp1.circuits()
        circs2 = exp2.circuits()

        for circ1, circ2 in zip(circs1, circs2):
            self.assertEqual(circ1.decompose(), circ2.decompose())

    def test_full_sampling(self):
        """Test if full sampling generates different circuits."""
        exp1 = rb.MirrorRB(
            qubits=(0, 1),
            lengths=[10, 20],
            seed=123,
            backend=self.backend,
            num_samples=1,
            full_sampling=True,
        )

        exp2 = rb.MirrorRB(
            qubits=(0, 1),
            lengths=[10, 20],
            seed=123,
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

    def test_target_bitstring(self):
        """Test if correct target bitstring is returned."""
        qc = QuantumCircuit(9)
        qc.z(0)
        qc.y(1)
        qc.y(2)
        qc.z(3)
        qc.y(4)
        qc.x(7)
        qc.y(8)
        exp = rb.MirrorRB(qubits=[0], lengths=[2], backend=self.backend)
        expected_tb = exp._clifford_utils.compute_target_bitstring(qc)
        actual_tb = "110010110"
        self.assertEqual(expected_tb, actual_tb)

    def test_zero_2q_gate_density(self):
        """Test that there are no two-qubit gates when the two-qubit gate
        density is set to 0."""
        exp = rb.MirrorRB(
            qubits=(0, 1),
            lengths=[40],
            seed=124,
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
            qubits=(0, 1, 2, 3),
            lengths=[40],
            seed=125,
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

    def test_local_clifford(self):
        """Test that the number of layers is correct depending on whether
        local_clifford is set to True or False by counting the number of barriers."""
        exp = rb.MirrorRB(
            qubits=(0,),
            lengths=[2],
            seed=126,
            backend=self.backend,
            num_samples=1,
            local_clifford=True,
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
            qubits=(0,),
            lengths=[2],
            seed=126,
            backend=self.backend,
            num_samples=1,
            local_clifford=False,
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
        """Test that a circuit with an inverting Pauli layer at the end (i.e.,
        a layer of Paulis before the final measurement that restores the output
        to |0>^num_qubits up to a global phase) composes to the identity (up to
        a global phase)"""
        exp = rb.MirrorRB(
            qubits=(0, 1, 2),
            lengths=[2],
            seed=127,
            backend=self.backend,
            num_samples=3,
            local_clifford=True,
            pauli_randomize=True,
            two_qubit_gate_density=0.2,
            inverting_pauli_layer=True,
        )
        self.assertAllIdentity(exp.circuits())

    @data(
        {
            "qubits": [3, 3],
            "lengths": [2, 4, 6, 8, 10],
            "num_samples": 1,
            "seed": 100,
            "backend": AerSimulator(coupling_map=[[0, 1], [1, 0]]),
        },  # repeated qubits
        {
            "qubits": [0, 1],
            "lengths": [2, 4, 6, -8, 10],
            "num_samples": 1,
            "seed": 100,
            "backend": AerSimulator(coupling_map=[[0, 1], [1, 0]]),
        },  # negative length
        {
            "qubits": [0, 1],
            "lengths": [2, 4, 6, 8, 10],
            "num_samples": -4,
            "seed": 100,
            "backend": AerSimulator(coupling_map=[[0, 1], [1, 0]]),
        },  # negative number of samples
        {
            "qubits": [0, 1],
            "lengths": [2, 4, 6, 8, 10],
            "num_samples": 0,
            "seed": 100,
            "backend": AerSimulator(coupling_map=[[0, 1], [1, 0]]),
        },  # zero samples
        {
            "qubits": [0, 1],
            "lengths": [2, 6, 6, 6, 10],
            "num_samples": 2,
            "seed": 100,
            "backend": AerSimulator(coupling_map=[[0, 1], [1, 0]]),
        },  # repeated lengths
        {
            "qubits": [0, 1],
            "lengths": [2, 4, 5, 8, 10],
            "num_samples": 2,
            "seed": 100,
            "backend": AerSimulator(coupling_map=[[0, 1], [1, 0]]),
        },  # odd length
        {
            "qubits": [0, 1],
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
            "qubits": [0, 1],
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

    @data(
        {
            "qubits": [0, 25],
            "lengths": [2, 4, 6, 8, 10],
            "num_samples": 1,
            "seed": 100,
            "backend": AerSimulator.from_backend(FakeParis()),
        },  # Uncoupled qubits to test edgegrab algorithm warning
        {
            "qubits": [0, 1],
            "lengths": [2, 4, 6, 8, 10],
            "num_samples": 1,
            "seed": 100,
            "two_qubit_gate_density": 0.6,
            "backend": AerSimulator(coupling_map=[[0, 1], [1, 0]]),
        },  # High two-qubit gate density warning
    )
    def test_warnings(self, configs):
        """Test raise warnings when creating experiment."""
        mirror_exp = rb.MirrorRB(**configs)
        self.assertWarns(Warning, mirror_exp.run)

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = rb.MirrorRB(qubits=(0,), lengths=[10, 20, 30], seed=123, backend=self.backend)
        loaded_exp = rb.MirrorRB.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.json_equiv(exp, loaded_exp))

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = rb.MirrorRB(qubits=(0,), lengths=[10, 20, 30], seed=123)
        self.assertRoundTripSerializable(exp, self.json_equiv)

    def test_analysis_config(self):
        """ "Test converting analysis to and from config works"""
        analysis = rb.RBAnalysis()
        loaded = rb.RBAnalysis.from_config(analysis.config())
        self.assertNotEqual(analysis, loaded)
        self.assertEqual(analysis.config(), loaded.config())

    def test_expdata_serialization(self):
        """Test serializing experiment data works."""
        exp = rb.MirrorRB(
            qubits=(0,),
            lengths=list(range(2, 200, 50)),
            seed=123,
            backend=self.backend,
            inverting_pauli_layer=False,
        )
        exp.set_transpile_options(**self.transpiler_options)
        expdata = exp.run()
        self.assertExperimentDone(expdata)
        self.assertRoundTripSerializable(expdata, check_func=self.experiment_data_equiv)
        self.assertRoundTripPickle(expdata, check_func=self.experiment_data_equiv)


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

    def test_poor_experiment_result(self):
        """Test edge case that tail of decay is not sampled.

        This is a special case that fit outcome is very sensitive to initial guess.
        Perhaps generated initial guess is close to a local minima.
        """
        from qiskit.providers.fake_provider import FakeVigoV2

        backend = FakeVigoV2()
        exp = rb.StandardRB(
            qubits=(0,),
            lengths=[100, 200, 300, 400],
            seed=123,
            backend=backend,
            num_samples=5,
        )
        exp.set_transpile_options(basis_gates=["x", "sx", "rz"], optimization_level=1)
        # Simulator seed must be fixed. This can be set via run option with FakeBackend.
        # pylint: disable=no-member
        exp.set_run_options(seed_simulator=456)
        expdata = exp.run()
        self.assertExperimentDone(expdata)

        overview = expdata.analysis_results(0).value
        # This yields bad fit due to poor data points, but still fit is not completely off.
        self.assertLess(overview.reduced_chisq, 10)

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

    def test_full_sampling(self):
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

    def test_expdata_serialization(self):
        """Test serializing experiment data works."""
        exp = rb.StandardRB(
            qubits=(0,),
            lengths=list(range(1, 200, 50)),
            seed=123,
            backend=self.backend,
        )
        exp.set_transpile_options(**self.transpiler_options)
        expdata = exp.run()
        self.assertExperimentDone(expdata)
        self.assertRoundTripSerializable(expdata, check_func=self.experiment_data_equiv)
        self.assertRoundTripPickle(expdata, check_func=self.experiment_data_equiv)


@ddt
class TestInterleavedRB(RBTestCase):
    """Test for interleaved RB."""

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

    def test_interleaving_delay(self):
        """Test delay instruction can be interleaved."""
        # See qiskit-experiments/#727 for details
        interleaved_element = Delay(10, unit="us")
        exp = rb.InterleavedRB(
            interleaved_element,
            qubits=[0],
            lengths=[1],
            num_samples=1,
        )
        # Not raises an error
        _, int_circ = exp.circuits()

        # barrier, clifford, barrier, "delay", barrier, ...
        self.assertEqual(int_circ.data[3][0], interleaved_element)

    def test_interleaving_circuit_with_delay(self):
        """Test circuit with delay can be interleaved."""
        delay_qc = QuantumCircuit(2)
        delay_qc.delay(10, [0], unit="us")
        delay_qc.x(1)

        exp = rb.InterleavedRB(
            interleaved_element=delay_qc, qubits=[1, 2], lengths=[1], seed=123, num_samples=1
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

    def test_expdata_serialization(self):
        """Test serializing experiment data works."""
        exp = rb.InterleavedRB(
            interleaved_element=SXGate(),
            qubits=(0,),
            lengths=list(range(1, 200, 50)),
            seed=123,
            backend=self.backend,
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
        )
        exp_1qrb_q0.set_transpile_options(**transpiler_options)
        expdata_1qrb_q0 = exp_1qrb_q0.run(analysis=None).block_for_results(timeout=300)

        exp_1qrb_q1 = rb.StandardRB(
            qubits=(1,),
            lengths=[1, 10, 30, 50, 80, 120, 150, 200],
            seed=123,
            backend=backend,
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
