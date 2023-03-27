# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test for mirror randomized benchmarking experiments."""
from test.base import QiskitExperimentsTestCase
from test.library.randomized_benchmarking.mixin import RBTestMixin

import copy

from ddt import ddt, data

from qiskit.circuit.library import CXGate, ECRGate
from qiskit.exceptions import QiskitError
from qiskit.providers.fake_provider import FakeManilaV2
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler import Layout, PassManager, CouplingMap
from qiskit.transpiler.exceptions import TranspilerError

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from qiskit_experiments.library import randomized_benchmarking as rb


from qiskit_experiments.library.randomized_benchmarking.clifford_utils import (
    compute_target_bitstring,
)


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

    def decr_dep_param(self, q, q_1, q_2, coupling_map):
        """Helper function to generate a one-qubit depolarizing channel whose
        parameter depends on coupling map distance in a backend"""
        d = min(coupling_map.distance(q, q_1), coupling_map.distance(q, q_2))
        return 0.0035 * 0.999**d

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
                        dep_param = self.decr_dep_param(qubit, qubit_1, qubit_2, self.coupling_map)
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
        self.backend = FakeManilaV2()

        self.basis_gates = ["sx", "rz", "cx"]

        self.transpiler_options = {
            "basis_gates": self.basis_gates,
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
        exp = rb.MirrorRB([0], lengths=[10, 20, 30], seed=123, two_qubit_gate=ECRGate())
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


@ddt
class TestRunMirrorRB(QiskitExperimentsTestCase, RBTestMixin):
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
            coupling_map=AerSimulator.from_backend(FakeManilaV2()).configuration().coupling_map,
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

        self.assertAlmostEqual(epc.value.n, epc_expected, delta=3 * epc.value.std_dev)

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
        self.assertAlmostEqual(epc.value.n, epc_expected, delta=3 * epc.value.std_dev)

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

        transpiler_options = {
            "basis_gates": basis_gates,
        }
        noise_backend = NoiseSimulator(
            noise_model=noise_model,
            seed_simulator=123,
            coupling_map=CouplingMap.from_grid(2, 1).get_edges(),
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
        self.assertAlmostEqual(epc.value.n, epc_expected, delta=3 * epc.value.std_dev)

    def test_three_qubit_nonlocal_noise(self):
        """Test three-qubit mirror RB on a nonlocal noise model"""
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

        transpiler_options = {
            "basis_gates": basis_gates,
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
        self.assertAlmostEqual(epc.value.n, epc_expected, delta=3 * epc.value.std_dev)

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
