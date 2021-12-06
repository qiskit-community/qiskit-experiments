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

"""
A Tester for the RB experiment
"""
from test.base import QiskitExperimentsTestCase
import itertools as it
import ddt
from qiskit import QuantumCircuit
import qiskit.quantum_info as qi
from qiskit.providers.aer import AerSimulator
from qiskit_experiments.framework import BatchExperiment, ParallelExperiment
from qiskit_experiments.library import StateTomography, ProcessTomography
from qiskit_experiments.library.tomography import StateTomographyAnalysis, ProcessTomographyAnalysis


# TODO: tests for CVXPY fitters
FITTERS = [None, "linear_inversion", "scipy_linear_lstsq", "scipy_gaussian_lstsq"]


def filter_results(analysis_results, name):
    """Filter list of analysis results by result name"""
    for result in analysis_results:
        if result.name == name:
            return result
    return None


@ddt.ddt
class TestStateTomography(QiskitExperimentsTestCase):
    """Test StateTomography"""

    @ddt.data(*list(it.product([1, 2], FITTERS)))
    @ddt.unpack
    def test_full_qst(self, num_qubits, fitter):
        """Test 1-qubit QST experiment"""
        backend = AerSimulator(seed_simulator=9000)
        seed = 1234
        f_threshold = 0.95
        target = qi.random_statevector(2 ** num_qubits, seed=seed)
        qstexp = StateTomography(target)
        if fitter:
            qstexp.analysis.set_options(fitter=fitter)
        expdata = qstexp.run(backend)
        results = expdata.analysis_results()

        # Check state is density matrix
        state = filter_results(results, "state").value
        self.assertTrue(
            isinstance(state, qi.DensityMatrix), msg="fitted state is not density matrix"
        )

        # Check fit state fidelity
        fid = filter_results(results, "state_fidelity").value
        self.assertGreater(fid, f_threshold, msg="fit fidelity is low")

        # Manually check fidelity
        target_fid = qi.state_fidelity(state, target, validate=False)
        self.assertAlmostEqual(fid, target_fid, places=6, msg="result fidelity is incorrect")

    def test_qst_teleport(self):
        """Test subset state tomography generation"""
        # NOTE: This test breaks transpiler. I think it is a bug with
        # conditionals in Terra.

        # Teleport qubit 0 -> 2
        backend = AerSimulator(seed_simulator=9000)
        exp = StateTomography(teleport_circuit(), measurement_qubits=[2])
        expdata = exp.run(backend)
        results = expdata.analysis_results()

        # Check result
        f_threshold = 0.95

        # Check state is density matrix
        state = filter_results(results, "state").value
        self.assertTrue(
            isinstance(state, qi.DensityMatrix), msg="fitted state is not a density matrix"
        )

        # Manually check fidelity
        fid = qi.state_fidelity(state, qi.Statevector([1, 0]), validate=False)
        self.assertGreater(fid, f_threshold, msg="fitted state fidelity is low")

    @ddt.data(
        [0],
        [1],
        [2],
        [0, 1],
        [1, 0],
        [0, 2],
        [2, 0],
        [1, 2],
        [2, 1],
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    )
    def test_exp_circuits_measurement_qubits(self, meas_qubits):
        """Test subset state tomography generation"""
        # Subsystem unitaries
        seed = 1111
        nq = 3
        ops = [qi.random_unitary(2, seed=seed + i) for i in range(nq)]

        # Preparation circuit
        circ = QuantumCircuit(nq)
        for i, op in enumerate(ops):
            circ.append(op, [i])

        num_meas = len(meas_qubits)
        exp = StateTomography(circ, measurement_qubits=meas_qubits)
        tomo_circuits = exp.circuits()

        # Check correct number of circuits are generated
        self.assertEqual(len(tomo_circuits), 3 ** num_meas)

        # Check circuit metadata is correct
        for circ in tomo_circuits:
            meta = circ.metadata
            clbits = meta.get("clbits")
            self.assertEqual(clbits, list(range(num_meas)), msg="metadata clbits is incorrect")

        # Check experiment target metadata is correct
        exp_meta = exp._metadata()
        target_state = exp_meta.get("target")

        target_circ = QuantumCircuit(num_meas)
        for i, qubit in enumerate(meas_qubits):
            target_circ.append(ops[qubit], [i])
        fid = qi.state_fidelity(target_state, qi.Statevector(target_circ))
        self.assertGreater(fid, 0.99, msg="target_state is incorrect")

    @ddt.data([0], [1], [2], [0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1])
    def test_full_exp_measurement_qubits(self, meas_qubits):
        """Test subset state tomography generation"""
        # Subsystem unitaries
        seed = 1111
        nq = 3
        ops = [qi.random_unitary(2, seed=seed + i) for i in range(nq)]

        # Target state
        target_circ = QuantumCircuit(len(meas_qubits))
        for i, qubit in enumerate(meas_qubits):
            target_circ.append(ops[qubit], [i])
        target = qi.Statevector(target_circ)

        # Preparation circuit
        circ = QuantumCircuit(nq)
        for i, op in enumerate(ops):
            circ.append(op, [i])

        # Run
        backend = AerSimulator(seed_simulator=9000)
        exp = StateTomography(circ, measurement_qubits=meas_qubits)
        expdata = exp.run(backend)
        results = expdata.analysis_results()

        # Check result
        f_threshold = 0.95

        # Check state is density matrix
        state = filter_results(results, "state").value
        self.assertTrue(
            isinstance(state, qi.DensityMatrix), msg="fitted state is not density matrix"
        )

        # Check fit state fidelity
        fid = filter_results(results, "state_fidelity").value
        self.assertGreater(fid, f_threshold, msg="fit fidelity is low")

        # Manually check fidelity
        target_fid = qi.state_fidelity(state, target, validate=False)
        self.assertAlmostEqual(fid, target_fid, places=6, msg="result fidelity is incorrect")

    def test_batch_exp(self):
        """Test batch state tomography experiment with measurement_qubits kwarg"""
        # Subsystem unitaries
        seed = 1111
        nq = 3
        ops = [qi.random_unitary(2, seed=seed + i) for i in range(nq)]

        # Preparation circuit
        circuit = QuantumCircuit(nq)
        for i, op in enumerate(ops):
            circuit.append(op, [i])

        # Component experiments
        exps = []
        targets = []
        for i in range(nq):
            targets.append(qi.Statevector(ops[i].to_instruction()))
            exps.append(StateTomography(circuit, measurement_qubits=[i]))

        # Run batch experiments
        backend = AerSimulator(seed_simulator=9000)
        batch_exp = BatchExperiment(exps)
        batch_data = batch_exp.run(backend).block_for_results()

        # Check target fidelity of component experiments
        f_threshold = 0.95
        for i in range(batch_exp.num_experiments):
            results = batch_data.child_data(i).analysis_results()

            # Check state is density matrix
            state = filter_results(results, "state").value
            self.assertTrue(
                isinstance(state, qi.DensityMatrix), msg="fitted state is not density matrix"
            )

            # Check fit state fidelity
            fid = filter_results(results, "state_fidelity").value
            self.assertGreater(fid, f_threshold, msg="fit fidelity is low")

            # Manually check fidelity
            target_fid = qi.state_fidelity(state, targets[i], validate=False)
            self.assertAlmostEqual(fid, target_fid, places=6, msg="result fidelity is incorrect")

    def test_parallel_exp(self):
        """Test parallel state tomography experiment"""
        # Subsystem unitaries
        seed = 1221
        nq = 4
        ops = [qi.random_unitary(2, seed=seed + i) for i in range(nq)]

        # Component experiments
        exps = []
        targets = []
        for i in range(nq):
            exps.append(StateTomography(ops[i], qubits=[i]))
            targets.append(qi.Statevector(ops[i].to_instruction()))

        # Run batch experiments
        backend = AerSimulator(seed_simulator=9000)
        par_exp = ParallelExperiment(exps)
        par_data = par_exp.run(backend).block_for_results()

        # Check target fidelity of component experiments
        f_threshold = 0.95
        for i in range(par_exp.num_experiments):
            results = par_data.child_data(i).analysis_results()

            # Check state is density matrix
            state = filter_results(results, "state").value
            self.assertTrue(
                isinstance(state, qi.DensityMatrix), msg="fitted state is not density matrix"
            )

            # Check fit state fidelity
            fid = filter_results(results, "state_fidelity").value
            self.assertGreater(fid, f_threshold, msg="fit fidelity is low")

            # Manually check fidelity
            target_fid = qi.state_fidelity(state, targets[i], validate=False)
            self.assertAlmostEqual(fid, target_fid, places=6, msg="result fidelity is incorrect")

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = StateTomography(QuantumCircuit(3), measurement_qubits=[0, 2], qubits=[5, 7, 1])
        loaded_exp = StateTomography.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.experiments_equiv(exp, loaded_exp))

    def test_analysis_config(self):
        """ "Test converting analysis to and from config works"""
        analysis = StateTomographyAnalysis()
        loaded = StateTomographyAnalysis.from_config(analysis.config())
        self.assertNotEqual(analysis, loaded)
        self.assertEqual(analysis.config(), loaded.config())


@ddt.ddt
class TestProcessTomography(QiskitExperimentsTestCase):
    """Test QuantumProcessTomography"""

    @ddt.data(*list(it.product([1, 2], FITTERS)))
    @ddt.unpack
    def test_full_qpt(self, num_qubits, fitter):
        """Test QPT experiment"""
        backend = AerSimulator(seed_simulator=9000)
        seed = 1234
        f_threshold = 0.94
        target = qi.random_unitary(2 ** num_qubits, seed=seed)
        qstexp = ProcessTomography(target)
        if fitter:
            qstexp.analysis.set_options(fitter=fitter)
        expdata = qstexp.run(backend)
        results = expdata.analysis_results()

        # Check state is density matrix
        state = filter_results(results, "state").value
        self.assertTrue(isinstance(state, qi.Choi), msg="fitted state is not a Choi matrix")

        # Check fit state fidelity
        fid = filter_results(results, "process_fidelity").value
        self.assertGreater(fid, f_threshold, msg="fit fidelity is low")
        # Manually check fidelity
        target_fid = qi.process_fidelity(state, target, require_tp=False, require_cp=False)
        self.assertAlmostEqual(fid, target_fid, places=6, msg="result fidelity is incorrect")

    @ddt.data([0], [1], [2], [0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1])
    def test_exp_measurement_preparation_qubits(self, qubits):
        """Test subset measurement process tomography generation"""
        # Subsystem unitaries
        seed = 1111
        nq = 3
        ops = [qi.random_unitary(2, seed=seed + i) for i in range(nq)]

        # Preparation circuit
        circ = QuantumCircuit(nq)
        for i, op in enumerate(ops):
            circ.append(op, [i])

        num_meas = len(qubits)
        exp = ProcessTomography(circ, measurement_qubits=qubits, preparation_qubits=qubits)
        tomo_circuits = exp.circuits()

        # Check correct number of circuits are generated
        size = 3 ** num_meas * 4 ** num_meas
        self.assertEqual(len(tomo_circuits), size)

        # Check circuit metadata is correct
        for circ in tomo_circuits:
            meta = circ.metadata
            clbits = meta.get("clbits")
            self.assertEqual(clbits, list(range(num_meas)), msg="metadata clbits is incorrect")

        # Check experiment target metadata is correct
        exp_meta = exp._metadata()
        target_state = exp_meta.get("target")

        target_circ = QuantumCircuit(num_meas)
        for i, qubit in enumerate(qubits):
            target_circ.append(ops[qubit], [i])
        fid = qi.process_fidelity(target_state, qi.Operator(target_circ))
        self.assertGreater(fid, 0.99, msg="target_state is incorrect")

    @ddt.data([0], [1], [1, 0])
    def test_full_exp_meas_prep_qubits(self, qubits):
        """Test subset state tomography generation"""
        # Subsystem unitaries
        seed = 1111
        nq = 3
        ops = [qi.random_unitary(2, seed=seed + i) for i in range(nq)]

        # Target state
        target_circ = QuantumCircuit(len(qubits))
        for i, qubit in enumerate(qubits):
            target_circ.append(ops[qubit], [i])
        target = qi.Operator(target_circ)

        # Preparation circuit
        circ = QuantumCircuit(nq)
        for i, op in enumerate(ops):
            circ.append(op, [i])

        # Run
        backend = AerSimulator(seed_simulator=9000)
        exp = ProcessTomography(circ, measurement_qubits=qubits, preparation_qubits=qubits)
        expdata = exp.run(backend)
        results = expdata.analysis_results()

        # Check result
        f_threshold = 0.95

        # Check state is density matrix
        state = filter_results(results, "state").value
        self.assertTrue(isinstance(state, qi.Choi), msg="fitted state is not a Choi matrix")

        # Check fit state fidelity
        fid = filter_results(results, "process_fidelity").value
        self.assertGreater(fid, f_threshold, msg="fit fidelity is low")

        # Manually check fidelity
        target_fid = qi.process_fidelity(state, target, require_tp=False, require_cp=False)
        self.assertAlmostEqual(fid, target_fid, places=6, msg="result fidelity is incorrect")

    def test_qpt_teleport(self):
        """Test subset state tomography generation"""
        # NOTE: This test breaks transpiler. I think it is a bug with
        # conditionals in Terra.

        # Teleport qubit 0 -> 2
        backend = AerSimulator(seed_simulator=9000)
        exp = ProcessTomography(teleport_circuit(), measurement_qubits=[2], preparation_qubits=[0])
        expdata = exp.run(backend, shots=10000)
        results = expdata.analysis_results()

        # Check result
        f_threshold = 0.95

        # Check state is density matrix
        state = filter_results(results, "state").value
        self.assertTrue(isinstance(state, qi.Choi), msg="fitted state is not a Choi matrix")

        # Manually check fidelity
        fid = qi.process_fidelity(state, require_tp=False, require_cp=False)
        self.assertGreater(fid, f_threshold, msg="fitted state fidelity is low")

    def test_batch_exp_with_measurement_qubits(self):
        """Test batch process tomography experiment with kwargs"""
        seed = 1111
        nq = 3
        ops = [qi.random_unitary(2, seed=seed + i) for i in range(nq)]

        # Preparation circuit
        circuit = QuantumCircuit(nq)
        for i, op in enumerate(ops):
            circuit.append(op, [i])

        # Component experiments
        exps = []
        targets = []
        for i in range(nq):
            targets.append(ops[i])
            exps.append(ProcessTomography(circuit, measurement_qubits=[i], preparation_qubits=[i]))

        # Run batch experiments
        backend = AerSimulator(seed_simulator=9000)
        batch_exp = BatchExperiment(exps)
        batch_data = batch_exp.run(backend).block_for_results()

        # Check target fidelity of component experiments
        f_threshold = 0.95
        for i in range(batch_exp.num_experiments):
            results = batch_data.child_data(i).analysis_results()

            # Check state is density matrix
            state = filter_results(results, "state").value
            self.assertTrue(isinstance(state, qi.Choi), msg="fitted state is not a Choi matrix")

            # Check fit state fidelity
            fid = filter_results(results, "process_fidelity").value
            self.assertGreater(fid, f_threshold, msg="fit fidelity is low")

            # Manually check fidelity
            target_fid = qi.process_fidelity(state, targets[i], require_tp=False, require_cp=False)
            self.assertAlmostEqual(fid, target_fid, places=6, msg="result fidelity is incorrect")

    def test_parallel_exp(self):
        """Test parallel process tomography experiment"""
        # Subsystem unitaries
        seed = 1221
        nq = 4
        ops = [qi.random_unitary(2, seed=seed + i) for i in range(nq)]

        # Component experiments
        exps = []
        targets = []
        for i in range(nq):
            exps.append(ProcessTomography(ops[i], qubits=[i]))
            targets.append(ops[i])

        # Run batch experiments
        backend = AerSimulator(seed_simulator=9000)
        par_exp = ParallelExperiment(exps)
        par_data = par_exp.run(backend).block_for_results()

        # Check target fidelity of component experiments
        f_threshold = 0.95
        for i in range(par_exp.num_experiments):
            results = par_data.child_data(i).analysis_results()

            # Check state is density matrix
            state = filter_results(results, "state").value
            self.assertTrue(isinstance(state, qi.Choi), msg="fitted state is not a Choi matrix")

            # Check fit state fidelity
            fid = filter_results(results, "process_fidelity").value
            self.assertGreater(fid, f_threshold, msg="fit fidelity is low")

            # Manually check fidelity
            target_fid = qi.process_fidelity(state, targets[i], require_tp=False, require_cp=False)
            self.assertAlmostEqual(fid, target_fid, places=6, msg="result fidelity is incorrect")

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = ProcessTomography(teleport_circuit(), measurement_qubits=[2], preparation_qubits=[0])
        loaded_exp = ProcessTomography.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.experiments_equiv(exp, loaded_exp))

    def test_analysis_config(self):
        """ "Test converting analysis to and from config works"""
        analysis = ProcessTomographyAnalysis()
        loaded = ProcessTomographyAnalysis.from_config(analysis.config())
        self.assertNotEqual(analysis, loaded)
        self.assertEqual(analysis.config(), loaded.config())


def teleport_circuit():
    """Teleport qubit 0 to qubit 2"""
    teleport = QuantumCircuit(3, 2)
    teleport.h(1)
    teleport.cx(1, 2)
    teleport.cx(0, 1)
    teleport.h(0)
    teleport.measure(0, 0)
    teleport.measure(1, 1)
    # Conditionals
    creg = teleport.cregs[0]
    teleport.z(2).c_if(creg[0], 1)
    teleport.x(2).c_if(creg[1], 1)
    return teleport
