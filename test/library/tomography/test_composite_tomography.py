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
Composite StateTomography and ProcessTomography experiment tests
"""
from test.base import QiskitExperimentsTestCase
from qiskit import QuantumCircuit
import qiskit.quantum_info as qi
from qiskit_aer import AerSimulator
from qiskit_experiments.framework import BatchExperiment, ParallelExperiment
from qiskit_experiments.library import StateTomography, ProcessTomography


class TestCompositeTomography(QiskitExperimentsTestCase):
    """Test composite tomography experiments"""

    def test_batch_qst_exp(self):
        """Test batch state tomography experiment with measurement_indices kwarg"""
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
            exps.append(StateTomography(circuit, measurement_indices=[i]))

        # Run batch experiments
        backend = AerSimulator(seed_simulator=9000)
        batch_exp = BatchExperiment(exps, flatten_results=False)
        batch_data = batch_exp.run(backend)
        self.assertExperimentDone(batch_data)

        # Check target fidelity of component experiments
        f_threshold = 0.95
        for i in range(batch_exp.num_experiments):
            results = batch_data.child_data(i).analysis_results(dataframe=True)

            # Check state is density matrix
            state = results[results.name == "state"].iloc[0].value
            self.assertTrue(
                isinstance(state, qi.DensityMatrix), msg="fitted state is not density matrix"
            )

            # Check fit state fidelity
            fid = results[results.name == "state_fidelity"].iloc[0].value
            self.assertGreater(fid, f_threshold, msg="fit fidelity is low")

            # Manually check fidelity
            target_fid = qi.state_fidelity(state, targets[i], validate=False)
            self.assertAlmostEqual(fid, target_fid, places=6, msg="result fidelity is incorrect")

    def test_parallel_qst_exp(self):
        """Test parallel state tomography experiment"""
        # Subsystem unitaries
        seed = 1221
        nq = 4
        ops = [qi.random_unitary(2, seed=seed + i) for i in range(nq)]

        # Component experiments
        exps = []
        targets = []
        for i in range(nq):
            exps.append(StateTomography(ops[i], physical_qubits=[i]))
            targets.append(qi.Statevector(ops[i].to_instruction()))

        # Run batch experiments
        backend = AerSimulator(seed_simulator=9000)
        par_exp = ParallelExperiment(exps, flatten_results=False)
        par_data = par_exp.run(backend)
        self.assertExperimentDone(par_data)

        # Check target fidelity of component experiments
        f_threshold = 0.95
        for i in range(par_exp.num_experiments):
            results = par_data.child_data(i).analysis_results(dataframe=True)

            # Check state is density matrix
            state = results[results.name == "state"].iloc[0].value
            self.assertTrue(
                isinstance(state, qi.DensityMatrix), msg="fitted state is not density matrix"
            )

            # Check fit state fidelity
            fid = results[results.name == "state_fidelity"].iloc[0].value
            self.assertGreater(fid, f_threshold, msg="fit fidelity is low")

            # Manually check fidelity
            target_fid = qi.state_fidelity(state, targets[i], validate=False)
            self.assertAlmostEqual(fid, target_fid, places=6, msg="result fidelity is incorrect")

    def test_batch_qpt_exp_with_measurement_indices(self):
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
            exps.append(
                ProcessTomography(circuit, measurement_indices=[i], preparation_indices=[i])
            )

        # Run batch experiments
        backend = AerSimulator(seed_simulator=9000)
        batch_exp = BatchExperiment(exps, flatten_results=False)
        batch_data = batch_exp.run(backend)
        self.assertExperimentDone(batch_data)

        # Check target fidelity of component experiments
        f_threshold = 0.95
        for i in range(batch_exp.num_experiments):
            results = batch_data.child_data(i).analysis_results(dataframe=True)

            # Check state is density matrix
            state = results[results.name == "state"].iloc[0].value
            self.assertTrue(isinstance(state, qi.Choi), msg="fitted state is not a Choi matrix")

            # Check fit state fidelity
            fid = results[results.name == "process_fidelity"].iloc[0].value
            self.assertGreater(fid, f_threshold, msg="fit fidelity is low")

            # Manually check fidelity
            target_fid = qi.process_fidelity(state, targets[i], require_tp=False, require_cp=False)
            self.assertAlmostEqual(fid, target_fid, places=6, msg="result fidelity is incorrect")

    def test_parallel_qpt_exp(self):
        """Test parallel process tomography experiment"""
        # Subsystem unitaries
        seed = 1221
        nq = 4
        ops = [qi.random_unitary(2, seed=seed + i) for i in range(nq)]

        # Component experiments
        exps = []
        targets = []
        for i in range(nq):
            exps.append(ProcessTomography(ops[i], physical_qubits=[i]))
            targets.append(ops[i])

        # Run batch experiments
        backend = AerSimulator(seed_simulator=9000)
        par_exp = ParallelExperiment(exps, flatten_results=False)
        par_data = par_exp.run(backend)
        self.assertExperimentDone(par_data)

        # Check target fidelity of component experiments
        f_threshold = 0.95
        for i in range(par_exp.num_experiments):
            results = par_data.child_data(i).analysis_results(dataframe=True)

            # Check state is density matrix
            state = results[results.name == "state"].iloc[0].value
            self.assertTrue(isinstance(state, qi.Choi), msg="fitted state is not a Choi matrix")

            # Check fit state fidelity
            fid = results[results.name == "process_fidelity"].iloc[0].value
            self.assertGreater(fid, f_threshold, msg="fit fidelity is low")

            # Manually check fidelity
            target_fid = qi.process_fidelity(state, targets[i], require_tp=False, require_cp=False)
            self.assertAlmostEqual(fid, target_fid, places=6, msg="result fidelity is incorrect")

    def test_mixed_batch_exp(self):
        """Test batch state and process tomography experiment"""
        # Subsystem unitaries
        state_op = qi.random_unitary(2, seed=321)
        chan_op = qi.random_unitary(2, seed=123)

        state_target = qi.Statevector(state_op.to_instruction())
        chan_target = qi.Choi(chan_op.to_instruction())

        state_exp = StateTomography(state_op)
        chan_exp = ProcessTomography(chan_op)
        batch_exp = BatchExperiment([state_exp, chan_exp], flatten_results=False)

        # Run batch experiments
        backend = AerSimulator(seed_simulator=9000)
        par_data = batch_exp.run(backend)
        self.assertExperimentDone(par_data)

        f_threshold = 0.95

        # Check state tomo results
        state_results = par_data.child_data(0).analysis_results(dataframe=True)
        state = state_results[state_results.name == "state"].iloc[0].value

        # Check fit state fidelity
        state_fid = state_results[state_results.name == "state_fidelity"].iloc[0].value
        self.assertGreater(state_fid, f_threshold, msg="fit fidelity is low")

        # Manually check fidelity
        target_fid = qi.state_fidelity(state, state_target, validate=False)
        self.assertAlmostEqual(state_fid, target_fid, places=6, msg="result fidelity is incorrect")

        # Check process tomo results
        chan_results = par_data.child_data(1).analysis_results(dataframe=True)
        chan = chan_results[chan_results.name == "state"].iloc[0].value

        # Check fit process fidelity
        chan_fid = chan_results[chan_results.name == "process_fidelity"].iloc[0].value
        self.assertGreater(chan_fid, f_threshold, msg="fit fidelity is low")

        # Manually check fidelity
        target_fid = qi.process_fidelity(chan, chan_target, require_cp=False, require_tp=False)
        self.assertAlmostEqual(chan_fid, target_fid, places=6, msg="result fidelity is incorrect")
