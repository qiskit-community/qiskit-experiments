# -*- coding: utf-8 -*-

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

import itertools as it
import ddt

from qiskit.test import QiskitTestCase
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
import qiskit.quantum_info as qi
from qiskit.providers.aer import AerSimulator
from qiskit_experiments.composite import BatchExperiment, ParallelExperiment
import qiskit_experiments.tomography as tomo

# TODO: tests for CVXPY fitters
FITTERS = [None, "linear_inversion", "scipy_linear_lstsq", "scipy_gaussian_lstsq"]


@ddt.ddt
class TestStateTomographyExperiment(QiskitTestCase):
    """Test StateTomographyExperiment"""

    @ddt.data(*list(it.product([1, 2], FITTERS)))
    @ddt.unpack
    def test_full_qst(self, num_qubits, fitter):
        """Test 1-qubit QST experiment"""
        backend = AerSimulator()
        seed = 1234
        f_threshold = 0.97
        target = qi.random_statevector(2 ** num_qubits, seed=seed)
        qstexp = tomo.StateTomographyExperiment(target)
        if fitter:
            qstexp.set_analysis_options(fitter=fitter)
        expdata = qstexp.run(backend)
        result = expdata.analysis_result(-1)

        self.assertTrue(result.get("success", False), msg="analysis failed")

        # Check state is density matrix
        state = result.get("state")
        self.assertTrue(
            isinstance(state, qi.DensityMatrix), msg="fitted state is not density matrix"
        )

        # Check fit state fidelity
        self.assertGreater(result.get("value", 0), f_threshold, msg="fit fidelity is low")

        # Manually check fidelity
        fid = qi.state_fidelity(state, target, validate=False)
        self.assertGreater(fid, f_threshold, msg="fitted state fidelity is low")

    def skip_test_qst_teleport(self):
        """Test subset state tomography generation"""
        # NOTE: This test breaks transpiler. I think it is a bug with
        # conditionals in Terra.

        # Teleport qubit 1 -> 2
        backend = AerSimulator()
        exp = tomo.StateTomographyExperiment(teleport_circuit(1, 2), measurement_qubits=[2])
        expdata = exp.run(backend)
        result = expdata.analysis_result(-1)

        # Check result
        f_threshold = 0.98
        self.assertTrue(result.get("success", False), msg="analysis failed")

        # Check state is density matrix
        state = result.get("state")
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
        exp = tomo.StateTomographyExperiment(circ, measurement_qubits=meas_qubits)
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
        target_state = exp_meta.get("target_state")

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
        backend = AerSimulator()
        exp = tomo.StateTomographyExperiment(circ, measurement_qubits=meas_qubits)
        expdata = exp.run(backend)
        result = expdata.analysis_result(-1)

        # Check result
        f_threshold = 0.97
        self.assertTrue(result.get("success", False), msg="analysis failed")

        # Check state is density matrix
        state = result.get("state")
        self.assertTrue(
            isinstance(state, qi.DensityMatrix), msg="fitted state is not density matrix"
        )

        # Check fit state fidelity
        self.assertGreater(result.get("value", 0), f_threshold, msg="fit fidelity is low")

        # Manually check fidelity
        fid = qi.state_fidelity(state, target, validate=False)
        self.assertGreater(fid, f_threshold, msg="fitted state fidelity is low")

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
            exps.append(tomo.StateTomographyExperiment(circuit, measurement_qubits=[i]))

        # Run batch experiments
        backend = AerSimulator()
        batch_exp = BatchExperiment(exps)
        batch_data = batch_exp.run(backend)
        batch_result = batch_data.analysis_result(-1)
        self.assertTrue(batch_result.get("success"), msg="BatchExperiment failed")

        # Check target fidelity of component experiments
        f_threshold = 0.97
        for i in range(batch_exp.num_experiments):
            result = batch_data.component_experiment_data(i).analysis_result(-1)
            self.assertTrue(result.get("success", False), msg="component analysis failed")

            # Check state is density matrix
            state = result.get("state")
            self.assertTrue(
                isinstance(state, qi.DensityMatrix), msg="fitted state is not density matrix"
            )

            # Check fit state fidelity
            self.assertGreater(result.get("value", 0), f_threshold, msg="fit fidelity is low")

            # Manually check fidelity
            fid = qi.state_fidelity(state, targets[i], validate=False)
            self.assertGreater(fid, f_threshold, msg="fitted state fidelity is low")

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
            exps.append(tomo.StateTomographyExperiment(ops[i], qubits=[i]))
            targets.append(qi.Statevector(ops[i].to_instruction()))

        # Run batch experiments
        backend = AerSimulator()
        par_exp = ParallelExperiment(exps)
        par_data = par_exp.run(backend)
        par_result = par_data.analysis_result(-1)
        self.assertTrue(par_result.get("success"), msg="ParallelExperiment failed")

        # Check target fidelity of component experiments
        f_threshold = 0.97
        for i in range(par_exp.num_experiments):
            result = par_data.component_experiment_data(i).analysis_result(-1)
            self.assertTrue(result.get("success", False), msg="component analysis failed")

            # Check state is density matrix
            state = result.get("state")
            self.assertTrue(
                isinstance(state, qi.DensityMatrix), msg="fitted state is not density matrix"
            )

            # Check fit state fidelity
            self.assertGreater(result.get("value", 0), f_threshold, msg="fit fidelity is low")

            # Manually check fidelity
            fid = qi.state_fidelity(state, targets[i], validate=False)
            self.assertGreater(fid, f_threshold, msg="fitted state fidelity is low")


@ddt.ddt
class TestProcessTomographyExperiment(QiskitTestCase):
    """Test QuantumProcessTomography"""

    @ddt.data(*list(it.product([1, 2], FITTERS)))
    @ddt.unpack
    def test_full_qpt(self, num_qubits, fitter):
        """Test QPT experiment"""
        backend = AerSimulator()
        seed = 1234
        f_threshold = 0.95
        target = qi.random_unitary(2 ** num_qubits, seed=seed)
        qstexp = tomo.ProcessTomographyExperiment(target)
        if fitter:
            qstexp.set_analysis_options(fitter=fitter)
        expdata = qstexp.run(backend)
        result = expdata.analysis_result(-1)

        self.assertTrue(result.get("success", False), msg="analysis failed")

        # Check state is density matrix
        state = result.get("state")
        self.assertTrue(isinstance(state, qi.Choi), msg="fitted state is not a Choi matrix")

        # Check fit state fidelity
        self.assertGreater(result.get("value", 0), f_threshold, msg="fit fidelity is low")

        # Manually check fidelity
        fid = qi.process_fidelity(state, target, require_tp=False, require_cp=False)
        self.assertGreater(fid, f_threshold, msg="fitted process fidelity is low")

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
        exp = tomo.ProcessTomographyExperiment(
            circ, measurement_qubits=qubits, preparation_qubits=qubits
        )
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
        target_state = exp_meta.get("target_state")

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
        backend = AerSimulator()
        exp = tomo.ProcessTomographyExperiment(
            circ, measurement_qubits=qubits, preparation_qubits=qubits
        )
        expdata = exp.run(backend)
        result = expdata.analysis_result(-1)

        # Check result
        f_threshold = 0.97
        self.assertTrue(result.get("success", False), msg="analysis failed")

        # Check state is density matrix
        state = result.get("state")
        self.assertTrue(isinstance(state, qi.Choi), msg="fitted state is not a Choi matrix")

        # Check fit state fidelity
        self.assertGreater(result.get("value", 0), f_threshold, msg="fit fidelity is low")

        # Manually check fidelity
        fid = qi.process_fidelity(state, target, require_tp=False, require_cp=False)
        self.assertGreater(fid, f_threshold, msg="fitted state fidelity is low")

    def skip_test_qpt_teleport(self):
        """Test subset state tomography generation"""
        # NOTE: This test breaks transpiler. I think it is a bug with
        # conditionals in Terra.

        # Teleport qubit 1 -> 2
        backend = AerSimulator()
        exp = tomo.ProcessTomographyExperiment(
            teleport_circuit(1, 2), measurement_qubits=[2], preparation_qubits=[1]
        )
        expdata = exp.run(backend)
        result = expdata.analysis_result(-1)

        # Check result
        f_threshold = 0.98
        self.assertTrue(result.get("success", False), msg="analysis failed")

        # Check state is density matrix
        state = result.get("state")
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
            exps.append(
                tomo.ProcessTomographyExperiment(
                    circuit, measurement_qubits=[i], preparation_qubits=[i]
                )
            )

        # Run batch experiments
        backend = AerSimulator()
        batch_exp = BatchExperiment(exps)
        batch_data = batch_exp.run(backend)
        batch_result = batch_data.analysis_result(-1)
        self.assertTrue(batch_result.get("success"), msg="BatchExperiment failed")

        # Check target fidelity of component experiments
        f_threshold = 0.95
        for i in range(batch_exp.num_experiments):
            result = batch_data.component_experiment_data(i).analysis_result(-1)
            self.assertTrue(result.get("success", False), msg="component analysis failed")

            # Check state is density matrix
            state = result.get("state")
            self.assertTrue(isinstance(state, qi.Choi), msg="fitted state is not a Choi matrix")

            # Check fit state fidelity
            self.assertGreater(result.get("value", 0), f_threshold, msg="fit fidelity is low")

            # Manually check fidelity
            fid = qi.process_fidelity(state, targets[i], require_tp=False, require_cp=False)
            self.assertGreater(fid, f_threshold, msg="fitted state fidelity is low")

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
            exps.append(tomo.ProcessTomographyExperiment(ops[i], qubits=[i]))
            targets.append(ops[i])

        # Run batch experiments
        backend = AerSimulator()
        par_exp = ParallelExperiment(exps)
        par_data = par_exp.run(backend)
        par_result = par_data.analysis_result(-1)
        self.assertTrue(par_result.get("success"), msg="ParallelExperiment failed")

        # Check target fidelity of component experiments
        f_threshold = 0.95
        for i in range(par_exp.num_experiments):
            result = par_data.component_experiment_data(i).analysis_result(-1)
            self.assertTrue(result.get("success", False), msg="component analysis failed")

            # Check state is density matrix
            state = result.get("state")
            self.assertTrue(isinstance(state, qi.Choi), msg="fitted state is not a Choi matrix")

            # Check fit state fidelity
            self.assertGreater(result.get("value", 0), f_threshold, msg="fit fidelity is low")

            # Manually check fidelity
            fid = qi.process_fidelity(state, targets[i], require_tp=False, require_cp=False)
            self.assertGreater(fid, f_threshold, msg="fitted state fidelity is low")


def teleport_circuit(qubit_in, qubit_out):
    """Teleport qubit_in to qubit_out"""
    # It would be nice to be able to do this without using
    # register objects

    # Teleport qubit 0 -> 2
    qr = QuantumRegister(3)
    c0 = ClassicalRegister(1)
    c1 = ClassicalRegister(1)
    teleport = QuantumCircuit(qr, c0, c1)
    teleport.h(qr[1])
    teleport.cx(qr[1], qr[2])
    teleport.cx(qr[0], qr[1])
    teleport.h(qr[0])
    teleport.measure(qr[0], c0[0])
    teleport.measure(qr[1], c1[0])
    teleport.z(qr[2]).c_if(c0, 1)
    teleport.x(qr[2]).c_if(c1, 1)

    # Return mapped circuit
    num_qubits = max(3, qubit_in + 1, qubit_out + 1)
    qubit_anc = [i for i in range(3) if i not in [qubit_in, qubit_out]][0]
    circ = QuantumCircuit(num_qubits, 2)
    circ = circ.compose(teleport, [qubit_in, qubit_anc, qubit_out], [0, 1])
    return circ
