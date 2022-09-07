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
StateTomography experiment tests
"""
from test.base import QiskitExperimentsTestCase
import ddt
from qiskit import QuantumCircuit
from qiskit.circuit.library import XGate
import qiskit.quantum_info as qi
from qiskit.providers.aer import AerSimulator
from qiskit_experiments.library import StateTomography
from qiskit_experiments.library.tomography import StateTomographyAnalysis
from .tomo_utils import FITTERS, filter_results, teleport_circuit


@ddt.ddt
class TestStateTomography(QiskitExperimentsTestCase):
    """Test StateTomography experiments"""

    @ddt.data(1, 2)
    def test_full_qst(self, num_qubits):
        """Test QST experiment"""
        seed = 1234
        shots = 5000
        f_threshold = 0.99

        # Generate tomography data without analysis
        backend = AerSimulator(seed_simulator=seed, shots=shots)
        target = qi.random_statevector(2**num_qubits, seed=seed)
        exp = StateTomography(target)
        expdata = exp.run(backend, analysis=None)
        self.assertExperimentDone(expdata)

        # Run each tomography fitter analysis as a subtest so
        # we don't have to re-run simulation data for each fitter
        for fitter in FITTERS:
            with self.subTest(fitter=fitter):
                if fitter:
                    exp.analysis.set_options(fitter=fitter)
                fitdata = exp.analysis.run(expdata)
                self.assertExperimentDone(fitdata)
                results = expdata.analysis_results()

                # Check state is density matrix
                state = filter_results(results, "state").value
                self.assertTrue(
                    isinstance(state, qi.DensityMatrix),
                    msg=f"{fitter} fitted state is not density matrix",
                )

                # Check fit state fidelity
                fid = filter_results(results, "state_fidelity").value
                self.assertGreater(fid, f_threshold, msg=f"{fitter} fit fidelity is low")

                # Manually check fidelity
                target_fid = qi.state_fidelity(state, target, validate=False)
                self.assertAlmostEqual(
                    fid, target_fid, places=6, msg=f"{fitter} result fidelity is incorrect"
                )

    def test_qst_teleport(self):
        """Test subset state tomography generation"""
        # NOTE: This test breaks transpiler. I think it is a bug with
        # conditionals in Terra.

        # Teleport qubit 0 -> 2
        backend = AerSimulator(seed_simulator=9000)
        exp = StateTomography(teleport_circuit(), measurement_qubits=[2])
        expdata = exp.run(backend)
        self.assertExperimentDone(expdata)
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
        self.assertEqual(len(tomo_circuits), 3**num_meas)

        # Check circuit metadata is correct
        for circ in tomo_circuits:
            meta = circ.metadata
            clbits = meta.get("clbits")
            self.assertEqual(clbits, list(range(num_meas)), msg="metadata clbits is incorrect")

        # Check analysis target is correct
        target_state = exp.analysis.options.target

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
        self.assertExperimentDone(expdata)
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

    def test_expdata_serialization(self):
        """Test serializing experiment data works."""
        backend = AerSimulator(seed_simulator=9000)
        exp = StateTomography(XGate())
        expdata = exp.run(backend)
        self.assertExperimentDone(expdata)
        self.assertRoundTripSerializable(expdata, check_func=self.experiment_data_equiv)
        self.assertRoundTripPickle(expdata, check_func=self.experiment_data_equiv)

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = StateTomography(QuantumCircuit(3), measurement_qubits=[0, 2], qubits=[5, 7, 1])
        loaded_exp = StateTomography.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.json_equiv(exp, loaded_exp))

    def test_analysis_config(self):
        """ "Test converting analysis to and from config works"""
        analysis = StateTomographyAnalysis()
        loaded = StateTomographyAnalysis.from_config(analysis.config())
        self.assertNotEqual(analysis, loaded)
        self.assertEqual(analysis.config(), loaded.config())
