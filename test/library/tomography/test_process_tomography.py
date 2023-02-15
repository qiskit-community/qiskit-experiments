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
ProcessTomography experiment tests
"""
from test.base import QiskitExperimentsTestCase
import ddt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import XGate, CXGate
from qiskit.result import LocalReadoutMitigator
import qiskit.quantum_info as qi
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_experiments.database_service import ExperimentEntryNotFound
from qiskit_experiments.library import ProcessTomography, MitigatedProcessTomography
from qiskit_experiments.library.tomography import ProcessTomographyAnalysis, basis
from .tomo_utils import (
    FITTERS,
    filter_results,
    teleport_circuit,
    teleport_bell_circuit,
    readout_noise_model,
)


@ddt.ddt
class TestProcessTomography(QiskitExperimentsTestCase):
    """Test QuantumProcessTomography experiments"""

    @ddt.data(1, 2)
    def test_full_qpt_random_unitary(self, num_qubits):
        """Test QPT experiment"""
        seed = 1234
        shots = 5000
        f_threshold = 0.98

        # Generate tomography data without analysis
        backend = AerSimulator(seed_simulator=seed, shots=shots)
        target = qi.random_unitary(2**num_qubits, seed=seed)
        exp = ProcessTomography(target)
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
                results = fitdata.analysis_results()

                # Check state is density matrix
                state = filter_results(results, "state").value
                self.assertTrue(
                    isinstance(state, qi.Choi), msg=f"{fitter} fitted state is not a Choi matrix"
                )

                # Check fit state fidelity
                fid = filter_results(results, "process_fidelity").value
                self.assertGreater(fid, f_threshold, msg=f"{fitter} fit fidelity is low")
                # Manually check fidelity
                target_fid = qi.process_fidelity(state, target, require_tp=False, require_cp=False)
                self.assertAlmostEqual(
                    fid, target_fid, places=6, msg=f"{fitter} result fidelity is incorrect"
                )

    def test_full_qpt_analysis_none(self):
        """Test QPT experiment without analysis"""
        seed = 4321
        shots = 1000
        # Generate tomography data without analysis
        backend = AerSimulator(seed_simulator=seed, shots=shots)
        target = qi.random_unitary(2, seed=seed)
        exp = ProcessTomography(target, backend=backend, analysis=None)
        self.assertEqual(exp.analysis, None)
        expdata = exp.run()
        self.assertExperimentDone(expdata)
        self.assertFalse(expdata.analysis_results())

    def test_cvxpy_gaussian_lstsq_cx(self):
        """Test fitter with high fidelity threshold"""
        seed = 1234
        shots = 3000
        f_threshold = 0.999
        fitter = "cvxpy_gaussian_lstsq"

        # Generate tomography data without analysis
        backend = AerSimulator(seed_simulator=seed, shots=shots)
        target = CXGate()
        exp = ProcessTomography(target)
        exp.analysis.set_options(fitter=fitter)
        expdata = exp.run(backend)
        self.assertExperimentDone(expdata)

        results = expdata.analysis_results()

        # Check state is density matrix
        state = filter_results(results, "state").value
        self.assertTrue(
            isinstance(state, qi.Choi), msg=f"{fitter} fitted state is not a Choi matrix"
        )

        # Check fit state fidelity
        fid = filter_results(results, "process_fidelity").value
        self.assertGreater(fid, f_threshold, msg=f"{fitter} fit fidelity is low")
        # Manually check fidelity
        target_fid = qi.process_fidelity(state, target, require_tp=False, require_cp=False)
        self.assertAlmostEqual(
            fid, target_fid, places=6, msg=f"{fitter} result fidelity is incorrect"
        )

    @ddt.data([0], [1], [2], [0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1])
    def test_exp_measurement_preparation_indices(self, qubits):
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
        exp = ProcessTomography(circ, measurement_indices=qubits, preparation_indices=qubits)
        tomo_circuits = exp.circuits()

        # Check correct number of circuits are generated
        size = 3**num_meas * 4**num_meas
        self.assertEqual(len(tomo_circuits), size)

        # Check circuit metadata is correct
        for circ in tomo_circuits:
            meta = circ.metadata
            clbits = meta.get("clbits")
            self.assertEqual(clbits, list(range(num_meas)), msg="metadata clbits is incorrect")

        # Check analysis target is correct
        target_state = exp.analysis.options.target

        target_circ = QuantumCircuit(num_meas)
        for i, qubit in enumerate(qubits):
            target_circ.append(ops[qubit], [i])
        fid = qi.process_fidelity(target_state, qi.Operator(target_circ))
        self.assertGreater(fid, 0.99, msg="target_state is incorrect")

    @ddt.data([0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1])
    @ddt.unpack
    def test_asymmetric_qubits(self, prep_qubit, meas_qubit):
        """Test subset measurement process tomography generation"""
        # Subsystem unitaries
        seed = 1111
        nq = 3
        ops = [qi.random_unitary(2, seed=seed + i) for i in range(nq)]

        # Preparation circuit
        circ = QuantumCircuit(nq)
        for i, op in enumerate(ops):
            circ.append(op, [i])

        exp = ProcessTomography(
            circ, measurement_indices=[meas_qubit], preparation_indices=[prep_qubit]
        )
        backend = AerSimulator(seed_simulator=9000)
        expdata = exp.run(backend, shots=5000)
        self.assertExperimentDone(expdata)

        # Check Choi matrix
        state = expdata.analysis_results("state").value
        self.assertTrue(isinstance(state, qi.Choi), msg="fitted state is not a Choi matrix")
        self.assertEqual(state.input_dims(), (2,), msg="fitted state has wrong input dims")
        self.assertEqual(state.output_dims(), (2,), msg="fitted state has wrong output dims")

        # Check fit state fidelity
        f_threshold = 0.99
        fid = expdata.analysis_results("process_fidelity").value
        self.assertGreater(fid, f_threshold, msg="fit fidelity is low")

    @ddt.data([(0,), (1, 2)], [(2, 0), (2,)])
    @ddt.unpack
    def test_asymmetric_dimensions(self, prep_qubits, meas_qubits):
        """Test subset measurement process tomography generation"""
        # Subsystem unitaries
        seed = 1111
        nq = 3
        ops = [qi.random_unitary(2, seed=seed + i) for i in range(nq)]

        # Preparation circuit
        circ = QuantumCircuit(nq)
        for i, op in enumerate(ops):
            circ.append(op, [i])

        exp = ProcessTomography(
            circ, measurement_indices=meas_qubits, preparation_indices=prep_qubits
        )
        backend = AerSimulator(seed_simulator=9000)
        expdata = exp.run(backend, shots=5000)
        self.assertExperimentDone(expdata)

        # Check Choi matrix
        state = expdata.analysis_results("state").value
        self.assertTrue(isinstance(state, qi.Choi), msg="fitted state is not a Choi matrix")
        self.assertEqual(
            state.input_dims(), len(prep_qubits) * (2,), msg="fitted state has wrong input dims"
        )
        self.assertEqual(
            state.output_dims(), len(meas_qubits) * (2,), msg="fitted state has wrong output dims"
        )

        # Check fit state fidelity
        f_threshold = 0.98
        fid = expdata.analysis_results("process_fidelity").value
        self.assertGreater(fid, f_threshold, msg="fit fidelity is low")

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
        exp = ProcessTomography(circ, measurement_indices=qubits, preparation_indices=qubits)
        expdata = exp.run(backend)
        self.assertExperimentDone(expdata)
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

    @ddt.data(True, False)
    def test_qpt_teleport(self, flatten_creg):
        """Test subset state tomography generation"""
        # Teleport qubit 0 -> 2
        backend = AerSimulator(seed_simulator=9000)
        exp = ProcessTomography(
            teleport_circuit(flatten_creg), measurement_indices=[2], preparation_indices=[0]
        )
        expdata = exp.run(backend, shots=1000)
        self.assertExperimentDone(expdata)
        results = expdata.analysis_results()

        # Check result
        f_threshold = 0.95

        # Check state is a Choi matrix
        state = filter_results(results, "state").value
        self.assertTrue(isinstance(state, qi.Choi), msg="fitted state is not a Choi matrix")

        # Manually check fidelity
        fid = qi.process_fidelity(state, require_tp=False, require_cp=False)
        self.assertGreater(fid, f_threshold, msg="fitted state fidelity is low")

    @ddt.data(True, False)
    def test_qpt_teleport_bell(self, flatten_creg):
        """Test subset state tomography generation"""
        # Teleport qubit 0 -> 2
        backend = AerSimulator(seed_simulator=9000)
        exp = ProcessTomography(
            teleport_bell_circuit(flatten_creg),
            measurement_indices=[2, 3],
            preparation_indices=[0, 3],
        )
        expdata = exp.run(backend, shots=1000)
        self.assertExperimentDone(expdata)
        results = expdata.analysis_results()

        # Check result
        f_threshold = 0.95

        # Check state is a Choi matrix
        state = filter_results(results, "state").value
        self.assertTrue(isinstance(state, qi.Choi), msg="fitted state is not a Choi matrix")

        # Target circuit
        target = QuantumCircuit(2)
        target.h(0)
        target.cx(0, 1)
        target = qi.Operator(target)

        # Manually check fidelity
        fid = qi.process_fidelity(state, target, require_tp=False, require_cp=False)
        self.assertGreater(fid, f_threshold, msg="fitted state fidelity is low")

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = ProcessTomography(
            teleport_circuit(), measurement_indices=[2], preparation_indices=[0]
        )
        loaded_exp = ProcessTomography.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.json_equiv(exp, loaded_exp))

    def test_analysis_config(self):
        """ "Test converting analysis to and from config works"""
        analysis = ProcessTomographyAnalysis()
        loaded = ProcessTomographyAnalysis.from_config(analysis.config())
        self.assertNotEqual(analysis, loaded)
        self.assertEqual(analysis.config(), loaded.config())

    def test_expdata_serialization(self):
        """Test serializing experiment data works."""
        backend = AerSimulator(seed_simulator=9000)
        exp = ProcessTomography(XGate())
        expdata = exp.run(backend)
        self.assertExperimentDone(expdata)
        self.assertRoundTripPickle(expdata, check_func=self.experiment_data_equiv)
        self.assertRoundTripSerializable(expdata, check_func=self.experiment_data_equiv)

    def test_target_none(self):
        """Test setting target=None disables fidelity calculation."""
        seed = 4343
        backend = AerSimulator(seed_simulator=seed)
        target = qi.random_unitary(2, seed=seed)
        exp = ProcessTomography(target, backend=backend, target=None)
        expdata = exp.run()
        self.assertExperimentDone(expdata)
        state = expdata.analysis_results("state").value
        self.assertTrue(
            isinstance(state, qi.Choi),
            msg="Fitted state is not Choi matrix",
        )
        with self.assertRaises(
            ExperimentEntryNotFound, msg="process_fidelity should not exist when target=None"
        ):
            expdata.analysis_results("process_fidelity")

    def test_qpt_spam_mitigated_basis(self):
        """Test QPT with SPAM mitigation basis"""
        num_qubits = 2
        noise_model = NoiseModel()

        # Reset noise model
        p_reset = 0.1
        reset_chans = [
            (1 - p_reset) * qi.SuperOp(np.eye(4))
            + p_reset * qi.random_quantum_channel(2, seed=100 + i)
            for i in range(num_qubits)
        ]
        qubit_states = {}
        for qubit, chan in enumerate(reset_chans):
            qubit_states[qubit] = [chan]
            noise_model.add_quantum_error(chan, "reset", [qubit])

        # Noisy preparation basis
        prep_basis = basis.LocalPreparationBasis(
            "NoisyPauliPrep",
            instructions=basis.PauliPreparationBasis()._instructions,
            qubit_states=qubit_states,
        )

        # Measurement noise model
        p_meas = 0.15
        meas_chans = [
            (1 - p_meas) * qi.SuperOp(np.eye(4))
            + p_meas * qi.random_quantum_channel(2, seed=200 + i)
            for i in range(num_qubits)
        ]
        qubit_povms = {}
        for qubit, chan in enumerate(meas_chans):
            qubit_povms[qubit] = [chan]
            noise_model.add_quantum_error(chan, "measure", [qubit])

        # Noisy measurement basis
        meas_basis = basis.LocalMeasurementBasis(
            "NoisyMeas",
            instructions=basis.PauliMeasurementBasis()._instructions,
            qubit_povms=qubit_povms,
        )

        # Noisy simulator
        backend = AerSimulator(noise_model=noise_model, seed_simulator=1337)

        exp = ProcessTomography(
            CXGate(), measurement_basis=meas_basis, preparation_basis=prep_basis
        )
        exp.backend = backend
        expdata = exp.run(shots=2000).block_for_results()
        self.assertExperimentDone(expdata)
        fid = expdata.analysis_results("process_fidelity").value
        self.assertGreater(fid, 0.95)

    def test_qpt_amat_pauli_basis(self):
        """Test QPT with A-matrix mitigation Pauli basis"""
        num_qubits = 4

        #  Construct a-matrices
        amats = []
        for qubit in range(num_qubits):
            p0g1 = 0.1 + 0.01 * qubit
            p1g0 = 0.05 + 0.01 * qubit
            amats.append(np.array([[1 - p1g0, p0g1], [p1g0, 1 - p0g1]]))

        # Construct noisy measurement basis
        mitigator = LocalReadoutMitigator(amats)
        meas_basis = basis.PauliMeasurementBasis(mitigator=mitigator)

        # Construct noisy simulator
        noise_model = NoiseModel()
        for qubit, amat in enumerate(amats):
            noise_model.add_readout_error(amat.T, [qubit])
        backend = AerSimulator(noise_model=noise_model)

        # Run experiment
        exp = ProcessTomography(CXGate(), measurement_basis=meas_basis)
        exp.backend = backend
        expdata = exp.run(shots=2000).block_for_results()
        self.assertExperimentDone(expdata)
        fid = expdata.analysis_results("process_fidelity").value
        self.assertGreater(fid, 0.95)

    @ddt.data((0,), (1,), (2,), (3,), (0, 1), (2, 0), (0, 3))
    def test_mitigated_full_qpt_random_unitary(self, qubits):
        """Test QPT experiment"""
        seed = 1234
        shots = 5000
        f_threshold = 0.95

        noise_model = readout_noise_model(4, seed=seed)
        backend = AerSimulator(seed_simulator=seed, shots=shots, noise_model=noise_model)
        target = qi.random_unitary(2 ** len(qubits), seed=seed)
        exp = MitigatedProcessTomography(target, backend=backend)
        exp.analysis.set_options(unmitigated_fit=True)
        expdata = exp.run(analysis=None)
        self.assertExperimentDone(expdata)

        for fitter in FITTERS:
            with self.subTest(fitter=fitter, qubits=qubits):
                if fitter:
                    exp.analysis.set_options(fitter=fitter)
                fitdata = exp.analysis.run(expdata)
                self.assertExperimentDone(fitdata)
                # Should be 2 results, mitigated and unmitigated
                states = expdata.analysis_results("state")
                self.assertEqual(len(states), 2)

                # Check state is density matrix
                for state in states:
                    self.assertTrue(
                        isinstance(state.value, qi.Choi),
                        msg=f"{fitter} fitted state is not density matrix for qubits {qubits}",
                    )

                # Check fit state fidelity
                fids = expdata.analysis_results("process_fidelity")
                self.assertEqual(len(fids), 2)
                mitfid, nomitfid = fids
                # Check mitigation improves fidelity
                self.assertTrue(
                    mitfid.value >= nomitfid.value,
                    msg="mitigated {} did not improve fidelity for qubits {} ({:.4f} < {:.4f})".format(
                        fitter, qubits, mitfid.value, nomitfid.value
                    ),
                )
                self.assertGreater(
                    mitfid.value,
                    f_threshold,
                    msg=f"{fitter} fit fidelity is low for qubits {qubits}",
                )
