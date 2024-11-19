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
from math import sqrt

import ddt
import numpy as np
from uncertainties import UFloat

import qiskit.quantum_info as qi
from qiskit import QuantumCircuit
from qiskit.circuit.library import XGate

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from qiskit_experiments.data_processing import LocalReadoutMitigator
from qiskit_experiments.database_service import ExperimentEntryNotFound
from qiskit_experiments.library import StateTomography, MitigatedStateTomography
from qiskit_experiments.library.tomography import StateTomographyAnalysis, basis
from .tomo_utils import (
    FITTERS,
    filter_results,
    teleport_circuit,
    teleport_bell_circuit,
    readout_noise_model,
)


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
        expdata = exp.run(backend, analysis=None, shots=shots)
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

    def test_full_qst_analysis_none(self):
        """Test QST experiment"""
        seed = 4321
        shots = 1000
        # Generate tomography data without analysis
        backend = AerSimulator(seed_simulator=seed, shots=shots)
        target = qi.random_statevector(2, seed=seed)
        exp = StateTomography(target, backend=backend, analysis=None)
        self.assertEqual(exp.analysis, None)
        expdata = exp.run()
        self.assertExperimentDone(expdata)
        self.assertFalse(expdata.analysis_results())

    @ddt.data(True, False)
    def test_qst_teleport(self, flatten_creg):
        """Test subset state tomography generation"""
        # Teleport qubit 0 -> 2
        backend = AerSimulator(seed_simulator=9000)
        exp = StateTomography(teleport_circuit(flatten_creg), measurement_indices=[2])
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

    @ddt.data(True, False)
    def test_qst_teleport_bell(self, flatten_creg):
        """Test subset state tomography generation"""
        # Teleport qubit 0 -> 2
        backend = AerSimulator(seed_simulator=9000)
        exp = StateTomography(teleport_bell_circuit(flatten_creg), measurement_indices=[2, 3])
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
        fid = qi.state_fidelity(state, qi.Statevector([1, 0, 0, 1]) / sqrt(2), validate=False)
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
    def test_exp_circuits_measurement_indices(self, meas_qubits):
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
        exp = StateTomography(circ, measurement_indices=meas_qubits)
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
    def test_full_exp_measurement_indices(self, meas_qubits):
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
        exp = StateTomography(circ, measurement_indices=meas_qubits)
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

    def test_circuit_roundtrip_serializable(self):
        """Test a simple roundtrip experiment serialization"""
        circ = QuantumCircuit(2)
        circ.h(0)
        circ.s(0)
        circ.cx(0, 1)

        exp = StateTomography(circ)
        self.assertRoundTripSerializable(exp._transpiled_circuits())

    def test_expdata_serialization(self):
        """Test serializing experiment data works."""
        backend = AerSimulator(seed_simulator=9000)
        exp = StateTomography(XGate())
        expdata = exp.run(backend)
        self.assertExperimentDone(expdata)
        self.assertRoundTripSerializable(expdata)
        self.assertRoundTripPickle(expdata)

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = StateTomography(
            QuantumCircuit(3), measurement_indices=[0, 2], physical_qubits=[5, 7, 1]
        )
        loaded_exp = StateTomography.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertEqualExtended(exp, loaded_exp)

    def test_analysis_config(self):
        """Test converting analysis to and from config works"""
        analysis = StateTomographyAnalysis()
        loaded = StateTomographyAnalysis.from_config(analysis.config())
        self.assertNotEqual(analysis, loaded)
        self.assertEqual(analysis.config(), loaded.config())

    def test_target_none(self):
        """Test setting target=None disables fidelity calculation."""
        seed = 4343
        backend = AerSimulator(seed_simulator=seed)
        target = qi.random_statevector(2, seed=seed)
        exp = StateTomography(target, backend=backend, target=None)
        expdata = exp.run()
        self.assertExperimentDone(expdata)
        state = expdata.analysis_results("state").value
        self.assertTrue(
            isinstance(state, qi.DensityMatrix),
            msg="Fitted state is not density matrix",
        )
        with self.assertRaises(
            ExperimentEntryNotFound, msg="state_fidelity should not exist when target=None"
        ):
            expdata.analysis_results("state_fidelity")

    def test_qst_spam_mitigated_basis(self):
        """Test QST with SPAM mitigation basis"""
        num_qubits = 4
        noise_model = NoiseModel()

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

        circ = QuantumCircuit(num_qubits)
        circ.h(0)
        for i in range(1, num_qubits):
            circ.cx(i - 1, i)
        exp = StateTomography(circ, measurement_basis=meas_basis)
        exp.backend = backend
        expdata = exp.run(shots=2000)
        self.assertExperimentDone(expdata)
        fid = expdata.analysis_results("state_fidelity").value
        self.assertGreater(fid, 0.95)

    def test_qst_amat_pauli_basis(self):
        """Test QST with A-matrix mitigation Pauli basis"""
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
        backend = AerSimulator(noise_model=noise_model, seed_simulator=1234)

        # Run experiment
        circ = QuantumCircuit(num_qubits)
        circ.h(0)
        for i in range(1, num_qubits):
            circ.cx(i - 1, i)
        exp = StateTomography(circ, measurement_basis=meas_basis)
        exp.backend = backend
        expdata = exp.run(shots=2000)
        self.assertExperimentDone(expdata)
        fid = expdata.analysis_results("state_fidelity").value
        self.assertGreater(fid, 0.945)

    @ddt.data((0,), (1,), (2,), (3,), (0, 1), (2, 0), (0, 3), (0, 3, 1))
    def test_mitigated_full_qst(self, qubits):
        """Test QST experiment"""
        seed = 1234
        shots = 5000
        f_threshold = 0.95

        noise_model = readout_noise_model(4, seed=seed)
        backend = AerSimulator(seed_simulator=seed, shots=shots, noise_model=noise_model)
        target = qi.random_statevector(2 ** len(qubits), seed=seed)
        exp = MitigatedStateTomography(target, physical_qubits=qubits, backend=backend)
        exp.analysis.set_options(unmitigated_fit=True)
        expdata = exp.run(analysis=None, shots=shots)
        self.assertExperimentDone(expdata)

        for fitter in FITTERS:
            with self.subTest(fitter=fitter, qubits=qubits):
                if fitter:
                    exp.analysis.set_options(fitter=fitter)
                fitdata = exp.analysis.run(expdata)
                self.assertExperimentDone(fitdata)
                # Should be 2 results, mitigated and unmitigated
                states = fitdata.analysis_results("state")
                self.assertEqual(len(states), 2)
                for state in states:
                    self.assertTrue(
                        isinstance(state.value, qi.DensityMatrix),
                        msg=f"{fitter} fitted state is not density matrix for qubits {qubits}",
                    )

                # Check fit state fidelity
                fids = expdata.analysis_results("state_fidelity")
                self.assertEqual(len(fids), 2)
                mitfid, nomitfid = fids
                # Check mitigation improves fidelity
                self.assertTrue(
                    mitfid.value >= nomitfid.value,
                    msg=(
                        f"mitigated {fitter} did not improve fidelity for qubits {qubits} "
                        f"({mitfid.value:.4f} < {nomitfid.value:.4f})"
                    ),
                )
                self.assertGreater(
                    mitfid.value,
                    f_threshold,
                    msg=f"{fitter} fit fidelity is low for qubits {qubits}",
                )
                self.assertTrue(mitfid.extra["mitigated"])
                self.assertFalse(nomitfid.extra["mitigated"])

    @ddt.data([None, 1], [True, 4], [[0], 2], [[1], 2], [[0, 1], 4])
    @ddt.unpack
    def test_qst_conditional_circuit(self, circuit_clbits, num_components):
        """Test subset state tomography generation"""
        # Preparation circuit
        prep_circ = QuantumCircuit(2)
        prep_circ.ry(np.pi / 3, 0)
        prep_circ.ry(-np.pi / 4, 1)

        # Calculate component probabilities and targets
        # measurements project to diagonal of density matrix
        prep_state = qi.DensityMatrix(np.diag(np.diag(qi.DensityMatrix(prep_circ))))
        if circuit_clbits is None:
            component_probs = [1]
            components = [qi.DensityMatrix(np.diag(prep_state.probabilities()))]
        elif circuit_clbits is True or len(circuit_clbits) == 2:
            component_probs = prep_state.probabilities()
            components = [qi.DensityMatrix.from_label(i) for i in ["00", "01", "10", "11"]]
        else:
            component_probs = prep_state.probabilities(circuit_clbits)
            components = [
                prep_state.evolve(qi.DensityMatrix.from_label(str(i)), circuit_clbits) / p
                for i, p in enumerate(component_probs)
            ]

        # Add measurements
        circ = prep_circ.copy()
        circ.measure_all()

        # Run experiment
        backend = AerSimulator(seed_simulator=7172)
        exp = StateTomography(
            circ,
            backend=backend,
            conditional_circuit_clbits=circuit_clbits,
        )
        expdata = exp.run(shots=2000, analysis=None)
        self.assertExperimentDone(expdata)

        for fitter in FITTERS:
            with self.subTest(fitter=fitter):
                if fitter:
                    exp.analysis.set_options(fitter=fitter)
                fitdata = exp.analysis.run(expdata)
                states = fitdata.analysis_results("state")
                if circuit_clbits is None:
                    states = [states]
                self.assertEqual(len(states), num_components)
                for state in states:
                    idx = state.extra.get("conditional_circuit_outcome", 0)
                    prob = state.extra["conditional_probability"]
                    fid = qi.state_fidelity(state.value, components[idx])
                    self.assertGreater(
                        fid,
                        0.95,
                        msg=f"fitter {fitter} fidelity is low for conditional outcome {idx}",
                    )
                    self.assertLess(
                        abs(prob - component_probs[idx]),
                        1e-2,
                        msg=f"fitter {fitter} probability is incorrect for conditional outcome {idx}",
                    )

    @ddt.data([0], [1], [0, 1])
    def test_qst_conditional_zeros_circuit(self, circuit_clbits):
        """Test subset state tomography generation"""
        # Preparation circuit
        circ = QuantumCircuit(2)
        circ.measure_all()

        # Run experiment
        backend = AerSimulator(seed_simulator=7172)
        exp = StateTomography(
            circ,
            backend=backend,
            conditional_circuit_clbits=circuit_clbits,
        )
        expdata = exp.run(shots=2000, analysis=None)
        self.assertExperimentDone(expdata)

        for fitter in FITTERS:
            with self.subTest(fitter=fitter):
                if fitter:
                    exp.analysis.set_options(fitter=fitter)
                fitdata = exp.analysis.run(expdata)
                states = fitdata.analysis_results("state")
                if circuit_clbits is None:
                    states = [states]
                self.assertEqual(len(states), 2 ** len(circuit_clbits))
                for state in states:
                    idx = state.extra.get("conditional_circuit_outcome", 0)
                    prob = state.extra["conditional_probability"]
                    if idx == 0:
                        self.assertTrue(
                            np.isclose(prob, 1, atol=1e-3),
                            msg=f"fitter {fitter} probability incorrect for component"
                            f" {idx} ({prob} != 1)",
                        )
                        fid = qi.state_fidelity(
                            state.value, qi.Statevector.from_label("0" * state.value.num_qubits)
                        )
                        self.assertGreater(
                            fid, 0.99, msg=f"fitter {fitter} fidelity is low for component {idx}"
                        )
                    else:
                        self.assertTrue(
                            np.isclose(prob, 0, atol=1e-3),
                            msg=f"fitter {fitter} probability incorrect for component"
                            f" {idx} ({prob} != 0)",
                        )

    @ddt.data(None, [0], [1], [0, 1])
    def test_qst_conditional_measurement(self, conditional_indices):
        """Test subset state tomography generation"""
        # Preparation circuit
        circ = QuantumCircuit(2)
        circ.ry(np.pi / 3, 0)
        circ.ry(-np.pi / 4, 1)
        circ_target = qi.DensityMatrix(circ)

        # Run experiment
        backend = AerSimulator(seed_simulator=7172)
        exp = StateTomography(circ, backend=backend)
        expdata = exp.run(shots=5000, analysis=None)
        self.assertExperimentDone(expdata)

        for fitter in FITTERS:
            with self.subTest(fitter=fitter):
                exp.analysis.set_options(conditional_measurement_indices=conditional_indices)
                if fitter:
                    exp.analysis.set_options(fitter=fitter)
                mbasis = exp.analysis.options.measurement_basis
                fitdata = exp.analysis.run(expdata)
                states = fitdata.analysis_results("state")
                if conditional_indices is None:
                    states = [states]
                for state in states:
                    prob = state.extra["conditional_probability"]
                    index = state.extra.get("conditional_measurement_index")
                    outcome = state.extra.get("conditional_measurement_outcome")
                    if index:
                        outcome_proj = qi.Operator(
                            mbasis.matrix(index, outcome, conditional_indices)
                        )
                        proj = qi.partial_trace(
                            circ_target.evolve(outcome_proj, conditional_indices),
                            conditional_indices,
                        )
                        target_prob = np.trace(proj).real
                        target_state = proj / target_prob
                    else:
                        target_prob = 1
                        target_state = circ_target
                    fid = qi.state_fidelity(state.value, target_state)
                    self.assertGreater(
                        fid,
                        0.94,
                        msg=f"fitter {fitter} fidelity is low for conditional measurement"
                        f" {index}, {outcome}",
                    )
                    self.assertLess(
                        abs(prob - target_prob),
                        2e-2,
                        msg=f"fitter {fitter} probability is incorrect for conditional"
                        f" outcome {index}, {outcome}",
                    )

    def test_bootstrap_qst(self):
        """Test QST experiment with bootstrapped error bars"""
        seed = 1234
        shots = 100
        bootstrap_samples = 10

        # Generate tomography data without analysis
        backend = AerSimulator(seed_simulator=seed, shots=shots)
        target = qi.Statevector([0, 1])
        exp = StateTomography(target)
        exp.analysis.set_options(target_bootstrap_samples=bootstrap_samples)
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

                # Check fit state fidelity
                fid = filter_results(results, "state_fidelity").value
                self.assertTrue(isinstance(fid, UFloat))
                self.assertGreater(fid.s, 0)
