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
A Tester for the Quantum Volume experiment
"""

from qiskit import Aer
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import readout_error
from qiskit.providers.aer.noise.errors.standard_errors import (
    depolarizing_error,
    thermal_relaxation_error,
)
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.test import QiskitTestCase

from qiskit_experiments.library import QuantumVolume

SEED = 42


def create_noise_model():
    """
    create noise model with depolarizing error, thermal error and readout error
    Returns:
        NoiseModel: noise model
    """
    noise_model = NoiseModel()
    p1q = 0.0004
    p2q = 0.01
    depol_sx = depolarizing_error(p1q, 1)
    depol_x = depolarizing_error(p1q, 1)
    depol_cx = depolarizing_error(p2q, 2)

    # Add T1/T2 noise to the simulation
    t_1 = 110e3
    t_2 = 120e3
    gate1q = 50
    gate2q = 100
    termal_sx = thermal_relaxation_error(t_1, t_2, gate1q)
    termal_x = thermal_relaxation_error(t_1, t_2, gate1q)
    termal_cx = thermal_relaxation_error(t_1, t_2, gate2q).tensor(
        thermal_relaxation_error(t_1, t_2, gate2q)
    )

    noise_model.add_all_qubit_quantum_error(depol_sx.compose(termal_sx), "sx")
    noise_model.add_all_qubit_quantum_error(depol_x.compose(termal_x), "x")
    noise_model.add_all_qubit_quantum_error(depol_cx.compose(termal_cx), "cx")

    read_err = readout_error.ReadoutError([[0.98, 0.02], [0.04, 0.96]])
    noise_model.add_all_qubit_readout_error(read_err)
    return noise_model


def create_high_noise_model():
    """
    create high noise model with depolarizing error, thermal error and readout error
    Returns:
        NoiseModel: noise model
    """
    noise_model = NoiseModel()
    p1q = 0.004
    p2q = 0.05
    depol_sx = depolarizing_error(p1q, 1)
    depol_x = depolarizing_error(p1q, 1)
    depol_cx = depolarizing_error(p2q, 2)

    # Add T1/T2 noise to the simulation
    t_1 = 110e2
    t_2 = 120e2
    gate1q = 50
    gate2q = 100
    termal_sx = thermal_relaxation_error(t_1, t_2, gate1q)
    termal_x = thermal_relaxation_error(t_1, t_2, gate1q)
    termal_cx = thermal_relaxation_error(t_1, t_2, gate2q).tensor(
        thermal_relaxation_error(t_1, t_2, gate2q)
    )

    noise_model.add_all_qubit_quantum_error(depol_sx.compose(termal_sx), "sx")
    noise_model.add_all_qubit_quantum_error(depol_x.compose(termal_x), "x")
    noise_model.add_all_qubit_quantum_error(depol_cx.compose(termal_cx), "cx")

    read_err = readout_error.ReadoutError([[0.98, 0.02], [0.04, 0.96]])
    noise_model.add_all_qubit_readout_error(read_err)
    return noise_model


class TestQuantumVolume(QiskitTestCase):
    """Test Quantum Volume experiment"""

    def test_qv_circuits_length(self):
        """
        Test circuit generation - check the number of circuits generated
        and the amount of qubits in each circuit
        """

        qubits_lists = [3, [0, 1, 2], [0, 1, 2, 4]]
        ntrials = [2, 3, 5]

        for qubits in qubits_lists:
            for trials in ntrials:
                qv_exp = QuantumVolume(qubits)
                qv_exp.set_experiment_options(trials=trials)
                qv_circs = qv_exp.circuits()

                self.assertEqual(
                    len(qv_circs),
                    trials,
                    "Number of circuits generated do not match the number of trials",
                )

                self.assertEqual(
                    len(qv_circs[0].qubits),
                    qv_exp.num_qubits,
                    "Number of qubits in the Quantum Volume circuit do not match the"
                    " number of qubits in the experiment",
                )

    def test_qv_ideal_probabilities(self):
        """
        Test the probabilities of ideal circuit
        Compare between simulation and statevector calculation
        and compare to pre-calculated probabilities with the same seed
        """
        num_of_qubits = 3
        qv_exp = QuantumVolume(num_of_qubits, seed=SEED)
        qv_exp.set_experiment_options(trials=20)
        qv_circs = qv_exp.circuits()
        simulation_probabilities = [
            list(qv_circ.metadata["ideal_probabilities"]) for qv_circ in qv_circs
        ]
        # create the circuits again, but this time disable simulation so the
        # ideal probabilities will be calculated using statevector
        qv_exp = QuantumVolume(num_of_qubits, seed=SEED)
        qv_exp.set_experiment_options(trials=20)
        qv_exp._simulation_backend = None
        qv_circs = qv_exp.circuits()
        statevector_probabilities = [
            qv_circ.metadata["ideal_probabilities"] for qv_circ in qv_circs
        ]

        self.assertTrue(
            matrix_equal(simulation_probabilities, statevector_probabilities),
            "probabilities calculated using simulation and " "statevector are not the same",
        )

    def test_qv_sigma_decreasing(self):
        """
        Test that the sigma is decreasing after adding more trials
        """
        num_of_qubits = 3
        backend = Aer.get_backend("aer_simulator")

        qv_exp = QuantumVolume(num_of_qubits, seed=SEED)
        # set number of trials to a low number to make the test faster
        qv_exp.set_experiment_options(trials=2)
        expdata1 = qv_exp.run(backend)
        expdata1.block_for_results()
        result_data1 = expdata1.analysis_results(0)
        expdata2 = qv_exp.run(backend, experiment_data=expdata1)
        expdata2.block_for_results()
        result_data2 = expdata2.analysis_results(0)

        self.assertTrue(result_data1.extra["trials"] == 2, "number of trials is incorrect")
        self.assertTrue(
            result_data2.extra["trials"] == 4,
            "number of trials is incorrect" " after adding more trials",
        )
        self.assertTrue(
            result_data2.value.stderr <= result_data1.value.stderr,
            "sigma did not decreased after adding more trials",
        )

    def test_qv_failure_insufficient_trials(self):
        """
        Test that the quantum volume is unsuccessful when:
            there is less than 100 trials
        """
        num_of_qubits = 3
        backend = Aer.get_backend("aer_simulator")

        qv_exp = QuantumVolume(num_of_qubits, seed=SEED)
        qv_exp.set_experiment_options(trials=70)
        qv_data = qv_exp.run(backend)
        qv_data.block_for_results()

        qv_exp.run_analysis(qv_data)
        qv_result = qv_data.analysis_results(1)
        self.assertTrue(
            qv_result.extra["success"] is False and qv_result.value == 1,
            "quantum volume is successful with less than 100 trials",
        )

    def test_qv_failure_insufficient_hop(self):
        """
        Test that the quantum volume is unsuccessful when:
            there are more than 100 trials, but the heavy output probability mean is less than 2/3
        """
        num_of_qubits = 4
        backend = Aer.get_backend("aer_simulator")
        basis_gates = ["id", "rz", "sx", "x", "cx", "reset"]
        noise = create_high_noise_model()

        qv_exp = QuantumVolume(num_of_qubits, seed=SEED)
        qv_exp.set_transpile_options(basis_gates=basis_gates)
        qv_data = qv_exp.run(backend, noise_model=noise, basis_gates=basis_gates)
        qv_data.block_for_results()

        qv_exp.run_analysis(qv_data)
        qv_result = qv_data.analysis_results(1)
        self.assertTrue(
            qv_result.extra["success"] is False and qv_result.value == 1,
            "quantum volume is successful with heavy output probability less than 2/3",
        )

    def test_qv_failure_insufficient_confidence(self):
        """
        Test that the quantum volume is unsuccessful when:
            there are more than 100 trials, the heavy output probability mean is more than 2/3
            but the confidence is not high enough
        """
        num_of_qubits = 4
        backend = Aer.get_backend("aer_simulator")
        basis_gates = ["id", "rz", "sx", "x", "cx", "reset"]
        noise = create_noise_model()

        qv_exp = QuantumVolume(num_of_qubits, seed=SEED)
        qv_exp.set_transpile_options(basis_gates=basis_gates)
        qv_data = qv_exp.run(backend, noise_model=noise, basis_gates=basis_gates)
        qv_data.block_for_results()

        qv_exp.run_analysis(qv_data)
        qv_result = qv_data.analysis_results(1)
        self.assertTrue(
            qv_result.extra["success"] is False and qv_result.value == 1,
            "quantum volume is successful with insufficient confidence",
        )

    def test_qv_success(self):
        """
        Test a successful run of quantum volume.
        Compare the results to a pre-run experiment
        """
        num_of_qubits = 4
        backend = Aer.get_backend("aer_simulator")
        basis_gates = ["id", "rz", "sx", "x", "cx", "reset"]
        noise = create_noise_model()

        qv_exp = QuantumVolume(num_of_qubits, seed=SEED)
        qv_exp.set_experiment_options(trials=300)
        qv_exp.set_transpile_options(basis_gates=basis_gates)
        qv_data = qv_exp.run(backend, noise_model=noise, basis_gates=basis_gates)
        qv_data.block_for_results()

        # check HOP entry
        mean_hop = qv_data.analysis_results("mean_HOP")

        ref_hop_value = 0.73146484375
        ref_hop_stderr = 0.025588019729799863
        ref_hop_two_sigma = 0.051176039459599726
        ref_hop_depth = 4
        ref_hop_trials = 300

        self.assertAlmostEqual(
            mean_hop.value.value,
            ref_hop_value,
            delta=1e-3,
            msg="result mean HOP value is not the same as precalculated analysis",
        )
        self.assertAlmostEqual(
            mean_hop.value.stderr,
            ref_hop_stderr,
            delta=1e-3,
            msg="result value is not the same as precalculated analysis",
        )
        self.assertAlmostEqual(
            mean_hop.extra["two_sigma"],
            ref_hop_two_sigma,
            delta=1e-3,
            msg="result two_sigma is not the same as precalculated analysis",
        )
        self.assertAlmostEqual(
            mean_hop.extra["two_sigma"],
            ref_hop_two_sigma,
            delta=1e-3,
            msg="result two_sigma is not the same as precalculated analysis",
        )
        self.assertEqual(
            mean_hop.extra["depth"],
            ref_hop_depth,
            msg="result depth is not the same as precalculated analysis",
        )
        self.assertEqual(
            mean_hop.extra["trials"],
            ref_hop_trials,
            msg="result trials is not the same as precalculated analysis",
        )

        # check QV entry
        quantum_volume = qv_data.analysis_results("quantum_volume")

        ref_qv_value = 16
        ref_qv_confidence = 0.9943351826324864

        self.assertEqual(
            quantum_volume.value,
            ref_qv_value,
            msg="result quantum volume value is not the same as precalculated analysis",
        )
        self.assertTrue(
            quantum_volume.extra["success"],
            msg="result quantum volume success is not the same as precalculated analysis",
        )
        self.assertAlmostEqual(
            quantum_volume.extra["confidence"],
            ref_qv_confidence,
            delta=1e-3,
            msg="result quantum volume confidence is not the same as precalculated analysis",
        )
