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

"""Test Rabi amplitude Experiment class."""

from typing import Dict, Tuple
import numpy as np

from qiskit import QuantumCircuit, execute
from qiskit.providers.basebackend import BaseBackend
from qiskit.providers.models import QasmBackendConfiguration
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit.test import QiskitTestCase
from qiskit.result import Result
from qiskit.providers import JobV1

from qiskit_experiments import ExperimentData
from qiskit_experiments.calibration.experiments.rabi import RabiAnalysis, Rabi
from qiskit_experiments.data_processing.data_processor import DataProcessor
from qiskit_experiments.data_processing.nodes import Probability


# TODO Reuse functionality from spectroscopy.
class TestJob(JobV1):
    """Job for testing."""

    def __init__(self, backend: BaseBackend, result: Dict):
        """Setup a job for testing."""
        super().__init__(backend, "test-id")
        self._result = result

    def result(self) -> Result:
        """Return a result."""
        return Result.from_dict(self._result)

    def submit(self):
        pass

    def status(self):
        pass

    def cancel(self):
        pass


class RabiBackend(BaseBackend):
    """
    A simple and primitive backend, to be run by the T1 tests
    """

    def __init__(
        self,
        iq_cluster_centers: Tuple[float, float, float, float] = (1.0, 1.0, -1.0, -1.0),
        iq_cluster_width: float = 1.0,
        amplitude_to_angle = np.pi,
    ):
        """
        Initialize the spectroscopy backend.
        """

        configuration = QasmBackendConfiguration(
            backend_name="rabi_simulator",
            backend_version="0",
            n_qubits=int(1),
            basis_gates=["rx"],
            gates=[],
            local=True,
            simulator=True,
            conditional=False,
            open_pulse=False,
            memory=True,
            max_shots=int(1e6),
            coupling_map=[],
            dt=0.1,
        )

        self._iq_cluster_centers = iq_cluster_centers
        self._iq_cluster_width = iq_cluster_width
        self._amplitude_to_angle = amplitude_to_angle

        super().__init__(configuration)

    def _draw_iq_shot(self, prob):
        """Produce an IQ shot."""

        rand_i = np.random.normal(0, self._iq_cluster_width)
        rand_q = np.random.normal(0, self._iq_cluster_width)

        if np.random.binomial(1, prob) > 0.5:
            return [[self._iq_cluster_centers[0] + rand_i, self._iq_cluster_centers[1] + rand_q]]
        else:
            return [[self._iq_cluster_centers[2] + rand_i, self._iq_cluster_centers[3] + rand_q]]

    # pylint: disable = arguments-differ
    def run(self, qobj):
        """Run the spectroscopy backend."""

        shots = qobj.config.shots

        result = {
            "backend_name": "spectroscopy backend",
            "backend_version": "0",
            "qobj_id": 0,
            "job_id": 0,
            "success": True,
            "results": [],
        }

        for circ in qobj.experiments:
            memory = []

            # Convert the amplitude to a rotation angle.
            angle = float(circ.instructions[0].params[0])*self._amplitude_to_angle

            es_prob = np.sin(angle)**2

            for _ in range(shots):
                memory.append(self._draw_iq_shot(es_prob))

            run_result = {
                "shots": shots,
                "success": True,
                "header": {"metadata": circ.header.metadata},
                "data": {"memory": memory},
                "meas_level": 1
            }

            result["results"].append(run_result)

        return TestJob(self, result)


class TestRabiEndToEnd(QiskitTestCase):
    """Test the rabi experiment."""

    def setUp(self):
        """Setup."""
        super().setUp()
        np.random.seed(seed=10)

    def test_rabi_end_to_end(self):
        """Test the Rabi experiment end to end."""

        backend = RabiBackend()

        spec = Rabi(3, np.linspace(-0.95, 0.95, 21))
        result = spec.run(backend, amp=0.05, shots=10).analysis_result(0)

        self.assertEqual(result["quality"], "computer_good")


class TestRabiAnalysis(QiskitTestCase):
    """Class to test the fitting."""

    def simulate_experiment_data(self, thetas, amplitudes, shots=1024):
        """Generate experiment data for Rx rotations with an arbitrary amplitude calibration."""
        circuits = []
        for theta in thetas:
            qc = QuantumCircuit(1)
            qc.rx(theta, 0)
            qc.measure_all()
            circuits.append(qc)

        sim = QasmSimulatorPy()
        result = execute(circuits, sim, shots=shots, seed_simulator=10).result()
        data = [
            {"counts": self._add_uncertainty(result.get_counts(i)),
             "metadata": {"xval": amplitudes[i]}} for i, theta in enumerate(thetas)
        ]
        return data

    def _add_uncertainty(self, counts):
        """Ensure that we always have a non-zero sigma in the test."""
        for label in ["0", "1"]:
            if label not in counts:
                counts[label] = 1

        return counts

    def test_analysis(self):
        """Test the Rabi analysis."""
        experiment_data = ExperimentData()

        thetas = np.linspace(-1.5*np.pi, 1.5*np.pi, 51)
        amplitudes = np.linspace(-0.95, 0.95, 51)

        for shots in [10, 1024]:
            data = self.simulate_experiment_data(thetas, amplitudes, shots=shots)
            experiment_data.add_data(data)

            rabi_analysis = RabiAnalysis()

            data_processor = DataProcessor("counts", [Probability(outcome="1")])

            result = rabi_analysis.run(experiment_data, data_processor=data_processor, plot=False)

            self.assertEqual(result["quality"], "computer_good")
