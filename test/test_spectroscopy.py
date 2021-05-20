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

"""Spectroscopy tests."""

from typing import Dict, Optional

import numpy as np
from qiskit.providers.basebackend import BaseBackend
from qiskit.providers.backend import BackendV1 as Backend
from qiskit.providers import JobV1
from qiskit.providers.models import QasmBackendConfiguration
from qiskit.result import Result
from qiskit.qobj.utils import MeasLevel
from qiskit.test import QiskitTestCase
from qiskit.exceptions import QiskitError

from qiskit_experiments.characterization.spectroscopy import Spectroscopy


class TestJob(JobV1):
    """Job for testing."""

    def __init__(self, backend: Backend, result: Dict):
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


class SpectroscopyBackend(Backend):
    """
    A simple and primitive backend, to be run by the T1 tests
    """

    def __init__(
        self,
        line_width: Optional[float] = None,
        freq_offset: Optional[float] = None,
    ):
        """
        Initialize the spectroscopy backend.
        """

        configuration = QasmBackendConfiguration(
            backend_name="spectroscopy_simulator",
            backend_version="0",
            n_qubits=int(1),
            basis_gates=["spec"],
            gates=[],
            local=True,
            simulator=True,
            conditional=False,
            open_pulse=False,
            memory=False,
            max_shots=int(1e6),
            coupling_map=[],
            dt=0.1,
        )

        self._linewidth = line_width if line_width else 2.0e-3
        self._freq_offset = freq_offset if freq_offset else 0

        super().__init__(configuration)

    def _default_options(self):
        """Default options of the test backend."""

    # pylint: disable = arguments-differ
    def run(self, circuits, shots=1024, meas_level=MeasLevel.KERNELED, **options):
        """Run the spectroscopy backend."""

        result = {
            "backend_name": "spectroscopy backend",
            "backend_version": "0",
            "qobj_id": 0,
            "job_id": 0,
            "success": True,
            "results": [],
        }

        for circ in circuits:
            print(circ.data[0][0].params)
            if meas_level == MeasLevel.CLASSIFIED:
                counts = {"1": 0, "0": 0}

                set_freq = float(circ.data[0][0].params[0])
                delta_freq = set_freq - self._freq_offset
                prob = np.exp(-(delta_freq ** 2) / (2 * self._linewidth ** 2))

                for _ in range(shots):
                    counts[str(np.random.binomial(1, prob))] += 1

                run_result = {
                    "shots": shots,
                    "success": True,
                    "header": {"metadata": circ.metadata},
                    "data": {"counts": counts},
                }

                result["results"].append(run_result)
            else:
                raise NotImplementedError

        return TestJob(self, result)


class TestSpectroscopy(QiskitTestCase):
    """Test spectroscopy experiment."""

    def setUp(self):
        """Setup."""
        super().setUp()
        np.random.seed(seed=10)

    def test_spectroscopy_end2end_classified(self):
        """End to end test of the spectroscopy experiment."""

        backend = SpectroscopyBackend(line_width=2e6)

        spec = Spectroscopy(3, np.linspace(-10.0, 10.0, 21), unit="MHz")
        result = spec.run(backend, amp=0.05, meas_level=MeasLevel.CLASSIFIED).analysis_result(0)

        print(result)

        self.assertTrue(abs(result["value"]) < 1e6)
        self.assertTrue(result["success"])
        self.assertEqual(result["quality"], "computer_good")

        # Test if we find still find the peak when it is shifted by 5 MHz.
        backend = SpectroscopyBackend(line_width=2.0e-3, freq_offset=5.0e-3)

        spec = Spectroscopy(3, np.linspace(-10.0, 10.0, 21), unit="MHz")
        result = spec.run(backend).analysis_result(0)

        self.assertTrue(result["value"] < 5.1e6)
        self.assertTrue(result["value"] > 4.9e6)
        self.assertEqual(result["quality"], "computer_good")
