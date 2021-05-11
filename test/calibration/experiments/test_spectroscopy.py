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

from typing import Optional

import numpy as np
from qiskit.providers import BaseBackend
from qiskit.providers.models import QasmBackendConfiguration
from qiskit.result import Result
from qiskit.test import QiskitTestCase
from qiskit.exceptions import QiskitError

from qiskit_experiments.calibration.experiments.spectroscopy import Spectroscopy


class SpectroscopyBackend(BaseBackend):
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
            counts = {"1": 0, "0": 0}

            if circ.config.calibrations.gates[0].instructions[0].name != "setf":
                raise QiskitError("Spectroscopy does not have a set frequency.")

            set_freq = circ.config.calibrations.gates[0].instructions[0].frequency
            delta_freq = set_freq - self._freq_offset
            prob = np.exp(-(delta_freq ** 2) / (2 * self._linewidth ** 2))

            for _ in range(shots):
                counts[str(np.random.binomial(1, prob))] += 1

            run_result = {
                "shots": shots,
                "success": True,
                "header": {"metadata": circ.header.metadata},
                "data": {"counts": counts},
            }

            result["results"].append(run_result)

        return Result.from_dict(result)


class TestSpectroscopy(QiskitTestCase):
    """Test spectroscopy experiment."""

    def setUp(self):
        """Setup."""
        super().setUp()
        np.random.seed(seed=10)

    def test_spectroscopy_end2end(self):
        """End to end test of the spectroscopy experiment."""

        backend = SpectroscopyBackend(line_width=2e-3)

        spec = Spectroscopy(3, np.linspace(-10.0, 10.0, 21), unit="MHz")
        result = spec.run(backend)

        self.assertTrue(abs(result.analysis_result(0)["value"]) < 1e6)

        # Test if we find still find the peak when it is shifted by 5 MHz.
        backend = SpectroscopyBackend(line_width=2.0e-3, freq_offset=5.0e-3)

        spec = Spectroscopy(3, np.linspace(-10.0, 10.0, 21), unit="MHz")
        result = spec.run(backend)

        self.assertTrue(result.analysis_result(0)["value"] < 6e6)
        self.assertTrue(result.analysis_result(0)["value"] > 4e6)
