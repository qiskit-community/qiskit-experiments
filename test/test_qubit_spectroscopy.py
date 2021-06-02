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

from typing import Tuple

import numpy as np
from qiskit.qobj.utils import MeasLevel
from qiskit.test import QiskitTestCase

from qiskit_experiments.characterization.qubit_spectroscopy import QubitSpectroscopy
from qiskit_experiments.test.mock_iq_backend import TestJob, IQTestBackend


class SpectroscopyBackend(IQTestBackend):
    """
    A simple and primitive backend to test spectroscopy experiments.
    """

    def __init__(
        self,
        line_width: float = 2e6,
        freq_offset: float = 0.0,
        iq_cluster_centers: Tuple[float, float, float, float] = (1.0, 1.0, -1.0, -1.0),
        iq_cluster_width: float = 0.2,
    ):
        """Initialize the spectroscopy backend."""

        self.__configuration__["basis_gates"] = ["spec"]

        self._linewidth = line_width
        self._freq_offset = freq_offset

        super().__init__(iq_cluster_centers, iq_cluster_width)

    # pylint: disable = arguments-differ
    def run(
        self, circuits, shots=1024, meas_level=MeasLevel.KERNELED, meas_return="single", **options
    ):
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

            run_result = {
                "shots": shots,
                "success": True,
                "header": {"metadata": circ.metadata},
            }

            set_freq = float(circ.data[0][0].params[0])
            delta_freq = set_freq - self._freq_offset
            prob = np.exp(-(delta_freq ** 2) / (2 * self._linewidth ** 2))

            if meas_level == MeasLevel.CLASSIFIED:
                counts = {"1": 0, "0": 0}

                for _ in range(shots):
                    counts[str(self._rng.binomial(1, prob))] += 1

                run_result["data"] = {"counts": counts}
            else:
                memory = [self._draw_iq_shot(prob) for _ in range(shots)]

                if meas_return == "avg":
                    memory = np.average(np.array(memory), axis=0).tolist()

                run_result["data"] = {"memory": memory}

            result["results"].append(run_result)

        return TestJob(self, result)


class TestQubitSpectroscopy(QiskitTestCase):
    """Test spectroscopy experiment."""

    def test_spectroscopy_end2end_classified(self):
        """End to end test of the spectroscopy experiment."""

        backend = SpectroscopyBackend(line_width=2e6)

        spec = QubitSpectroscopy(3, np.linspace(-10.0, 10.0, 21), unit="MHz")
        spec.set_run_options(meas_level=MeasLevel.CLASSIFIED)
        result = spec.run(backend).analysis_result(0)

        self.assertTrue(abs(result["value"]) < 1e6)
        self.assertTrue(result["success"])
        self.assertEqual(result["quality"], "computer_good")

        # Test if we find still find the peak when it is shifted by 5 MHz.
        backend = SpectroscopyBackend(line_width=2e6, freq_offset=5.0e6)

        spec = QubitSpectroscopy(3, np.linspace(-10.0, 10.0, 21), unit="MHz")
        spec.set_run_options(meas_level=MeasLevel.CLASSIFIED)
        result = spec.run(backend).analysis_result(0)

        self.assertTrue(result["value"] < 5.1e6)
        self.assertTrue(result["value"] > 4.9e6)
        self.assertEqual(result["quality"], "computer_good")

    def test_spectroscopy_end2end_kerneled(self):
        """End to end test of the spectroscopy experiment on IQ data."""

        backend = SpectroscopyBackend(line_width=2e6)

        spec = QubitSpectroscopy(3, np.linspace(-10.0, 10.0, 21), unit="MHz")
        result = spec.run(backend).analysis_result(0)

        self.assertTrue(abs(result["value"]) < 1e6)
        self.assertTrue(result["success"])
        self.assertEqual(result["quality"], "computer_good")

        # Test if we find still find the peak when it is shifted by 5 MHz.
        backend = SpectroscopyBackend(line_width=2e6, freq_offset=5.0e6)

        spec = QubitSpectroscopy(3, np.linspace(-10.0, 10.0, 21), unit="MHz")
        result = spec.run(backend).analysis_result(0)

        self.assertTrue(result["value"] < 5.1e6)
        self.assertTrue(result["value"] > 4.9e6)
        self.assertEqual(result["quality"], "computer_good")
        self.assertTrue(result["ydata_err"] is not None)

        spec.set_run_options(meas_return="avg")
        result = spec.run(backend).analysis_result(0)

        self.assertTrue(result["value"] < 5.1e6)
        self.assertTrue(result["value"] > 4.9e6)
        self.assertEqual(result["quality"], "computer_good")
        self.assertTrue(result["ydata_err"] is None)
