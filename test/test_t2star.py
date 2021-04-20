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
Test T2Star experiment
"""
import unittest
import numpy as np
from typing import Tuple
from qiskit.providers import BaseBackend
from qiskit.providers.models import QasmBackendConfiguration
from qiskit.result import Result
from qiskit_experiments.experiment_data import ExperimentData
from qiskit_experiments.composite import ParallelExperiment
from qiskit_experiments.characterization import T2StarExperiment

from qiskit.test import QiskitTestCase

# Fix seed for simulations
SEED = 9000

# from Yael
class T2Backend(BaseBackend):
    """
    A simple and primitive backend, to be run by the T1 tests
    """

    def __init__(
            self, t2, initial_prob1=None, readout0to1=None, readout1to0=None, dt_factor_in_microsec=1e6
    ):
        """
        Initialize the T2 backend
        """

        configuration = QasmBackendConfiguration(
            backend_name="t1_simulator",
            backend_version="0",
            n_qubits=int(1e6),
            basis_gates=["barrier", "x", "h", "p", "delay", "measure"],
            gates=[],
            local=True,
            simulator=True,
            conditional=False,
            open_pulse=False,
            memory=False,
            max_shots=int(1e6),
            coupling_map=None,
            dt=dt_factor_in_microsec * 1000,
        )

        self._t2 = t2
        self._initial_prob1 = initial_prob1
        self._readout0to1 = readout0to1
        self._readout1to0 = readout1to0
        self._dt_factor_in_microsec = dt_factor_in_microsec
        super().__init__(configuration)

    # pylint: disable = arguments-differ
    def run(self, qobj):
        """
        Run the T1 backend
        """

        shots = qobj.config.shots

        result = {
            "backend_name": "T1 backend",
            "backend_version": "0",
            "qobj_id": 0,
            "job_id": 0,
            "success": True,
            "results": [],
        }

        for circ in qobj.experiments:
            nqubits = circ.config.n_qubits
            counts = dict()

            if self._readout0to1 is None:
                ro01 = np.zeros(nqubits)
            else:
                ro01 = self._readout0to1

            if self._readout1to0 is None:
                ro10 = np.zeros(nqubits)
            else:
                ro10 = self._readout1to0

            for _ in range(shots):
                if self._initial_prob1 is None:
                    prob1 = np.zeros(nqubits)
                else:
                    prob1 = self._initial_prob1.copy()

                clbits = np.zeros(circ.config.memory_slots, dtype=int)

                for op in circ.instructions:
                    qubit = op.qubits[0]
                    if op.name == "x":
                        prob1 = 1 - prob1
                    if op.name == "h":
                        prob1 = (1 / np.sqrt(2)) * prob1
                    if op.name == "p":
                        prob1 = prob1 * np.exp(complex(0.0, op.params[0].real))
                    if op.name == "delay":
                        delay = op.params[0] * self._dt_factor_in_microsec
                        prob1 = prob1 * np.exp(-delay / self._t2)
                    if op.name == "measure":
                        meas_res = np.random.binomial(
                            1, prob1 * (1 - ro10) + (1 - prob1) * ro01
                        )
                        clbits[op.memory[0]] = meas_res
                        prob1 = meas_res

                clstr = ""
                for clbit in clbits[::-1]:
                    clstr = clstr + str(clbit)

                if clstr in counts:
                    counts[clstr] += 1
                else:
                    counts[clstr] = 1

            result["results"].append(
                {
                    "shots": shots,
                    "success": True,
                    "header": {"metadata": circ.header.metadata},
                    "data": {"counts": counts},
                }
            )
        print(counts)
        return Result.from_dict(result)


class TestT2Star(QiskitTestCase):
    """ Test T2Star experiment"""
    def test_t2star_generate_circuits(self):
        """
        Test T2Star experiment using a simulator.
        Currently only verifies that there is no exception,
        but does not verify accuracy of the estimate.
        """
        t2star = 25
        # Set up the circuits
        qubit = 0
        delays = np.append(
            (np.linspace(1.0, 15.0, num=15)).astype(float),
            (np.linspace(16.0, 45.0, num=59)).astype(float))
        print(delays)
        exp = T2StarExperiment(qubit, delays, nosc=1)
        circs, xdata, omega = exp.circuits()
        self.assertEqual(len(circs), 74)
        self.assertEqual(omega, (1. / 45.))
        print("xdata = "+str(xdata))
        print("omega = " + str(omega))
        p0, bounds = exp.T2Star_default_params(T2star=t2star, osc_freq=omega)
        print(p0)
        print(bounds)
        #self.assertEqual(p0, [0.5, 25, 0.022222222222222223, 0, 0.5])
        #self.assertEqual(bounds, ([-0.5, 1.5], [0, np.inf], [0.011111111111111112, 0.03333333333333333], [0, 2 * np.pi], [-0.5, 1.5]))

    def test_t2star_run(self):
        #run backend
        dt_factor_in_microsec = 0.0002
        t2star = 25
        # Set up the circuits
        qubit = 0
        delays = np.append(
              (np.linspace(1.0, 15.0, num=15)).astype(float),
              (np.linspace(16.0, 45.0, num=59)).astype(float))
        exp = T2StarExperiment(qubit, delays, nosc=1)
        circs, xdata, omega = exp.circuits()
        p0, bounds = exp.T2Star_default_params(T2star=t2star, osc_freq=omega)
        backend = T2Backend(
            [t2star],
            initial_prob1=[0.02],
            readout0to1=[0.02],
            readout1to0=[0.02],
            dt_factor_in_microsec=dt_factor_in_microsec,
        )

        # dummy numbers to avoid exception triggering
        instruction_durations = [
            ("measure", [0], 3 / dt_factor_in_microsec, "dt"),
            ("x", [0], 3 / dt_factor_in_microsec, "dt"),
        ]
        circs, xdata, omega = exp.circuits(backend=backend)
        #run circuit
        exp.T2Star_default_params(T2star=t2star, osc_freq=omega)

        res = exp.run(
                backend = backend,
                A_guess = 1.,
                T2star_guess=25,
                osc_guess=0.02,
                phi_guess=0,
                B_guess = 0.5,
                shots=1000
            )
        #data = exp.run(backend, noise_model=noise_model,
         #              fit_p0=p0, fit_bounds=bounds,
        #               instruction_durations=instruction_durations)


if __name__ == '__main__':
    unittest.main()
