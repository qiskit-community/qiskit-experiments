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
            self, t2star=None, initial_amplitude1=None, f_guess=None, phi_guess=None, B_guess=None,
            readout0to1=None, readout1to0=None, dt_factor_in_microsec=1e6
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
            dt=dt_factor_in_microsec * 1000
        )

        self._t2star = t2star
        self._f_guess = f_guess
        self._phi_guess = phi_guess
        self._B_guess = B_guess
        self._initial_amplitude1 = initial_amplitude1
        self._readout0to1 = readout0to1
        self._readout1to0 = readout1to0
        self._dt_factor_in_microsec = dt_factor_in_microsec
        super().__init__(configuration)

    # pylint: disable = arguments-differ
    def run(self, qobj, **kwargs):
        """
        Run the T1 backend
        """
        #print("in test run, qobj = " + str(qobj))
        #shots = qobj.config.shots
        shots = 1
        #print("shots = " + str(shots))
        result = {
            "backend_name": "T1 backend",
            "backend_version": "0",
            "qobj_id": 0,
            "job_id": 0,
            "success": True,
            "results": [],
        }
        print("len(qobj.experiments) = " +str(len(qobj.experiments)))
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
                if self._initial_amplitude1 is None:
                    amplitude1 = np.zeros(nqubits)
                else:
                    amplitude1= self._initial_amplitude1.copy()

                clbits = np.zeros(circ.config.memory_slots, dtype=int)
                print("len(circ.instructions) = " + str(len(circ.instructions)))
                for op in circ.instructions:
                    qubit = op.qubits[0]
                    if op.name == "h":
                        if amplitude1[qubit] == 0:
                            amplitude1[qubit] = 1 / np.sqrt(2)
                        else:
                            amplitude1[qubit] = (1 / np.sqrt(2)) * amplitude1[qubit]
                    if op.name == "p":
                        amplitude1[qubit] = amplitude1[qubit] * np.exp(complex(0.0, op.params[0].real))
                    if op.name == "delay":
                        delay = op.params[0] * self._dt_factor_in_microsec
                        print("delay = " + str(delay))
                        amplitude1[qubit] = \
                            amplitude1[qubit] * np.exp(-delay / self._t2star[qubit]) * \
                                np.cos(2 * np.pi * self._f_guess[qubit] * delay + self._phi_guess[qubit]) + self._B_guess[qubit]
                    if op.name == "measure":
                        prob1 = np.absolute(amplitude1[qubit])
                        meas_res = np.random.binomial(
                            1, prob1 * (1 - ro10[qubit]) + (1 - prob1) * ro01[qubit]
                        )
                        clbits[op.memory[0]] = meas_res

                clstr = ""
                for clbit in clbits[::-1]:
                    clstr = clstr + str(clbit)

                if clstr in counts:
                    counts[clstr] += 1
                else:
                    counts[clstr] = 1
            print(counts)
            result["results"].append(
                {
                    "shots": shots,
                    "success": True,
                    "header": {"metadata":circ.header.metadata},
                    "data": {"counts": counts},
                }
            )

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
        #print(delays)
        exp = T2StarExperiment(qubit, delays, nosc=1)
        circs = exp.circuits()
        self.assertEqual(len(circs), 74)
        p0, bounds = exp.T2Star_default_params(T2star=t2star, osc_freq=exp._nosc)
        #print(p0)
        #print(bounds)
        #self.assertEqual(p0, [0.5, 25, 0.022222222222222223, 0, 0.5])
        #self.assertEqual(bounds, ([-0.5, 1.5], [0, np.inf], [0.011111111111111112, 0.03333333333333333], [0, 2 * np.pi], [-0.5, 1.5]))

    def test_t2star_run(self):
        #run backend
        dt_factor_in_microsec = 0.0002
        t2star = 25
        f_guess = 0.02
        phi_guess = 0
        B_guess = 0.5
        # Set up the circuits
        qubit = 0
        delays = np.append(
              (np.linspace(1.0, 15.0, num=15)).astype(float),
              (np.linspace(16.0, 45.0, num=59)).astype(float))
        print(delays)
        exp = T2StarExperiment(qubit, delays, nosc=1)
        circs = exp.circuits()
        #print(circs[0])
        #print(circs[-1])
        backend = T2Backend(
            t2star=[t2star],
            initial_amplitude1=[0.0],
            f_guess=[f_guess],
            phi_guess=[phi_guess],
            B_guess=[B_guess],
            readout0to1=[0.02],
            readout1to0=[0.02],
            dt_factor_in_microsec=dt_factor_in_microsec,
        )

        # dummy numbers to avoid exception triggering
        instruction_durations = [
            ("measure", [0], 3 / dt_factor_in_microsec, "dt"),
            ("x", [0], 3 / dt_factor_in_microsec, "dt"),
        ]
        exp.circuits(backend=backend)
        p0, bounds = exp.T2Star_default_params(T2star=t2star, osc_freq=exp._nosc)

        #run circuit
        exp.T2Star_default_params(T2star=t2star, osc_freq=exp._nosc)

        res = exp.run(
                backend = backend,
                A_guess = 1.,
                T2star_guess=25,
                osc_guess=0.02,
                phi_guess=0,
                B_guess = 0.5,
                shots=1
            )
        #data = exp.run(backend, noise_model=noise_model,
         #              fit_p0=p0, fit_bounds=bounds,
        #               instruction_durations=instruction_durations)


if __name__ == '__main__':
    unittest.main()

