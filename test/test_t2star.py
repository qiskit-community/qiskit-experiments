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
import random
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
            self, p0=None, initial_prob_plus=None,
            readout0to1=None, readout1to0=None, dt_factor=1
    ):
        """
        Initialize the T2star backend
        """

        configuration = QasmBackendConfiguration(
            backend_name="t2star_simulator",
            backend_version="0",
            n_qubits=int(1e6),
            basis_gates=["barrier", "h", "p", "delay", "measure"],
            gates=[],
            local=True,
            simulator=True,
            conditional=False,
            open_pulse=False,
            memory=False,
            max_shots=int(1e6),
            coupling_map=None,
            dt=dt_factor
        )

        self._t2star = p0['t2star']
        self._A_guess = p0['A_guess']
        self._f_guess = p0['f_guess']
        self._phi_guess = p0['phi_guess']
        self._B_guess = p0['B_guess']
        self._initial_prob_plus = initial_prob_plus
        self._readout0to1 = readout0to1
        self._readout1to0 = readout1to0
        self._dt_factor = dt_factor
        super().__init__(configuration)

    # pylint: disable = arguments-differ
    def run(self, qobj, **kwargs):
        """
        Run the T2star backend
        """
        shots = qobj.config.shots
        result = {
            "backend_name": "T2star backend",
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
                if self._initial_prob_plus is None:
                    prob_plus = np.ones(nqubits)
                else:
                    prob_plus = self._initial_prob_plus.copy()

                clbits = np.zeros(circ.config.memory_slots, dtype=int)
                for op in circ.instructions:
                    qubit = op.qubits[0]

                    if op.name == "delay":
                        #print("op.params[0] = " + str(op.params[0]))
                        delay = op.params[0]
                        prob_plus[qubit] = \
                            self._A_guess[qubit] * np.exp(-delay / self._t2star[qubit]) * \
                            np.cos(2 * np.pi * self._f_guess[qubit] * delay + self._phi_guess[qubit]) + self._B_guess[qubit]
                        
                    if op.name == "measure":
                        # measure in |+> basis
                        meas_res = np.random.binomial(
                            1, prob_plus[qubit] * (1 - ro10[qubit]) + (1 - prob_plus[qubit]) * ro01[qubit]
                        )
                        clbits[op.memory[0]] = meas_res

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
                    "header": {"metadata":circ.header.metadata},
                    "data": {"counts": counts},
                }
            )
        return Result.from_dict(result)


class TestT2Star(QiskitTestCase):
    """ Test T2Star experiment"""
    def atest_t2star_generate_circuits(self):
        """
        Test T2Star experiment using a simulator.
        Currently only verifies that there is no exception,
        but does not verify accuracy of the estimate.
        """
        t2star = 10
        estimated_freq = 5 / 45

        # Set up the circuits
        qubit = 0
        delays = np.append(
            (np.linspace(1.0, 15.0, num=15)).astype(float),
            (np.linspace(16.0, 45.0, num=59)).astype(float))

        exp = T2StarExperiment(qubit, delays, osc_freq=estimated_freq, unit='us')
        circs = exp.circuits()
        self.assertEqual(len(circs), 74)
        #p0, bounds = exp.T2Star_default_params(T2star=t2star, osc_freq=exp._osc_freq)
        print(p0)
        print(bounds)
        self.assertEqual(list(p0.values()), [0.5, t2star, estimated_freq, 0.0, 0.5])
        self.assertEqual(bounds, ([-0.5, 0, 0.5 * estimated_freq, 0, -0.5], [1.5, np.inf, 1.5 * estimated_freq,  2 * np.pi, 1.5]))

    def test_t2star_run(self):
        #run backend
        dt_factor = 1
        estimated_t2star = 20
        estimated_freq = 0.1
        # Set up the circuits
        qubit = 0
        delays = np.append(
              (np.linspace(1.0, 15.0, num=15)).astype(float),
              (np.linspace(16.0, 45.0, num=59)).astype(float))
        exp = T2StarExperiment(qubit, delays)
        circs = exp.circuits()
        
        backend = T2Backend(
            p0 = {'A_guess':[0.5], 't2star':[estimated_t2star],
                  'f_guess':[estimated_freq], 'phi_guess':[-np.pi/20], 'B_guess':[0.5]},
            initial_prob_plus = [0.0],
            readout0to1=[0.02],
            readout1to0=[0.02],
            dt_factor=dt_factor,
        )

        exp.circuits(backend=backend)
        #t2star = 10
        #p0, bounds = exp.T2Star_default_params(t2star=t2star, osc_freq=5 / 45)
        #p0 = [A=None, t2star=10, osc_freq=None, phi=None, B=None]
        #run circuit
        result = exp.run(
                backend = backend,
                p0=None,
                bounds=None,
                #plot=False,
                shots=2000
            )
        #self.assertEqual(result["quality"], "computer_good")
        t2star_res = result._analysis_results[0]['popt'][1]
        frequency_res = result._analysis_results[0]['popt'][2]
        print("result t2star = " + str(t2star_res))
        print("result freq = " + str(frequency_res))
        self.assertAlmostEqual(t2star_res, estimated_t2star, delta=1)
        self.assertAlmostEqual(frequency_res, estimated_freq, delta=0.01)



if __name__ == '__main__':
    unittest.main()

