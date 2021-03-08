# -*- coding: utf-8 -*-

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
Test T1 experiment
"""

import unittest
import numpy as np
from qiskit.providers import BaseBackend
from qiskit.providers.models import BackendConfiguration
from qiskit.result import Result
from qiskit_experiments.composite import ParallelExperiment
from qiskit_experiments.characterization import T1Experiment


class T1Backend(BaseBackend):
    """
    A simple and primitive backend, to be run by the T1 tests
    """

    def __init__(self, t1,
                 initial_prob1=None,
                 readout0to1=None,
                 readout1to0=None):
        """
        Initialize the T1 backend
        """

        configuration = BackendConfiguration(
            't1_simulator', '0', int(1e6),
            ['barrier', 'x', 'delay', 'measure'],
            [], True, True, False, False, False,
            int(1e6), None)

        self._t1 = t1
        self._initial_prob1 = initial_prob1
        self._readout0to1 = readout0to1
        self._readout1to0 = readout1to0
        super().__init__(configuration)

    def run(self, qobj, **kwargs):
        """
        Run the T1 backend
        """

        shots = qobj.config.shots

        result = {
            'backend_name': 'T1 backend',
            'backend_version': '0',
            'qobj_id': 0,
            'job_id': 0,
            'success': True,
            'results': []
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
                        prob1[qubit] = 1 - prob1[qubit]
                    if op.name == "delay":
                        if self._t1[qubit] is not None:
                            prob1[qubit] = prob1[qubit] * np.exp(-op.params[0] / self._t1[qubit])
                    if op.name == "measure":
                        meas_res = np.random.binomial(1, prob1[qubit] * (1 - ro10[qubit]) + (1 - prob1[qubit]) * ro01[qubit])
                        clbits[op.memory[0]] = meas_res
                        prob1[qubit] = meas_res

                clstr = ''
                for clbit in clbits[::-1]:
                    clstr = clstr + str(clbit)

                if clstr in counts:
                    counts[clstr] += 1
                else:
                    counts[clstr] = 1

            result['results'].append({'shots': shots,
                                      'success': True,
                                      'header': {'metadata': circ.header.metadata},
                                      'data': {'counts': counts}})
            
        return Result.from_dict(result)        
        

class TestT1(unittest.TestCase):
    """
    Test measurement of T1
    """

    def test_t1(self):
        """
        Test T1 experiment using a simulator.
        """

        t1 = 25

        delays = list(range(1, 33, 6))
        p0 = [1, t1, 0]
        bounds = ([0, 0, -1], [2, 40, 1])

        # dummy numbers to avoid exception triggerring
        instruction_durations = [("measure", [0], 3, "dt"), ("x", [0], 3, "dt")]

        exp = T1Experiment(0, delays)
        res = exp.run(
            T1Backend([t1], initial_prob1=[0.1],
                      readout0to1=[0.1],
                      readout1to0=[0.1]),
            p0=p0, bounds=bounds,
            instruction_durations=instruction_durations,
            shots=10000
            )

        self.assertAlmostEqual(res.analysis_result(0)['value'], t1, delta=3)

    def test_t1_parallel(self):
        """
        Test parallel experiments of T1 using a simulator.
        """

        t1 = [25, 15]

        delays = list(range(1, 33, 6))
        p0 = [1, t1[0], 0]
        bounds = ([0, 0, -1], [2, 40, 1])

        # dummy numbers to avoid exception triggerring
        instruction_durations = [("measure", [0], 3, "dt"), ("x", [0], 3, "dt")]

        exp0 = T1Experiment(0, delays)
        exp2 = T1Experiment(2, delays)
        par_exp = ParallelExperiment([exp0, exp2])
        res = par_exp.run(
            T1Backend([t1[0], None, t1[1]]),
            p0=p0, bounds=bounds,
            instruction_durations=instruction_durations,
            shots=10000
            )

        for i in range(2):
            self.assertAlmostEqual(
                res.component_experiment_data(i).analysis_result(0)['value'],
                t1[i], delta=3
                )


if __name__ == "__main__":
    unittest.main()
