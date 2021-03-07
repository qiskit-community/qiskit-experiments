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
from qiskit_experiments.characterization import T1Experiment


class T1Backend(BaseBackend):
    """
    A simple and primitive backend, to be run by the T1 tests
    """

    def __init__(self, t1):
        """
        Initialize the T1 backend
        """

        configuration = BackendConfiguration(
            't1_simulator', '0', int(1e6),
            ['barrier', 'x', 'delay', 'measure'],
            [], True, True, False, False, False,
            int(1e6), None)

        self._t1 = t1
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
            counts = dict()

            for _ in range(shots):
                prob1 = np.zeros(circ.config.n_qubits)
                clbits = np.zeros(circ.config.memory_slots, dtype=int)
                           
                for op in circ.instructions:
                    qubit = op.qubits[0]
                    if op.name == "x":
                        prob1[qubit] = 1 - prob1[qubit]
                    if op.name == "delay":
                        prob1[qubit] = prob1[qubit] * np.exp(-op.params[0] / self._t1)
                    if op.name == "measure":
                        meas_res = np.random.binomial(1, prob1[qubit])
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
        instruction_durations = [("measure", [0], 3, "dt"), ("x", [0], 3, "dt")]

        exp = T1Experiment(0, delays)
        res = exp.run(
            T1Backend(t1),
            p0=p0, bounds=bounds,
            instruction_durations=instruction_durations
        )

        self.assertAlmostEqual(res.analysis_result(0)['value'], t1, delta=2)


if __name__ == "__main__":
    unittest.main()
