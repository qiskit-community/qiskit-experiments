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

from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise.errors.standard_errors import thermal_relaxation_error
from qiskit.providers.aer.noise import NoiseModel

from qiskit_experiments.characterization import T1Experiment

BACKEND = QasmSimulator()
INSTRUCTION_DURATIONS = [("measure", [0], 3, "dt"), ("x", [0], 3, "dt")]
GATE_TIME = 0.1

# Fix seed for simulations
SEED = 9000


class TestT1(unittest.TestCase):
    """
    Test measurement of T1
    """

    def test_t1(self):
        """
        Test T1 experiment using a simulator.
        Currently only verifies that there is no exception,
        but does not verify accuracy of the estimate.
        """

        t1 = 25

        noise_model = NoiseModel()
        noise_model.add_quantum_error(thermal_relaxation_error(t1, 2 * t1, GATE_TIME), "delay", [0])

        delays = list(range(1, 33, 6))
        p0 = [1, t1, 0]
        bounds = ([0, 0, -1], [2, 40, 1])

        exp = T1Experiment(0, delays)
        exp.run(
            BACKEND,
            noise_model=noise_model,
            fit_p0=p0,
            fit_bounds=bounds,
            instruction_durations=INSTRUCTION_DURATIONS,
        )


if __name__ == "__main__":
    unittest.main()
