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
Test T2Hahn experiment
"""
import numpy as np

from qiskit.utils import apply_prefix
from qiskit.test import QiskitTestCase
from qiskit_experiments.library.characterization.t2hahn import T2Hahn
from qiskit_experiments.test.t2hahn_backend import T2HahnBackend
import unittest


class TestT2Hahn(QiskitTestCase):
    """Test T2Hahn experiment"""

    __tolerance__ = 0.1

    def test_t2hahn_run_end2end(self):
        """
        Run the T2Hahn backend on all possible units
        """
        for unit in ["s"]:
            if unit in ("s", "dt"):
                dt_factor = 1
            else:
                dt_factor = apply_prefix(1, unit)
            osc_freq = 0.1 / dt_factor
            estimated_t2hahn = 20
            # Set up the circuits
            qubit = 0
            if unit == "dt":  # dt requires integer values for delay
                delays = list(range(1, 46))
            else:
                delays = np.append(
                    (np.linspace(1.0, 15.0, num=15)).astype(float),
                    (np.linspace(16.0, 45.0, num=59)).astype(float),
                )
            exp = T2Hahn(qubit, delays, unit=unit)
            default_p0 = {
                "A": 0.5,
                "T2": estimated_t2hahn,
                "B": 0.5,
            }
            backend = T2HahnBackend(
                t2hahn=[estimated_t2hahn],
                frequency=[osc_freq],
                initialization_error=[0.0],
                readout0to1=[0.02],
                readout1to0=[0.02],
                conversion_factor=dt_factor,
            )

            for user_p0 in [default_p0, dict()]:
                # exp.set_analysis_options(user_p0=user_p0, plot=True)
                exp.set_analysis_options(p0={"amp": 0.5, "tau": estimated_t2hahn / dt_factor, "base": 0.5}, plot=True)
                expdata = exp.run(backend=backend, shots=1000)
                expdata.block_for_results()  # Wait for job/analysis to finish.
                result = expdata.analysis_results("T2")
                fitval = result.value
                self.assertEqual(result.quality, "good")
                self.assertAlmostEqual(fitval.value, estimated_t2hahn, delta=3)
                self.assertEqual(fitval.unit, "s")

    def test_t2hahn_concat_2_experiments(self):
        """
        Concatenate the data from 2 separate experiments
        """
        unit = "s"
        estimated_t2hahn = 30
        # First experiment
        qubit = 0
        delays0 = list(range(1, 60, 2))
        osc_freq = 0.08
        dt_factor = 1

        exp0 = T2Hahn(qubit, delays0, unit=unit)
        exp0.set_analysis_options(p0={"amp": 0.5, "tau": estimated_t2hahn / dt_factor, "base": 0.5}, plot=True)
        backend = T2HahnBackend(
            t2hahn=[estimated_t2hahn],
            frequency=[osc_freq],
            initialization_error=[0.0],
            readout0to1=[0.02],
            readout1to0=[0.02],
            conversion_factor=1,
        )

        # run circuits
        expdata0 = exp0.run(backend=backend, shots=1000).block_for_results()
        expdata0.block_for_results()

        res_t2_0 = expdata0.analysis_results('T2')

        # second experiment
        delays1 = list(range(2, 65, 2))
        exp1 = T2Hahn(qubit, delays1, unit=unit)
        exp1.set_analysis_options(p0={"amp": 0.5, "tau": estimated_t2hahn / dt_factor, "base": 0.5}, plot=True)
        expdata1 = exp1.run(backend=backend, analysis=False, shots=1000).block_for_results()
        expdata1.add_data(expdata0.data())
        exp1.run_analysis(expdata1).block_for_results()

        res_t2_1 = expdata1.analysis_results("T2")

        fitval = res_t2_1.value
        self.assertEqual(res_t2_1.quality, "good")
        self.assertAlmostEqual(res_t2_1.value.value, estimated_t2hahn, delta=3)
        self.assertEqual(fitval.unit, "s")

        self.assertAlmostEqual(
            res_t2_1.value.value,
            estimated_t2hahn,
            delta=TestT2Hahn.__tolerance__ * res_t2_1.value.value,
        )

        self.assertLessEqual(res_t2_1.value.stderr, res_t2_0.value.stderr)
        self.assertEqual(len(expdata1.data()), len(delays0) + len(delays1))

if __name__ == '__main__':
    unittest.main()