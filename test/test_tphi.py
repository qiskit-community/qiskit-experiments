# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Test T2Ramsey experiment
"""

from test.base import QiskitExperimentsTestCase
from qiskit_experiments.library import Tphi
from qiskit_experiments.test.tphi_backend import TphiBackend
from qiskit_experiments.library.characterization.analysis.tphi_analysis import TphiAnalysis


class TestTphi(QiskitExperimentsTestCase):
    """Test Tphi experiment"""

    __tolerance__ = 0.1

    def test_tphi_end_to_end(self):
        """
        Run the Tphi backend
        """
        delays_t1 = list(range(1, 40, 3))
        delays_t2 = list(range(1, 51, 2))
        exp = Tphi(qubit=0, delays_t1=delays_t1, delays_t2=delays_t2, osc_freq=0.1)

        t1 = 20
        t2ramsey = 25
        backend = TphiBackend(t1=t1, t2ramsey=t2ramsey, freq=0.1)

        expdata = exp.run(backend=backend, analysis=TphiAnalysis())
        self.assertExperimentDone(expdata)
        result = expdata.analysis_results("T_phi")
        estimated_tphi = 1 / ((1 / t2ramsey) - (1 / (2 * t1)))
        self.assertAlmostEqual(
            result.value.value,
            estimated_tphi,
            delta=TestTphi.__tolerance__ * result.value.value,
        )
        self.assertEqual(result.quality, "good", "Result quality bad")

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = Tphi(0, [1], [2], 3)
        self.assertRoundTripSerializable(exp, self.json_equiv)

    def test_analysis_config(self):
        """ "Test converting analysis to and from config works"""
        analysis = TphiAnalysis()
        loaded = TphiAnalysis.from_config(analysis.config())
        self.assertNotEqual(analysis, loaded)
        self.assertEqual(analysis.config(), loaded.config())
