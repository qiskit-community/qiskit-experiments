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
        Run a complete Tphi experiment on a fake Tphi backend
        """
        delays_t1 = list(range(1, 40, 3))
        delays_t2 = list(range(1, 51, 2))
        exp = Tphi(qubit=0, delays_t1=delays_t1, delays_t2=delays_t2, osc_freq=0.1)

        t1 = 20
        t2ramsey = 25
        backend = TphiBackend(t1=t1, t2ramsey=t2ramsey, freq=0.1)
        expdata = exp.run(backend=backend)
        self.assertExperimentDone(expdata)
        self.assertRoundTripSerializable(expdata, check_func=self.experiment_data_equiv)
        self.assertRoundTripPickle(expdata, check_func=self.experiment_data_equiv)
        result = expdata.analysis_results("T_phi")
        estimated_tphi = 1 / ((1 / t2ramsey) - (1 / (2 * t1)))
        self.assertAlmostEqual(
            result.value.nominal_value,
            estimated_tphi,
            delta=TestTphi.__tolerance__ * result.value.nominal_value,
        )
        self.assertEqual(result.quality, "good", "Result quality bad")

    def test_tphi_with_changing_params(self):
        """
        Run Tphi experiment, then set new delay values in set_experiment_options, and check
        that the new experiment has the correct delay values.
        """
        delays_t1 = list(range(1, 40, 3))
        delays_t2 = list(range(1, 50, 2))
        exp = Tphi(qubit=0, delays_t1=delays_t1, delays_t2=delays_t2, osc_freq=0.1)

        t1 = 20
        t2ramsey = 25
        backend = TphiBackend(t1=t1, t2ramsey=t2ramsey, freq=0.1)
        expdata = exp.run(backend=backend)
        self.assertExperimentDone(expdata)

        # Extract x values from metadata
        x_values_t1 = []
        x_values_t2 = []
        for datum in expdata.data():
            comp_meta = datum["metadata"]["composite_metadata"][0]
            if comp_meta["experiment_type"] == "T1":
                x_values_t1.append(comp_meta["xval"])
            else:
                x_values_t2.append(comp_meta["xval"])
        self.assertListEqual(x_values_t1, delays_t1, "Incorrect delays_t1")
        self.assertListEqual(x_values_t2, delays_t2, "Incorrect delays_t2")

        new_delays_t1 = list(range(1, 45, 3))
        new_delays_t2 = list(range(1, 55, 2))
        new_osc_freq = 0.2

        exp.set_experiment_options(
            delays_t1=new_delays_t1, delays_t2=new_delays_t2, osc_freq=new_osc_freq
        )
        expdata = exp.run(backend=backend)
        self.assertExperimentDone(expdata)

        # Extract x values from metadata
        x_values_t1 = []
        x_values_t2 = []
        new_freq_t2 = None
        for datum in expdata.data():
            comp_meta = datum["metadata"]["composite_metadata"][0]
            if comp_meta["experiment_type"] == "T1":
                x_values_t1.append(comp_meta["xval"])
            else:
                x_values_t2.append(comp_meta["xval"])
                if new_freq_t2 is None:
                    new_freq_t2 = comp_meta["osc_freq"]
        self.assertListEqual(x_values_t1, new_delays_t1, "Incorrect delays_t1")
        self.assertListEqual(x_values_t2, new_delays_t2, "Incorrect delays_t2")
        self.assertEqual(new_freq_t2, new_osc_freq, "Option osc_freq not set correctly")

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = Tphi(0, [1], [2], 3)
        self.assertRoundTripSerializable(exp, self.json_equiv)

    def test_analysis_config(self):
        """Test converting analysis to and from config works"""
        analysis = TphiAnalysis()
        loaded_analysis = analysis.from_config(analysis.config())
        self.assertNotEqual(analysis, loaded_analysis)
        self.assertEqual(analysis.config(), loaded_analysis.config())
