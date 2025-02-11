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
Test Tphi experiment.
"""
from test.base import QiskitExperimentsTestCase
from qiskit.exceptions import QiskitError
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_experiments.library import Tphi, T2Hahn, T2Ramsey
from qiskit_experiments.test.noisy_delay_aer_simulator import NoisyDelayAerBackend
from qiskit_experiments.library.characterization.analysis import (
    TphiAnalysis,
    T2RamseyAnalysis,
    T2HahnAnalysis,
)


class TestTphi(QiskitExperimentsTestCase):
    """Test Tphi experiment."""

    __tolerance__ = 0.1

    def test_tphi_ramsey_end_to_end(self):
        """
        Run a complete Tphi experiment with T2ramsey on a fake Tphi backend.
        """
        delays_t1 = list(range(1, 40, 3))
        delays_t2 = list(range(1, 51, 2))
        exp = Tphi(
            physical_qubits=[0],
            delays_t1=delays_t1,
            delays_t2=delays_t2,
            t2type="ramsey",
            osc_freq=0.1,
        )

        t1 = 20
        t2ramsey = 25
        backend = NoisyDelayAerBackend([t1], [t2ramsey])
        expdata = exp.run(backend=backend, seed_simulator=1)
        self.assertExperimentDone(expdata)
        self.assertRoundTripSerializable(expdata)
        self.assertRoundTripPickle(expdata)
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
        exp = Tphi(
            physical_qubits=[0],
            delays_t1=delays_t1,
            delays_t2=delays_t2,
            t2type="ramsey",
            osc_freq=0.1,
        )

        t1 = 20
        t2ramsey = 25
        backend = NoisyDelayAerBackend([t1], [t2ramsey])
        expdata = exp.run(backend=backend, seed_simulator=1)
        self.assertExperimentDone(expdata)

        # Extract x values from metadata
        x_values_t1 = []
        x_values_t2 = []
        for datum in expdata.data():
            metadata = datum["metadata"]
            xval = metadata["composite_metadata"][0]["xval"]
            if metadata["composite_index"][0] == 0:
                x_values_t1.append(xval)
            else:
                x_values_t2.append(xval)
        self.assertListEqual(x_values_t1, delays_t1, "Incorrect delays_t1")
        self.assertListEqual(x_values_t2, delays_t2, "Incorrect delays_t2")

        new_delays_t1 = list(range(1, 45, 3))
        new_delays_t2 = list(range(1, 55, 2))
        new_osc_freq = 0.2

        exp.set_experiment_options(
            delays_t1=new_delays_t1, delays_t2=new_delays_t2, osc_freq=new_osc_freq
        )
        expdata = exp.run(backend=backend, seed_simulator=1)
        self.assertExperimentDone(expdata)

        # Extract x values from metadata
        x_values_t1 = []
        x_values_t2 = []
        new_freq_t2 = expdata.metadata["component_metadata"][1]["osc_freq"]
        for datum in expdata.data():
            metadata = datum["metadata"]
            xval = metadata["composite_metadata"][0]["xval"]
            if metadata["composite_index"][0] == 0:
                x_values_t1.append(xval)
            else:
                x_values_t2.append(xval)
        self.assertListEqual(x_values_t1, new_delays_t1, "Incorrect delays_t1")
        self.assertListEqual(x_values_t2, new_delays_t2, "Incorrect delays_t2")
        self.assertEqual(new_freq_t2, new_osc_freq, "Option osc_freq not set correctly")

    def test_tphi_t2_option(self):
        """Test that Tphi switches between T2Ramsey and T2Hahn correctly."""

        delays_t1 = list(range(1, 40, 3))
        delays_t2 = list(range(1, 50, 2))

        exp = Tphi(physical_qubits=[0], delays_t1=delays_t1, delays_t2=delays_t2, t2type="ramsey")
        self.assertTrue(isinstance(exp.component_experiment(1), T2Ramsey))
        self.assertTrue(isinstance(exp.analysis.component_analysis(1), T2RamseyAnalysis))
        with self.assertRaises(QiskitError):  # T2Ramsey should not allow a T2Hahn option
            exp.set_experiment_options(num_echoes=1)

        exp = Tphi(physical_qubits=[0], delays_t1=delays_t1, delays_t2=delays_t2)
        self.assertTrue(isinstance(exp.component_experiment(1), T2Hahn))
        self.assertTrue(isinstance(exp.analysis.component_analysis(1), T2HahnAnalysis))
        with self.assertRaises(QiskitError):  # T2Hahn should not allow a T2ramsey option
            exp.set_experiment_options(osc_freq=0.0)

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = Tphi([0], [1], [2])
        self.assertRoundTripSerializable(exp)
        exp = Tphi([0], [1], [2], "hahn", 3)
        self.assertRoundTripSerializable(exp)
        exp = Tphi([0], [1], [2], "ramsey", 0)
        self.assertRoundTripSerializable(exp)

    def test_circuits_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        backend = GenericBackendV2(num_qubits=2)
        exp = Tphi([0], [1e-6], [2e-6], backend=backend)
        self.assertRoundTripSerializable(exp._transpiled_circuits())
        exp = Tphi([0], [1e-6], [2e-6], "hahn", 3, backend=backend)
        self.assertRoundTripSerializable(exp._transpiled_circuits())
        exp = Tphi([0], [1e-6], [2e-6], "ramsey", 0, backend=backend)
        self.assertRoundTripSerializable(exp._transpiled_circuits())

    def test_analysis_config(self):
        """Test converting analysis to and from config works"""
        analysis = TphiAnalysis()
        loaded_analysis = analysis.from_config(analysis.config())
        self.assertNotEqual(analysis, loaded_analysis)
        self.assertEqual(analysis.config(), loaded_analysis.config())
