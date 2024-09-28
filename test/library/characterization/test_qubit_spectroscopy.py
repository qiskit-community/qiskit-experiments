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

"""Spectroscopy tests."""
from test.base import QiskitExperimentsTestCase
import numpy as np

from qiskit.qobj.utils import MeasLevel
from qiskit.circuit.library import XGate
from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2
from qiskit_experiments.framework import ParallelExperiment

from qiskit_experiments.framework import BackendData
from qiskit_experiments.library import QubitSpectroscopy, EFSpectroscopy
from qiskit_experiments.test.mock_iq_backend import MockIQBackend, MockIQParallelBackend
from qiskit_experiments.test.mock_iq_helpers import MockIQSpectroscopyHelper as SpectroscopyHelper
from qiskit_experiments.test.mock_iq_helpers import (
    MockIQParallelExperimentHelper as ParallelExperimentHelper,
)


class TestQubitSpectroscopy(QiskitExperimentsTestCase):
    """Test spectroscopy experiment."""

    def test_spectroscopy_end2end_classified(self):
        """End to end test of the spectroscopy experiment."""

        exp_helper = SpectroscopyHelper(
            line_width=2e6,
            iq_cluster_centers=[((-1.0, -1.0), (1.0, 1.0))],
            iq_cluster_width=[0.2],
        )
        backend = MockIQBackend(
            experiment_helper=exp_helper,
        )
        backend.target.add_instruction(XGate(), properties={(0,): None})

        qubit = 1
        freq01 = BackendData(backend).drive_freqs[qubit]
        frequencies = np.linspace(freq01 - 10.0e6, freq01 + 10.0e6, 21)

        spec = QubitSpectroscopy([qubit], frequencies)
        spec.set_run_options(meas_level=MeasLevel.CLASSIFIED)
        expdata = spec.run(backend)
        self.assertExperimentDone(expdata)
        result = expdata.analysis_results("f01")
        self.assertRoundTripSerializable(result.value)

        self.assertAlmostEqual(result.value.n, freq01, delta=1e6)
        self.assertEqual(result.quality, "good")
        self.assertEqual(str(result.device_components[0]), f"Q{qubit}")

        # Test if we find still find the peak when it is shifted by 5 MHz.
        exp_helper.freq_offset = 5.0e6
        spec = QubitSpectroscopy([qubit], frequencies)
        spec.set_run_options(meas_level=MeasLevel.CLASSIFIED)
        expdata = spec.run(backend)
        self.assertExperimentDone(expdata)
        result = expdata.analysis_results("f01")
        self.assertRoundTripSerializable(result.value)

        self.assertAlmostEqual(result.value.n, freq01 + 5e6, delta=1e6)
        self.assertEqual(result.quality, "good")

    def test_spectroscopy_end2end_kerneled(self):
        """End to end test of the spectroscopy experiment on IQ data."""

        exp_helper = SpectroscopyHelper(
            line_width=2e6,
            iq_cluster_centers=[((-1.0, -1.0), (1.0, 1.0))],
            iq_cluster_width=[0.2],
        )
        backend = MockIQBackend(
            experiment_helper=exp_helper,
        )
        backend.target.add_instruction(XGate(), properties={(0,): None})

        qubit = 0
        freq01 = BackendData(backend).drive_freqs[qubit]
        frequencies = np.linspace(freq01 - 10.0e6, freq01 + 10.0e6, 21)

        spec = QubitSpectroscopy([qubit], frequencies)
        expdata = spec.run(backend)
        self.assertExperimentDone(expdata)
        result = expdata.analysis_results("f01")
        self.assertRoundTripSerializable(result.value)

        self.assertTrue(freq01 - 2e6 < result.value.n < freq01 + 2e6)
        self.assertEqual(result.quality, "good")

        exp_helper.freq_offset = 5.0e6
        exp_helper.iq_cluster_centers = [((1.0, 1.0), (-1.0, -1.0))]

        spec = QubitSpectroscopy([qubit], frequencies)
        expdata = spec.run(backend)
        self.assertExperimentDone(expdata)
        result = expdata.analysis_results("f01")
        self.assertRoundTripSerializable(result.value)

        self.assertTrue(freq01 + 3e6 < result.value.n < freq01 + 8e6)
        self.assertEqual(result.quality, "good")

        spec.set_run_options(meas_return="avg")
        expdata = spec.run(backend)
        self.assertExperimentDone(expdata)
        result = expdata.analysis_results("f01")
        self.assertRoundTripSerializable(result.value)

        self.assertTrue(freq01 + 3e6 < result.value.n < freq01 + 8e6)
        self.assertEqual(result.quality, "good")

    def test_spectroscopy12_end2end_classified(self):
        """End to end test of the spectroscopy experiment with an x pulse."""

        backend = MockIQBackend(
            experiment_helper=SpectroscopyHelper(
                line_width=2e6,
                iq_cluster_centers=[((-1.0, -1.0), (1.0, 1.0))],
                iq_cluster_width=[0.2],
            ),
        )
        backend.target.add_instruction(XGate(), properties={(0,): None})
        qubit = 0
        freq01 = BackendData(backend).drive_freqs[qubit]
        frequencies = np.linspace(freq01 - 10.0e6, freq01 + 10.0e6, 21)

        # Note that the backend is not sophisticated enough to simulate an e-f
        # transition so we run the test with g-e.
        spec = EFSpectroscopy([qubit], frequencies)
        spec.backend = backend
        spec.set_run_options(meas_level=MeasLevel.CLASSIFIED)
        expdata = spec.run(backend)
        self.assertExperimentDone(expdata)
        result = expdata.analysis_results("f12")
        self.assertRoundTripSerializable(result.value)

        self.assertTrue(freq01 - 2e6 < result.value.n < freq01 + 2e6)
        self.assertEqual(result.quality, "good")

        # Test the circuits
        circ = spec.circuits()[0]
        self.assertEqual(circ.data[0].operation.name, "x")
        self.assertEqual(circ.data[1].operation.name, "Spec")

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = QubitSpectroscopy([1], np.linspace(100, 150, 20) * 1e6)
        loaded_exp = QubitSpectroscopy.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertEqualExtended(exp, loaded_exp)

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = QubitSpectroscopy([1], np.linspace(int(100e6), int(150e6), 4))
        # Checking serialization of the experiment
        self.assertRoundTripSerializable(exp)

    def test_expdata_serialization(self):
        """Test experiment data and analysis data JSON serialization"""
        exp_helper = SpectroscopyHelper(
            line_width=2e6,
            iq_cluster_centers=[((-1.0, -1.0), (1.0, 1.0))],
            iq_cluster_width=[0.2],
        )
        backend = MockIQBackend(
            experiment_helper=exp_helper,
        )
        backend.target.add_instruction(XGate(), properties={(0,): None})

        qubit = 1
        freq01 = BackendData(backend).drive_freqs[qubit]
        frequencies = np.linspace(freq01 - 10.0e6, freq01 + 10.0e6, 21)
        exp = QubitSpectroscopy([qubit], frequencies)

        exp.set_run_options(meas_level=MeasLevel.CLASSIFIED, shots=1024)
        expdata = exp.run(backend)
        self.assertExperimentDone(expdata)

        # Checking serialization of the experiment data obj
        self.assertRoundTripSerializable(expdata)

        # Checking serialization of the analysis
        self.assertRoundTripSerializable(expdata.analysis_results("f01"))

    def test_kerneled_expdata_serialization(self):
        """Test experiment data and analysis data JSON serialization"""
        exp_helper = SpectroscopyHelper(
            line_width=2e6,
            iq_cluster_centers=[((-1.0, -1.0), (1.0, 1.0))],
            iq_cluster_width=[0.2],
        )
        backend = MockIQBackend(
            experiment_helper=exp_helper,
        )
        backend.target.add_instruction(XGate(), properties={(0,): None})

        qubit = 1
        freq01 = BackendData(backend).drive_freqs[qubit]
        frequencies = np.linspace(freq01 - 10.0e6, freq01 + 10.0e6, 21)
        exp = QubitSpectroscopy([qubit], frequencies)

        exp.set_run_options(meas_level=MeasLevel.KERNELED, shots=1024)
        expdata = exp.run(backend)
        self.assertExperimentDone(expdata)

        # Checking serialization of the experiment data
        self.assertRoundTripSerializable(expdata)

        # Checking serialization of the analysis
        self.assertRoundTripSerializable(expdata.analysis_results("f01"))

    def test_parallel_experiment(self):
        """Test for parallel experiment"""
        # backend initialization
        iq_cluster_centers = [
            ((-1.0, 0.0), (1.0, 0.0)),
            ((0.0, -1.0), (0.0, 1.0)),
            ((3.0, 0.0), (5.0, 0.0)),
        ]

        parallel_backend = MockIQParallelBackend(
            experiment_helper=None,
            rng_seed=0,
        )
        parallel_backend.target.add_instruction(
            XGate(),
            properties={(0,): None, (1,): None},
        )

        # experiment hyper parameters
        qubit1 = 0
        qubit2 = 1
        backend_data = BackendData(parallel_backend)
        freq01 = backend_data.drive_freqs[qubit1]
        freq02 = backend_data.drive_freqs[qubit2]

        # experiments initialization
        frequencies1 = np.linspace(freq01 - 10.0e6, freq01 + 10.0e6, 23)
        frequencies2 = np.linspace(freq02 - 10.0e6, freq02 + 10.0e6, 21)

        exp_list = [
            QubitSpectroscopy(
                [qubit1],
                frequencies1,
            ),
            QubitSpectroscopy(
                [qubit2],
                frequencies2,
            ),
        ]

        exp_helper_list = [
            SpectroscopyHelper(iq_cluster_centers=iq_cluster_centers),
            SpectroscopyHelper(iq_cluster_centers=iq_cluster_centers),
        ]
        parallel_helper = ParallelExperimentHelper(exp_list, exp_helper_list)

        parallel_backend.experiment_helper = parallel_helper

        # initializing parallel experiment
        par_experiment = ParallelExperiment(
            exp_list, flatten_results=False, backend=parallel_backend
        )
        par_experiment.set_run_options(
            meas_level=MeasLevel.KERNELED, meas_return="single", shots=20
        )

        par_data = par_experiment.run()
        self.assertExperimentDone(par_data)

        # since under _experiment in kwargs there is an argument of the backend which isn't serializable.
        par_data._experiment = None
        # Checking serialization of the experiment data
        self.assertRoundTripSerializable(par_data)

        for child_data in par_data.child_data():
            self.assertRoundTripSerializable(child_data)

    def test_circuit_roundtrip_serializable(self):
        """Test circuits round trip JSON serialization"""
        backend = FakeWashingtonV2()
        qubit = 1
        freq01 = BackendData(backend).drive_freqs[qubit]
        frequencies = np.linspace(freq01 - 10.0e6, freq01 + 10.0e6, 3)
        exp = QubitSpectroscopy([1], frequencies, backend=backend)
        # Checking serialization of the experiment
        self.assertRoundTripSerializable(exp._transpiled_circuits())
