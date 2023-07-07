# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Stark P1 spectroscopy experiment."""

from test.base import QiskitExperimentsTestCase

import numpy as np
from qiskit import pulse
from qiskit.circuit import QuantumCircuit, Gate
from qiskit.providers import QubitProperties
from qiskit.providers.fake_provider import FakeHanoiV2

from qiskit_experiments.framework import ExperimentData, AnalysisResultData
from qiskit_experiments.library import StarkP1Spectroscopy
from qiskit_experiments.library.characterization.analysis import StarkP1SpectAnalysis
from qiskit_experiments.test import FakeService


class StarkP1SpectAnalysisReturnXvals(StarkP1SpectAnalysis):
    """A test analysis class that returns x values."""

    def _run_spect_analysis(
        self,
        xdata: np.ndarray,
        ydata: np.ndarray,
        ydata_err: np.ndarray,
    ):
        return [
            AnalysisResultData(
                name="xvals",
                value=xdata,
            )
        ]


class TestStarkP1Spectroscopy(QiskitExperimentsTestCase):
    """Test case for the Stark P1 Spectroscopy experiment."""

    def setUp(self):
        super().setUp()

        self.service = FakeService()

        self.service.create_experiment(
            experiment_type="StarkRamseyXYAmpScan",
            backend_name="fake_hanoi",
            experiment_id="123456789",
        )

        self.coeffs = {
            "stark_pos_coef_o1": 5e6,
            "stark_pos_coef_o2": 200e6,
            "stark_pos_coef_o3": -50e6,
            "stark_neg_coef_o1": 5e6,
            "stark_neg_coef_o2": -180e6,
            "stark_neg_coef_o3": -40e6,
            "stark_ferr": 100e3,
        }
        for i, (key, value) in enumerate(self.coeffs.items()):
            self.service.create_analysis_result(
                experiment_id="123456789",
                result_data={"value": value},
                result_type=key,
                device_components=["Q0"],
                tags=[],
                quality="Good",
                verified=False,
                result_id=str(i),
            )

    def test_linear_spaced_parameters(self):
        """Test generating parameters with linear spacing."""
        exp = StarkP1Spectroscopy((0,))
        exp.set_experiment_options(
            min_stark_amp=-1,
            max_stark_amp=1,
            num_stark_amps=5,
            spacing="linear",
        )
        params = exp.parameters()
        ref = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

        np.testing.assert_array_almost_equal(params, ref)

    def test_quadratic_spaced_parameters(self):
        """Test generating parameters with quadratic spacing."""
        exp = StarkP1Spectroscopy((0,))
        exp.set_experiment_options(
            min_stark_amp=-1,
            max_stark_amp=1,
            num_stark_amps=5,
            spacing="quadratic",
        )
        params = exp.parameters()
        ref = np.array([-1.0, -0.25, 0.0, 0.25, 1.0])

        np.testing.assert_array_almost_equal(params, ref)

    def test_invalid_spacing(self):
        """Test setting invalid spacing option."""
        exp = StarkP1Spectroscopy((0,))
        with self.assertRaises(ValueError):
            exp.set_experiment_options(spacing="invalid_option")

    def test_circuits(self):
        """Test generated circuits."""
        backend = FakeHanoiV2()

        # For simplicity of the test
        backend.target.dt = 1
        backend.target.granularity = 1
        backend.target.pulse_alignment = 1
        backend.target.acquire_alignment = 1
        backend.target.qubit_properties = [QubitProperties(frequency=1e9)]

        exp = StarkP1Spectroscopy((0,), backend)
        exp.set_experiment_options(
            stark_amps=[-0.5, 0.5],
            stark_freq_offset=10e6,
            t1_delay=100,
            stark_sigma=15,
            stark_risefall=2,
        )
        circs = exp.circuits()

        # amp = -0.5
        with pulse.build() as sched1:
            # Red shift: must be greater than f01
            pulse.set_frequency(1.01e9, pulse.DriveChannel(0))
            pulse.play(
                # Always positive amplitude
                pulse.GaussianSquare(100, 0.5, 15, 40),
                pulse.DriveChannel(0),
            )
        qc1 = QuantumCircuit(1, 1)
        qc1.x(0)
        qc1.append(Gate("Stark", 1, [-0.5]), [0])
        qc1.measure(0, 0)
        qc1.add_calibration("Stark", (0,), sched1, [-0.5])

        # amp = +0.5
        with pulse.build() as sched2:
            # Blue shift: Must be lower than f01
            pulse.set_frequency(0.99e9, pulse.DriveChannel(0))
            pulse.play(
                # Always positive amplitude
                pulse.GaussianSquare(100, 0.5, 15, 40),
                pulse.DriveChannel(0),
            )
        qc2 = QuantumCircuit(1, 1)
        qc2.x(0)
        qc2.append(Gate("Stark", 1, [0.5]), [0])
        qc2.measure(0, 0)
        qc2.add_calibration("Stark", (0,), sched2, [0.5])

        self.assertEqual(circs[0], qc1)
        self.assertEqual(circs[1], qc2)

    def test_retrieve_coefficients(self):
        """Test retrieving Stark coefficients from the experiment service."""
        retrieved_coeffs = StarkP1SpectAnalysis.retrieve_coefficients_from_service(
            service=self.service,
            qubit=0,
            backend="fake_hanoi",
        )
        self.assertDictEqual(
            retrieved_coeffs,
            self.coeffs,
        )

    def test_running_analysis_without_service(self):
        """Test running analysis without setting service to the experiment data.

        This uses input xvals as-is.
        """
        analysis = StarkP1SpectAnalysisReturnXvals()

        xvals = np.linspace(-1, 1, 11)
        exp_data = ExperimentData()
        for x in xvals:
            exp_data.add_data({"counts": {"0": 1000, "1": 0}, "metadata": {"xval": x}})
        analysis.run(exp_data, replace_results=True)
        test_xvals = exp_data.analysis_results("xvals").value
        ref_xvals = xvals
        np.testing.assert_array_almost_equal(test_xvals, ref_xvals)

    def test_running_analysis_with_service(self):
        """Test running analysis by setting service to the experiment data.

        This must convert x-axis into frequencies with the Stark coefficients.
        """
        analysis = StarkP1SpectAnalysisReturnXvals()

        xvals = np.linspace(-1, 1, 11)
        exp_data = ExperimentData(
            service=self.service,
            backend=FakeHanoiV2(),
        )
        exp_data.metadata.update({"physical_qubits": [0]})
        for x in xvals:
            exp_data.add_data({"counts": {"0": 1000, "1": 0}, "metadata": {"xval": x}})
        analysis.run(exp_data, replace_results=True)
        test_xvals = exp_data.analysis_results("xvals").value
        ref_xvals = np.where(
            xvals > 0,
            (
                self.coeffs["stark_pos_coef_o1"] * xvals
                + self.coeffs["stark_pos_coef_o2"] * xvals**2
                + self.coeffs["stark_pos_coef_o3"] * xvals**3
                + self.coeffs["stark_ferr"]
            ),
            (
                self.coeffs["stark_neg_coef_o1"] * xvals
                + self.coeffs["stark_neg_coef_o2"] * xvals**2
                + self.coeffs["stark_neg_coef_o3"] * xvals**3
                + self.coeffs["stark_ferr"]
            ),
        )
        np.testing.assert_array_almost_equal(test_xvals, ref_xvals)

    def test_running_analysis_with_user_provided_coeffs(self):
        """Test running analysis by manually providing Stark coefficients.

        This must convert x-axis into frequencies with the provided coefficients.
        """
        analysis = StarkP1SpectAnalysisReturnXvals()
        analysis.set_options(
            stark_coefficients={
                "stark_pos_coef_o1": 0.0,
                "stark_pos_coef_o2": 200e6,
                "stark_pos_coef_o3": 0.0,
                "stark_neg_coef_o1": 0.0,
                "stark_neg_coef_o2": -200e6,
                "stark_neg_coef_o3": 0.0,
                "stark_ferr": 0.0,
            }
        )

        xvals = np.linspace(-1, 1, 11)
        exp_data = ExperimentData()
        for x in xvals:
            exp_data.add_data({"counts": {"0": 1000, "1": 0}, "metadata": {"xval": x}})
        analysis.run(exp_data, replace_results=True)
        test_xvals = exp_data.analysis_results("xvals").value
        ref_xvals = np.where(xvals > 0, 200e6 * xvals**2, -200e6 * xvals**2)
        np.testing.assert_array_almost_equal(test_xvals, ref_xvals)
