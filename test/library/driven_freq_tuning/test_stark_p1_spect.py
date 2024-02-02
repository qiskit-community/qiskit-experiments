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

from ddt import ddt, named_data, unpack
import numpy as np
from qiskit import pulse
from qiskit.circuit import QuantumCircuit, Gate
from qiskit.providers import QubitProperties
from qiskit_ibm_runtime.fake_provider import FakeHanoiV2

from qiskit_experiments.framework import ExperimentData, AnalysisResultData
from qiskit_experiments.library import StarkP1Spectroscopy
from qiskit_experiments.library.driven_freq_tuning.p1_spect_analysis import StarkP1SpectAnalysis
from qiskit_experiments.library.driven_freq_tuning.coefficients import StarkCoefficients
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


@ddt
class TestStarkP1Spectroscopy(QiskitExperimentsTestCase):
    """Test case for the Stark P1 Spectroscopy experiment."""

    def test_linear_spaced_parameters(self):
        """Test generating parameters with linear spacing."""
        exp = StarkP1Spectroscopy((0,))
        exp.set_experiment_options(
            min_xval=-1,
            max_xval=1,
            num_xvals=5,
            spacing="linear",
        )
        params = exp.parameters()
        ref = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

        np.testing.assert_array_almost_equal(params, ref)

    def test_quadratic_spaced_parameters(self):
        """Test generating parameters with quadratic spacing."""
        exp = StarkP1Spectroscopy((0,))
        exp.set_experiment_options(
            min_xval=-1,
            max_xval=1,
            num_xvals=5,
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

    def test_raises_scanning_frequency_without_service(self):
        """Test raises error when frequency is set without no coefficients.

        This covers following situations:
        - stark_coefficients options is None
        - backend object doesn't provide experiment service
        """
        exp = StarkP1Spectroscopy((0,), backend=FakeHanoiV2())
        exp.set_experiment_options(
            xvals=[-100e6, -50e6, 0, 50e6, 100e6],
            xval_type="frequency",
        )
        with self.assertRaises(RuntimeError):
            exp.parameters()

    def test_scanning_frequency_with_coeffs(self):
        """Test scanning frequency with manually provided Stark coefficients."""
        coeffs = StarkCoefficients(
            pos_coef_o1=5e6,
            pos_coef_o2=200e6,
            pos_coef_o3=-50e6,
            neg_coef_o1=5e6,
            neg_coef_o2=-180e6,
            neg_coef_o3=-40e6,
            offset=100e3,
        )
        exp = StarkP1Spectroscopy((0,), backend=FakeHanoiV2())

        ref_amps = np.array([-0.50, -0.25, 0.0, 0.25, 0.50], dtype=float)
        test_freqs = coeffs.convert_amp_to_freq(ref_amps)
        exp.set_experiment_options(
            xvals=test_freqs,
            xval_type="frequency",
            stark_coefficients=coeffs,
        )
        params = exp.parameters()
        np.testing.assert_array_almost_equal(params, ref_amps)

    def test_scanning_frequency_around_zero(self):
        """Test scanning frequency around zero."""
        coeffs = StarkCoefficients(
            pos_coef_o1=5e6,
            pos_coef_o2=100e6,
            pos_coef_o3=10e6,
            neg_coef_o1=-5e6,
            neg_coef_o2=-100e6,
            neg_coef_o3=-10e6,
            offset=500e3,
        )
        exp = StarkP1Spectroscopy((0,), backend=FakeHanoiV2())
        exp.set_experiment_options(
            xvals=[0, 500e3],
            xval_type="frequency",
            stark_coefficients=coeffs,
        )
        params = exp.parameters()
        # Frequency offset is 500 kHz and we need negative shift to tune frequency at zero.
        self.assertLess(params[0], 0)

        # Frequency offset is 500 kHz and we don't need tone.
        self.assertAlmostEqual(params[1], 0)

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
            xvals=[-0.5, 0.5],
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

    def test_running_analysis_without_service(self):
        """Test running analysis without setting service to the experiment data.

        This uses input xvals as-is.
        """
        analysis = StarkP1SpectAnalysisReturnXvals()

        xvals = np.linspace(-1, 1, 11)
        ref_xvals = xvals
        exp_data = ExperimentData()
        for x in xvals:
            exp_data.add_data({"counts": {"0": 1000, "1": 0}, "metadata": {"xval": x}})
        analysis.run(exp_data, replace_results=True)
        test_xvals = exp_data.analysis_results("xvals").value
        np.testing.assert_array_almost_equal(test_xvals, ref_xvals)

    @named_data(
        ["ordinary", 5e6, 200e6, -50e6, 5e6, -180e6, -40e6, 100e3],
        ["asymmetric_inflection_1st_ord", 10e6, 200e6, -20e6, -50e6, -180e6, -20e6, -10e6],
        ["inflection_3st_ord", 10e6, 200e6, -80e6, 80e6, -180e6, -200e6, 100e3],
    )
    @unpack
    def test_running_analysis_with_service(self, po1, po2, po3, no1, no2, no3, ferr):
        """Test running analysis by setting service to the experiment data.

        This must convert x-axis into frequencies with the Stark coefficients.
        """
        mock_experiment_id = "6453f3d1-04ef-4e3b-82c6-1a92e3e066eb"
        mock_result_id = "d067ae34-96db-4e8e-adc8-030305d3d404"
        mock_backend = FakeHanoiV2().name

        coeffs = StarkCoefficients(
            pos_coef_o1=po1,
            pos_coef_o2=po2,
            pos_coef_o3=po3,
            neg_coef_o1=no1,
            neg_coef_o2=no2,
            neg_coef_o3=no3,
            offset=ferr,
        )

        service = FakeService()
        service.create_experiment(
            experiment_type="StarkRamseyXYAmpScan",
            backend_name=mock_backend,
            experiment_id=mock_experiment_id,
        )
        service.create_analysis_result(
            experiment_id=mock_experiment_id,
            result_data={"value": coeffs},
            result_type="stark_coefficients",
            device_components=["Q0"],
            tags=[],
            quality="Good",
            verified=False,
            result_id=mock_result_id,
        )

        analysis = StarkP1SpectAnalysisReturnXvals()

        xvals = np.linspace(-1, 1, 11)
        ref_fvals = coeffs.convert_amp_to_freq(xvals)

        exp_data = ExperimentData(
            service=service,
            backend=FakeHanoiV2(),
        )
        exp_data.metadata.update({"physical_qubits": [0]})
        for x in xvals:
            exp_data.add_data({"counts": {"0": 1000, "1": 0}, "metadata": {"xval": x}})
        analysis.run(exp_data, replace_results=True).block_for_results()
        test_fvals = exp_data.analysis_results("xvals").value
        np.testing.assert_array_almost_equal(test_fvals, ref_fvals)

    def test_running_analysis_with_user_provided_coeffs(self):
        """Test running analysis by manually providing Stark coefficients.

        This must convert x-axis into frequencies with the provided coefficients.
        This is just a difference of API from the test_running_analysis_with_service.
        Data driven test is omitted here.
        """
        coeffs = StarkCoefficients(
            pos_coef_o1=5e6,
            pos_coef_o2=200e6,
            pos_coef_o3=-50e6,
            neg_coef_o1=5e6,
            neg_coef_o2=-180e6,
            neg_coef_o3=-40e6,
            offset=100e3,
        )

        analysis = StarkP1SpectAnalysisReturnXvals()
        analysis.set_options(stark_coefficients=coeffs)

        xvals = np.linspace(-1, 1, 11)
        ref_fvals = coeffs.convert_amp_to_freq(xvals)

        exp_data = ExperimentData()
        for x in xvals:
            exp_data.add_data({"counts": {"0": 1000, "1": 0}, "metadata": {"xval": x}})
        analysis.run(exp_data, replace_results=True).block_for_results()
        test_fvals = exp_data.analysis_results("xvals").value
        np.testing.assert_array_almost_equal(test_fvals, ref_fvals)
