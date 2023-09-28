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
T1 Analysis class.
"""
from typing import Union, Tuple, List, Dict

import numpy as np
from qiskit_ibm_experiment import IBMExperimentService
from qiskit_ibm_experiment.exceptions import IBMApiError
from uncertainties import unumpy as unp

import qiskit_experiments.curve_analysis as curve
import qiskit_experiments.data_processing as dp
import qiskit_experiments.visualization as vis
from qiskit_experiments.data_processing.exceptions import DataProcessorError
from qiskit_experiments.database_service.device_component import Qubit
from qiskit_experiments.framework import BaseAnalysis, ExperimentData, AnalysisResultData, Options


class T1Analysis(curve.DecayAnalysis):
    """A class to analyze T1 experiments."""

    @classmethod
    def _default_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_options()
        options.plotter.set_figure_options(
            xlabel="Delay",
            ylabel="P(1)",
            xval_unit="s",
        )
        options.result_parameters = [curve.ParameterRepr("tau", "T1", "s")]

        return options

    def _evaluate_quality(self, fit_data: curve.CurveFitResult) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared lower than three
            - absolute amp is within [0.9, 1.1]
            - base is less than 0.1
            - amp error is less than 0.1
            - tau error is less than its value
            - base error is less than 0.1
        """
        amp = fit_data.ufloat_params["amp"]
        tau = fit_data.ufloat_params["tau"]
        base = fit_data.ufloat_params["base"]

        criteria = [
            fit_data.reduced_chisq < 3,
            abs(amp.nominal_value - 1.0) < 0.1,
            abs(base.nominal_value) < 0.1,
            curve.utils.is_error_not_significant(amp, absolute=0.1),
            curve.utils.is_error_not_significant(tau),
            curve.utils.is_error_not_significant(base, absolute=0.1),
        ]

        if all(criteria):
            return "good"

        return "bad"


class T1KerneledAnalysis(curve.DecayAnalysis):
    """A class to analyze T1 experiments with kerneled data."""

    @classmethod
    def _default_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_options()
        options.plotter.set_figure_options(
            xlabel="Delay",
            ylabel="Normalized Projection on the Main Axis",
            xval_unit="s",
        )
        options.result_parameters = [curve.ParameterRepr("tau", "T1", "s")]
        options.normalization = True

        return options

    def _evaluate_quality(self, fit_data: curve.CurveFitResult) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared lower than three
            - absolute amp is within [0.9, 1.1]
            - base is less than 0.1
            - amp error is less than 0.1
            - tau error is less than its value
            - base error is less than 0.1
        """
        amp = fit_data.ufloat_params["amp"]
        tau = fit_data.ufloat_params["tau"]
        base = fit_data.ufloat_params["base"]

        criteria = [
            fit_data.reduced_chisq < 3,
            abs(amp.nominal_value - 1.0) < 0.1,
            abs(base.nominal_value) < 0.1,
            curve.utils.is_error_not_significant(amp, absolute=0.1),
            curve.utils.is_error_not_significant(tau),
            curve.utils.is_error_not_significant(base, absolute=0.1),
        ]

        if all(criteria):
            return "good"

        return "bad"

    def _format_data(
        self,
        curve_data: curve.ScatterTable,
    ) -> curve.ScatterTable:
        """Postprocessing for the processed dataset.

        Args:
            curve_data: Processed dataset created from experiment results.

        Returns:
            Formatted data.
        """
        # check if the SVD decomposition categorized 0 as 1 by calculating the average slope
        diff_y = np.diff(curve_data.yval)
        avg_slope = sum(diff_y) / len(diff_y)
        if avg_slope > 0:
            curve_data.yval = 1 - curve_data.yval
        return super()._format_data(curve_data)


class StarkP1SpectAnalysis(BaseAnalysis):
    """Analysis class for StarkP1Spectroscopy.

    # section: overview

        The P1 landscape is hardly predictable because of the random appearance of
        lossy TLS notches, and hence this analysis doesn't provide any
        generic mathematical model to fit the measurement data.
        A developer may subclass this to conduct own analysis.

        This analysis just visualizes the measured P1 values against Stark tone amplitudes.
        The tone amplitudes can be converted into the amount of Stark shift
        when the calibrated coefficients are provided in the analysis option,
        or the calibration experiment results are available in the result database.

    # section: see_also
        :class:`qiskit_experiments.library.characterization.ramsey_xy.StarkRamseyXYAmpScan`

    """

    stark_coefficients_names = [
        "stark_pos_coef_o1",
        "stark_pos_coef_o2",
        "stark_pos_coef_o3",
        "stark_neg_coef_o1",
        "stark_neg_coef_o2",
        "stark_neg_coef_o3",
        "stark_ferr",
    ]

    @property
    def plotter(self) -> vis.CurvePlotter:
        """Curve plotter instance."""
        return self.options.plotter

    @classmethod
    def _default_options(cls) -> Options:
        """Default analysis options.

        Analysis Options:
            plotter (Plotter): Plotter to visualize P1 landscape.
            data_processor (DataProcessor): Data processor to compute P1 value.
            stark_coefficients (Union[Dict, str]): Dictionary of Stark shift coefficients to
                convert tone amplitudes into amount of Stark shift. This dictionary must include
                all keys defined in :attr:`.StarkP1SpectAnalysis.stark_coefficients_names`,
                which are calibrated with :class:`.StarkRamseyXYAmpScan`.
                Alternatively, it searches for these coefficients in the result database
                when "latest" is set. This requires having the experiment service set in
                the experiment data to analyze.
            x_key (str): Key of the circuit metadata to represent x value.
        """
        options = super()._default_options()

        p1spect_plotter = vis.CurvePlotter(vis.MplDrawer())
        p1spect_plotter.set_figure_options(
            xlabel="Stark amplitude",
            ylabel="P(1)",
            xscale="quadratic",
        )

        options.update_options(
            plotter=p1spect_plotter,
            data_processor=dp.DataProcessor("counts", [dp.Probability("1")]),
            stark_coefficients="latest",
            x_key="xval",
        )
        return options

    # pylint: disable=unused-argument
    def _run_spect_analysis(
        self,
        xdata: np.ndarray,
        ydata: np.ndarray,
        ydata_err: np.ndarray,
    ) -> List[AnalysisResultData]:
        """Run further analysis on the spectroscopy data.

        .. note::
            A subclass can overwrite this method to conduct analysis.

        Args:
            xdata: X values. This is either amplitudes or frequencies.
            ydata: Y values. This is P1 values measured at different Stark tones.
            ydata_err: Sampling error of the Y values.

        Returns:
            A list of analysis results.
        """
        return []

    @classmethod
    def retrieve_coefficients_from_service(
        cls,
        service: IBMExperimentService,
        qubit: int,
        backend: str,
    ) -> Dict:
        """Retrieve stark coefficient dictionary from the experiment service.

        Args:
            service: A valid experiment service instance.
            qubit: Qubit index.
            backend: Name of the backend.

        Returns:
            A dictionary of Stark coefficients to convert amplitude to frequency.
            None value is returned when the dictionary is incomplete.
        """
        out = {}
        try:
            for name in cls.stark_coefficients_names:
                results = service.analysis_results(
                    device_components=[str(Qubit(qubit))],
                    result_type=name,
                    backend_name=backend,
                    sort_by=["creation_datetime:desc"],
                )
                if len(results) == 0:
                    return None
                result_data = getattr(results[0], "result_data")
                out[name] = result_data["value"]
        except (IBMApiError, ValueError, KeyError, AttributeError):
            return None
        return out

    def _convert_axis(
        self,
        xdata: np.ndarray,
        coefficients: Dict[str, float],
    ) -> np.ndarray:
        """A helper method to convert x-axis.

        Args:
            xdata: An array of Stark tone amplitude.
            coefficients: Stark coefficients to convert amplitudes into frequencies.

        Returns:
            An array of amount of Stark shift.
        """
        names = self.stark_coefficients_names  # alias
        positive = np.poly1d([coefficients[names[idx]] for idx in [2, 1, 0, 6]])
        negative = np.poly1d([coefficients[names[idx]] for idx in [5, 4, 3, 6]])

        new_xdata = np.where(xdata > 0, positive(xdata), negative(xdata))
        self.plotter.set_figure_options(
            xlabel="Stark shift",
            xval_unit="Hz",
            xscale="linear",
        )
        return new_xdata

    def _run_analysis(
        self,
        experiment_data: ExperimentData,
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:

        x_key = self.options.x_key

        # Get calibrated Stark tone coefficients
        if self.options.stark_coefficients == "latest" and experiment_data.service is not None:
            # Get value from service
            stark_coeffs = self.retrieve_coefficients_from_service(
                service=experiment_data.service,
                qubit=experiment_data.metadata["physical_qubits"][0],
                backend=experiment_data.backend_name,
            )
        elif isinstance(self.options.stark_coefficients, dict):
            # Get value from experiment options
            missing = set(self.stark_coefficients_names) - self.options.stark_coefficients.keys()
            if any(missing):
                raise KeyError(
                    "Following coefficient data is missing in the "
                    f"'stark_coefficients' dictionary: {missing}."
                )
            stark_coeffs = self.options.stark_coefficients
        else:
            # No calibration is available
            stark_coeffs = None

        # Compute P1 value and sampling error
        data = experiment_data.data()
        try:
            xdata = np.asarray([datum["metadata"][x_key] for datum in data], dtype=float)
        except KeyError as ex:
            raise DataProcessorError(
                f"X value key {x_key} is not defined in circuit metadata."
            ) from ex
        ydata_ufloat = self.options.data_processor(data)
        ydata = unp.nominal_values(ydata_ufloat)
        ydata_err = unp.std_devs(ydata_ufloat)

        # Convert x-axis of amplitudes into Stark shift by consuming calibrated parameters.
        if stark_coeffs:
            xdata = self._convert_axis(xdata, stark_coeffs)

        # Draw figures and create analysis results.
        self.plotter.set_series_data(
            series_name="stark_p1",
            x_formatted=xdata,
            y_formatted=ydata,
            y_formatted_err=ydata_err,
            x_interp=xdata,
            y_interp=ydata,
        )
        analysis_results = self._run_spect_analysis(xdata, ydata, ydata_err)

        return analysis_results, [self.plotter.figure()]
