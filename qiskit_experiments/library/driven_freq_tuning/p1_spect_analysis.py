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
"""P1 spectroscopy analyses."""

from __future__ import annotations

import numpy as np
from uncertainties import unumpy as unp

from qiskit.utils.deprecation import deprecate_func

import qiskit_experiments.data_processing as dp
import qiskit_experiments.visualization as vis
from qiskit_experiments.data_processing.exceptions import DataProcessorError
from qiskit_experiments.framework import BaseAnalysis, ExperimentData, AnalysisResultData, Options
from .coefficients import (
    StarkCoefficients,
    retrieve_coefficients_from_service,
)


class StarkP1SpectAnalysis(BaseAnalysis):
    """Analysis class for StarkP1Spectroscopy.

    # section: overview

        The P1 landscape is hardly predictable because of the random appearance of
        lossy TLS notches, and hence this analysis doesn't provide any
        generic mathematical model to fit the measurement data.
        A developer may subclass this to conduct own analysis.
        The :meth:`StarkP1SpectAnalysis._run_spect_analysis` is a hook method where
        you can define a custom analysis protocol.

        By default, this analysis just visualizes the measured P1 values against Stark tone amplitudes.
        The tone amplitudes can be converted into the amount of Stark shift
        when the calibrated coefficients are provided in the analysis option,
        or the calibration experiment results are available in the result database.

    # section: see_also
        :class:`qiskit_experiments.library.driven_freq_tuning.StarkRamseyXYAmpScan`

    """

    @deprecate_func(
        since="0.8",
        package_name="qiskit-experiments",
        additional_msg=(
            "Due to the deprecation of Qiskit Pulse, experiments and related classses "
            "involving pulse gate calibrations like this one have been deprecated."
        ),
    )
    def __init__(self):
        """Initialize the analysis object."""
        # Pass through to parent. This method is only here to be decorated by
        # deprecate_func
        super().__init__()

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
            stark_coefficients=None,
            x_key="xval",
        )
        options.set_validator("stark_coefficients", StarkCoefficients)

        return options

    # pylint: disable=unused-argument
    def _run_spect_analysis(
        self,
        xdata: np.ndarray,
        ydata: np.ndarray,
        ydata_err: np.ndarray,
    ) -> list[AnalysisResultData]:
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

    def _run_analysis(
        self,
        experiment_data: ExperimentData,
    ) -> tuple[list[AnalysisResultData], list["matplotlib.figure.Figure"]]:

        x_key = self.options.x_key

        # Get calibrated Stark tone coefficients
        if self.options.stark_coefficients is None and experiment_data.service is not None:
            # Get value from service
            stark_coeffs = retrieve_coefficients_from_service(
                service=experiment_data.service,
                backend_name=experiment_data.backend_name,
                qubit=experiment_data.metadata["physical_qubits"][0],
            )
        else:
            stark_coeffs = self.options.stark_coefficients

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
        if isinstance(stark_coeffs, StarkCoefficients):
            xdata = stark_coeffs.convert_amp_to_freq(amps=xdata)
            self.plotter.set_figure_options(
                xlabel="Stark shift",
                xval_unit="Hz",
                xscale="linear",
            )

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
