# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Stark P1 spectroscopy experiment analysis.
"""

from typing import List, Tuple

import numpy as np
from uncertainties import unumpy as unp

import qiskit_experiments.data_processing as dp
import qiskit_experiments.visualization as vis
from qiskit_experiments.data_processing.exceptions import DataProcessorError
from qiskit_experiments.framework import BaseAnalysis, ExperimentData, AnalysisResultData, Options


class StarkP1SpectroscopyAnalysis(BaseAnalysis):
    """Analysis class for P1 spectroscopy.

    Because P1 spectral landscape is hardly predictable with the numerical function
    due to random appearance of the TLS notches, this analysis class just
    visualizes the P1 values against qubit frequency.
    """

    @property
    def plotter(self) -> vis.CurvePlotter:
        """Return curve plotter."""
        return self.options.plotter

    @classmethod
    def _default_options(cls) -> Options:
        """Return the default analysis options.

        Experiment Options:
            plotter (Plotter): Plotter to visualize P1 landscape.
            data_processor (DataProcessor): Data processor to compute P1 value.
            x_key (str): Key of the circuit metadata to represent x value.
        """
        p1_plotter = vis.CurvePlotter(vis.MplDrawer())
        p1_plotter.set_figure_options(
            xlabel="Stark shift",
            ylabel="P(1)",
            xval_unit="Hz",
        )

        options = super()._default_options()
        options.update_options(
            plotter=p1_plotter,
            data_processor=dp.DataProcessor("counts", [dp.Probability("1")]),
            x_key="xval",
        )
        return options

    def _run_analysis(
        self,
        experiment_data: ExperimentData,
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:

        x_key = self.options.x_key
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

        self.plotter.set_series_data(
            series_name="stark_p1",
            x_formatted=xdata,
            y_formatted=ydata,
            y_formatted_err=ydata_err,
            x_interp=xdata,
            y_interp=ydata,
        )

        return [], [self.plotter.figure()]
