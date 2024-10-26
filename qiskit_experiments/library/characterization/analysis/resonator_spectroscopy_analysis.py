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

"""Spectroscopy analysis class for resonators."""

from typing import List, Optional, Tuple
import numpy as np

from qiskit.utils.deprecation import deprecate_func

import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.framework import AnalysisResultData, ExperimentData
from qiskit_experiments.framework.matplotlib import get_non_gui_ax
from qiskit_experiments.data_processing.nodes import ProjectorType
from qiskit_experiments.database_service.device_component import Resonator


class ResonatorSpectroscopyAnalysis(curve.ResonanceAnalysis):
    """Class to analysis resonator spectroscopy."""

    @deprecate_func(
        since="0.8",
        package_name="qiskit-experiments",
        additional_msg=(
            "Due to the deprecation of Qiskit Pulse, experiments and related classses "
            "involving pulse gate calibrations like this one have been deprecated."
        ),
    )
    def __init__(
        self,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

    @classmethod
    def _default_options(cls):
        """Return default analysis options.

        Analysis Options:
            dimensionality_reduction (ProjectorType): Type of the data processor node
                that will reduce the two-dimensional data to one dimension.
            plot_iq_data (bool): Set True to generate IQ plot.
        """
        options = super()._default_options()
        options.dimensionality_reduction = ProjectorType.ABS
        options.result_parameters = [
            curve.ParameterRepr("freq", "res_freq0", "Hz"),
            curve.ParameterRepr("kappa", "kappa", "Hz"),
        ]
        options.plot_iq_data = True
        return options

    def _get_experiment_components(self, experiment_data: ExperimentData):
        """Return resonators as experiment components."""
        return [Resonator(qubit) for qubit in experiment_data.metadata["physical_qubits"]]

    def _run_analysis(
        self, experiment_data: ExperimentData
    ) -> Tuple[List[AnalysisResultData], List["pyplot.Figure"]]:
        """Wrap the analysis to optionally plot the IQ data."""
        analysis_results, figures = super()._run_analysis(experiment_data)

        if self.options.plot_iq_data:
            axis = get_non_gui_ax()
            figure = axis.get_figure()
            # TODO: Move plotting to a new IQPlotter class.
            figure.set_size_inches(*self.plotter.drawer.style["figsize"])

            iqs = []

            for datum in experiment_data.data():
                if "memory" in datum:
                    mem = np.array(datum["memory"])

                    # Average single-shot data.
                    if len(mem.shape) == 3:
                        for idx in range(mem.shape[1]):
                            iqs.append(np.average(mem[:, idx, :], axis=0))
                    else:
                        iqs.append(mem)

            if len(iqs) > 0:
                iqs = np.vstack(iqs)
                axis.scatter(iqs[:, 0], iqs[:, 1], color="b")
                axis.set_xlabel(
                    "In phase [arb. units]", fontsize=self.plotter.drawer.style["axis_label_size"]
                )
                axis.set_ylabel(
                    "Quadrature [arb. units]", fontsize=self.plotter.drawer.style["axis_label_size"]
                )
                axis.tick_params(labelsize=self.plotter.drawer.style["tick_label_size"])
                axis.grid(True)

                figures.append(figure)

        return analysis_results, figures
