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
Readout Angle Analysis class.
"""

from typing import List, Optional
import numpy as np

from qiskit_experiments.framework import BaseAnalysis, AnalysisResultData, Options
from qiskit_experiments.framework.matplotlib import get_non_gui_ax


class ReadoutAngleAnalysis(BaseAnalysis):
    """
    A class to analyze readout angle experiments
    """

    @classmethod
    def _default_options(cls) -> Options:
        """Return default analysis options.

        Analysis Options:
            plot (bool): Set ``True`` to create figure for fit result.
            ax(AxesSubplot): Optional. A matplotlib axis object to draw.
        """
        options = super()._default_options()
        options.plot = True
        options.ax = None
        return options

    def _run_analysis(self, experiment_data):
        angles = []
        radii = []
        centers = []
        for i in range(2):
            center = complex(*experiment_data.data(i)["memory"][0])
            angles.append(np.angle(center))
            radii.append(np.absolute(center))
            centers.append(center)

        angle = (angles[0] + angles[1]) / 2
        if (np.abs(angles[0] - angles[1])) % (2 * np.pi) > np.pi:
            angle += np.pi

        extra_results = {}
        extra_results["readout_angle_0"] = angles[0]
        extra_results["readout_angle_1"] = angles[1]
        extra_results["readout_radius_0"] = radii[0]
        extra_results["readout_radius_1"] = radii[1]

        analysis_results = [
            AnalysisResultData(name="readout_angle", value=angle, extra=extra_results)
        ]

        if self.options.plot:
            ax = self._format_plot(centers, ax=self.options.ax)
            figures = [ax.get_figure()]
        else:
            figures = None

        return analysis_results, figures

    @staticmethod
    def _format_plot(centers: List[complex], ax: Optional["matplotlib.pyplot.AxesSubplot"] = None):
        """Format the readout_angle plot

        Args:
            centers: the two centers of the level 1 measurements for 0 and for 1.
            ax: matplotlib axis to add plot to.

        Returns:
            AxesSubPlot: the matplotlib axes containing the plot.
        """
        largest_extent = (
            np.max([np.max(np.abs(np.real(centers))), np.max(np.abs(np.imag(centers)))]) * 1.1
        )

        ax = get_non_gui_ax()
        ax.plot(np.real(centers[0]), np.imag(centers[0]), "ro", markersize=24)
        ax.plot(np.real(centers[1]), np.imag(centers[1]), "bo", markersize=24)
        ax.set_xlim([-largest_extent, largest_extent])
        ax.set_ylim([-largest_extent, largest_extent])
        ax.set_xlabel("I [arb.]")
        ax.set_ylabel("Q [arb.]")
        ax.set_title("Centroid Positions")
        ax.legend(["0", "1"])
        return ax
