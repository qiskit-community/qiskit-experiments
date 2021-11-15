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

import numpy as np

from qiskit_experiments.framework import BaseAnalysis, AnalysisResultData


class ReadoutAngleAnalysis(BaseAnalysis):
    """
    A class to analyze readout angle experiments
    """

    # pylint: disable=unused-argument
    def _run_analysis(self, experiment_data, **kwargs):
        center0 = complex(*experiment_data.data(0)["memory"][0])
        center1 = complex(*experiment_data.data(1)["memory"][0])

        angle = (np.angle(center0) + np.angle(center1)) / 2
        if np.abs(np.angle(center0) - np.angle(center1)) > np.pi:
            angle += np.pi

        analysis_results = [AnalysisResultData(name="ReadoutAngle", value=angle)]

        return analysis_results, []
